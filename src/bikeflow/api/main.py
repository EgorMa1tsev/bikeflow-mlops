"""BikeFlow HTTP API."""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from bikeflow.api import metrics
from bikeflow.api.dependencies import (
    get_drift_report_path,
    get_predictor,
    get_retraining_manager,
    get_store,
)
from bikeflow.api.retraining import RetrainingManager
from bikeflow.api.schemas import (
    ActualRequest,
    DriftCheckResponse,
    HealthResponse,
    ModelInfo,
    PredictionRecord,
    PredictionRequest,
    PredictionResponse,
    RetrainingStatus,
)
from bikeflow.api.storage import PredictionStore, StoredPrediction
from bikeflow.config import get_settings
from bikeflow.ml.config import load_config
from bikeflow.ml.features import FeatureValidationError
from bikeflow.model.protocol import Predictor
from bikeflow.monitoring.drift import InsufficientDataError, check_drift, journal_to_frame


class JsonFormatter(logging.Formatter):
    """Render application log records as JSON without request payloads."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def configure_logging() -> None:
    """Configure the BikeFlow logger once."""

    logger = logging.getLogger("bikeflow")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(get_settings().log_level.upper())
    logger.propagate = False


configure_logging()
logger = logging.getLogger("bikeflow.api")
SEOUL = ZoneInfo("Asia/Seoul")

app = FastAPI(
    title="BikeFlow API",
    version="0.1.0",
    description="Development API for hourly bicycle-rental demand predictions.",
)


@app.middleware("http")
async def count_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Count every request and time it, labelled by route rather than by URL."""

    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    if path != "/metrics":
        metrics.HTTP_REQUESTS.labels(request.method, path, response.status_code).inc()
        metrics.HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - started)
    return response


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Metrics for Prometheus to scrape."""

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health() -> HealthResponse:
    """Report that the HTTP process is alive."""

    return HealthResponse(status="ok")


@app.get("/model", response_model=ModelInfo, status_code=status.HTTP_200_OK)
def model_info(
    predictor: Annotated[Predictor, Depends(get_predictor)],
    retraining: Annotated[RetrainingManager, Depends(get_retraining_manager)],
) -> ModelInfo:
    """Which model is being served, and the validation MAE it was accepted with."""

    reference_source = getattr(predictor, "monitoring_reference", None)
    _, reference_mae = reference_source() if reference_source else (None, None)
    return ModelInfo(
        model_version=predictor.model_version,
        source="registry" if get_settings().model_uri else "file",
        reference_mae=reference_mae,
        retraining_enabled=retraining.enabled,
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(
    request: PredictionRequest,
    predictor: Annotated[Predictor, Depends(get_predictor)],
    store: Annotated[PredictionStore, Depends(get_store)],
) -> PredictionResponse:
    """Predict hourly rentals and record the prediction in the journal."""

    features = request.to_features()
    try:
        prediction = predictor.predict(features)
    except FeatureValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    prediction_id = store.add(
        request.prediction_time, features, prediction, predictor.model_version
    )
    logger.info(
        "prediction_completed id=%s model_version=%s", prediction_id, predictor.model_version
    )
    metrics.PREDICTIONS.labels(predictor.model_version).inc()
    metrics.PREDICTED_RENTALS.observe(prediction)
    return PredictionResponse(
        prediction_id=prediction_id,
        prediction_time=request.prediction_time,
        predicted_rentals=prediction,
        model_version=predictor.model_version,
    )


def _to_record(stored: StoredPrediction) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=stored.id,
        created_at=stored.created_at,
        prediction_time=stored.prediction_time,
        features=stored.features,
        predicted_rentals=stored.predicted_rentals,
        model_version=stored.model_version,
        actual_rentals=stored.actual_rentals,
        absolute_error=stored.absolute_error,
    )


@app.post(
    "/predictions/{prediction_id}/actual",
    response_model=PredictionRecord,
    status_code=status.HTTP_200_OK,
)
def record_actual(
    prediction_id: int,
    request: ActualRequest,
    store: Annotated[PredictionStore, Depends(get_store)],
) -> PredictionRecord:
    """Report the demand actually observed for a previously predicted hour."""

    stored = store.record_actual(prediction_id, request.actual_rentals)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found.",
        )
    metrics.ACTUALS.inc()
    if stored.absolute_error is not None:
        metrics.ABSOLUTE_ERROR.observe(stored.absolute_error)
    return _to_record(stored)


@app.get("/predictions", response_model=list[PredictionRecord], status_code=status.HTTP_200_OK)
def recent_predictions(
    store: Annotated[PredictionStore, Depends(get_store)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[PredictionRecord]:
    """The most recent predictions, newest first, with actual demand when known."""

    return [_to_record(stored) for stored in store.recent(limit)]


def _cooldown_over(store: PredictionStore, window_end: datetime) -> bool:
    """True once `retraining.cooldown_hours` of new data followed the last retraining.

    Measured in data time, not wall time, so replayed traffic behaves like live.
    """
    last = store.latest_retraining()
    if last is None:
        return True
    last_end = datetime.fromisoformat(last["holdout_end"])
    current = window_end.astimezone(SEOUL).replace(tzinfo=None)
    return current >= last_end + timedelta(hours=load_config()["retraining"]["cooldown_hours"])


@app.post("/drift/check", response_model=DriftCheckResponse, status_code=status.HTTP_200_OK)
def run_drift_check(
    predictor: Annotated[Predictor, Depends(get_predictor)],
    store: Annotated[PredictionStore, Depends(get_store)],
    report_path: Annotated[Path, Depends(get_drift_report_path)],
    retraining: Annotated[RetrainingManager, Depends(get_retraining_manager)],
) -> DriftCheckResponse:
    """Check data, target and concept drift over the latest journal window.

    Writes an Evidently HTML report available at `GET /drift/report`. Concept drift
    starts an automatic retraining when that is enabled.
    """

    reference_source = getattr(predictor, "monitoring_reference", None)
    reference, reference_mae = reference_source() if reference_source else (None, None)
    if reference is None or reference_mae is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The served model has no drift reference. Retrain it to enable monitoring.",
        )

    window = store.monitoring_window(
        predictor.model_version, int(load_config()["monitoring"]["window_hours"])
    )
    try:
        result = check_drift(
            journal_to_frame(window),
            reference,
            reference_mae,
            predictor.model_version,
            report_path=report_path,
        )
    except InsufficientDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    payload = result.to_dict()
    check_id = store.add_drift_check(payload)
    metrics.record_drift_check(payload)
    logger.info(
        "drift_checked id=%s data=%s target=%s concept=%s mae_ratio=%.2f",
        check_id,
        result.data_drift,
        result.target_drift,
        result.concept_drift,
        result.mae_ratio,
    )

    retraining_started = False
    if (
        result.concept_drift
        and retraining.enabled
        and load_config()["retraining"]["auto"]
        and _cooldown_over(store, result.window_end)
    ):
        retraining_started = retraining.start("drift")
    return DriftCheckResponse(check_id=check_id, retraining_started=retraining_started, **payload)


@app.post("/retrain", response_model=RetrainingStatus, status_code=status.HTTP_202_ACCEPTED)
def start_retraining(
    retraining: Annotated[RetrainingManager, Depends(get_retraining_manager)],
) -> RetrainingStatus:
    """Start retraining in the background; poll `GET /retrain/status` for the outcome."""

    if not retraining.enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Retraining needs the model to be served from the MLflow registry: "
                "set BIKEFLOW_MODEL_URI, e.g. models:/bikeflow-demand@champion."
            ),
        )
    if not retraining.start("manual"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Retraining is already running."
        )
    return RetrainingStatus(**retraining.status())


@app.get("/retrain/status", response_model=RetrainingStatus, status_code=status.HTTP_200_OK)
def retraining_status(
    retraining: Annotated[RetrainingManager, Depends(get_retraining_manager)],
    store: Annotated[PredictionStore, Depends(get_store)],
) -> RetrainingStatus:
    """The running retraining, or the last finished one if none is running."""

    current = retraining.status()
    if current["state"] == "idle":
        current["result"] = store.latest_retraining()
    return RetrainingStatus(**current)


@app.get("/drift/latest", response_model=DriftCheckResponse, status_code=status.HTTP_200_OK)
def latest_drift_check(
    store: Annotated[PredictionStore, Depends(get_store)],
) -> DriftCheckResponse:
    """The most recent drift check."""

    latest = store.latest_drift_check()
    if latest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No drift check yet.")
    return DriftCheckResponse(**latest)


@app.get("/drift/report", response_class=FileResponse, status_code=status.HTTP_200_OK)
def drift_report(report_path: Annotated[Path, Depends(get_drift_report_path)]) -> FileResponse:
    """The latest Evidently drift report as an HTML page."""

    if not report_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No drift report yet.")
    return FileResponse(report_path, media_type="text/html")
