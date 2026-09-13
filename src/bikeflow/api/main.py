"""BikeFlow HTTP API."""

import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse

from bikeflow.api.dependencies import get_drift_report_path, get_predictor, get_store
from bikeflow.api.schemas import (
    ActualRequest,
    DriftCheckResponse,
    HealthResponse,
    PredictionRecord,
    PredictionRequest,
    PredictionResponse,
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

app = FastAPI(
    title="BikeFlow API",
    version="0.1.0",
    description="Development API for hourly bicycle-rental demand predictions.",
)


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health() -> HealthResponse:
    """Report that the HTTP process is alive."""

    return HealthResponse(status="ok")


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
    return _to_record(stored)


@app.get("/predictions", response_model=list[PredictionRecord], status_code=status.HTTP_200_OK)
def recent_predictions(
    store: Annotated[PredictionStore, Depends(get_store)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[PredictionRecord]:
    """The most recent predictions, newest first, with actual demand when known."""

    return [_to_record(stored) for stored in store.recent(limit)]


@app.post("/drift/check", response_model=DriftCheckResponse, status_code=status.HTTP_200_OK)
def run_drift_check(
    predictor: Annotated[Predictor, Depends(get_predictor)],
    store: Annotated[PredictionStore, Depends(get_store)],
    report_path: Annotated[Path, Depends(get_drift_report_path)],
) -> DriftCheckResponse:
    """Check data, target and concept drift over the latest journal window.

    Writes an Evidently HTML report available at `GET /drift/report`.
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
    logger.info(
        "drift_checked id=%s data=%s target=%s concept=%s mae_ratio=%.2f",
        check_id,
        result.data_drift,
        result.target_drift,
        result.concept_drift,
        result.mae_ratio,
    )
    return DriftCheckResponse(check_id=check_id, **payload)


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
