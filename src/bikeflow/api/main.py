"""BikeFlow HTTP API."""

import json
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status

from bikeflow.api.dependencies import get_predictor, get_store
from bikeflow.api.schemas import (
    ActualRequest,
    HealthResponse,
    PredictionRecord,
    PredictionRequest,
    PredictionResponse,
)
from bikeflow.api.storage import PredictionStore, StoredPrediction
from bikeflow.config import get_settings
from bikeflow.ml.features import FeatureValidationError
from bikeflow.model.protocol import Predictor


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
