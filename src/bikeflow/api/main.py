"""BikeFlow HTTP API."""

import json
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from bikeflow.api.dependencies import get_predictor
from bikeflow.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
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
) -> PredictionResponse:
    """Predict hourly rentals with the configured inference implementation."""

    try:
        prediction = predictor.predict(request.to_features())
    except FeatureValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    logger.info("prediction_completed model_version=%s", predictor.model_version)
    return PredictionResponse(
        prediction_time=request.prediction_time,
        predicted_rentals=prediction,
        model_version=predictor.model_version,
    )
