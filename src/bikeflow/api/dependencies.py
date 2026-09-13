"""FastAPI dependencies for replaceable runtime components."""

from functools import lru_cache
from pathlib import Path

from bikeflow.api.storage import PredictionStore
from bikeflow.config import get_settings
from bikeflow.model.adapter import BikeflowPredictor
from bikeflow.model.protocol import Predictor


@lru_cache
def get_predictor() -> Predictor:
    """Provide the current predictor implementation."""

    return BikeflowPredictor(get_settings().model_path)


@lru_cache
def get_store() -> PredictionStore:
    """Provide the prediction journal."""

    return PredictionStore(get_settings().db_path)


def get_drift_report_path() -> Path:
    """Where the latest Evidently drift report is written."""

    return get_settings().monitoring_dir / "drift_report.html"
