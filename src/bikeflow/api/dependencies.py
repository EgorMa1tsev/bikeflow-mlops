"""FastAPI dependencies for replaceable runtime components."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from bikeflow.api.retraining import RetrainingManager
from bikeflow.api.storage import PredictionStore
from bikeflow.config import get_settings
from bikeflow.model.adapter import BikeflowPredictor
from bikeflow.model.protocol import Predictor


@lru_cache
def get_predictor() -> Predictor:
    """Provide the current predictor implementation."""

    settings = get_settings()
    return BikeflowPredictor(settings.model_path, model_uri=settings.model_uri)


@lru_cache
def get_store() -> PredictionStore:
    """Provide the prediction journal."""

    return PredictionStore(get_settings().db_path)


def get_drift_report_path() -> Path:
    """Where the latest Evidently drift report is written."""

    return get_settings().monitoring_dir / "drift_report.html"


def retraining_job(trigger: str) -> dict[str, Any]:
    """Retrain on the journal; on promotion, make the API reload the new champion."""

    from bikeflow.ml.training.retrain import retrain

    store = get_store()
    champion = get_predictor().ml_predictor()
    result = retrain(store.training_rows(), champion, trigger, get_settings().retraining_dir)
    payload = result.to_dict()
    store.add_retraining(payload)
    if result.promoted:
        # The alias now points at the challenger; the next request loads it.
        get_predictor.cache_clear()
    return payload


@lru_cache
def get_retraining_manager() -> RetrainingManager:
    """Retraining is only possible when the model is served from the MLflow registry."""

    return RetrainingManager(retraining_job, enabled=get_settings().model_uri is not None)
