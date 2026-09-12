"""FastAPI dependencies for replaceable runtime components."""

from functools import lru_cache

from bikeflow.config import get_settings
from bikeflow.model.adapter import BikeflowPredictor
from bikeflow.model.protocol import Predictor


@lru_cache
def get_predictor() -> Predictor:
    """Provide the current predictor implementation."""

    return BikeflowPredictor(get_settings().model_path)
