"""Prediction interfaces and implementations."""

from bikeflow.model.protocol import FeatureValue, Predictor
from bikeflow.model.stub import StubPredictor

__all__ = ["FeatureValue", "Predictor", "StubPredictor"]
