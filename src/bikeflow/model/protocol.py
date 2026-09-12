"""Small inference-only boundary between the API and a model artifact."""

from collections.abc import Mapping
from typing import Protocol, TypeAlias, runtime_checkable

FeatureValue: TypeAlias = float | int | bool | str


@runtime_checkable
class Predictor(Protocol):
    """Predict demand for one feature row and expose the artifact version."""

    @property
    def model_version(self) -> str:
        """Return the immutable model artifact version."""
        ...

    def predict(self, features: Mapping[str, FeatureValue]) -> float:
        """Return a non-negative demand prediction for one feature row."""
        ...
