"""Adapter between the API's `Predictor` protocol and the trained ML artifact.

This is the only place where the API contract meets the training pipeline. It
translates the request field names used by the service into the canonical
feature names used during training, then defers every transformation to
`bikeflow.ml.features` — the same module the training run used, so online and
offline preprocessing cannot drift apart.

Training, drift detection and retraining stay outside this boundary, as the
model/API contract requires.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bikeflow.ml.features import api_input_contract
from bikeflow.model.protocol import FeatureValue

#: API request field -> canonical training feature name.
FIELD_ALIASES: dict[str, str] = {
    public_name: spec["canonical_name"] for public_name, spec in api_input_contract().items()
}

#: Fields whose names already match the training pipeline.
PASSTHROUGH: frozenset[str] = frozenset({"hour", "season", "day_of_week", "date"})


def to_canonical_row(features: Mapping[str, FeatureValue]) -> dict[str, FeatureValue]:
    """Rename API fields to the names the trained model expects."""
    row: dict[str, FeatureValue] = {}
    unknown: list[str] = []

    for key, value in features.items():
        if key in FIELD_ALIASES:
            row[FIELD_ALIASES[key]] = value
        elif key in PASSTHROUGH:
            row[key] = value
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"Unknown feature field(s) for the trained model: {sorted(unknown)}. "
            f"Known: {sorted(set(FIELD_ALIASES) | set(PASSTHROUGH))}"
        )
    return row


class BikeflowPredictor:
    """Serve predictions from a trained BikeFlow artifact.

    Implements the `Predictor` protocol: one feature mapping in, one
    non-negative float out, plus an immutable artifact version.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = model_path
        self._predictor: Any | None = None
        self._model_version: str | None = None

    def _load(self) -> None:
        """Delay disk I/O until after FastAPI has validated the request body."""

        if self._predictor is not None:
            return
        from bikeflow.ml.inference import Predictor as MLPredictor

        self._predictor = MLPredictor.load(self._model_path)
        self._model_version = str(self._predictor.metadata["model_version"])

    @property
    def model_version(self) -> str:
        """Identify the exact inference artifact."""
        self._load()
        assert self._model_version is not None
        return self._model_version

    def predict(self, features: Mapping[str, FeatureValue]) -> float:
        """Return a non-negative demand prediction for one feature row."""
        self._load()
        assert self._predictor is not None
        prediction = self._predictor.predict(to_canonical_row(features))
        return float(prediction[0])
