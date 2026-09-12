"""Serializable online preprocessing and model pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .features import build_features, coerce_input


class InferencePipeline:
    """Keep validation, feature building and the fitted estimator together."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, records: Any) -> np.ndarray:
        """Validate raw canonical records and return non-negative predictions."""

        frame = coerce_input(records)
        features = build_features(frame)
        predictions = np.asarray(self.model.predict(features), dtype="float64")
        predictions = np.where(frame["is_functioning"].to_numpy(), predictions, 0.0)
        return np.clip(predictions, 0.0, None)
