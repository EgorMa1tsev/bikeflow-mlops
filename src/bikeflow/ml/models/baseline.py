"""Seasonal median baseline: the bar every real model must clear."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config


class SeasonalMedianBaseline:
    """Predict the median demand seen for the same (hour, day_of_week) in training.

    Falls back through progressively coarser keys so an unseen combination still
    produces a sensible number instead of NaN:
        (hour, day_of_week) -> hour -> global median.
    """

    kind = "seasonal_median"

    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys or list(load_config()["models"]["baseline"]["keys"])
        self.table_: pd.Series | None = None
        self.hour_table_: pd.Series | None = None
        self.global_: float | None = None

    def fit(self, features: pd.DataFrame, y) -> SeasonalMedianBaseline:
        frame = features[self.keys].copy()
        frame["_y"] = np.asarray(y, dtype="float64")
        self.table_ = frame.groupby(self.keys, observed=True)["_y"].median()
        self.hour_table_ = frame.groupby("hour", observed=True)["_y"].median()
        self.global_ = float(frame["_y"].median())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.table_ is None:
            raise RuntimeError("Baseline is not fitted.")

        index = pd.MultiIndex.from_frame(features[self.keys])
        primary = self.table_.reindex(index).to_numpy(dtype="float64")

        fallback_hour = self.hour_table_.reindex(features["hour"]).to_numpy(dtype="float64")
        predictions = np.where(np.isnan(primary), fallback_hour, primary)
        predictions = np.where(np.isnan(predictions), self.global_, predictions)
        return np.clip(predictions, 0.0, None)
