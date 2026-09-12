"""Prediction interface for one or many observations.

The model is not autonomous at this stage: nothing here schedules, serves or
retrains anything. It is called explicitly, either from Python or from the CLI.

    from bikeflow.inference import Predictor
    p = Predictor.load("models/model.joblib")
    p.predict({...})           # one observation
    p.predict([{...}, {...}])  # a batch
    p.predict(dataframe)       # a frame
    p.predict("hours.csv")     # a csv / parquet file
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .models.registry import load_bundle


class Predictor:
    """A loaded model artifact plus the shared preprocessing in front of it."""

    def __init__(self, bundle: dict[str, Any], source: Path | None = None) -> None:
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.model = self.pipeline.model
        self.kind = bundle["kind"]
        self.metadata = bundle["metadata"]
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> Predictor:
        target = path or load_config()["paths"]["production_model"]
        return cls(load_bundle(target), Path(target))

    def predict(self, records: Any, return_frame: bool = False):
        """Predict hourly demand for one or several observations.

        Returns a float ndarray by default, or the input frame with a
        `predicted_demand` column when `return_frame` is set.
        """
        predictions = self.pipeline.predict(records)

        if return_frame:
            from .features import coerce_input

            out = coerce_input(records)
            out["predicted_demand"] = predictions
            return out
        return predictions

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        trained = self.metadata.get("trained_at", "unknown")
        return f"<Predictor kind={self.kind} trained_at={trained}>"


_CACHE: dict[str, Predictor] = {}


def predict(records: Any, model_path: str | Path | None = None, return_frame: bool = False):
    """Module-level shortcut that reuses a cached Predictor."""
    key = str(model_path or load_config()["paths"]["production_model"])
    if key not in _CACHE:
        _CACHE[key] = Predictor.load(key)
    return _CACHE[key].predict(records, return_frame=return_frame)


def predict_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score a whole csv/parquet file and optionally write the result."""
    predictor = Predictor.load(model_path)
    result = predictor.predict(str(input_path), return_frame=True)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        print(f"[predict] {len(result)} row(s) -> {destination}")
    return result
