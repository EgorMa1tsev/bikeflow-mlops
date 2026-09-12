"""Rolling-origin cross-validation for model selection.

A single validation window is a fragile basis for choosing a model: whichever
model happens to suit those two months wins, and the choice does not generalise.
Here every fold trains on all data up to a cutoff and scores on the following
window, so time order is respected throughout, and the selection signal is an
average over several periods instead of one.

The test split is never read. `assert_excludes_test` enforces that.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import load_config
from ..data.split import split_bounds
from ..features import TARGET, build_features
from ..metrics import mae, wape
from ..models.baseline import SeasonalMedianBaseline
from ..models.hgb import HGBModel
from ..models.torch_mlp import TorchMLPRegressor


class CVConfigError(ValueError):
    """Raised when the configured folds would leak the test period."""


def build_candidates() -> dict[str, Any]:
    """Fresh, unfitted instances of every model that may become champion."""
    return {
        "seasonal_median": SeasonalMedianBaseline(),
        "hgb": HGBModel(),
        "mlp_onehot": TorchMLPRegressor(encoding="onehot"),
        "mlp_embedding": TorchMLPRegressor(encoding="embedding"),
    }


def fold_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    folds = load_config()["selection"]["folds"]
    return [
        (pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)) for start, end in folds
    ]


def assert_excludes_test(windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Timestamp:
    """Guarantee no fold reaches into the test period. Returns the hard cutoff."""
    test_start = split_bounds()["test"][0]
    for start, end in windows:
        if end >= test_start:
            raise CVConfigError(
                f"Fold {start.date()}..{end.date()} overlaps the test split, which "
                f"starts {test_start.date()}. Model selection must never read test."
            )
    return test_start


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    return build_features(frame), frame[TARGET].to_numpy(dtype="float64")


def run_cv(frame: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    """Score every candidate on every fold. One row per (fold, model)."""
    cfg = load_config()["selection"]
    windows = fold_windows()
    test_start = assert_excludes_test(windows)

    if frame is None:
        from ..data.preprocess import load_processed

        frame = load_processed()

    usable = frame[frame["is_functioning"] & (frame["timestamp"] < test_start)]
    usable = usable.reset_index(drop=True)
    inner = pd.Timedelta(weeks=int(cfg["inner_weeks"]))

    rows = []
    for number, (start, end) in enumerate(windows, 1):
        outer = usable[usable["timestamp"].between(start, end)]
        history = usable[usable["timestamp"] < start]
        cut = start - inner
        fit_part = history[history["timestamp"] < cut]
        eval_part = history[history["timestamp"] >= cut]

        if outer.empty or fit_part.empty or eval_part.empty:
            raise CVConfigError(
                f"Fold {number} ({start.date()}..{end.date()}) leaves an empty "
                "training, early-stopping or scoring window."
            )

        if verbose:
            print(
                f"[cv] fold {number}: fit<{cut.date()} ({len(fit_part)}), "
                f"stop<{start.date()} ({len(eval_part)}), "
                f"score {start.date()}..{end.date()} ({len(outer)})"
            )

        x_fit, y_fit = _xy(fit_part)
        x_stop, y_stop = _xy(eval_part)
        x_out, y_out = _xy(outer)

        for name, model in build_candidates().items():
            if isinstance(model, SeasonalMedianBaseline):
                model.fit(x_fit, y_fit)
            else:
                model.fit(x_fit, y_fit, x_stop, y_stop)
            predicted = model.predict(x_out)
            rows.append(
                {
                    "fold": number,
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "model": name,
                    "n": len(outer),
                    "mae": mae(y_out, predicted),
                    "wape": wape(y_out, predicted),
                }
            )

    return pd.DataFrame(rows)


def summarise_cv(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and worst fold per model, ordered best first."""
    metric = load_config()["selection"]["metric"]
    table = (
        scores.groupby("model", observed=True)[metric]
        .agg(mean_score="mean", worst_score="max", best_score="min")
        .reset_index()
        .sort_values("mean_score")
        .reset_index(drop=True)
    )
    table.insert(1, "metric", metric)
    return table


def choose_by_cv(scores: pd.DataFrame) -> str:
    """Champion = best mean score across folds, over every candidate."""
    return str(summarise_cv(scores).iloc[0]["model"])
