"""Evaluation metrics.

MAPE is deliberately absent: the series contains hours with very small demand,
where dividing by the actual value makes the metric explode and stop being
comparable between splits. WAPE is used instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype="float64").ravel()
    pred = np.asarray(y_pred, dtype="float64").ravel()
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        raise ValueError("Cannot compute metrics on an empty sample.")
    return true, pred


def mae(y_true, y_pred) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.abs(true - pred).mean())


def rmse(y_true, y_pred) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.sqrt(((true - pred) ** 2).mean()))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|y-p| / sum|y|."""
    true, pred = _arrays(y_true, y_pred)
    denominator = np.abs(true).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(true - pred).sum() / denominator)


def r2(y_true, y_pred) -> float:
    true, pred = _arrays(y_true, y_pred)
    total = ((true - true.mean()) ** 2).sum()
    if total == 0:
        return float("nan")
    return float(1.0 - ((true - pred) ** 2).sum() / total)


def weighted_cost_error(
    y_true, y_pred, under_weight: float | None = None, over_weight: float | None = None
) -> float:
    """Business metric: underforecasting demand costs more than overforecasting.

    Underforecasting leaves riders without bikes (lost revenue and trust);
    overforecasting only wastes redistribution effort.
    """
    cfg = load_config()["business_metric"]
    w_under = cfg["under_weight"] if under_weight is None else under_weight
    w_over = cfg["over_weight"] if over_weight is None else over_weight

    true, pred = _arrays(y_true, y_pred)
    under = np.clip(true - pred, 0.0, None)
    over = np.clip(pred - true, 0.0, None)
    return float((w_under * under + w_over * over).mean())


def evaluate(y_true, y_pred) -> dict[str, float]:
    """All headline metrics at once."""
    true, pred = _arrays(y_true, y_pred)
    return {
        "n": int(true.size),
        "mae": mae(true, pred),
        "wape": wape(true, pred),
        "rmse": rmse(true, pred),
        "r2": r2(true, pred),
        "wce": weighted_cost_error(true, pred),
        "mean_actual": float(true.mean()),
        "mean_predicted": float(pred.mean()),
        "bias": float((pred - true).mean()),
    }


def evaluate_by_slice(frame: pd.DataFrame, y_true, y_pred, by: str) -> pd.DataFrame:
    """Metrics per level of `by` (a column of `frame`), sorted by that level."""
    true, pred = _arrays(y_true, y_pred)
    if len(frame) != true.size:
        raise ValueError(f"frame has {len(frame)} rows but {true.size} predictions were given.")

    work = frame.reset_index(drop=True)
    if by not in work.columns:
        raise KeyError(f"Column '{by}' not found; available: {list(work.columns)}")

    rows = []
    for level, index in work.groupby(by, dropna=False, observed=True).groups.items():
        positions = work.index.get_indexer(index)
        row = {by: level}
        row.update(evaluate(true[positions], pred[positions]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def rain_flag(frame: pd.DataFrame) -> pd.Series:
    """Convenience slice: hours with any precipitation."""
    return (frame["rainfall"] > 0) | (frame["snowfall"] > 0)
