"""Data, target and concept drift of the served model over the prediction journal.

Reference: the validation split together with the served model's own predictions,
stored inside the model artifact at training time. Current: the latest
`window_hours` of journal entries that already have an actual demand, served by
the same model version.

- Data drift: weather features compared with Evidently (normed Wasserstein).
- Target drift: the distribution of actual demand, compared the same way.
- Concept drift: the model's error grows. Window MAE is divided by the validation
  MAE; the threshold comes from the spread of that ratio across validation weeks.

Data and target drift are informative here: weather and demand are seasonal, so a
week of one season always differs from the reference period. Concept drift is
the signal that the model itself has degraded and should be retrained.

Evidently is imported inside functions, so training code can build a reference
without loading it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bikeflow.ml.config import load_config
from bikeflow.ml.features import TARGET, api_input_contract

WEATHER = [
    "temperature",
    "humidity",
    "wind_speed",
    "visibility",
    "dew_point",
    "solar_radiation",
    "rainfall",
    "snowfall",
]
PREDICTED = "predicted"
REFERENCE_COLUMNS = [*WEATHER, "hour", TARGET, PREDICTED]

_THRESHOLD = re.compile(r"threshold=([0-9.]+)")


class InsufficientDataError(ValueError):
    """Too few journal entries with an actual demand to judge drift."""


@dataclass(frozen=True)
class DriftResult:
    checked_at: datetime
    model_version: str
    window_start: datetime
    window_end: datetime
    rows: int
    data_drift: bool
    drifted_feature_share: float
    drifted_features: list[str]
    target_drift: bool
    target_drift_score: float
    concept_drift: bool
    current_mae: float
    reference_mae: float
    mae_ratio: float
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("checked_at", "window_start", "window_end"):
            data[key] = data[key].isoformat()
        return data


def build_reference(frame: pd.DataFrame, predictions: Any) -> pd.DataFrame:
    """Validation rows with the model's predictions, in the monitoring layout."""
    reference = frame[[*WEATHER, "hour", TARGET]].reset_index(drop=True).astype("float64")
    reference[PREDICTED] = np.asarray(predictions, dtype="float64")
    return reference


def journal_to_frame(entries: Iterable[Any]) -> pd.DataFrame:
    """Journal entries with a known actual demand for operating hours.

    Entries store request fields under their public API names; they are mapped
    back to the canonical names the reference uses. Closed hours are dropped,
    as they are from the reference: their demand is zero by rule, not by model.
    """
    canonical = {public: spec["canonical_name"] for public, spec in api_input_contract().items()}
    rows = []
    for entry in entries:
        if entry.actual_rentals is None:
            continue
        features = {canonical.get(key, key): value for key, value in entry.features.items()}
        if not features.get("is_functioning", True):
            continue
        row = {column: float(features[column]) for column in WEATHER}
        row["hour"] = float(features["hour"])
        row[TARGET] = float(entry.actual_rentals)
        row[PREDICTED] = float(entry.predicted_rentals)
        row["prediction_time"] = entry.prediction_time
        rows.append(row)
    return pd.DataFrame(rows, columns=[*REFERENCE_COLUMNS, "prediction_time"])


def _drift_score(metrics: list[dict[str, Any]], column: str) -> tuple[float, float]:
    prefix = f"ValueDrift(column={column},"
    for metric in metrics:
        if metric["metric_name"].startswith(prefix):
            threshold = float(_THRESHOLD.search(metric["metric_name"]).group(1))
            return float(metric["value"]), threshold
    raise KeyError(f"Evidently returned no drift score for {column!r}")


def check_drift(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    reference_mae: float,
    model_version: str,
    report_path: str | Path | None = None,
) -> DriftResult:
    """Compare the current window with the reference and optionally save an HTML report."""
    cfg = load_config()["monitoring"]
    if len(current) < int(cfg["min_rows"]):
        raise InsufficientDataError(
            f"Need at least {cfg['min_rows']} predictions with an actual demand from model "
            f"{model_version}, have {len(current)}."
        )

    from evidently import DataDefinition, Dataset, Regression, Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset, RegressionPreset

    threshold = float(cfg["drift_threshold"])
    columns = [*WEATHER, TARGET, PREDICTED]
    definition = DataDefinition(
        numerical_columns=columns,
        regression=[Regression(target=TARGET, prediction=PREDICTED)],
    )
    report = Report(
        [
            DataDriftPreset(
                columns=WEATHER,
                num_method="wasserstein",
                num_threshold=threshold,
                drift_share=float(cfg["data_drift_share"]),
            ),
            ValueDrift(column=TARGET, method="wasserstein", threshold=threshold),
            RegressionPreset(),
        ]
    )
    snapshot = report.run(
        Dataset.from_pandas(current[columns].reset_index(drop=True), data_definition=definition),
        Dataset.from_pandas(reference[columns].reset_index(drop=True), data_definition=definition),
    )
    metrics = snapshot.dict()["metrics"]

    drifted = []
    for column in WEATHER:
        score, column_threshold = _drift_score(metrics, column)
        if score >= column_threshold:
            drifted.append(column)
    share = len(drifted) / len(WEATHER)
    target_score, target_threshold = _drift_score(metrics, TARGET)

    current_mae = float((current[TARGET] - current[PREDICTED]).abs().mean())
    ratio = current_mae / float(reference_mae)

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(".tmp.html")
        snapshot.save_html(str(staging))
        staging.replace(path)

    times = pd.to_datetime(current["prediction_time"]) if "prediction_time" in current else None
    now = datetime.now(UTC)
    return DriftResult(
        checked_at=now,
        model_version=model_version,
        window_start=times.min().to_pydatetime() if times is not None else now,
        window_end=times.max().to_pydatetime() if times is not None else now,
        rows=len(current),
        data_drift=share >= float(cfg["data_drift_share"]),
        drifted_feature_share=share,
        drifted_features=drifted,
        target_drift=target_score >= target_threshold,
        target_drift_score=target_score,
        concept_drift=ratio > float(cfg["concept_drift_mae_ratio"]),
        current_mae=current_mae,
        reference_mae=float(reference_mae),
        mae_ratio=ratio,
        thresholds={
            "drift_threshold": threshold,
            "data_drift_share": float(cfg["data_drift_share"]),
            "concept_drift_mae_ratio": float(cfg["concept_drift_mae_ratio"]),
        },
    )
