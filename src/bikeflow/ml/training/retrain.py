"""Retrain the served model on history plus the production journal, behind a quality gate.

The challenger is the same kind of model as the champion, trained on the original
dataset before the journal starts plus every journal hour that already has an
actual demand — except the last two weeks. The week before last drives early
stopping; the last week is a holdout neither model has learned from.

Quality gate: the challenger becomes the champion only if it lowers the holdout
MAE by at least `retraining.min_improvement`. Every challenger is registered in
MLflow for the record; only a promoted one receives the serving alias.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bikeflow.monitoring.drift import WEATHER, build_reference

from ..config import ensure_dir, load_config
from ..data.download import download_raw
from ..data.preprocess import load_raw, to_canonical
from ..features import TARGET, api_input_contract, build_features
from ..inference import Predictor
from ..metrics import evaluate
from ..models.baseline import SeasonalMedianBaseline
from ..models.registry import save_bundle
from .cv import build_candidates

SEOUL = "Asia/Seoul"
COLUMNS = [
    "timestamp",
    "date",
    "hour",
    *WEATHER,
    "season",
    "is_holiday",
    "is_functioning",
    "day_of_week",
    TARGET,
]


class NotEnoughJournalError(ValueError):
    """The journal does not yet cover the holdout and early-stopping weeks."""


@dataclass(frozen=True)
class RetrainResult:
    trigger: str
    started_at: datetime
    finished_at: datetime
    champion_model_version: str
    challenger_model_version: str
    registered_version: str
    mlflow_run_id: str
    rows_fit: int
    rows_early_stopping: int
    rows_holdout: int
    holdout_start: datetime
    holdout_end: datetime
    champion_mae: float
    challenger_mae: float
    improvement: float
    min_improvement: float
    promoted: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("started_at", "finished_at", "holdout_start", "holdout_end"):
            data[key] = data[key].isoformat()
        return data


def journal_frame(entries: Iterable[Any]) -> pd.DataFrame:
    """Journal hours with an actual demand, as canonical training rows.

    Timestamps become naive Seoul time like the original dataset. A replay run
    twice yields the same hour twice; the latest report wins.
    """
    canonical = {public: spec["canonical_name"] for public, spec in api_input_contract().items()}
    rows = []
    for entry in entries:
        if entry.actual_rentals is None:
            continue
        features = {canonical.get(key, key): value for key, value in entry.features.items()}
        moment = pd.Timestamp(entry.prediction_time).tz_convert(SEOUL).tz_localize(None)
        row = {column: features[column] for column in (*WEATHER, "hour", "season", "day_of_week")}
        row["is_holiday"] = bool(features.get("is_holiday", False))
        row["is_functioning"] = bool(features.get("is_functioning", True))
        row["timestamp"] = moment
        row["date"] = moment.normalize()
        row[TARGET] = float(entry.actual_rentals)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(rows)[COLUMNS].sort_values("timestamp")
    return frame.drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def history_before(moment: pd.Timestamp) -> pd.DataFrame:
    """The original dataset up to the first journal hour."""
    download_raw()
    frame = to_canonical(load_raw())
    return frame.loc[frame["timestamp"] < moment, COLUMNS].reset_index(drop=True)


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Hash of the exact rows a challenger learned from, so its version is unique."""
    hashed = pd.util.hash_pandas_object(frame[COLUMNS], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _mae(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.abs(actual.to_numpy(dtype="float64") - predicted).mean())


def retrain(
    entries: Iterable[Any],
    champion: Predictor,
    trigger: str,
    work_dir: str | Path,
    history: pd.DataFrame | None = None,
) -> RetrainResult:
    """Train a challenger, compare it with the champion and promote it if it passes the gate."""
    cfg = load_config()["retraining"]
    started = datetime.now(UTC)

    journal = journal_frame(entries)
    journal = journal[journal["is_functioning"]].reset_index(drop=True)
    holdout_rows = int(cfg["holdout_hours"])
    early_rows = int(cfg["early_stopping_hours"])
    if len(journal) < holdout_rows + early_rows:
        raise NotEnoughJournalError(
            f"Retraining needs at least {holdout_rows + early_rows} journal hours with an actual "
            f"demand, have {len(journal)}."
        )

    holdout = journal.tail(holdout_rows).reset_index(drop=True)
    early = journal.iloc[-(holdout_rows + early_rows) : -holdout_rows].reset_index(drop=True)
    recent = journal.iloc[: -(holdout_rows + early_rows)]

    if history is None:
        history = history_before(journal["timestamp"].min())
    history = history[history["is_functioning"]]
    fit = pd.concat([history[COLUMNS], recent[COLUMNS]], ignore_index=True)

    challenger_model = build_candidates()[champion.kind]
    fit_features = build_features(fit)
    if isinstance(challenger_model, SeasonalMedianBaseline):
        challenger_model.fit(fit_features, fit[TARGET])
    else:
        challenger_model.fit(fit_features, fit[TARGET], build_features(early), early[TARGET])

    holdout_predictions = challenger_model.predict(build_features(holdout))
    everything = pd.concat([fit, early, holdout], ignore_index=True)
    bundle_path = save_bundle(
        ensure_dir(work_dir) / "challenger.joblib",
        challenger_model,
        metrics={
            "train": evaluate(fit[TARGET], challenger_model.predict(fit_features)),
            "validation": evaluate(holdout[TARGET], holdout_predictions),
        },
        data_sha256=data_fingerprint(everything),
        train_period=(str(fit["timestamp"].min()), str(fit["timestamp"].max())),
        training_params=getattr(challenger_model, "params", {}),
        extra={"retraining_trigger": trigger},
        reference=build_reference(holdout, holdout_predictions),
    )
    challenger = Predictor.load(bundle_path)

    champion_mae = _mae(holdout[TARGET], champion.predict(holdout))
    challenger_mae = _mae(holdout[TARGET], challenger.predict(holdout))
    improvement = 1.0 - challenger_mae / champion_mae if champion_mae else 0.0
    promoted = improvement >= float(cfg["min_improvement"])

    run_id, version = _log(
        trigger=trigger,
        champion=champion,
        challenger=challenger,
        bundle_path=bundle_path,
        holdout=holdout,
        rows=(len(fit), len(early), len(holdout)),
        maes=(champion_mae, challenger_mae, improvement),
        promoted=promoted,
    )

    return RetrainResult(
        trigger=trigger,
        started_at=started,
        finished_at=datetime.now(UTC),
        champion_model_version=str(champion.metadata.get("model_version")),
        challenger_model_version=str(challenger.metadata.get("model_version")),
        registered_version=version,
        mlflow_run_id=run_id,
        rows_fit=len(fit),
        rows_early_stopping=len(early),
        rows_holdout=len(holdout),
        holdout_start=holdout["timestamp"].min().to_pydatetime(),
        holdout_end=holdout["timestamp"].max().to_pydatetime(),
        champion_mae=champion_mae,
        challenger_mae=challenger_mae,
        improvement=improvement,
        min_improvement=float(cfg["min_improvement"]),
        promoted=promoted,
    )


def _log(
    trigger: str,
    champion: Predictor,
    challenger: Predictor,
    bundle_path: Path,
    holdout: pd.DataFrame,
    rows: tuple[int, int, int],
    maes: tuple[float, float, float],
    promoted: bool,
) -> tuple[str, str]:
    """Record the retraining in MLflow, register the challenger, move the alias if promoted."""
    import mlflow

    from .tracking import register_bundle, set_champion, start_tracking

    cfg = load_config()
    client = start_tracking()
    gate = "passed" if promoted else "rejected"
    with mlflow.start_run(run_name=f"retrain {trigger} ({gate})") as run:
        mlflow.log_params(
            {
                "trigger": trigger,
                "model_kind": challenger.kind,
                "holdout_hours": cfg["retraining"]["holdout_hours"],
                "early_stopping_hours": cfg["retraining"]["early_stopping_hours"],
                "min_improvement": cfg["retraining"]["min_improvement"],
                "rows_fit": rows[0],
                "rows_early_stopping": rows[1],
                "rows_holdout": rows[2],
            }
        )
        mlflow.log_metrics(
            {
                "holdout_mae_champion": maes[0],
                "holdout_mae_challenger": maes[1],
                "improvement": maes[2],
            }
        )
        mlflow.set_tags(
            {
                "quality_gate": gate,
                "champion_model_version": str(champion.metadata.get("model_version")),
                "model_version": str(challenger.metadata.get("model_version")),
                "data_sha256": str(challenger.metadata.get("data_sha256")),
            }
        )
        version = register_bundle(bundle_path, holdout)

    client.set_model_version_tag(cfg["tracking"]["registered_model"], version, "quality_gate", gate)
    if promoted:
        set_champion(client, version)
    return run.info.run_id, version
