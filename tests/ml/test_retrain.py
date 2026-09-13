"""Retraining on the journal: the challenger replaces the champion only through the gate."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient  # noqa: E402

from bikeflow.api.storage import StoredPrediction  # noqa: E402
from bikeflow.ml.config import load_config  # noqa: E402
from bikeflow.ml.features import TARGET, build_features  # noqa: E402
from bikeflow.ml.inference import Predictor  # noqa: E402
from bikeflow.ml.models.baseline import SeasonalMedianBaseline  # noqa: E402
from bikeflow.ml.models.registry import save_bundle  # noqa: E402
from bikeflow.ml.training.retrain import (  # noqa: E402
    COLUMNS,
    NotEnoughJournalError,
    journal_frame,
    retrain,
)
from bikeflow.monitoring.drift import WEATHER  # noqa: E402

SEOUL = ZoneInfo("Asia/Seoul")
HISTORY_START = pd.Timestamp("2018-08-01")
WEEK = 168
HOLDOUT = int(load_config()["retraining"]["holdout_hours"])
EARLY = int(load_config()["retraining"]["early_stopping_hours"])


def base_demand(hour: int) -> float:
    return 100.0 + 10.0 * hour


def canonical_hours(start: pd.Timestamp, count: int, factor: float) -> pd.DataFrame:
    stamps = pd.date_range(start, periods=count, freq="h")
    frame = pd.DataFrame({column: 5.0 for column in WEATHER}, index=range(count))
    frame["timestamp"] = stamps
    frame["date"] = stamps.normalize()
    frame["hour"] = stamps.hour
    frame["season"] = "Autumn"
    frame["is_holiday"] = False
    frame["is_functioning"] = True
    frame["day_of_week"] = stamps.dayofweek
    frame[TARGET] = [base_demand(hour) * factor for hour in stamps.hour]
    return frame[COLUMNS]


def to_entries(frame: pd.DataFrame, first_id: int = 1) -> list[StoredPrediction]:
    entries = []
    for offset, row in enumerate(frame.itertuples(index=False)):
        entries.append(
            StoredPrediction(
                id=first_id + offset,
                created_at=datetime.now(UTC),
                prediction_time=row.timestamp.tz_localize(SEOUL).to_pydatetime(),
                features={
                    "temperature_c": 5.0,
                    "humidity_pct": 5.0,
                    "wind_speed_m_s": 5.0,
                    "visibility_10m": 5.0,
                    "dew_point_c": 5.0,
                    "solar_radiation_mj_m2": 5.0,
                    "rainfall_mm": 5.0,
                    "snowfall_cm": 5.0,
                    "holiday": False,
                    "functioning_day": True,
                    "hour": int(row.hour),
                    "season": "Autumn",
                    "day_of_week": int(row.day_of_week),
                },
                predicted_rentals=0.0,
                model_version="champion",
                actual_rentals=float(getattr(row, TARGET)),
                actual_recorded_at=None,
            )
        )
    return entries


@pytest.fixture
def store(tmp_path, monkeypatch):
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    # Explicit URI: a bare MlflowClient() would reuse the URI a previous test set globally.
    return MlflowClient(tracking_uri=uri)


@pytest.fixture
def history():
    return canonical_hours(HISTORY_START, 2 * WEEK, factor=1.0)


@pytest.fixture
def champion(tmp_path, history):
    model = SeasonalMedianBaseline().fit(build_features(history), history[TARGET])
    return Predictor.load(save_bundle(tmp_path / "champion.joblib", model, metrics={}))


def journal(weeks: list[float]) -> list[StoredPrediction]:
    """Consecutive journal weeks after the history, each with its own demand factor."""
    start = HISTORY_START + pd.Timedelta(weeks=2)
    frames = [
        canonical_hours(start + pd.Timedelta(weeks=index), WEEK, factor)
        for index, factor in enumerate(weeks)
    ]
    return to_entries(pd.concat(frames, ignore_index=True))


def test_challenger_that_learned_the_new_demand_is_promoted(store, tmp_path, champion, history):
    # Demand tripled for six weeks: most of them to learn from, then early stopping and holdout.
    result = retrain(journal([3.0] * 6), champion, "drift", tmp_path / "work", history=history)

    assert result.promoted is True
    assert result.challenger_mae == pytest.approx(0.0)
    assert result.champion_mae > 0
    assert result.improvement == pytest.approx(1.0)
    assert (result.rows_fit, result.rows_early_stopping, result.rows_holdout) == (
        2 * WEEK + 6 * WEEK - EARLY - HOLDOUT,
        EARLY,
        HOLDOUT,
    )
    assert result.challenger_model_version != result.champion_model_version

    tracking = load_config()["tracking"]
    alias = store.get_model_version_by_alias(tracking["registered_model"], tracking["alias"])
    assert str(alias.version) == result.registered_version
    version = store.get_model_version(tracking["registered_model"], result.registered_version)
    assert version.tags["quality_gate"] == "passed"


def test_challenger_that_is_worse_is_registered_but_not_promoted(
    store, tmp_path, champion, history
):
    # It learns a temporary 1.5x surge, but the holdout week is back to normal.
    result = retrain(
        journal([1.5, 1.5, 1.5, 1.5, 1.0, 1.02]),
        champion,
        "manual",
        tmp_path / "work",
        history=history,
    )

    assert result.promoted is False
    assert result.improvement < 0
    assert result.trigger == "manual"

    tracking = load_config()["tracking"]
    registered = store.get_registered_model(tracking["registered_model"])
    assert tracking["alias"] not in registered.aliases
    version = store.get_model_version(tracking["registered_model"], result.registered_version)
    assert version.tags["quality_gate"] == "rejected"


def test_too_short_journal_is_refused(tmp_path, champion, history):
    short = to_entries(
        canonical_hours(HISTORY_START + pd.Timedelta(weeks=2), HOLDOUT + EARLY - 1, 1.0)
    )

    with pytest.raises(NotEnoughJournalError, match="at least"):
        retrain(short, champion, "manual", tmp_path / "work", history=history)


def test_journal_frame_keeps_the_latest_report_for_a_repeated_hour():
    hours = canonical_hours(pd.Timestamp("2018-11-01"), 2, factor=1.0)
    first = to_entries(hours)
    repeated = to_entries(hours.head(1).assign(**{TARGET: 999.0}), first_id=10)

    frame = journal_frame(first + repeated)

    assert len(frame) == 2
    assert frame.loc[frame["hour"] == 0, TARGET].item() == 999.0
    assert frame["timestamp"].dt.tz is None


def test_journal_frame_skips_entries_without_actual_demand():
    entry = to_entries(canonical_hours(pd.Timestamp("2018-11-01"), 1, factor=1.0))[0]
    pending = StoredPrediction(**{**entry.__dict__, "actual_rentals": None})

    assert journal_frame([pending]).empty


def test_result_serialises_with_iso_times(store, tmp_path, champion, history):
    data = retrain(
        journal([3.0] * 6), champion, "drift", tmp_path / "work", history=history
    ).to_dict()

    assert datetime.fromisoformat(data["holdout_end"]) - datetime.fromisoformat(
        data["holdout_start"]
    ) == timedelta(hours=HOLDOUT - 1)
    assert np.isfinite(data["improvement"])
