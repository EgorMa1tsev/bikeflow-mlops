"""Drift checks on synthetic data where the right answer is known."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")

from bikeflow.api.storage import StoredPrediction  # noqa: E402
from bikeflow.ml.features import TARGET  # noqa: E402
from bikeflow.monitoring.drift import (  # noqa: E402
    PREDICTED,
    WEATHER,
    InsufficientDataError,
    build_reference,
    check_drift,
    journal_to_frame,
)

SEOUL = ZoneInfo("Asia/Seoul")


def hours(count: int, seed: int, weather_shift: float = 0.0, demand_factor: float = 1.0):
    """Hourly rows whose predictions are right up to noise, unless demand is scaled."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({column: rng.normal(10.0, 3.0, count) for column in WEATHER})
    frame[WEATHER] += weather_shift
    frame["hour"] = np.arange(count) % 24
    predicted = rng.normal(700.0, 150.0, count)
    frame[PREDICTED] = predicted
    frame[TARGET] = predicted * demand_factor + rng.normal(0.0, 50.0, count)
    frame["prediction_time"] = pd.date_range("2018-11-01", periods=count, freq="h", tz=SEOUL)
    return frame


@pytest.fixture(scope="module")
def reference():
    return hours(1000, seed=1).drop(columns="prediction_time")


def reference_mae(reference):
    return float((reference[TARGET] - reference[PREDICTED]).abs().mean())


def test_healthy_window_raises_no_model_drift(reference):
    result = check_drift(hours(300, seed=2), reference, reference_mae(reference), "v1")

    assert result.concept_drift is False
    assert result.mae_ratio == pytest.approx(1.0, abs=0.15)
    assert result.data_drift is False
    assert result.rows == 300
    assert result.model_version == "v1"


def test_changed_weather_is_data_drift_but_not_model_drift(reference):
    result = check_drift(
        hours(300, seed=3, weather_shift=10.0), reference, reference_mae(reference), "v1"
    )

    assert result.data_drift is True
    assert set(result.drifted_features) == set(WEATHER)
    assert result.drifted_feature_share == 1.0
    assert result.concept_drift is False


def test_grown_error_is_concept_and_target_drift(reference):
    result = check_drift(
        hours(300, seed=4, demand_factor=2.0), reference, reference_mae(reference), "v1"
    )

    assert result.concept_drift is True
    assert result.mae_ratio > 1.6
    assert result.target_drift is True


def test_too_few_rows_are_refused(reference):
    with pytest.raises(InsufficientDataError, match="at least"):
        check_drift(hours(10, seed=5), reference, reference_mae(reference), "v1")


def test_report_is_written_as_html(reference, tmp_path):
    path = tmp_path / "nested" / "drift_report.html"

    check_drift(hours(300, seed=6), reference, reference_mae(reference), "v1", report_path=path)

    assert path.exists()
    assert "<html" in path.read_text(encoding="utf-8", errors="ignore")[:5000].lower()


def test_result_serialises_with_iso_times(reference):
    data = check_drift(hours(300, seed=7), reference, reference_mae(reference), "v1").to_dict()

    assert datetime.fromisoformat(data["window_start"]) < datetime.fromisoformat(data["window_end"])
    assert data["thresholds"]["concept_drift_mae_ratio"] == 1.6


def test_build_reference_keeps_monitoring_columns():
    frame = hours(24, seed=8).drop(columns=["prediction_time", PREDICTED])

    built = build_reference(frame, np.full(24, 500.0))

    assert list(built.columns) == [*WEATHER, "hour", TARGET, PREDICTED]
    assert (built[PREDICTED] == 500.0).all()


def entry(identifier, actual, functioning=True):
    return StoredPrediction(
        id=identifier,
        created_at=datetime(2026, 9, 13, tzinfo=SEOUL),
        prediction_time=datetime(2018, 11, 1, tzinfo=SEOUL) + timedelta(hours=identifier),
        features={
            "temperature_c": 5.0,
            "humidity_pct": 50.0,
            "wind_speed_m_s": 1.0,
            "visibility_10m": 2000.0,
            "dew_point_c": -2.0,
            "solar_radiation_mj_m2": 0.1,
            "rainfall_mm": 0.0,
            "snowfall_cm": 0.0,
            "holiday": False,
            "functioning_day": functioning,
            "hour": identifier % 24,
            "season": "Autumn",
            "day_of_week": 3,
        },
        predicted_rentals=400.0,
        model_version="v1",
        actual_rentals=actual,
        actual_recorded_at=None,
    )


def test_journal_entries_map_to_reference_columns():
    frame = journal_to_frame([entry(1, 420.0), entry(2, None), entry(3, 0.0, functioning=False)])

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["temperature"] == 5.0
    assert row["visibility"] == 2000.0
    assert row[TARGET] == 420.0
    assert row[PREDICTED] == 400.0
    assert row["hour"] == 1.0
