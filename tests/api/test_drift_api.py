"""Drift endpoints over the prediction journal."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("evidently")

from bikeflow.api.dependencies import get_predictor  # noqa: E402
from bikeflow.api.main import app  # noqa: E402
from bikeflow.ml.features import TARGET  # noqa: E402
from bikeflow.model.stub import StubPredictor  # noqa: E402
from bikeflow.monitoring.drift import PREDICTED, WEATHER  # noqa: E402

SEOUL = ZoneInfo("Asia/Seoul")
PREDICTION = 700.0

REQUEST = {
    "temperature_c": 10.0,
    "humidity_pct": 50.0,
    "wind_speed_m_s": 1.5,
    "visibility_10m": 1800,
    "dew_point_c": 3.0,
    "solar_radiation_mj_m2": 0.5,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
    "holiday": False,
    "functioning_day": True,
}


class MonitoredStub(StubPredictor):
    """A constant predictor that carries a drift reference like a real artifact."""

    def __init__(self, reference: pd.DataFrame, validation_mae: float) -> None:
        super().__init__(PREDICTION)
        self._reference = reference
        self._validation_mae = validation_mae

    def monitoring_reference(self):
        return self._reference, self._validation_mae


@pytest.fixture(scope="module")
def reference():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({column: rng.normal(8.0, 4.0, 500) for column in WEATHER})
    frame["hour"] = np.arange(500) % 24
    frame[PREDICTED] = PREDICTION
    frame[TARGET] = PREDICTION + rng.normal(0.0, 60.0, 500)
    return frame


def use(predictor):
    app.dependency_overrides[get_predictor] = lambda: predictor
    return TestClient(app)


def serve_hours(client, count, actual):
    """Predict `count` consecutive hours and report the same actual demand for each."""
    start = datetime(2018, 11, 1, tzinfo=SEOUL)
    for offset in range(count):
        moment = (start + timedelta(hours=offset)).isoformat()
        prediction = client.post("/predict", json={**REQUEST, "prediction_time": moment})
        assert prediction.status_code == 200
        reported = client.post(
            f"/predictions/{prediction.json()['prediction_id']}/actual",
            json={"actual_rentals": actual(offset)},
        )
        assert reported.status_code == 200


@pytest.fixture(autouse=True)
def clear_predictor():
    yield
    app.dependency_overrides.pop(get_predictor, None)


def test_model_without_reference_cannot_be_monitored():
    client = use(StubPredictor(PREDICTION))

    response = client.post("/drift/check")

    assert response.status_code == 409


def test_not_enough_actuals_is_refused(reference):
    client = use(MonitoredStub(reference, 50.0))
    serve_hours(client, 5, lambda _: PREDICTION)

    response = client.post("/drift/check")

    assert response.status_code == 422
    assert "at least" in response.json()["detail"]


def test_accurate_model_is_not_flagged(reference):
    client = use(MonitoredStub(reference, 50.0))
    serve_hours(client, 96, lambda hour: PREDICTION + (40 if hour % 2 else -40))

    response = client.post("/drift/check")

    assert response.status_code == 200
    body = response.json()
    assert body["concept_drift"] is False
    assert body["rows"] == 96
    assert body["current_mae"] == pytest.approx(40.0)
    assert body["model_version"] == "stub-v0"


def test_degraded_model_is_flagged_and_the_result_is_kept(reference):
    client = use(MonitoredStub(reference, 50.0))
    serve_hours(client, 96, lambda _: PREDICTION * 2)

    checked = client.post("/drift/check").json()
    latest = client.get("/drift/latest")
    report = client.get("/drift/report")

    assert checked["concept_drift"] is True
    assert checked["mae_ratio"] == pytest.approx(14.0)
    assert latest.status_code == 200
    assert latest.json()["check_id"] == checked["check_id"]
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/html")


def test_nothing_to_show_before_the_first_check():
    client = use(StubPredictor(PREDICTION))

    assert client.get("/drift/latest").status_code == 404
    assert client.get("/drift/report").status_code == 404


def test_window_only_uses_the_served_model_version(journal):
    start = datetime(2018, 11, 1, tzinfo=SEOUL)
    for offset, version in [(0, "old"), (1, "new"), (2, "new"), (3, "old")]:
        identifier = journal.add(start + timedelta(hours=offset), {"hour": offset}, 1.0, version)
        journal.record_actual(identifier, 2.0)
    journal.add(start + timedelta(hours=4), {"hour": 4}, 1.0, "new")

    window = journal.monitoring_window("new", hours=168)

    assert [entry.features["hour"] for entry in window] == [1, 2]


def test_window_ends_at_the_latest_predicted_hour(journal):
    start = datetime(2018, 11, 1, tzinfo=SEOUL)
    for offset in range(10):
        identifier = journal.add(start + timedelta(hours=offset), {"hour": offset}, 1.0, "v")
        journal.record_actual(identifier, 1.0)

    window = journal.monitoring_window("v", hours=3)

    assert [entry.features["hour"] for entry in window] == [7, 8, 9]
