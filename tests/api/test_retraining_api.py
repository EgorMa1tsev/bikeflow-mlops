"""Retraining endpoints, the automatic trigger on concept drift, and the background manager."""

import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("evidently")

from bikeflow.api.dependencies import get_predictor, get_retraining_manager  # noqa: E402
from bikeflow.api.main import app  # noqa: E402
from bikeflow.api.retraining import RetrainingManager  # noqa: E402
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


class FakeManager:
    """Records starts instead of training."""

    def __init__(self, enabled=True, accepts=True):
        self.enabled = enabled
        self.accepts = accepts
        self.triggers = []

    def start(self, trigger):
        if not self.accepts:
            return False
        self.triggers.append(trigger)
        return True

    def status(self):
        state = "running" if self.triggers else "idle"
        trigger = self.triggers[-1] if self.triggers else None
        return {
            "state": state,
            "trigger": trigger,
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }


class MonitoredStub(StubPredictor):
    def __init__(self, reference):
        super().__init__(PREDICTION)
        self._reference = reference

    def monitoring_reference(self):
        return self._reference, 50.0


@pytest.fixture
def reference():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({column: rng.normal(8.0, 4.0, 500) for column in WEATHER})
    frame["hour"] = np.arange(500) % 24
    frame[PREDICTED] = PREDICTION
    frame[TARGET] = PREDICTION + rng.normal(0.0, 60.0, 500)
    return frame


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.pop(get_predictor, None)
    app.dependency_overrides.pop(get_retraining_manager, None)


def client_with(manager, predictor=None):
    app.dependency_overrides[get_retraining_manager] = lambda: manager
    app.dependency_overrides[get_predictor] = lambda: predictor or StubPredictor(PREDICTION)
    return TestClient(app)


def serve_hours(client, count, actual):
    start = datetime(2018, 11, 1, tzinfo=SEOUL)
    for offset in range(count):
        moment = (start + timedelta(hours=offset)).isoformat()
        prediction_id = client.post("/predict", json={**REQUEST, "prediction_time": moment}).json()[
            "prediction_id"
        ]
        client.post(f"/predictions/{prediction_id}/actual", json={"actual_rentals": actual})


def test_manual_retraining_needs_the_registry():
    response = client_with(FakeManager(enabled=False)).post("/retrain")

    assert response.status_code == 409
    assert "BIKEFLOW_MODEL_URI" in response.json()["detail"]


def test_manual_retraining_starts_in_the_background():
    manager = FakeManager()

    response = client_with(manager).post("/retrain")

    assert response.status_code == 202
    assert response.json()["state"] == "running"
    assert manager.triggers == ["manual"]


def test_second_start_while_running_is_refused():
    response = client_with(FakeManager(accepts=False)).post("/retrain")

    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


def test_status_shows_the_last_finished_run_when_idle(journal):
    journal.add_retraining({"promoted": True, "registered_version": "3"})

    response = client_with(FakeManager()).get("/retrain/status")

    assert response.status_code == 200
    assert response.json()["state"] == "idle"
    assert response.json()["result"] == {"promoted": True, "registered_version": "3"}


def test_concept_drift_starts_retraining_automatically(reference):
    manager = FakeManager()
    client = client_with(manager, MonitoredStub(reference))
    serve_hours(client, 96, actual=PREDICTION * 2)

    body = client.post("/drift/check").json()

    assert body["concept_drift"] is True
    assert body["retraining_started"] is True
    assert manager.triggers == ["drift"]


def test_healthy_model_does_not_retrain(reference):
    manager = FakeManager()
    client = client_with(manager, MonitoredStub(reference))
    serve_hours(client, 96, actual=PREDICTION + 30)

    body = client.post("/drift/check").json()

    assert body["concept_drift"] is False
    assert body["retraining_started"] is False
    assert manager.triggers == []


def test_manager_runs_one_job_at_a_time_and_reports_the_result():
    release = threading.Event()

    def job(trigger):
        release.wait(5)
        return {"promoted": True, "trigger": trigger}

    manager = RetrainingManager(job)

    assert manager.start("manual") is True
    assert manager.status()["state"] == "running"
    assert manager.start("drift") is False

    release.set()
    manager.wait(5)
    status = manager.status()
    assert status["state"] == "finished"
    assert status["result"] == {"promoted": True, "trigger": "manual"}
    assert status["finished_at"] is not None


def test_manager_reports_a_failed_job():
    def job(trigger):
        raise RuntimeError("no data")

    manager = RetrainingManager(job)
    manager.start("manual")
    manager.wait(5)

    status = manager.status()
    assert status["state"] == "failed"
    assert status["error"] == "RuntimeError: no data"
    assert manager.start("manual") is True


def test_automatic_retraining_waits_for_new_data_after_the_last_one(reference, journal):
    # The drift window ends at 2018-11-04 23:00; a retraining just evaluated up to 2018-11-03.
    journal.add_retraining({"holdout_end": "2018-11-03T23:00:00", "promoted": False})
    manager = FakeManager()
    client = client_with(manager, MonitoredStub(reference))
    serve_hours(client, 96, actual=PREDICTION * 2)

    body = client.post("/drift/check").json()

    assert body["concept_drift"] is True
    assert body["retraining_started"] is False
    assert manager.triggers == []


def test_automatic_retraining_resumes_once_the_cooldown_has_passed(reference, journal):
    journal.add_retraining({"holdout_end": "2018-10-20T23:00:00", "promoted": False})
    manager = FakeManager()
    client = client_with(manager, MonitoredStub(reference))
    serve_hours(client, 96, actual=PREDICTION * 2)

    body = client.post("/drift/check").json()

    assert body["retraining_started"] is True
    assert manager.triggers == ["drift"]
