"""The replay scenario started from the API: endpoints and the background manager."""

import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import bikeflow.replay as replay_module
from bikeflow.api.dependencies import get_predictor, get_replay_manager
from bikeflow.api.main import app
from bikeflow.api.simulation import ReplayManager
from bikeflow.model.stub import StubPredictor


def canonical_hours(count: int) -> pd.DataFrame:
    stamps = pd.date_range("2018-10-01 06:00", periods=count, freq="h")
    weather = {
        "temperature": 15.0,
        "humidity": 55,
        "wind_speed": 1.5,
        "visibility": 2000,
        "dew_point": 5.0,
        "solar_radiation": 0.5,
        "rainfall": 0.0,
        "snowfall": 0.0,
    }
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "hour": stamps.hour,
            "rented_bike_count": 100,
            **weather,
            "is_holiday": False,
            "is_functioning": True,
        }
    )


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.pop(get_replay_manager, None)
    app.dependency_overrides.pop(get_predictor, None)


def client_with(manager):
    app.dependency_overrides[get_replay_manager] = lambda: manager
    return TestClient(app)


def test_default_body_is_the_concept_drift_demo():
    started = []
    manager = ReplayManager(lambda params, log: started.append(params) or {})

    response = client_with(manager).post("/replay")
    manager.wait(5)

    assert response.status_code == 202
    assert started == [
        {"check_every": 24, "evening_boost": 2.5, "boost_from": "2018-11-01", "hours": None}
    ]


def test_second_start_while_running_is_refused():
    release = threading.Event()
    manager = ReplayManager(lambda params, log: release.wait(5) and {})
    client = client_with(manager)

    first = client.post("/replay", json={"hours": 3})
    second = client.post("/replay", json={"hours": 3})
    release.set()
    manager.wait(5)

    assert first.status_code == 202
    assert first.json()["state"] == "running"
    assert second.status_code == 409


def test_status_carries_the_log_and_the_result():
    def job(params, log):
        log("[replay] 3 ч")
        log("[replay] готово")
        return {"predictions": 3}

    manager = ReplayManager(job)
    client = client_with(manager)
    client.post("/replay", json={"hours": 3})
    manager.wait(5)

    status = client.get("/replay/status").json()

    assert status["state"] == "finished"
    assert status["log"] == ["[replay] 3 ч", "[replay] готово"]
    assert status["result"] == {"predictions": 3}


def test_a_failed_replay_is_reported():
    def job(params, log):
        raise RuntimeError("API unreachable")

    manager = ReplayManager(job)
    manager.start({})
    manager.wait(5)

    status = manager.status()
    assert status["state"] == "failed"
    assert status["error"] == "RuntimeError: API unreachable"


def test_scenario_replays_the_period_through_the_api(monkeypatch, journal):
    app.dependency_overrides[get_predictor] = lambda: StubPredictor(42.0)
    client = TestClient(app)
    monkeypatch.setattr(replay_module, "load_hours", lambda start, end: canonical_hours(30))
    monkeypatch.setattr(
        replay_module,
        "http_post",
        lambda api: lambda path, body: client.post(path, json=body).json(),
    )
    monkeypatch.setattr(replay_module, "http_get", lambda api: lambda path: client.get(path).json())
    messages = []

    stats = replay_module.run_scenario("http://api", hours=10, log=messages.append)

    assert stats.predictions == stats.actuals == 10
    assert len(journal.recent()) == 10
    assert messages[0].startswith("[replay] 10 ч")
    assert messages[-1].startswith("[replay] готово")
    assert pd.Timestamp(journal.recent()[-1].prediction_time).hour == 6
