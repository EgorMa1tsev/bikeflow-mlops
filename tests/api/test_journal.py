"""Prediction journal: every prediction is recorded and later receives its actual demand."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from bikeflow.api.dependencies import get_predictor
from bikeflow.api.main import app
from bikeflow.api.storage import PredictionStore
from bikeflow.model.stub import StubPredictor

REQUEST = {
    "prediction_time": "2026-07-15T08:00:00+09:00",
    "temperature_c": 24.5,
    "humidity_pct": 61.0,
    "wind_speed_m_s": 1.8,
    "visibility_10m": 1800,
    "dew_point_c": 16.4,
    "solar_radiation_mj_m2": 1.2,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
    "holiday": False,
    "functioning_day": True,
}


@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = lambda: StubPredictor(42.0)
    yield TestClient(app)
    app.dependency_overrides.pop(get_predictor, None)


def predict(client) -> int:
    response = client.post("/predict", json=REQUEST)
    assert response.status_code == 200
    return response.json()["prediction_id"]


def test_prediction_is_recorded_with_its_inputs(client, journal):
    stored = journal.get(predict(client))

    assert stored.predicted_rentals == 42.0
    assert stored.model_version == "stub-v0"
    assert stored.prediction_time == datetime(2026, 7, 15, 8, tzinfo=ZoneInfo("Asia/Seoul"))
    assert stored.features["temperature_c"] == 24.5
    assert stored.features["hour"] == 8
    assert stored.features["season"] == "Summer"
    assert stored.features["day_of_week"] == datetime(2026, 7, 15).weekday()
    assert stored.actual_rentals is None
    assert stored.absolute_error is None


def test_actual_demand_is_attached_later(client, journal):
    prediction_id = predict(client)

    response = client.post(f"/predictions/{prediction_id}/actual", json={"actual_rentals": 50})

    assert response.status_code == 200
    assert response.json()["actual_rentals"] == 50.0
    assert response.json()["absolute_error"] == 8.0
    assert journal.get(prediction_id).actual_rentals == 50.0


def test_actual_for_unknown_prediction_is_404(client):
    response = client.post("/predictions/999/actual", json={"actual_rentals": 10})

    assert response.status_code == 404


def test_negative_actual_is_rejected(client):
    prediction_id = predict(client)

    response = client.post(f"/predictions/{prediction_id}/actual", json={"actual_rentals": -1})

    assert response.status_code == 422


def test_recent_predictions_are_newest_first_and_limited(client):
    ids = [predict(client) for _ in range(3)]

    response = client.get("/predictions", params={"limit": 2})

    assert response.status_code == 200
    assert [row["prediction_id"] for row in response.json()] == [ids[2], ids[1]]


def test_limit_outside_bounds_is_rejected(client):
    assert client.get("/predictions", params={"limit": 0}).status_code == 422
    assert client.get("/predictions", params={"limit": 1001}).status_code == 422


def test_rejected_request_is_not_recorded(client, journal):
    response = client.post("/predict", json={**REQUEST, "humidity_pct": 150})

    assert response.status_code == 422
    assert journal.recent() == []


def test_journal_survives_a_restart(tmp_path):
    path = tmp_path / "predictions.db"
    first = PredictionStore(path)
    prediction_id = first.add(
        datetime(2018, 10, 1, 8, tzinfo=ZoneInfo("Asia/Seoul")), {"hour": 8}, 100.0, "v1"
    )
    first.record_actual(prediction_id, 112.0)

    reopened = PredictionStore(path).get(prediction_id)

    assert reopened.actual_rentals == 112.0
    assert reopened.absolute_error == 12.0
    assert reopened.features == {"hour": 8}
