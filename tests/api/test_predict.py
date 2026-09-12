from fastapi.testclient import TestClient

from bikeflow.api.dependencies import get_predictor
from bikeflow.api.main import app
from bikeflow.model.adapter import BikeflowPredictor
from bikeflow.model.stub import StubPredictor

VALID_REQUEST = {
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


def test_stub_is_available_only_through_dependency_injection() -> None:
    app.dependency_overrides[get_predictor] = lambda: StubPredictor(42.0)
    try:
        response = TestClient(app).post("/predict", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["predicted_rentals"] == 42.0
    assert response.json()["model_version"] == "stub-v0"


def test_predict_uses_a_real_mlp_artifact(real_mlp_artifact) -> None:
    app.dependency_overrides[get_predictor] = lambda: BikeflowPredictor(real_mlp_artifact)
    try:
        response = TestClient(app).post("/predict", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_rentals"] > 0
    assert payload["model_version"].startswith("mlp_embedding-")


def test_prediction_time_is_normalized_to_seoul(real_mlp_artifact) -> None:
    app.dependency_overrides[get_predictor] = lambda: BikeflowPredictor(real_mlp_artifact)
    try:
        response = TestClient(app).post(
            "/predict",
            json={**VALID_REQUEST, "prediction_time": "2026-07-14T23:00:00Z"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["prediction_time"] == "2026-07-15T08:00:00+09:00"


def test_predict_rejects_naive_time() -> None:
    response = TestClient(app).post(
        "/predict", json={**VALID_REQUEST, "prediction_time": "2026-07-15T08:00:00"}
    )
    assert response.status_code == 422


def test_predict_rejects_values_outside_the_shared_contract() -> None:
    for field, value in (("humidity_pct", 101), ("temperature_c", -41), ("visibility_10m", 2001)):
        response = TestClient(app).post("/predict", json={**VALID_REQUEST, field: value})
        assert response.status_code == 422, field
