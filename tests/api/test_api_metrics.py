"""`GET /metrics` exposes what Prometheus scrapes."""

import pytest
from fastapi.testclient import TestClient

from bikeflow.api import metrics
from bikeflow.api.dependencies import get_predictor
from bikeflow.api.main import app
from bikeflow.model.stub import StubPredictor

client = TestClient(app)

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


@pytest.fixture(autouse=True)
def stub_predictor():
    app.dependency_overrides[get_predictor] = lambda: StubPredictor(42.0)
    yield
    app.dependency_overrides.pop(get_predictor, None)


def scrape() -> str:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return response.text


def sample(text: str, name: str) -> float:
    """The first sample whose line starts with `name`; 0 while the series is unused.

    A counter with labels appears only after its first increment, exactly as a
    Prometheus client is supposed to behave.
    """
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_are_exposed_in_the_prometheus_text_format():
    text = scrape()

    assert "# HELP bikeflow_predictions_total" in text
    assert "# TYPE bikeflow_http_requests_total counter" in text


def test_a_prediction_is_counted_with_its_model_version():
    counter = 'bikeflow_predictions_total{model_version="stub-v0"}'
    before = scrape()
    served = client.post("/predict", json=VALID_REQUEST)

    after = scrape()

    assert served.status_code == 200
    assert sample(after, counter) == sample(before, counter) + 1
    assert sample(after, "bikeflow_predicted_rentals_count") >= 1


def test_an_actual_report_records_the_absolute_error():
    served = client.post("/predict", json=VALID_REQUEST).json()
    before = scrape()

    client.post(f"/predictions/{served['prediction_id']}/actual", json={"actual_rentals": 500})

    after = scrape()
    assert sample(after, "bikeflow_actuals_total") == sample(before, "bikeflow_actuals_total") + 1
    assert (
        sample(after, "bikeflow_absolute_error_count")
        == sample(before, "bikeflow_absolute_error_count") + 1
    )


def test_requests_are_counted_by_route_and_the_scrape_itself_is_not():
    client.post("/predict", json=VALID_REQUEST)

    text = scrape()

    assert sample(text, 'bikeflow_http_requests_total{method="POST",route="/predict"') >= 1
    assert 'route="/metrics"' not in text


def test_a_drift_check_publishes_its_gauges():
    metrics.record_drift_check(
        {
            "concept_drift": True,
            "data_drift": False,
            "target_drift": True,
            "mae_ratio": 2.1,
            "current_mae": 385.0,
            "drifted_feature_share": 0.25,
            "target_drift_score": 0.48,
            "thresholds": {"concept_drift_mae_ratio": 1.6},
        }
    )

    text = scrape()

    assert "\nbikeflow_data_drift 0.0\n" in text
    assert sample(text, "bikeflow_concept_drift ") == 1.0
    assert sample(text, "bikeflow_drift_mae_ratio ") == 2.1
    assert sample(text, "bikeflow_drift_mae_ratio_threshold ") == 1.6
    assert sample(text, "bikeflow_window_mae ") == 385.0
