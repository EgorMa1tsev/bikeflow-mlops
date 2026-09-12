"""Replay: historical hours flow through the API, actual demand arrives later."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bikeflow.api.dependencies import get_predictor
from bikeflow.api.main import app
from bikeflow.api.schemas import PredictionRequest
from bikeflow.model.stub import StubPredictor
from bikeflow.replay import observed_demand, replay, to_request


def canonical_hours(count: int, start: str = "2018-10-01 06:00") -> pd.DataFrame:
    stamps = pd.date_range(start, periods=count, freq="h")
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "date": stamps.normalize(),
            "hour": stamps.hour,
            "rented_bike_count": [100 + 10 * i for i in range(count)],
            "temperature": 15.0,
            "humidity": 55,
            "wind_speed": 1.5,
            "visibility": 2000,
            "dew_point": 5.0,
            "solar_radiation": 0.5,
            "rainfall": 0.0,
            "snowfall": 0.0,
            "season": "Autumn",
            "is_holiday": False,
            "is_functioning": True,
        }
    )


def rows(frame: pd.DataFrame):
    return (row for _, row in frame.iterrows())


@pytest.fixture
def api_post():
    """Send replay traffic into the real FastAPI app instead of over the network."""
    app.dependency_overrides[get_predictor] = lambda: StubPredictor(42.0)
    client = TestClient(app)

    def post(path, body):
        response = client.post(path, json=body)
        assert response.status_code == 200, response.text
        return response.json()

    yield post
    app.dependency_overrides.pop(get_predictor, None)


def test_request_body_is_accepted_by_the_api_schema():
    row = canonical_hours(1, "2018-10-01 18:00").iloc[0]

    body = to_request(row)
    request = PredictionRequest.model_validate(body)

    assert body["prediction_time"] == "2018-10-01T18:00:00+09:00"
    assert request.to_features()["hour"] == 18
    assert request.to_features()["season"] == "Autumn"
    assert body["visibility_10m"] == 2000.0
    assert body["functioning_day"] is True


def test_replay_fills_the_journal_with_predictions_and_actuals(api_post, journal):
    frame = canonical_hours(6)

    stats = replay(rows(frame), api_post, delay_hours=1, report_every=0)

    entries = sorted(journal.recent(), key=lambda entry: entry.id)
    assert stats.predictions == stats.actuals == 6
    assert [entry.actual_rentals for entry in entries] == frame["rented_bike_count"].tolist()
    assert all(entry.predicted_rentals == 42.0 for entry in entries)
    expected_mae = (frame["rented_bike_count"] - 42.0).abs().mean()
    assert stats.mae == pytest.approx(expected_mae)


def recording_post(calls):
    """A fake API that numbers predictions and records the order of calls."""
    next_id = iter(range(1, 100))

    def post(path, body):
        calls.append(path)
        if path == "/predict":
            return {"prediction_id": next(next_id), "predicted_rentals": 1.0}
        return {}

    return post


def test_actual_is_reported_only_after_the_delay():
    calls = []

    replay(rows(canonical_hours(3)), recording_post(calls), delay_hours=1, report_every=0)

    assert calls == [
        "/predict",
        "/predict",
        "/predictions/1/actual",
        "/predict",
        "/predictions/2/actual",
        "/predictions/3/actual",
    ]


def test_zero_delay_reports_right_after_each_prediction():
    calls = []

    replay(rows(canonical_hours(2)), recording_post(calls), delay_hours=0, report_every=0)

    assert calls == ["/predict", "/predictions/1/actual", "/predict", "/predictions/2/actual"]


def test_evening_boost_inflates_only_evening_hours_after_its_start():
    frame = canonical_hours(24, "2018-10-31 00:00")
    evening_before = frame[(frame["hour"] == 18)].iloc[0]
    next_day = canonical_hours(24, "2018-11-01 00:00")
    evening_after = next_day[next_day["hour"] == 18].iloc[0]
    morning_after = next_day[next_day["hour"] == 9].iloc[0]
    since = pd.Timestamp("2018-11-01")

    assert observed_demand(evening_before, 1.5, since) == evening_before["rented_bike_count"]
    assert observed_demand(evening_after, 1.5, since) == round(
        evening_after["rented_bike_count"] * 1.5
    )
    assert observed_demand(morning_after, 1.5, since) == morning_after["rented_bike_count"]
    assert observed_demand(evening_after, 1.0, since) == evening_after["rented_bike_count"]
