"""The Streamlit interface, driven headlessly against a fake API."""

from typing import Any

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "src/bikeflow/ui/app.py"

MODEL = {
    "model_version": "mlp_embedding-abc",
    "source": "registry",
    "reference_mae": 180.0,
    "retraining_enabled": True,
}
DRIFT = {
    "check_id": 3,
    "model_version": "mlp_embedding-abc",
    "window_start": "2018-11-08T00:00:00+09:00",
    "window_end": "2018-11-14T23:00:00+09:00",
    "rows": 168,
    "data_drift": True,
    "drifted_feature_share": 0.875,
    "drifted_features": ["temperature"],
    "target_drift": True,
    "target_drift_score": 0.48,
    "concept_drift": True,
    "current_mae": 385.0,
    "reference_mae": 180.0,
    "mae_ratio": 2.14,
    "thresholds": {"drift_threshold": 0.1, "data_drift_share": 0.5, "concept_drift_mae_ratio": 1.6},
    "checked_at": "2026-09-13T10:00:00+00:00",
}
PREDICTIONS = [
    {
        "prediction_id": 2,
        "created_at": "2026-09-13T10:00:00+00:00",
        "prediction_time": "2018-11-14T18:00:00+09:00",
        "features": {},
        "predicted_rentals": 900.0,
        "model_version": "mlp_embedding-abc",
        "actual_rentals": 2000.0,
        "absolute_error": 1100.0,
    },
    {
        "prediction_id": 1,
        "created_at": "2026-09-13T09:00:00+00:00",
        "prediction_time": "2018-11-14T17:00:00+09:00",
        "features": {},
        "predicted_rentals": 800.0,
        "model_version": "mlp_embedding-abc",
        "actual_rentals": 820.0,
        "absolute_error": 20.0,
    },
]
IDLE = {
    "state": "idle",
    "trigger": None,
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else ""

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def api(monkeypatch):
    """A fake API plus the log of what the page asked it to do."""
    import requests

    responses: dict[str, Any] = {
        "/model": MODEL,
        "/drift/latest": DRIFT,
        "/predictions": PREDICTIONS,
        "/retrain/status": IDLE,
        "/replay/status": {**IDLE, "params": None, "log": []},
    }
    calls: list[tuple[str, str]] = []

    def route(path: str) -> str:
        return path.split("?")[0]

    def fake_get(url: str, **_: Any) -> FakeResponse:
        path = route(url.split("8000", 1)[-1])
        calls.append(("GET", path))
        payload = responses.get(path)
        return FakeResponse(payload, 200 if payload is not None else 404)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        path = route(url.split("8000", 1)[-1])
        calls.append(("POST", path))
        responses["BODY " + path] = kwargs.get("json")
        payload = responses.get("POST " + path, {"detail": "ok"})
        return FakeResponse(payload, responses.get("POST_CODE " + path, 200))

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    return responses, calls


def test_the_page_warns_about_concept_drift_and_lists_the_tabs(api):
    app = AppTest.from_file(APP, default_timeout=60).run()

    assert not app.exception
    assert len(app.tabs) == 4
    assert "Дрейф модели" in app.error[0].value


def test_a_healthy_model_is_reported_as_such(api):
    responses, _ = api
    responses["/drift/latest"] = {**DRIFT, "concept_drift": False}

    app = AppTest.from_file(APP, default_timeout=60).run()

    assert not app.error
    assert "модель в норме" in app.success[0].value


def test_a_large_error_is_flagged_as_an_anomaly(api):
    app = AppTest.from_file(APP, default_timeout=60).run()

    table = app.dataframe[0].value
    assert list(table["Аномалия"]) == ["🔴", ""]
    assert table.loc[0, "Ошибка"] == 1100


def test_the_forecast_form_sends_a_prediction_request(api):
    responses, calls = api
    responses["POST /predict"] = {
        "prediction_id": 7,
        "prediction_time": "2018-12-01T18:00:00+09:00",
        "predicted_rentals": 412.7,
        "model_version": "mlp_embedding-abc",
    }
    app = AppTest.from_file(APP, default_timeout=60).run()

    app.button[0].click().run()

    assert ("POST", "/predict") in calls
    assert app.metric[0].value.startswith("413")


def test_the_retrain_button_is_hidden_without_the_registry(api):
    responses, calls = api
    responses["/model"] = {**MODEL, "retraining_enabled": False, "source": "file"}

    app = AppTest.from_file(APP, default_timeout=60).run()

    assert any("реестра MLflow" in warning.value for warning in app.warning)
    assert ("POST", "/retrain") not in calls


def test_the_drift_report_is_fetched_through_the_api(api):
    responses, calls = api
    responses["/drift/report"] = "<html><body>Evidently report</body></html>"
    app = AppTest.from_file(APP, default_timeout=60).run()

    app.toggle[0].set_value(True).run()

    assert not app.exception
    assert ("GET", "/drift/report") in calls


def test_the_replay_button_starts_the_drift_demo(api):
    responses, calls = api
    responses["POST_CODE /replay"] = 202
    app = AppTest.from_file(APP, default_timeout=60).run()

    next(button for button in app.button if button.label == "Запустить поток данных").click().run()

    assert not app.exception
    assert ("POST", "/replay") in calls
    assert responses["BODY /replay"] == {
        "check_every": 24,
        "evening_boost": 2.5,
        "boost_from": "2018-11-01",
    }


def test_a_finished_replay_shows_its_totals_and_log(api):
    responses, _ = api
    responses["/replay/status"] = {
        **IDLE,
        "state": "finished",
        "params": {},
        "result": {
            "predictions": 1464,
            "actuals": 1464,
            "mae": 300.4,
            "drift_checks": 61,
            "concept_drift_alerts": 27,
            "retrainings_started": 4,
            "retrainings_promoted": 1,
        },
        "log": ["[replay] 1464 ч", "[replay] готово"],
    }

    app = AppTest.from_file(APP, default_timeout=60).run()

    assert not app.exception
    assert any("прогнозов 1464" in message.value for message in app.success)
    assert any("[replay] готово" in block.value for block in app.code)


def test_a_running_replay_shows_its_progress(api):
    responses, _ = api
    responses["/replay/status"] = {
        **IDLE,
        "state": "running",
        "params": {},
        "log": ["[replay] 1464 ч"],
    }

    app = AppTest.from_file(APP, default_timeout=60).run()

    assert not app.exception
    assert any("Поток идёт" in message.value for message in app.info)
