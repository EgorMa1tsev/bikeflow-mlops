"""Every API test writes to its own temporary journal and report folder."""

import pytest

from bikeflow.api.dependencies import get_drift_report_path, get_store
from bikeflow.api.main import app
from bikeflow.api.storage import PredictionStore


@pytest.fixture(autouse=True)
def journal(tmp_path):
    store = PredictionStore(tmp_path / "predictions.db")
    app.dependency_overrides[get_store] = lambda: store
    yield store
    app.dependency_overrides.pop(get_store, None)


@pytest.fixture(autouse=True)
def drift_report_path(tmp_path):
    path = tmp_path / "monitoring" / "drift_report.html"
    app.dependency_overrides[get_drift_report_path] = lambda: path
    yield path
    app.dependency_overrides.pop(get_drift_report_path, None)
