"""Every API test writes to its own temporary journal, never to data/predictions.db."""

import pytest

from bikeflow.api.dependencies import get_store
from bikeflow.api.main import app
from bikeflow.api.storage import PredictionStore


@pytest.fixture(autouse=True)
def journal(tmp_path):
    store = PredictionStore(tmp_path / "predictions.db")
    app.dependency_overrides[get_store] = lambda: store
    yield store
    app.dependency_overrides.pop(get_store, None)
