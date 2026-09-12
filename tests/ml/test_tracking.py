"""MLflow tracking: one run per training, champion registered under its alias."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from bikeflow.ml.config import load_config  # noqa: E402
from bikeflow.ml.features import build_features, coerce_input  # noqa: E402
from bikeflow.ml.inference import Predictor  # noqa: E402
from bikeflow.ml.models.baseline import SeasonalMedianBaseline  # noqa: E402
from bikeflow.ml.models.registry import save_bundle  # noqa: E402
from bikeflow.ml.training.tracking import (  # noqa: E402
    artifact_location,
    flatten,
    input_example,
    log_training_run,
    tracking_uri,
)

METRICS = {
    "n": 24,
    "mae": 12.0,
    "wape": 0.1,
    "rmse": 15.0,
    "r2": 0.9,
    "wce": 20.0,
    "mean_actual": 120.0,
    "mean_predicted": 121.0,
    "bias": 1.0,
}

RESULTS = {
    "seasonal_median": {
        "train": METRICS,
        "validation": METRICS,
        "test": {**METRICS, "mae": 14.0},
    },
    "hgb": {"train": METRICS, "validation": {**METRICS, "mae": 9.0}},
}

CV_SCORES = pd.DataFrame(
    [
        {"fold": 1, "model": "seasonal_median", "mae": 300.0, "wape": 0.30},
        {"fold": 2, "model": "seasonal_median", "mae": 320.0, "wape": 0.32},
        {"fold": 1, "model": "hgb", "mae": 400.0, "wape": 0.40},
        {"fold": 2, "model": "hgb", "mae": 420.0, "wape": 0.42},
    ]
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An isolated SQLite tracking store, never the project's mlflow.db."""
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


@pytest.fixture
def trained(tmp_path, hgb_rows):
    """A real fitted model saved as a bundle, plus the reports a training writes."""
    features = build_features(coerce_input(hgb_rows))
    target = 100.0 + 10.0 * features["hour"].to_numpy(dtype="float64")
    model = SeasonalMedianBaseline().fit(features, target)
    bundle = save_bundle(tmp_path / "model.joblib", model, metrics={}, data_sha256="a" * 64)

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text('{"main_model": "seasonal_median"}\n', encoding="utf-8")
    (reports / "model_selection.md").write_text("# Выбор модели\n", encoding="utf-8")
    return bundle, reports


def log(trained, hgb_rows):
    bundle, reports = trained
    return log_training_run(RESULTS, CV_SCORES, "seasonal_median", bundle, hgb_rows, reports)


def test_run_records_params_metrics_tags_and_reports(store, trained, hgb_rows):
    run_id, _ = log(trained, hgb_rows)
    run = MlflowClient(store).get_run(run_id)

    assert run.data.params["seed"] == str(load_config()["seed"])
    assert run.data.params["selection.strategy"] == "rolling_cv"
    assert "models.mlp.hidden" in run.data.params

    metrics = run.data.metrics
    assert metrics["validation_mae"] == 12.0
    assert metrics["test_mae"] == 14.0
    assert metrics["cv_mae_seasonal_median"] == 310.0
    assert metrics["cv_mae_hgb"] == 410.0
    assert metrics["cv_mae"] == 310.0

    assert run.data.tags["champion"] == "seasonal_median"
    assert run.data.tags["data_sha256"] == "a" * 64
    assert run.data.tags["model_version"].startswith("seasonal_median-")

    logged = {item.path for item in MlflowClient(store).list_artifacts(run_id, "reports")}
    assert logged == {"reports/metrics.json", "reports/model_selection.md"}


def test_champion_is_registered_and_predicts_like_the_bundle(store, trained, hgb_rows):
    _, version = log(trained, hgb_rows)
    tracking = load_config()["tracking"]

    client = MlflowClient(store)
    alias = client.get_model_version_by_alias(tracking["registered_model"], tracking["alias"])
    assert str(alias.version) == version == "1"

    example = input_example(hgb_rows)
    registered = mlflow.pyfunc.load_model(
        f"models:/{tracking['registered_model']}@{tracking['alias']}"
    )
    from_registry = np.asarray(registered.predict(example), dtype="float64")
    from_bundle = Predictor.load(trained[0]).predict(example)

    assert np.allclose(from_registry, from_bundle)
    assert (from_registry >= 0).all()


def test_next_training_gets_a_new_version_and_the_alias(store, trained, hgb_rows):
    log(trained, hgb_rows)
    _, second = log(trained, hgb_rows)
    tracking = load_config()["tracking"]

    alias = MlflowClient(store).get_model_version_by_alias(
        tracking["registered_model"], tracking["alias"]
    )
    assert second == "2"
    assert str(alias.version) == "2"


def test_local_store_keeps_artifacts_beside_the_database(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking_uri().startswith("sqlite:///")
    assert tracking_uri().endswith("/mlflow.db")

    assert artifact_location("sqlite:///C:/store/mlflow.db").endswith("/store/mlruns")
    assert artifact_location("http://mlflow:5000") is None


def test_flatten_turns_sections_into_dotted_names():
    flat = flatten({"mlp": {"hidden": [128, 64], "dropout": 0.1}, "seed": 42})

    assert flat == {"mlp.hidden": "[128, 64]", "mlp.dropout": 0.1, "seed": 42}


def test_input_example_uses_request_columns_and_iso_dates(hgb_rows):
    example = input_example(hgb_rows, rows=3)

    assert len(example) == 3
    assert "rented_bike_count" not in example.columns
    assert example["date"].iloc[0] == "2018-07-01"
