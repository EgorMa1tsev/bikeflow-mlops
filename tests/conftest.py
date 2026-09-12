"""Fast, deterministic fixtures shared by API and ML integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bikeflow.ml.config import load_config
from bikeflow.ml.features import build_features, coerce_input
from bikeflow.ml.models.hgb import HGBModel
from bikeflow.ml.models.registry import save_bundle
from bikeflow.ml.models.torch_mlp import TorchMLPRegressor


@pytest.fixture(scope="session")
def hgb_rows() -> pd.DataFrame:
    """Small canonical dataset with enough variation to fit a real HGB."""

    rows = []
    for index in range(96):
        hour = index % 24
        rows.append(
            {
                "date": (pd.Timestamp("2018-07-01") + pd.Timedelta(hours=index)).date().isoformat(),
                "hour": hour,
                "temperature": 18.0 + 8.0 * np.sin(hour / 24 * 2 * np.pi),
                "humidity": 55.0 + index % 20,
                "wind_speed": 1.0 + (index % 5) / 10,
                "visibility": 1800.0,
                "dew_point": 10.0,
                "solar_radiation": max(0.0, np.sin((hour - 6) / 12 * np.pi)),
                "rainfall": 1.0 if index % 19 == 0 else 0.0,
                "snowfall": 0.0,
                "season": "Summer",
                "is_holiday": False,
                "is_functioning": True,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def real_hgb_artifact(tmp_path, hgb_rows):
    features = build_features(coerce_input(hgb_rows))
    target = 180 + 12 * features["hour"].to_numpy() + 2 * hgb_rows["temperature"].to_numpy()
    params = {
        "loss": "poisson",
        "max_iter": 25,
        "learning_rate": 0.1,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 2,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "n_iter_no_change": 25,
    }
    model = HGBModel(params=params, seed=42).fit(features, target)
    return save_bundle(
        tmp_path / "hgb.joblib",
        model,
        metrics={"validation": {"mae": 1.0, "wape": 0.01}},
        data_sha256="a" * 64,
        training_params=params,
    )


@pytest.fixture(scope="session")
def real_mlp_artifact(tmp_path_factory, hgb_rows):
    """A small production-shaped MLP bundle with preprocessing and weights."""
    features = build_features(coerce_input(hgb_rows))
    target = 180 + 12 * features["hour"].to_numpy() + 2 * hgb_rows["temperature"].to_numpy()
    params = dict(load_config()["models"]["mlp"])
    params.update(
        hidden=[16, 8],
        dropout=0.0,
        learning_rate=0.01,
        batch_size=32,
        max_epochs=12,
        patience=4,
    )
    model = TorchMLPRegressor(encoding="embedding", params=params, seed=42)
    model.fit(features.iloc[:72], target[:72], features.iloc[72:], target[72:])
    destination = tmp_path_factory.mktemp("mlp-artifact") / "model.joblib"
    return save_bundle(
        destination,
        model,
        metrics={"validation": {"mae": 1.0, "wape": 0.01}},
        data_sha256="b" * 64,
        training_params=params,
    )
