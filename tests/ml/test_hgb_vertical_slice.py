"""The selected model can be trained, packaged, loaded, and served."""

import numpy as np

from bikeflow.ml.features import build_features, coerce_input
from bikeflow.ml.inference import Predictor
from bikeflow.ml.models.hgb import HGBModel
from bikeflow.ml.models.registry import load_bundle, save_bundle
from bikeflow.model.adapter import BikeflowPredictor


def test_hgb_trains_on_a_small_sample(hgb_rows) -> None:
    features = build_features(coerce_input(hgb_rows))
    target = 100 + 10 * features["hour"].to_numpy()
    model = HGBModel(
        params={
            "loss": "poisson",
            "max_iter": 10,
            "learning_rate": 0.1,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 2,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "n_iter_no_change": 10,
        },
        seed=42,
    ).fit(features, target)

    predictions = model.predict(features.head(3))
    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()


def test_train_save_load_predict_round_trip(real_hgb_artifact, hgb_rows) -> None:
    predictor = Predictor.load(real_hgb_artifact)
    before = predictor.predict(hgb_rows.iloc[0].to_dict())
    bundle = load_bundle(real_hgb_artifact)
    second_path = save_bundle(
        real_hgb_artifact.parent / "roundtrip.joblib",
        bundle["pipeline"].model,
        metrics=bundle["metrics"],
        data_sha256="a" * 64,
        training_params=bundle["metadata"]["training_params"],
    )
    after = Predictor.load(second_path).predict(hgb_rows.iloc[0].to_dict())
    assert np.allclose(before, after)


def test_real_adapter_wraps_hgb_artifact(real_hgb_artifact) -> None:
    predictor = BikeflowPredictor(real_hgb_artifact)
    prediction = predictor.predict(
        {
            "hour": 8,
            "day_of_week": 2,
            "season": "Summer",
            "temperature_c": 24.5,
            "humidity_pct": 61,
            "wind_speed_m_s": 1.8,
            "visibility_10m": 1800,
            "dew_point_c": 16.4,
            "solar_radiation_mj_m2": 1.2,
            "rainfall_mm": 0,
            "snowfall_cm": 0,
            "holiday": False,
            "functioning_day": True,
        }
    )
    assert predictor.model_version.startswith("hgb-")
    assert prediction > 0
