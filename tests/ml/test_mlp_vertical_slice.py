"""The production MLP can be trained, packaged, loaded and served."""

import numpy as np

from bikeflow.ml.inference import Predictor
from bikeflow.ml.models.registry import load_bundle, save_bundle
from bikeflow.model.adapter import BikeflowPredictor


def test_mlp_train_save_load_predict_round_trip(real_mlp_artifact, hgb_rows) -> None:
    before = Predictor.load(real_mlp_artifact).predict(hgb_rows.head(4))
    bundle = load_bundle(real_mlp_artifact)
    assert bundle["kind"] == "mlp_embedding"
    assert bundle["feature_spec"]["scaler"] is not None

    second_path = save_bundle(
        real_mlp_artifact.parent / "roundtrip.joblib",
        bundle["pipeline"].model,
        metrics=bundle["metrics"],
        data_sha256="b" * 64,
        training_params=bundle["metadata"]["training_params"],
    )
    after = Predictor.load(second_path).predict(hgb_rows.head(4))
    assert np.allclose(before, after, rtol=1e-6, atol=1e-5)


def test_adapter_loads_once_and_never_retrains(real_mlp_artifact, monkeypatch) -> None:
    from bikeflow.ml.inference import Predictor as MLPredictor
    from bikeflow.ml.models.torch_mlp import TorchMLPRegressor

    original_load = MLPredictor.load.__func__
    calls = 0

    def counted_load(cls, path=None):
        nonlocal calls
        calls += 1
        return original_load(cls, path)

    monkeypatch.setattr(MLPredictor, "load", classmethod(counted_load))

    def fail_fit(*args, **kwargs):
        raise AssertionError("retraining in API")

    monkeypatch.setattr(TorchMLPRegressor, "fit", fail_fit)
    predictor = BikeflowPredictor(real_mlp_artifact)
    features = {
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
    assert predictor.predict(features) > 0
    assert predictor.predict(features) > 0
    assert predictor.model_version.startswith("mlp_embedding-")
    assert calls == 1
