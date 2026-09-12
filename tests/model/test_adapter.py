"""The seam between the API contract and the trained artifact."""

import pytest

from bikeflow.api.schemas import PredictionRequest
from bikeflow.model.protocol import Predictor

pytest.importorskip("pandas", reason="requires the `ml` extra")

from bikeflow.ml.features import build_features, coerce_input  # noqa: E402
from bikeflow.ml.models.baseline import SeasonalMedianBaseline  # noqa: E402
from bikeflow.ml.models.registry import save_bundle  # noqa: E402
from bikeflow.model.adapter import BikeflowPredictor, to_canonical_row  # noqa: E402

REQUEST = {
    "prediction_time": "2018-12-01T18:00:00+09:00",
    "temperature_c": 3.5,
    "humidity_pct": 45,
    "wind_speed_m_s": 1.2,
    "visibility_10m": 2000,
    "dew_point_c": -7.0,
    "solar_radiation_mj_m2": 0.0,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
    "holiday": False,
    "functioning_day": True,
}


@pytest.fixture
def artifact(tmp_path):
    """A cheap fitted model stored in the real artifact format."""
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "date": "2018-12-01",
                "hour": hour,
                "temperature": 3.5,
                "humidity": 45,
                "wind_speed": 1.2,
                "visibility": 2000,
                "dew_point": -7.0,
                "solar_radiation": 0.0,
                "rainfall": 0.0,
                "snowfall": 0.0,
                "season": "Winter",
            }
            for hour in range(24)
        ]
    )
    features = build_features(coerce_input(rows))
    model = SeasonalMedianBaseline().fit(features, [100.0 * (h + 1) for h in range(24)])
    return save_bundle(tmp_path / "model.joblib", model, metrics={}, data_sha256="a" * 64)


def test_request_field_names_map_onto_training_names():
    row = to_canonical_row(PredictionRequest(**REQUEST).to_features())

    assert row["temperature"] == 3.5
    assert row["humidity"] == 45
    assert row["is_holiday"] is False
    assert row["is_functioning"] is True
    assert row["hour"] == 18
    assert row["season"] == "Winter"
    # 2018-12-01 was a Saturday.
    assert row["day_of_week"] == 5


def test_every_derived_field_is_recognised():
    """No field produced by the API may fall through unmapped."""
    to_canonical_row(PredictionRequest(**REQUEST).to_features())


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="Unknown feature field"):
        to_canonical_row({"temperature_c": 1.0, "wind_direction_deg": 180})


def test_adapter_satisfies_the_protocol(artifact):
    predictor = BikeflowPredictor(artifact)
    assert isinstance(predictor, Predictor)


def test_adapter_returns_one_non_negative_float(artifact):
    predictor = BikeflowPredictor(artifact)
    prediction = predictor.predict(PredictionRequest(**REQUEST).to_features())

    assert isinstance(prediction, float)
    assert prediction >= 0.0


def test_model_version_is_present_and_stable(artifact):
    predictor = BikeflowPredictor(artifact)

    assert predictor.model_version.startswith("seasonal_median-aaaaaaaa-")
    assert predictor.model_version == predictor.model_version


def test_closed_day_predicts_zero_through_the_api_contract(artifact):
    predictor = BikeflowPredictor(artifact)
    closed = predictor.predict(
        PredictionRequest(**{**REQUEST, "functioning_day": False}).to_features()
    )
    assert closed == 0.0
