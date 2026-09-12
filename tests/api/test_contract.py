import pytest
from pydantic import ValidationError

from bikeflow.api.schemas import PredictionRequest
from bikeflow.ml.features import FeatureValidationError, api_input_contract, coerce_input
from bikeflow.model.adapter import to_canonical_row


def test_api_and_model_reject_the_same_numeric_bounds() -> None:
    base = {
        "prediction_time": "2026-07-15T08:00:00+09:00",
        "temperature_c": 24.5,
        "humidity_pct": 61,
        "wind_speed_m_s": 1.8,
        "visibility_10m": 1800,
        "dew_point_c": 16.4,
        "solar_radiation_mj_m2": 1.2,
        "rainfall_mm": 0,
        "snowfall_cm": 0,
    }
    for api_name, spec in api_input_contract().items():
        if "max" not in spec:
            continue
        invalid = spec["max"] + 1
        with pytest.raises(ValidationError):
            PredictionRequest.model_validate({**base, api_name: invalid})

        valid = PredictionRequest.model_validate(base)
        canonical = to_canonical_row(valid.to_features())
        canonical[spec["canonical_name"]] = invalid
        with pytest.raises(FeatureValidationError):
            coerce_input(canonical)
