import pytest

from bikeflow.model.protocol import Predictor
from bikeflow.model.stub import StubPredictor


def test_stub_implements_predictor_and_returns_fixed_value() -> None:
    predictor = StubPredictor(fixed_prediction=17.5)

    assert isinstance(predictor, Predictor)
    assert predictor.predict({"hour": 8}) == 17.5
    assert predictor.model_version == "stub-v0"


def test_stub_rejects_negative_prediction() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        StubPredictor(fixed_prediction=-1)
