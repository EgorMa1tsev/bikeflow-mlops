"""Development-only predictor used until the ML artifact is available."""

from collections.abc import Mapping

from bikeflow.model.protocol import FeatureValue


class StubPredictor:
    """Return a fixed prediction while honoring the production interface."""

    def __init__(self, fixed_prediction: float = 42.0) -> None:
        if fixed_prediction < 0:
            raise ValueError("fixed_prediction must be non-negative")
        self._fixed_prediction = float(fixed_prediction)

    @property
    def model_version(self) -> str:
        """Identify the development stub."""

        return "stub-v0"

    def predict(self, features: Mapping[str, FeatureValue]) -> float:
        """Return the configured constant prediction."""

        del features
        return self._fixed_prediction
