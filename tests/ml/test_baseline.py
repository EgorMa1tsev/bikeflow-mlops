import numpy as np
import pandas as pd
import pytest

from bikeflow.ml.models.baseline import SeasonalMedianBaseline


def _features(pairs) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["hour", "day_of_week"])


def test_returns_the_median_of_the_matching_group():
    features = _features([(8, 0), (8, 0), (8, 0), (18, 0)])
    model = SeasonalMedianBaseline().fit(features, [100, 200, 300, 900])

    predictions = model.predict(_features([(8, 0), (18, 0)]))
    assert predictions.tolist() == [200.0, 900.0]


def test_falls_back_to_the_hour_when_the_weekday_is_unseen():
    features = _features([(8, 0), (8, 1)])
    model = SeasonalMedianBaseline().fit(features, [100, 300])

    # (8, 5) was never observed, but hour 8 was: median of 100 and 300.
    assert model.predict(_features([(8, 5)]))[0] == 200.0


def test_falls_back_to_the_global_median_when_the_hour_is_unseen():
    features = _features([(8, 0), (9, 0), (10, 0)])
    model = SeasonalMedianBaseline().fit(features, [100, 200, 600])

    assert model.predict(_features([(23, 6)]))[0] == 200.0


def test_predictions_are_never_negative_or_nan():
    features = _features([(h, d) for h in range(24) for d in range(7)])
    model = SeasonalMedianBaseline().fit(features, np.zeros(len(features)))

    predictions = model.predict(features)
    assert not np.isnan(predictions).any()
    assert (predictions >= 0).all()


def test_softplus_inverse_survives_large_means():
    """Regression: log(expm1(x)) overflows past ~709, and demand means get there."""
    import math

    pytest.importorskip("torch", reason="the production MLP test requires the mlp extra")
    from bikeflow.ml.models.torch_mlp import softplus_inverse

    # Small values keep the exact formula.
    assert softplus_inverse(1.0) == pytest.approx(math.log(math.expm1(1.0)))
    # Large ones must not raise, and softplus(bias) must still return the mean.
    for mean in (645.0, 729.0, 5000.0):
        bias = softplus_inverse(mean)
        assert math.isfinite(bias)
        assert bias == pytest.approx(mean, rel=1e-9)
