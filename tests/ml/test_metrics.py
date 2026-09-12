import numpy as np
import pandas as pd
import pytest

from bikeflow.ml.metrics import (
    evaluate,
    evaluate_by_slice,
    mae,
    rmse,
    wape,
    weighted_cost_error,
)

# Hand-checked example: errors are -10, +20, -30, +0
ACTUAL = [100.0, 200.0, 300.0, 400.0]
PREDICTED = [90.0, 220.0, 270.0, 400.0]


def test_mae_matches_hand_calculation():
    # (10 + 20 + 30 + 0) / 4
    assert mae(ACTUAL, PREDICTED) == pytest.approx(15.0)


def test_wape_matches_hand_calculation():
    # 60 / 1000
    assert wape(ACTUAL, PREDICTED) == pytest.approx(0.06)


def test_rmse_matches_hand_calculation():
    # sqrt((100 + 400 + 900 + 0) / 4)
    assert rmse(ACTUAL, PREDICTED) == pytest.approx(np.sqrt(350.0))


def test_weighted_cost_penalises_underforecast_more():
    # under = 10 + 30 = 40, over = 20; (3*40 + 1*20) / 4
    assert weighted_cost_error(ACTUAL, PREDICTED, 3.0, 1.0) == pytest.approx(35.0)
    # With symmetric weights it collapses to MAE.
    assert weighted_cost_error(ACTUAL, PREDICTED, 1.0, 1.0) == pytest.approx(mae(ACTUAL, PREDICTED))


def test_perfect_prediction():
    scores = evaluate(ACTUAL, ACTUAL)
    assert scores["mae"] == 0.0
    assert scores["wape"] == 0.0
    assert scores["r2"] == pytest.approx(1.0)
    assert scores["bias"] == 0.0


def test_bias_sign():
    assert evaluate([100.0], [150.0])["bias"] == pytest.approx(50.0)
    assert evaluate([100.0], [50.0])["bias"] == pytest.approx(-50.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mae([1.0, 2.0], [1.0])


def test_empty_sample_is_rejected():
    with pytest.raises(ValueError, match="empty sample"):
        mae([], [])


def test_slice_splits_the_sample():
    frame = pd.DataFrame({"season": ["Winter", "Winter", "Summer", "Summer"]})
    result = evaluate_by_slice(frame, ACTUAL, PREDICTED, "season")

    assert list(result["season"]) == ["Summer", "Winter"]
    assert result.set_index("season").loc["Winter", "n"] == 2
    # Winter rows: errors 10 and 20 -> MAE 15
    assert result.set_index("season").loc["Winter", "mae"] == pytest.approx(15.0)
    # Summer rows: errors 30 and 0 -> MAE 15
    assert result.set_index("season").loc["Summer", "mae"] == pytest.approx(15.0)


def test_slice_length_mismatch_is_rejected():
    frame = pd.DataFrame({"season": ["Winter"]})
    with pytest.raises(ValueError, match="predictions"):
        evaluate_by_slice(frame, ACTUAL, PREDICTED, "season")
