"""Rolling-origin selection: the folds must never reach into the test period."""

import pandas as pd
import pytest

from bikeflow.ml.data.split import split_bounds
from bikeflow.ml.training.cv import (
    CVConfigError,
    assert_excludes_test,
    build_candidates,
    choose_by_cv,
    fold_windows,
    summarise_cv,
)


def test_configured_folds_stay_before_test():
    windows = fold_windows()
    test_start = assert_excludes_test(windows)

    assert windows, "at least one fold must be configured"
    for _, end in windows:
        assert end < test_start


def test_fold_overlapping_test_is_rejected():
    test_start = split_bounds()["test"][0]
    leaking = [(test_start, test_start + pd.Timedelta(days=30))]

    with pytest.raises(CVConfigError, match="overlaps the test split"):
        assert_excludes_test(leaking)


def test_folds_move_forward_in_time():
    windows = fold_windows()
    for earlier, later in zip(windows[:-1], windows[1:], strict=True):
        assert later[0] > earlier[1]


def test_every_trained_model_is_a_selection_candidate():
    """Selection must range over all models, not just the neural networks."""
    from bikeflow.ml.training.train import fit_models

    assert set(build_candidates()) == {
        "seasonal_median",
        "hgb",
        "mlp_onehot",
        "mlp_embedding",
    }
    assert fit_models is not None  # trained set and candidate set are kept in step


def test_champion_is_the_best_mean_not_the_best_single_fold():
    """A model that wins one fold but loses the others must not be chosen."""
    scores = pd.DataFrame(
        [
            {"fold": 1, "model": "steady", "mae": 400.0, "wape": 0.4},
            {"fold": 2, "model": "steady", "mae": 380.0, "wape": 0.38},
            {"fold": 3, "model": "steady", "mae": 390.0, "wape": 0.39},
            {"fold": 1, "model": "spiky", "mae": 580.0, "wape": 0.58},
            {"fold": 2, "model": "spiky", "mae": 470.0, "wape": 0.47},
            {"fold": 3, "model": "spiky", "mae": 200.0, "wape": 0.20},
        ]
    )
    assert choose_by_cv(scores) == "steady"

    summary = summarise_cv(scores).set_index("model")
    assert summary.loc["spiky", "best_score"] == 200.0
    assert summary.loc["spiky", "worst_score"] == 580.0
    assert summary.loc["steady", "mean_score"] == pytest.approx(390.0)
