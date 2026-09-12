"""Gradient boosting reference model.

Kept alongside the neural network as an honest point of comparison and as a
ready-made Challenger for the Champion/Challenger gate in stage 7.

sklearn's built-in early stopping carves its validation set out at random, which
would break the temporal discipline of this project, so stopping is driven here
by our own chronological validation split instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import load_config
from ..features import CATEGORY_LEVELS, categorical_columns
from ..metrics import mae

_EVAL_STEP = 25


def as_categorical(features: pd.DataFrame) -> pd.DataFrame:
    """Give the categorical columns a pandas category dtype with fixed levels."""
    out = features.copy()
    for name in categorical_columns():
        out[name] = pd.Categorical(out[name], categories=CATEGORY_LEVELS[name])
    return out


class HGBModel:
    """Thin wrapper giving HistGradientBoostingRegressor our fit/predict shape."""

    kind = "hgb"

    def __init__(self, params: dict | None = None, seed: int | None = None) -> None:
        cfg = load_config()
        self.params = dict(params or cfg["models"]["hgb"])
        self.seed = cfg["seed"] if seed is None else seed
        self.model_: HistGradientBoostingRegressor | None = None
        self.best_iter_: int | None = None

    def _make(self, max_iter: int, warm_start: bool) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss=self.params["loss"],
            max_iter=max_iter,
            learning_rate=self.params["learning_rate"],
            max_leaf_nodes=self.params["max_leaf_nodes"],
            min_samples_leaf=self.params["min_samples_leaf"],
            l2_regularization=self.params["l2_regularization"],
            categorical_features="from_dtype",
            early_stopping=False,
            warm_start=warm_start,
            random_state=self.seed,
        )

    def fit(
        self,
        features: pd.DataFrame,
        y,
        eval_features: pd.DataFrame | None = None,
        eval_y=None,
    ) -> HGBModel:
        x_train = as_categorical(features)
        y_train = np.asarray(y, dtype="float64")
        max_iter = int(self.params["max_iter"])

        if eval_features is None:
            self.best_iter_ = max_iter
        else:
            x_eval = as_categorical(eval_features)
            y_eval = np.asarray(eval_y, dtype="float64")
            patience = int(self.params["n_iter_no_change"])

            probe = self._make(max_iter=_EVAL_STEP, warm_start=True)
            best_score, best_iter, stale = np.inf, _EVAL_STEP, 0
            for n_iter in range(_EVAL_STEP, max_iter + 1, _EVAL_STEP):
                probe.set_params(max_iter=n_iter)
                probe.fit(x_train, y_train)
                score = mae(y_eval, probe.predict(x_eval))
                if score < best_score - 1e-9:
                    best_score, best_iter, stale = score, n_iter, 0
                else:
                    stale += _EVAL_STEP
                    if stale >= patience:
                        break
            self.best_iter_ = best_iter
            print(f"[hgb] best_iter={best_iter} (validation MAE {best_score:.2f})")

        self.model_ = self._make(max_iter=self.best_iter_, warm_start=False)
        self.model_.fit(x_train, y_train)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("HGB model is not fitted.")
        return np.clip(self.model_.predict(as_categorical(features)), 0.0, None)
