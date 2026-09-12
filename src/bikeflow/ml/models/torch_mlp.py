"""PyTorch MLP: the main model.

Two encodings of the categorical features are supported and both are trained, so
the choice between them is made on the validation split rather than by taste:

  onehot     hour/dow/month/season -> 47 indicator columns
  embedding  each level gets a small trainable vector, so the network can learn
             that 08:00 and 09:00 behave alike instead of treating every hour as
             an unrelated category

Everything runs on CPU and is seeded end to end: a rerun must reproduce the same
metrics, otherwise the numbers in the report mean nothing.
"""

from __future__ import annotations

import io
import math
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import load_config
from ..features import CATEGORY_LEVELS, categorical_columns, numeric_columns
from ..metrics import mae

DEVICE = torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def softplus_inverse(value: float) -> float:
    """Bias that makes softplus(bias) == value.

    log(expm1(x)) overflows above x ~= 709, and demand means of that size are
    perfectly reachable if the split dates move. Above ~20 the two branches agree
    to double precision anyway, so fall through to the identity there.
    """
    value = max(float(value), 1e-3)
    return value if value > 20.0 else math.log(math.expm1(value))


def make_loss(name: str) -> nn.Module:
    """Training objective.

    `l1` optimises the conditional median, which is exactly what MAE — our
    selection and reporting metric — rewards. `poisson` optimises the
    conditional mean instead; on right-skewed count data the two disagree, so
    the choice is made on the validation split and recorded in params.yaml.
    """
    if name == "l1":
        return nn.L1Loss()
    if name == "poisson":
        return nn.PoissonNLLLoss(log_input=False, full=False, eps=1e-8)
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss {name!r}; use 'l1', 'poisson' or 'mse'.")


def encode_categoricals(features: pd.DataFrame) -> np.ndarray:
    """Map each categorical column onto contiguous integer codes."""
    columns = []
    for name in categorical_columns():
        lookup = {level: code for code, level in enumerate(CATEGORY_LEVELS[name])}
        codes = features[name].map(lookup)
        if codes.isna().any():
            bad = features[name][codes.isna()].iloc[0]
            raise ValueError(f"Feature '{name}': unknown level {bad!r}.")
        columns.append(codes.to_numpy(dtype="int64"))
    return np.stack(columns, axis=1)


class _Net(nn.Module):
    def __init__(
        self,
        encoding: str,
        cardinalities: list[int],
        embedding_dims: list[int],
        n_numeric: int,
        hidden: list[int],
        dropout: float,
        output_bias: float,
    ) -> None:
        super().__init__()
        self.encoding = encoding
        self.cardinalities = cardinalities

        if encoding == "embedding":
            self.embeddings = nn.ModuleList(
                nn.Embedding(card, dim)
                for card, dim in zip(cardinalities, embedding_dims, strict=True)
            )
            cat_width = sum(embedding_dims)
        elif encoding == "onehot":
            self.embeddings = None
            cat_width = sum(cardinalities)
        else:
            raise ValueError(f"Unknown encoding {encoding!r}; use 'onehot' or 'embedding'.")

        layers: list[nn.Module] = []
        width = cat_width + n_numeric
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        head = nn.Linear(width, 1)
        # Start the network at roughly the average demand so the Poisson loss does
        # not have to climb from ~0 up to several hundred rentals.
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, output_bias)
        layers.append(head)

        self.body = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    @torch.no_grad()
    def neutralise_unseen(self, seen: list[set[int]]) -> int:
        """Zero the weights belonging to categorical levels absent from training.

        A temporal split guarantees some levels never appear before the cutoff —
        season "Autumn" occurs only from October, i.e. on every single test row.
        Those weights receive no gradient, so whatever the initialiser happened
        to put there would leak seed-dependent noise into exactly the rows we
        care about. Zeroing them makes an unseen level mean "no information",
        which is deterministic and explainable.
        """
        zeroed = 0
        if self.embeddings is not None:
            for index, embedding in enumerate(self.embeddings):
                for level in range(self.cardinalities[index]):
                    if level not in seen[index]:
                        embedding.weight[level].zero_()
                        zeroed += 1
            return zeroed

        first = next(layer for layer in self.body if isinstance(layer, nn.Linear))
        offset = 0
        for index, cardinality in enumerate(self.cardinalities):
            for level in range(cardinality):
                if level not in seen[index]:
                    first.weight[:, offset + level].zero_()
                    zeroed += 1
            offset += cardinality
        return zeroed

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        if self.embeddings is not None:
            pieces = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        else:
            pieces = [
                nn.functional.one_hot(x_cat[:, i], num_classes=card).float()
                for i, card in enumerate(self.cardinalities)
            ]
        x = torch.cat(pieces + [x_num], dim=1)
        return self.softplus(self.body(x)).squeeze(1)


class TorchMLPRegressor:
    """fit/predict wrapper around `_Net`, picklable through joblib."""

    def __init__(self, encoding: str, params: dict | None = None, seed: int | None = None):
        cfg = load_config()
        self.encoding = encoding
        self.params = dict(params or cfg["models"]["mlp"])
        self.seed = cfg["seed"] if seed is None else seed
        self.kind = f"mlp_{encoding}"

        self.net_: _Net | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.history_: list[dict[str, float]] = []
        self.best_epoch_: int | None = None
        self.seen_levels_: list[set[int]] | None = None

    # ---- internals -------------------------------------------------------

    def _tensors(self, features: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        cat = torch.from_numpy(encode_categoricals(features))
        num = features[numeric_columns()].to_numpy(dtype="float64")
        num = (num - self.mean_) / self.std_
        return cat, torch.from_numpy(num.astype("float32"))

    # ---- public API ------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        y,
        eval_features: pd.DataFrame,
        eval_y,
    ) -> TorchMLPRegressor:
        set_seed(self.seed)

        numeric = features[numeric_columns()].to_numpy(dtype="float64")
        self.mean_ = numeric.mean(axis=0)
        self.std_ = np.where(numeric.std(axis=0) < 1e-9, 1.0, numeric.std(axis=0))

        y_train = np.asarray(y, dtype="float32")
        y_eval = np.asarray(eval_y, dtype="float64")

        x_cat, x_num = self._tensors(features)
        e_cat, e_num = self._tensors(eval_features)
        target = torch.from_numpy(y_train)

        cardinalities = [len(CATEGORY_LEVELS[n]) for n in categorical_columns()]
        embedding_dims = [self.params["embedding_dims"][n] for n in categorical_columns()]
        mean_target = float(y_train.mean())

        self.net_ = _Net(
            encoding=self.encoding,
            cardinalities=cardinalities,
            embedding_dims=embedding_dims,
            n_numeric=len(numeric_columns()),
            hidden=list(self.params["hidden"]),
            dropout=float(self.params["dropout"]),
            # so the initial prediction equals the training mean
            output_bias=softplus_inverse(mean_target),
        ).to(DEVICE)

        optimiser = torch.optim.Adam(
            self.net_.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        loss_fn = make_loss(str(self.params.get("loss", "poisson")))

        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(x_cat, x_num, target),
            batch_size=int(self.params["batch_size"]),
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

        patience = int(self.params["patience"])
        best_score, best_state, stale = math.inf, None, 0

        for epoch in range(1, int(self.params["max_epochs"]) + 1):
            self.net_.train()
            running = 0.0
            for batch_cat, batch_num, batch_y in loader:
                optimiser.zero_grad()
                predicted = self.net_(batch_cat, batch_num)
                loss = loss_fn(predicted, batch_y)
                loss.backward()
                optimiser.step()
                running += loss.detach().item() * len(batch_y)

            self.net_.eval()
            with torch.no_grad():
                eval_pred = self.net_(e_cat, e_num).numpy().astype("float64")
            score = mae(y_eval, eval_pred)
            self.history_.append(
                {"epoch": epoch, "train_loss": running / len(target), "val_mae": score}
            )

            if score < best_score - 1e-6:
                best_score, stale = score, 0
                self.best_epoch_ = epoch
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)

        train_codes = encode_categoricals(features)
        self.seen_levels_ = [
            set(np.unique(train_codes[:, i]).tolist()) for i in range(train_codes.shape[1])
        ]
        zeroed = self.net_.neutralise_unseen(self.seen_levels_)

        self.net_.eval()
        print(
            f"[{self.kind}] stopped at epoch {len(self.history_)}, "
            f"best epoch {self.best_epoch_}, validation MAE {best_score:.2f}, "
            f"{zeroed} unseen level(s) neutralised"
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError(f"{self.kind} is not fitted.")
        cat, num = self._tensors(features)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(cat, num).numpy().astype("float64")
        return np.clip(out, 0.0, None)

    def n_parameters(self) -> int:
        if self.net_ is None:
            return 0
        return sum(p.numel() for p in self.net_.parameters())

    def embedding_table(self, column: str) -> np.ndarray | None:
        """Learned vectors for one categorical column (embedding encoding only)."""
        if self.net_ is None or self.net_.embeddings is None:
            return None
        index = categorical_columns().index(column)
        return self.net_.embeddings[index].weight.detach().numpy()

    # ---- serialisation ---------------------------------------------------
    # Store a state_dict rather than the live module: joblib then handles every
    # model kind the same way, and loading does not depend on torch internals.

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        net = state.pop("net_")
        if net is None:
            state["_net_blob"] = None
            state["_net_config"] = None
        else:
            buffer = io.BytesIO()
            torch.save(net.state_dict(), buffer)
            state["_net_blob"] = buffer.getvalue()
            state["_net_config"] = {
                "encoding": net.encoding,
                "cardinalities": net.cardinalities,
                "embedding_dims": [self.params["embedding_dims"][n] for n in categorical_columns()],
                "n_numeric": len(numeric_columns()),
                "hidden": list(self.params["hidden"]),
                "dropout": float(self.params["dropout"]),
                "output_bias": 0.0,
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        blob = state.pop("_net_blob", None)
        config = state.pop("_net_config", None)
        self.__dict__.update(state)
        if blob is None or config is None:
            self.net_ = None
            return
        net = _Net(**config)
        net.load_state_dict(torch.load(io.BytesIO(blob), weights_only=True))
        net.eval()
        self.net_ = net
