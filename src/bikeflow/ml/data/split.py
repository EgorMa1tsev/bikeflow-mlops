"""Temporal train/validation/test split.

The data is a single hourly series, so splitting is done strictly by date. Never
shuffle: a random split would leak future information into training.
"""

from __future__ import annotations

import pandas as pd

from ..config import ensure_dir, load_config, resolve
from ..features import TARGET

SPLIT_NAMES = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised when the configured split boundaries are inconsistent."""


def split_bounds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    cfg = load_config()["split"]
    return {name: (pd.Timestamp(cfg[name][0]), pd.Timestamp(cfg[name][1])) for name in SPLIT_NAMES}


def temporal_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cut the frame into the three configured date windows."""
    bounds = split_bounds()
    parts: dict[str, pd.DataFrame] = {}

    for name, (start, end) in bounds.items():
        # `end` is an inclusive date, so extend it to the last hour of that day.
        end_inclusive = end + pd.Timedelta(hours=23)
        mask = frame["timestamp"].between(start, end_inclusive)
        part = frame.loc[mask].sort_values("timestamp").reset_index(drop=True)
        if part.empty:
            raise SplitError(f"Split '{name}' ({start.date()}..{end.date()}) is empty.")
        parts[name] = part

    previous_name, previous = None, None
    for name in SPLIT_NAMES:
        current = parts[name]
        if previous is not None and current["timestamp"].min() <= previous["timestamp"].max():
            raise SplitError(
                f"Split '{name}' overlaps '{previous_name}': "
                f"{current['timestamp'].min()} <= {previous['timestamp'].max()}"
            )
        previous_name, previous = name, current

    covered = sum(len(p) for p in parts.values())
    if covered != len(frame):
        print(
            f"[split] note: {len(frame) - covered} row(s) fall outside the configured "
            "windows and are not used"
        )
    return parts


def summarise(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, part in parts.items():
        working = part[part["is_functioning"]]
        rows.append(
            {
                "split": name,
                "start": part["timestamp"].min().date(),
                "end": part["timestamp"].max().date(),
                "rows": len(part),
                "functioning": len(working),
                "mean_target": round(float(working[TARGET].mean()), 1),
                "seasons": ", ".join(sorted(part["season"].unique())),
            }
        )
    return pd.DataFrame(rows)


def run_split(frame: pd.DataFrame | None = None, save: bool = True) -> dict[str, pd.DataFrame]:
    if frame is None:
        from .preprocess import load_processed

        frame = load_processed()

    parts = temporal_split(frame)

    if save:
        out_dir = ensure_dir(load_config()["data"]["processed_dir"])
        for name, part in parts.items():
            part.to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"[split] wrote {', '.join(SPLIT_NAMES)} to {out_dir}")

    print(summarise(parts).to_string(index=False))
    return parts


def load_splits() -> dict[str, pd.DataFrame]:
    out_dir = resolve(load_config()["data"]["processed_dir"])
    parts = {}
    for name in SPLIT_NAMES:
        path = out_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m bikeflow split` first.")
        parts[name] = pd.read_parquet(path)
    return parts
