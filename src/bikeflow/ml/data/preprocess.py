"""Raw CSV -> canonical hourly frame, with a hard data contract check."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import ensure_dir, load_config, resolve
from ..features import SEASONS, TARGET, add_calendar_features
from .download import raw_csv_path

#: Raw UCI header -> canonical snake_case name.
COLUMN_RENAMES: dict[str, str] = {
    "Date": "date",
    "Rented Bike Count": TARGET,
    "Hour": "hour",
    "Temperature(\N{DEGREE SIGN}C)": "temperature",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "wind_speed",
    "Visibility (10m)": "visibility",
    "Dew point temperature(\N{DEGREE SIGN}C)": "dew_point",
    "Solar Radiation (MJ/m2)": "solar_radiation",
    "Rainfall(mm)": "rainfall",
    "Snowfall (cm)": "snowfall",
    "Seasons": "season",
    "Holiday": "is_holiday",
    "Functioning Day": "is_functioning",
}

CANONICAL_COLUMNS = [
    "timestamp",
    "date",
    "hour",
    TARGET,
    "temperature",
    "humidity",
    "wind_speed",
    "visibility",
    "dew_point",
    "solar_radiation",
    "rainfall",
    "snowfall",
    "season",
    "is_holiday",
    "is_functioning",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
]


class DataContractError(ValueError):
    """Raised when the dataset violates an assumption the pipeline relies on."""


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV. The file is cp1252-encoded and uses DD/MM/YYYY dates."""
    cfg = load_config()["data"]
    source = Path(path) if path is not None else raw_csv_path()
    if not source.exists():
        raise FileNotFoundError(f"{source} not found. Run `python -m bikeflow download` first.")
    return pd.read_csv(source, encoding=cfg["encoding"])


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename, retype and enrich the raw frame."""
    unknown = set(COLUMN_RENAMES) - set(raw.columns)
    if unknown:
        raise DataContractError(
            f"Raw file is missing expected column(s): {sorted(unknown)}. Found: {list(raw.columns)}"
        )

    frame = raw.rename(columns=COLUMN_RENAMES)[list(COLUMN_RENAMES.values())].copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y")
    frame["hour"] = frame["hour"].astype("int64")
    frame[TARGET] = frame[TARGET].astype("int64")
    frame["season"] = frame["season"].astype(str).str.strip()
    frame["is_holiday"] = frame["is_holiday"].astype(str).str.strip().eq("Holiday")
    frame["is_functioning"] = frame["is_functioning"].astype(str).str.strip().eq("Yes")
    frame["timestamp"] = frame["date"] + pd.to_timedelta(frame["hour"], unit="h")

    frame = add_calendar_features(frame)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[CANONICAL_COLUMNS]


def validate(frame: pd.DataFrame) -> None:
    """Fail loudly if any downstream assumption is broken."""
    cfg = load_config()["data"]
    problems: list[str] = []

    if len(frame) != cfg["expected_rows"]:
        problems.append(f"expected {cfg['expected_rows']} rows, got {len(frame)}")

    nulls = frame.columns[frame.isna().any()].tolist()
    if nulls:
        problems.append(f"null values in {nulls}")

    duplicates = int(frame["timestamp"].duplicated().sum())
    if duplicates:
        problems.append(f"{duplicates} duplicated timestamps")

    gaps = frame["timestamp"].diff().dropna()
    if not gaps.empty and not (gaps == pd.Timedelta(hours=1)).all():
        bad = int((gaps != pd.Timedelta(hours=1)).sum())
        problems.append(f"hourly grid has {bad} gap(s)")

    if not frame["hour"].between(0, 23).all():
        problems.append("hour outside [0, 23]")

    if (frame[TARGET] < 0).any():
        problems.append("negative target values")

    seasons = set(frame["season"].unique())
    if seasons != set(SEASONS):
        problems.append(f"unexpected seasons {sorted(seasons)}, expected {list(SEASONS)}")

    # The rule the inference path relies on: closed hours are always exactly zero.
    closed = frame.loc[~frame["is_functioning"], TARGET]
    if len(closed) and (closed != 0).any():
        problems.append(
            f"{int((closed != 0).sum())} non-functioning hour(s) have a non-zero target"
        )
    open_hours = frame.loc[frame["is_functioning"], TARGET]
    if len(open_hours) and (open_hours == 0).any():
        problems.append(
            f"{int((open_hours == 0).sum())} functioning hour(s) have a zero target; "
            "the closed-hours rule may no longer hold"
        )

    if problems:
        raise DataContractError("Data contract violated: " + "; ".join(problems))


def preprocess(save: bool = True) -> pd.DataFrame:
    """Full raw -> processed step."""
    frame = to_canonical(load_raw())
    validate(frame)

    if save:
        out_dir = ensure_dir(load_config()["data"]["processed_dir"])
        destination = out_dir / "dataset.parquet"
        frame.to_parquet(destination, index=False)
        print(f"[preprocess] {len(frame)} rows -> {destination}")
        print(
            f"[preprocess] period {frame['timestamp'].min()} .. {frame['timestamp'].max()}, "
            f"{int((~frame['is_functioning']).sum())} non-functioning hours"
        )
    return frame


def load_processed() -> pd.DataFrame:
    path = resolve(load_config()["data"]["processed_dir"]) / "dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m bikeflow preprocess` first.")
    return pd.read_parquet(path)
