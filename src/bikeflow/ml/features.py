"""Shared feature contract.

This module is imported by BOTH the training pipeline and (from stage 4 on) the
FastAPI service, so it must stay free of training-only concerns. Anything that
changes the meaning of a column here is a breaking change to the API contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config

SEASONS: tuple[str, ...] = ("Winter", "Spring", "Summer", "Autumn")

TARGET = "rented_bike_count"

#: Columns a caller must supply for a single observation, with allowed ranges.
#: Ranges are physical sanity bounds, deliberately wider than the training data.
FEATURE_CONTRACT: dict[str, dict[str, Any]] = {
    # ISO 8601 only (YYYY-MM-DD). The raw UCI file uses DD/MM/YYYY, but that is
    # parsed with an explicit format in preprocess; accepting both here would
    # make 01/12 silently ambiguous for callers of the API.
    #
    # Either `date` or `day_of_week` must be supplied. Training data carries the
    # full date; the API layer derives the weekday from `prediction_time` and
    # sends that instead, since the model uses no other calendar information.
    "date": {"kind": "date", "required": False, "default": None, "derived": True},
    "day_of_week": {
        "kind": "int",
        "required": False,
        "min": 0,
        "max": 6,
        "default": None,
        "derived": True,
    },
    "hour": {"kind": "int", "required": True, "min": 0, "max": 23, "derived": True},
    "temperature": {
        "api_name": "temperature_c",
        "kind": "float",
        "unit": "°C",
        "required": True,
        "min": -40.0,
        "max": 50.0,
    },
    "humidity": {
        "api_name": "humidity_pct",
        "kind": "float",
        "unit": "%",
        "required": True,
        "min": 0.0,
        "max": 100.0,
    },
    "wind_speed": {
        "api_name": "wind_speed_m_s",
        "kind": "float",
        "unit": "m/s",
        "required": True,
        "min": 0.0,
        "max": 50.0,
    },
    "visibility": {
        "api_name": "visibility_10m",
        "kind": "float",
        "unit": "10 m",
        "required": True,
        "min": 0.0,
        "max": 2000.0,
    },
    "dew_point": {
        "api_name": "dew_point_c",
        "kind": "float",
        "unit": "°C",
        "required": True,
        "min": -40.0,
        "max": 40.0,
    },
    "solar_radiation": {
        "api_name": "solar_radiation_mj_m2",
        "kind": "float",
        "unit": "MJ/m²",
        "required": True,
        "min": 0.0,
        "max": 10.0,
    },
    "rainfall": {
        "api_name": "rainfall_mm",
        "kind": "float",
        "unit": "mm",
        "required": True,
        "min": 0.0,
        "max": 200.0,
    },
    "snowfall": {
        "api_name": "snowfall_cm",
        "kind": "float",
        "unit": "cm",
        "required": True,
        "min": 0.0,
        "max": 100.0,
    },
    "season": {"kind": "category", "required": True, "choices": SEASONS, "derived": True},
    "is_holiday": {
        "api_name": "holiday",
        "kind": "bool",
        "unit": "boolean",
        "required": False,
        "default": False,
    },
    "is_functioning": {
        "api_name": "functioning_day",
        "kind": "bool",
        "unit": "boolean",
        "required": False,
        "default": True,
    },
}

# Backwards-compatible public name used by preprocessing and validation. The API
# also reads this exact object, so bounds and types cannot drift independently.
RAW_INPUT_SCHEMA = FEATURE_CONTRACT


def api_input_contract() -> dict[str, dict[str, Any]]:
    """Return API field definitions keyed by their public request names."""

    return {
        spec["api_name"]: {"canonical_name": canonical_name, **spec}
        for canonical_name, spec in FEATURE_CONTRACT.items()
        if "api_name" in spec
    }


#: Categories of every categorical feature, in the order the encoders use.
CATEGORY_LEVELS: dict[str, list[Any]] = {
    "hour": list(range(24)),
    "day_of_week": list(range(7)),
    "month": list(range(1, 13)),
    "season": list(SEASONS),
}


class FeatureValidationError(ValueError):
    """Raised when caller input does not satisfy RAW_INPUT_SCHEMA."""


def categorical_columns() -> list[str]:
    return list(load_config()["features"]["categorical"])


def numeric_columns() -> list[str]:
    return list(load_config()["features"]["numeric"])


def feature_columns() -> list[str]:
    """Full feature vector in a fixed, contract-stable order."""
    return categorical_columns() + numeric_columns()


def _as_frame(records: Any) -> pd.DataFrame:
    """Accept a dict, an iterable of dicts, a DataFrame or a file path."""
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, Mapping):
        return pd.DataFrame([dict(records)])
    if isinstance(records, str | Path):
        path = Path(records)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    if isinstance(records, Iterable):
        rows = [dict(r) for r in records]
        if not rows:
            raise FeatureValidationError("Empty input: nothing to predict.")
        return pd.DataFrame(rows)
    raise FeatureValidationError(
        f"Unsupported input type {type(records).__name__}. Pass a dict, a list of "
        "dicts, a DataFrame, or a path to a .csv/.parquet file."
    )


def _coerce_bool(series: pd.Series, column: str) -> pd.Series:
    truthy = {"yes", "true", "1", "holiday", "y"}
    falsy = {"no", "false", "0", "no holiday", "n"}

    def convert(value: Any) -> bool:
        if isinstance(value, bool | np.bool_):
            return bool(value)
        if isinstance(value, int | float | np.integer | np.floating) and not pd.isna(value):
            return bool(value)
        text = str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        raise FeatureValidationError(f"Column '{column}': cannot interpret {value!r} as a boolean.")

    return series.map(convert)


def coerce_input(records: Any) -> pd.DataFrame:
    """Normalise arbitrary caller input into a validated frame.

    Fills optional columns with their defaults, casts types, and checks ranges.
    Raises FeatureValidationError naming the offending column.
    """
    frame = _as_frame(records)
    if frame.empty:
        raise FeatureValidationError("Empty input: nothing to predict.")

    frame.columns = [str(c).strip() for c in frame.columns]

    missing = [
        name
        for name, spec in RAW_INPUT_SCHEMA.items()
        if spec["required"] and name not in frame.columns
    ]
    if missing:
        raise FeatureValidationError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            f"Expected: {', '.join(RAW_INPUT_SCHEMA)}"
        )

    if "date" not in frame.columns and "day_of_week" not in frame.columns:
        raise FeatureValidationError(
            "Supply either 'date' (ISO, e.g. '2018-12-01') or 'day_of_week' (0=Monday)."
        )

    out = pd.DataFrame(index=frame.index)
    for name, spec in RAW_INPUT_SCHEMA.items():
        if name not in frame.columns:
            out[name] = spec["default"]
            continue

        column = frame[name]
        kind = spec["kind"]

        # A batch may mix records that carry an optional field with records that
        # omit it; pandas leaves NaN in the gaps, which is the default in disguise.
        if not spec["required"] and column.isna().any():
            column = column.fillna(spec["default"])

        if kind == "date":
            if pd.api.types.is_datetime64_any_dtype(column):
                parsed = column
            else:
                parsed = pd.to_datetime(column.astype(str), errors="coerce", format="ISO8601")
            if parsed.isna().any():
                bad = column[parsed.isna()].iloc[0]
                raise FeatureValidationError(
                    f"Column 'date': cannot parse {bad!r}. Use ISO format, e.g. '2018-12-01'."
                )
            out[name] = parsed.dt.normalize()
            continue

        if kind == "bool":
            out[name] = _coerce_bool(column, name)
            continue

        if kind == "category":
            values = column.astype(str).str.strip().str.capitalize()
            unknown = sorted(set(values) - set(spec["choices"]))
            if unknown:
                raise FeatureValidationError(
                    f"Column '{name}': unknown value(s) {unknown}. Allowed: {list(spec['choices'])}"
                )
            out[name] = values
            continue

        numeric = pd.to_numeric(column, errors="coerce")
        if numeric.isna().any():
            bad = column[numeric.isna()].iloc[0]
            raise FeatureValidationError(f"Column '{name}': {bad!r} is not numeric.")
        below = numeric < spec["min"]
        above = numeric > spec["max"]
        if below.any() or above.any():
            offending = numeric[below | above].iloc[0]
            raise FeatureValidationError(
                f"Column '{name}': value {offending} outside allowed range "
                f"[{spec['min']}, {spec['max']}]."
            )
        out[name] = numeric.astype("int64" if kind == "int" else "float64")

    return out.reset_index(drop=True)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar columns from `date`. Safe to call twice.

    When `date` is absent the caller must already carry `day_of_week` — the only
    calendar column the model actually consumes. `month`, `day_of_year` and
    `is_weekend` are reporting slices, so they are simply left out in that case.
    """
    out = frame.copy()
    has_date = "date" in out.columns and out["date"].notna().all()

    if not has_date:
        if "day_of_week" not in out.columns or out["day_of_week"].isna().any():
            raise FeatureValidationError(
                "Neither 'date' nor 'day_of_week' is available to derive calendar features."
            )
        out["day_of_week"] = out["day_of_week"].astype("int64")
        return out

    dates = pd.to_datetime(out["date"])
    out["day_of_week"] = dates.dt.dayofweek.astype("int64")
    out["month"] = dates.dt.month.astype("int64")
    out["day_of_year"] = dates.dt.dayofyear.astype("int64")
    out["is_weekend"] = out["day_of_week"] >= 5
    return out


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature matrix in contract order.

    Input must already be canonical (i.e. the output of `coerce_input` or of the
    preprocessing pipeline).
    """
    enriched = add_calendar_features(frame)
    columns = feature_columns()

    absent = [c for c in columns if c not in enriched.columns]
    if absent:
        raise FeatureValidationError(
            f"Cannot build features, column(s) missing after derivation: {absent}"
        )

    features = enriched[columns].copy()
    for name in categorical_columns():
        levels = CATEGORY_LEVELS[name]
        unknown = sorted(set(features[name].unique()) - set(levels))
        if unknown:
            raise FeatureValidationError(f"Feature '{name}': unexpected level(s) {unknown}.")
    for name in numeric_columns():
        features[name] = features[name].astype("float64")
    return features
