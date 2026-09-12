"""Shared fixtures. Tests never touch the network or the real dataset."""

from __future__ import annotations

import pytest

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    # Installed without the `ml` extra (pandas, scikit-learn, torch). Skip this
    # directory instead of failing collection, so `pip install -e ".[dev]"`
    # keeps working for anyone who only touches the API.
    pd = None
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture
def raw_rows() -> pd.DataFrame:
    """A handful of rows in the exact shape of the UCI csv."""
    return pd.DataFrame(
        [
            {
                "Date": "01/12/2017",
                "Rented Bike Count": 254,
                "Hour": 0,
                "Temperature(\N{DEGREE SIGN}C)": -5.2,
                "Humidity(%)": 37,
                "Wind speed (m/s)": 2.2,
                "Visibility (10m)": 2000,
                "Dew point temperature(\N{DEGREE SIGN}C)": -17.6,
                "Solar Radiation (MJ/m2)": 0.0,
                "Rainfall(mm)": 0.0,
                "Snowfall (cm)": 0.0,
                "Seasons": "Winter",
                "Holiday": "No Holiday",
                "Functioning Day": "Yes",
            },
            {
                "Date": "01/12/2017",
                "Rented Bike Count": 204,
                "Hour": 1,
                "Temperature(\N{DEGREE SIGN}C)": -5.5,
                "Humidity(%)": 38,
                "Wind speed (m/s)": 0.8,
                "Visibility (10m)": 2000,
                "Dew point temperature(\N{DEGREE SIGN}C)": -17.6,
                "Solar Radiation (MJ/m2)": 0.0,
                "Rainfall(mm)": 0.0,
                "Snowfall (cm)": 0.0,
                "Seasons": "Winter",
                "Holiday": "No Holiday",
                "Functioning Day": "Yes",
            },
            {
                "Date": "02/12/2017",
                "Rented Bike Count": 0,
                "Hour": 0,
                "Temperature(\N{DEGREE SIGN}C)": -6.0,
                "Humidity(%)": 40,
                "Wind speed (m/s)": 1.0,
                "Visibility (10m)": 1800,
                "Dew point temperature(\N{DEGREE SIGN}C)": -18.0,
                "Solar Radiation (MJ/m2)": 0.0,
                "Rainfall(mm)": 0.0,
                "Snowfall (cm)": 0.0,
                "Seasons": "Winter",
                "Holiday": "Holiday",
                "Functioning Day": "No",
            },
        ]
    )


@pytest.fixture
def observation() -> dict:
    """One well-formed prediction request."""
    return {
        "date": "2018-12-01",
        "hour": 18,
        "temperature": 3.5,
        "humidity": 45,
        "wind_speed": 1.2,
        "visibility": 2000,
        "dew_point": -7.0,
        "solar_radiation": 0.0,
        "rainfall": 0.0,
        "snowfall": 0.0,
        "season": "Winter",
        "is_holiday": False,
        "is_functioning": True,
    }
