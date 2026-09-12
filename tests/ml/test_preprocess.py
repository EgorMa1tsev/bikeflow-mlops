import pandas as pd
import pytest

from bikeflow.ml.data.preprocess import DataContractError, to_canonical, validate


def test_renames_and_types(raw_rows):
    frame = to_canonical(raw_rows)

    assert frame["date"].iloc[0] == pd.Timestamp("2017-12-01")
    assert frame["timestamp"].iloc[1] == pd.Timestamp("2017-12-01 01:00")
    assert frame["rented_bike_count"].iloc[0] == 254
    assert frame["temperature"].iloc[0] == -5.2
    assert frame["is_holiday"].tolist() == [False, False, True]
    assert frame["is_functioning"].tolist() == [True, True, False]
    # 2017-12-01 was a Friday.
    assert frame["day_of_week"].iloc[0] == 4
    assert not frame["is_weekend"].iloc[0]


def test_dates_are_day_first(raw_rows):
    """02/12/2017 is 2 December, not 12 February."""
    frame = to_canonical(raw_rows)
    assert frame["date"].iloc[2] == pd.Timestamp("2017-12-02")
    assert frame["month"].iloc[2] == 12


def test_missing_column_is_rejected(raw_rows):
    with pytest.raises(DataContractError, match="missing expected column"):
        to_canonical(raw_rows.drop(columns=["Seasons"]))


def _valid_year() -> pd.DataFrame:
    """A synthetic full year that satisfies every contract rule."""
    stamps = pd.date_range("2017-12-01", periods=8760, freq="h")
    frame = pd.DataFrame({"timestamp": stamps})
    frame["date"] = stamps.normalize()
    frame["hour"] = stamps.hour
    frame["rented_bike_count"] = 100
    for column in (
        "temperature",
        "humidity",
        "wind_speed",
        "visibility",
        "dew_point",
        "solar_radiation",
        "rainfall",
        "snowfall",
    ):
        frame[column] = 1.0
    frame["season"] = "Winter"
    frame.loc[2200:4400, "season"] = "Spring"
    frame.loc[4401:6600, "season"] = "Summer"
    frame.loc[6601:, "season"] = "Autumn"
    frame["is_holiday"] = False
    frame["is_functioning"] = True
    frame["day_of_week"] = stamps.dayofweek
    frame["month"] = stamps.month
    frame["day_of_year"] = stamps.dayofyear
    frame["is_weekend"] = frame["day_of_week"] >= 5
    return frame


def test_valid_year_passes():
    validate(_valid_year())


@pytest.mark.parametrize(
    "break_it, message",
    [
        (lambda f: f.iloc[:-1], "expected 8760 rows"),
        (lambda f: f.assign(hour=f["hour"].where(f.index != 5, 99)), "hour outside"),
        (
            lambda f: f.assign(rented_bike_count=f["rented_bike_count"].where(f.index != 7, -1)),
            "negative target",
        ),
        (lambda f: f.assign(season="Winter"), "unexpected seasons"),
        (
            lambda f: f.assign(is_functioning=f["is_functioning"].where(f.index != 3, False)),
            "non-functioning hour",
        ),
        (
            lambda f: f.assign(rented_bike_count=f["rented_bike_count"].where(f.index != 9, 0)),
            "zero target",
        ),
    ],
)
def test_each_contract_rule_fires(break_it, message):
    with pytest.raises(DataContractError, match=message):
        validate(break_it(_valid_year()))


def test_duplicate_timestamp_is_caught():
    frame = _valid_year()
    frame.loc[10, "timestamp"] = frame.loc[9, "timestamp"]
    with pytest.raises(DataContractError, match="duplicated timestamps"):
        validate(frame)
