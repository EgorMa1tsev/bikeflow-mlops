import pandas as pd
import pytest

from bikeflow.ml.data.split import SPLIT_NAMES, SplitError, split_bounds, temporal_split


def _year() -> pd.DataFrame:
    stamps = pd.date_range("2017-12-01", periods=8760, freq="h")
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "date": stamps.normalize(),
            "hour": stamps.hour,
            "rented_bike_count": 100,
            "season": "Winter",
            "is_functioning": True,
        }
    )


def test_boundaries_match_config():
    parts = temporal_split(_year())
    bounds = split_bounds()

    for name in SPLIT_NAMES:
        start, end = bounds[name]
        assert parts[name]["timestamp"].min() == start
        # The configured end date is inclusive, down to its last hour.
        assert parts[name]["timestamp"].max() == end + pd.Timedelta(hours=23)


def test_splits_are_ordered_and_disjoint():
    parts = temporal_split(_year())
    train, validation, test = (parts[n] for n in SPLIT_NAMES)

    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    assert set(train["timestamp"]).isdisjoint(validation["timestamp"])
    assert set(validation["timestamp"]).isdisjoint(test["timestamp"])


def test_expected_sizes():
    parts = temporal_split(_year())
    assert [len(parts[n]) for n in SPLIT_NAMES] == [5832, 1464, 1464]
    assert sum(len(parts[n]) for n in SPLIT_NAMES) == 8760


def test_empty_window_is_rejected():
    short = _year().head(24)
    with pytest.raises(SplitError, match="is empty"):
        temporal_split(short)
