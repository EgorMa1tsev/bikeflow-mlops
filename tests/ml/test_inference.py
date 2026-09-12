"""The contract role B builds the API on: one observation or many, same answer."""

import numpy as np
import pandas as pd
import pytest

from bikeflow.ml.features import (
    FeatureValidationError,
    build_features,
    coerce_input,
    feature_columns,
)
from bikeflow.ml.inference import Predictor
from bikeflow.ml.models.baseline import SeasonalMedianBaseline
from bikeflow.ml.models.registry import load_bundle, save_bundle


@pytest.fixture
def predictor(tmp_path, observation):
    """A cheap fitted model saved through the real artifact format."""
    rows = pd.DataFrame([observation] * 8)
    rows["hour"] = list(range(8))
    features = build_features(coerce_input(rows))

    model = SeasonalMedianBaseline().fit(features, np.arange(100, 900, 100))
    path = save_bundle(tmp_path / "model.joblib", model, metrics={})
    return Predictor(load_bundle(path), path)


def test_single_observation_matches_the_batch(predictor, observation):
    single = predictor.predict(observation)
    batch = predictor.predict([observation, observation])

    assert single.shape == (1,)
    assert batch.shape == (2,)
    assert single[0] == pytest.approx(batch[0])


def test_accepts_a_dataframe_and_a_file(predictor, observation, tmp_path):
    frame = pd.DataFrame([observation, observation])
    from_frame = predictor.predict(frame)

    csv = tmp_path / "hours.csv"
    frame.to_csv(csv, index=False)
    from_csv = predictor.predict(str(csv))

    assert np.allclose(from_frame, from_csv)


def test_closed_hours_predict_exactly_zero(predictor, observation):
    closed = {**observation, "is_functioning": False}
    assert predictor.predict(closed)[0] == 0.0
    # The rule applies per row, not to the whole batch.
    both = predictor.predict([observation, closed])
    assert both[1] == 0.0
    assert both[0] > 0.0


def test_predictions_are_never_negative(predictor, observation):
    assert (predictor.predict([observation] * 5) >= 0).all()


def test_return_frame_keeps_the_input(predictor, observation):
    result = predictor.predict(observation, return_frame=True)
    assert "predicted_demand" in result.columns
    assert result["hour"].iloc[0] == observation["hour"]


def test_missing_column_names_the_field(predictor, observation):
    broken = {k: v for k, v in observation.items() if k != "temperature"}
    with pytest.raises(FeatureValidationError, match="temperature"):
        predictor.predict(broken)


def test_optional_columns_get_defaults(observation):
    minimal = {k: v for k, v in observation.items() if k not in {"is_holiday", "is_functioning"}}
    frame = coerce_input(minimal)
    assert frame["is_holiday"].iloc[0] is np.False_ or not frame["is_holiday"].iloc[0]
    assert frame["is_functioning"].iloc[0]


def test_out_of_range_value_is_rejected(observation):
    with pytest.raises(FeatureValidationError, match="humidity"):
        coerce_input({**observation, "humidity": 250})


def test_unknown_season_is_rejected(observation):
    with pytest.raises(FeatureValidationError, match="season"):
        coerce_input({**observation, "season": "Monsoon"})


def test_non_numeric_value_is_rejected(observation):
    with pytest.raises(FeatureValidationError, match="temperature"):
        coerce_input({**observation, "temperature": "warm"})


def test_unparseable_date_is_rejected(observation):
    with pytest.raises(FeatureValidationError, match="date"):
        coerce_input({**observation, "date": "not-a-date"})


def test_iso_date_is_not_read_day_first(observation):
    """'2018-12-01' is 1 December, never 12 January."""
    frame = coerce_input({**observation, "date": "2018-12-01"})
    assert frame["date"].iloc[0] == pd.Timestamp("2018-12-01")
    assert build_features(frame)["season"].iloc[0] == observation["season"]


def test_ambiguous_slash_date_is_rejected(observation):
    """DD/MM vs MM/DD must not be guessed silently at the API boundary."""
    with pytest.raises(FeatureValidationError, match="ISO format"):
        coerce_input({**observation, "date": "01/12/2018"})


def test_batch_may_mix_present_and_absent_optional_fields(predictor, observation):
    """One record sets is_functioning, the other leaves it out."""
    with_flag = {**observation, "is_functioning": False}
    without_flag = {k: v for k, v in observation.items() if k != "is_functioning"}

    predictions = predictor.predict([without_flag, with_flag])
    assert predictions[0] > 0.0
    assert predictions[1] == 0.0


def test_string_booleans_are_understood(observation):
    frame = coerce_input({**observation, "is_holiday": "Holiday", "is_functioning": "Yes"})
    assert bool(frame["is_holiday"].iloc[0]) is True
    assert bool(frame["is_functioning"].iloc[0]) is True


def test_feature_order_is_stable(observation):
    features = build_features(coerce_input(observation))
    assert list(features.columns) == feature_columns()


def test_artifact_round_trip_preserves_predictions(predictor, observation, tmp_path):
    before = predictor.predict([observation] * 3)

    path = save_bundle(tmp_path / "again.joblib", predictor.model, metrics={})
    reloaded = Predictor(load_bundle(path), path)

    assert np.allclose(before, reloaded.predict([observation] * 3))


def test_artifact_rejects_a_stale_feature_contract(tmp_path, monkeypatch, observation):
    features = build_features(coerce_input(observation))
    model = SeasonalMedianBaseline().fit(features, [500.0])
    path = save_bundle(tmp_path / "stale.joblib", model, metrics={})

    import bikeflow.ml.models.registry as registry

    monkeypatch.setattr(registry, "feature_columns", lambda: ["something", "else"])
    with pytest.raises(ValueError, match="Feature contract mismatch"):
        load_bundle(path)
