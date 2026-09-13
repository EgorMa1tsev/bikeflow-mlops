"""Centralized preliminary request and response schemas."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from bikeflow.ml.features import SEASONS, api_input_contract
from bikeflow.model.protocol import FeatureValue


class Season(StrEnum):
    """Meteorological season derived from prediction time."""

    WINTER = SEASONS[0]
    SPRING = SEASONS[1]
    SUMMER = SEASONS[2]
    AUTUMN = SEASONS[3]


def season_from_month(month: int) -> Season:
    """Map a calendar month to a meteorological season."""

    if month in (12, 1, 2):
        return Season.WINTER
    if month in (3, 4, 5):
        return Season.SPRING
    if month in (6, 7, 8):
        return Season.SUMMER
    return Season.AUTUMN


class PredictionRequestBase(BaseModel):
    """Preliminary Seoul Bike Sharing feature contract for one prediction."""

    model_config = ConfigDict(extra="forbid")

    prediction_time: datetime = Field(
        description="Timezone-aware instant; normalized to Asia/Seoul before prediction."
    )

    @field_validator("prediction_time")
    @classmethod
    def require_timezone_and_normalize(cls, value: datetime) -> datetime:
        """Reject ambiguous local time and normalize valid instants to Seoul."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("prediction_time must include a UTC offset or timezone")
        return value.astimezone(ZoneInfo("Asia/Seoul"))

    def to_features(self) -> dict[str, FeatureValue]:
        """Build one model row, deriving calendar fields from prediction_time."""

        features: dict[str, FeatureValue] = self.model_dump(exclude={"prediction_time"})
        features["hour"] = self.prediction_time.hour
        features["season"] = season_from_month(self.prediction_time.month).value
        # Weekday is a trained feature: weekend demand differs sharply from
        # weekday demand (test MAE 143 vs 194). Derived here alongside hour and
        # season so clients still send only prediction_time.
        features["day_of_week"] = self.prediction_time.weekday()
        return features


_PYTHON_TYPES = {"float": float, "int": int, "bool": bool}
_api_fields = {}
for _name, _spec in api_input_contract().items():
    _default = ... if _spec["required"] else _spec["default"]
    _api_fields[_name] = (
        _PYTHON_TYPES[_spec["kind"]],
        Field(
            default=_default,
            ge=_spec.get("min"),
            le=_spec.get("max"),
            description=f"{_spec['canonical_name']} ({_spec['unit']})",
        ),
    )

PredictionRequest = create_model("PredictionRequest", __base__=PredictionRequestBase, **_api_fields)


class PredictionResponse(BaseModel):
    """Response returned by the demand endpoint."""

    prediction_id: int = Field(description="Journal id; use it to report the actual demand.")
    prediction_time: datetime
    predicted_rentals: float = Field(ge=0)
    model_version: str


class ActualRequest(BaseModel):
    """Observed demand for an hour that has already been predicted."""

    model_config = ConfigDict(extra="forbid")

    actual_rentals: float = Field(ge=0, description="Rentals actually observed in that hour.")


class PredictionRecord(BaseModel):
    """One journal entry: what was asked, what was predicted, and what happened."""

    prediction_id: int
    created_at: datetime
    prediction_time: datetime
    features: dict[str, FeatureValue]
    predicted_rentals: float
    model_version: str
    actual_rentals: float | None
    absolute_error: float | None


class DriftCheckResponse(BaseModel):
    """Outcome of comparing the latest journal window with the model's reference."""

    check_id: int
    checked_at: datetime
    model_version: str
    window_start: datetime
    window_end: datetime
    rows: int = Field(description="Predictions with an actual demand in the window.")
    data_drift: bool = Field(description="Weather differs from the reference period.")
    drifted_feature_share: float
    drifted_features: list[str]
    target_drift: bool = Field(description="Distribution of actual demand has shifted.")
    target_drift_score: float
    concept_drift: bool = Field(description="Model error grew; retraining is warranted.")
    current_mae: float
    reference_mae: float
    mae_ratio: float
    thresholds: dict[str, float]
    retraining_started: bool = Field(
        default=False, description="Concept drift started an automatic retraining."
    )


class RetrainingStatus(BaseModel):
    """State of the background retraining and the outcome of the last finished run."""

    state: str = Field(description="idle, running, finished or failed")
    trigger: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Liveness response."""

    status: str
