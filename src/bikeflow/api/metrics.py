"""Prometheus metrics exposed at `GET /metrics`.

The API runs as a single process, so the default registry of `prometheus_client`
holds everything. Gauges describe the latest drift check and start empty after a
restart until the next check.
"""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "bikeflow_http_requests_total",
    "HTTP requests by route and status code.",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "bikeflow_http_request_duration_seconds",
    "HTTP request duration by route.",
    ["method", "route"],
)

PREDICTIONS = Counter(
    "bikeflow_predictions_total",
    "Predictions served, by model version.",
    ["model_version"],
)
PREDICTED_RENTALS = Histogram(
    "bikeflow_predicted_rentals",
    "Predicted hourly rentals.",
    buckets=(0, 100, 250, 500, 750, 1000, 1500, 2000, 3000),
)
ACTUALS = Counter("bikeflow_actuals_total", "Actual demand values reported for predictions.")
ABSOLUTE_ERROR = Histogram(
    "bikeflow_absolute_error",
    "Absolute error of a prediction once its actual demand is known.",
    buckets=(10, 25, 50, 100, 200, 300, 500, 750, 1000, 2000),
)

DRIFT_CHECKS = Counter("bikeflow_drift_checks_total", "Drift checks run.")
CONCEPT_DRIFT = Gauge("bikeflow_concept_drift", "1 if the latest check found concept drift.")
DATA_DRIFT = Gauge("bikeflow_data_drift", "1 if the latest check found data drift.")
TARGET_DRIFT = Gauge("bikeflow_target_drift", "1 if the latest check found target drift.")
MAE_RATIO = Gauge("bikeflow_drift_mae_ratio", "Window MAE divided by the reference MAE.")
MAE_RATIO_THRESHOLD = Gauge(
    "bikeflow_drift_mae_ratio_threshold", "Window MAE ratio above which concept drift fires."
)
CURRENT_MAE = Gauge("bikeflow_window_mae", "MAE over the latest monitoring window.")
DRIFTED_FEATURE_SHARE = Gauge(
    "bikeflow_drifted_feature_share", "Share of weather features that drifted."
)
TARGET_DRIFT_SCORE = Gauge("bikeflow_target_drift_score", "Drift score of the actual demand.")

RETRAININGS = Counter(
    "bikeflow_retrainings_total",
    "Finished retrainings by outcome: promoted, rejected or failed.",
    ["outcome"],
)


def record_drift_check(payload: dict) -> None:
    """Publish the result of a drift check."""

    DRIFT_CHECKS.inc()
    CONCEPT_DRIFT.set(int(payload["concept_drift"]))
    DATA_DRIFT.set(int(payload["data_drift"]))
    TARGET_DRIFT.set(int(payload["target_drift"]))
    MAE_RATIO.set(payload["mae_ratio"])
    MAE_RATIO_THRESHOLD.set(payload["thresholds"]["concept_drift_mae_ratio"])
    CURRENT_MAE.set(payload["current_mae"])
    DRIFTED_FEATURE_SHARE.set(payload["drifted_feature_share"])
    TARGET_DRIFT_SCORE.set(payload["target_drift_score"])
