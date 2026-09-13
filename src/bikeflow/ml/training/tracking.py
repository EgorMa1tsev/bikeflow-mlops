"""Log a training run to MLflow and register the champion model.

One MLflow run per training. Parameters come from params.yaml, metrics from model
selection and the held-out test, and the champion is logged as an MLflow pyfunc
model that wraps the same joblib bundle the API serves. The model is then
registered in the Model Registry and receives the alias from params.yaml.

The tracking store defaults to a local SQLite file next to params.yaml, as in
the course lecture. Set MLFLOW_TRACKING_URI to use an MLflow server instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from ..config import load_config, resolve  # noqa: E402
from ..inference import Predictor  # noqa: E402
from .cv import summarise_cv  # noqa: E402

SQLITE_PREFIX = "sqlite:///"

#: params.yaml sections that describe how a model was trained.
LOGGED_PARAM_SECTIONS = ("seed", "split", "features", "models", "selection")

#: Raw request columns accepted by the model, used for the input example.
EXAMPLE_COLUMNS = [
    "date",
    "hour",
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
]

REPORT_FILES = ("metrics.json", "model_selection.md", "cv_summary.csv", "cv_folds.csv")


class BikeflowModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper around the joblib bundle, so both load the same model.

    The methods carry no type hints on purpose: MLflow would try to derive the
    model signature from them, while the signature is logged explicitly instead.
    """

    def load_context(self, context):
        self._predictor = Predictor.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        return self._predictor.predict(model_input)


def tracking_uri() -> str:
    """MLFLOW_TRACKING_URI if set, otherwise a SQLite file in the project root."""
    return os.environ.get("MLFLOW_TRACKING_URI") or SQLITE_PREFIX + resolve("mlflow.db").as_posix()


def artifact_location(uri: str) -> str | None:
    """Keep artifacts beside a local SQLite store; a remote server chooses its own."""
    if not uri.startswith(SQLITE_PREFIX):
        return None
    database = Path(uri[len(SQLITE_PREFIX) :])
    return (database.parent / "mlruns").resolve().as_uri()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Turn nested params.yaml sections into flat MLflow parameter names."""
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, inner in value.items():
            flat.update(flatten(inner, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value if isinstance(value, str | int | float | bool) else str(value)}


def input_example(frame: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    """A few real observations in the request format, with ISO dates."""
    example = frame[EXAMPLE_COLUMNS].head(rows).reset_index(drop=True).copy()
    example["date"] = pd.to_datetime(example["date"]).dt.strftime("%Y-%m-%d")
    return example


def collect_metrics(
    results: dict[str, dict[str, dict[str, float]]],
    cv_scores: pd.DataFrame,
    main_kind: str,
) -> dict[str, float]:
    """Champion validation and test metrics plus the CV score of every candidate."""
    metrics: dict[str, float] = {}
    for split in ("validation", "test"):
        for name in ("mae", "wape", "rmse", "r2", "wce"):
            value = results[main_kind].get(split, {}).get(name)
            if value is not None:
                metrics[f"{split}_{name}"] = float(value)

    for row in summarise_cv(cv_scores).itertuples(index=False):
        metrics[f"cv_mae_{row.model}"] = float(row.mean_score)
    metrics["cv_mae"] = metrics[f"cv_mae_{main_kind}"]
    return metrics


def start_tracking() -> MlflowClient:
    """Point MLflow at the configured store and experiment, creating it if needed."""
    tracking = load_config()["tracking"]
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    if client.get_experiment_by_name(tracking["experiment"]) is None:
        client.create_experiment(tracking["experiment"], artifact_location=artifact_location(uri))
    mlflow.set_experiment(tracking["experiment"])
    return client


def register_bundle(model_path: str | Path, example_frame: pd.DataFrame) -> str:
    """Log a joblib bundle as a pyfunc model in the active run and register a new version."""
    model_path = Path(model_path)
    predictor = Predictor.load(model_path)
    example = input_example(example_frame)
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=BikeflowModel(),
        artifacts={"bundle": str(model_path.resolve())},
        signature=infer_signature(example, predictor.predict(example)),
        input_example=example,
        registered_model_name=load_config()["tracking"]["registered_model"],
    )
    return str(info.registered_model_version)


def set_champion(client: MlflowClient, version: str) -> None:
    """Move the serving alias to a registered version."""
    tracking = load_config()["tracking"]
    client.set_registered_model_alias(tracking["registered_model"], tracking["alias"], version)


def load_registered(model_uri: str) -> Predictor:
    """Load a registered model, e.g. `models:/bikeflow-demand@champion`, as a Predictor."""
    mlflow.set_tracking_uri(tracking_uri())
    return mlflow.pyfunc.load_model(model_uri).unwrap_python_model()._predictor


def log_training_run(
    results: dict[str, dict[str, dict[str, float]]],
    cv_scores: pd.DataFrame,
    main_kind: str,
    model_path: str | Path,
    example_frame: pd.DataFrame,
    reports_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Record one training run, register the champion and point the alias at it.

    Returns the MLflow run id and the registered model version.
    """
    cfg = load_config()
    client = start_tracking()
    predictor = Predictor.load(model_path)

    params: dict[str, Any] = {}
    for section in LOGGED_PARAM_SECTIONS:
        params.update(flatten(cfg[section], section))

    reports = resolve(reports_dir or cfg["paths"]["reports_dir"])

    with mlflow.start_run(run_name=f"train {main_kind}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(collect_metrics(results, cv_scores, main_kind))
        mlflow.set_tags(
            {
                "champion": main_kind,
                "model_version": predictor.metadata.get("model_version", "unknown"),
                "git_commit": predictor.metadata.get("git_commit", "unknown"),
                "data_sha256": predictor.metadata.get("data_sha256") or "unknown",
            }
        )
        for name in REPORT_FILES:
            if (reports / name).exists():
                mlflow.log_artifact(str(reports / name), artifact_path="reports")

        version = register_bundle(model_path, example_frame)

    set_champion(client, version)
    return run.info.run_id, version
