"""Application configuration loaded from BIKEFLOW_* environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API scaffold."""

    model_config = SettingsConfigDict(env_prefix="BIKEFLOW_", env_file=".env")

    log_level: str = "INFO"
    model_path: Path = Path("models/model.joblib")
    # Serve the model from the MLflow registry instead of `model_path`, e.g.
    # "models:/bikeflow-demand@champion". Retraining needs it: promoting a model
    # moves the alias, and the API reloads whatever the alias points to.
    model_uri: str | None = None
    retraining_dir: Path = Path("data/retraining")
    # SQLite journal of served predictions and the actual demand reported later.
    db_path: Path = Path("data/predictions.db")
    # Latest Evidently drift report.
    monitoring_dir: Path = Path("data/monitoring")
    # Where the Streamlit interface looks for the API.
    api_url: str = "http://127.0.0.1:8000"


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
