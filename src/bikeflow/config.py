"""Application configuration loaded from BIKEFLOW_* environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API scaffold."""

    model_config = SettingsConfigDict(env_prefix="BIKEFLOW_", env_file=".env")

    log_level: str = "INFO"
    model_path: Path = Path("models/model.joblib")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
