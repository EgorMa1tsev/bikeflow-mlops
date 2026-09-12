"""Configuration loading. Every tunable value comes from params.yaml."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Locate the repository root (the directory holding params.yaml)."""
    override = os.environ.get("BIKEFLOW_ROOT")
    if override:
        return Path(override).resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "params.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "params.yaml not found in any parent of "
        f"{here}. Set BIKEFLOW_ROOT to point at the project root."
    )


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Read params.yaml once and cache it."""
    with (project_root() / "params.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative: str | Path) -> Path:
    """Turn a params.yaml path into an absolute path under the project root."""
    path = Path(relative)
    return path if path.is_absolute() else project_root() / path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) and return it."""
    directory = resolve(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
