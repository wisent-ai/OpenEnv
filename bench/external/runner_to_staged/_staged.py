"""Resolve benchmark artifacts materialized by the trusted Stado agent."""
from __future__ import annotations

import os
from pathlib import Path


def staged_file(environment_name: str) -> Path:
    """Return one required absolute local file path from the environment."""
    raw = os.environ.get(environment_name, "").strip()
    if not raw or "://" in raw:
        raise RuntimeError(
            f"{environment_name} must name a machine-staged local file"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{environment_name} must be an absolute path")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"{environment_name} is not a file: {path}")
    return path


def staged_directory(environment_name: str) -> Path:
    """Return one required absolute local directory path from the environment."""
    raw = os.environ.get(environment_name, "").strip()
    if not raw or "://" in raw:
        raise RuntimeError(
            f"{environment_name} must name a machine-staged local directory"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{environment_name} must be an absolute path")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise RuntimeError(f"{environment_name} is not a directory: {path}")
    return path


def configure_offline_lm_eval() -> None:
    """Point lm-eval at its staged cache and prohibit Hub network access."""
    cache = staged_directory("OPENENV_LM_EVAL_CACHE_PATH")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "true"
    os.environ["HF_DATASETS_OFFLINE"] = "true"
    os.environ["TRANSFORMERS_OFFLINE"] = "true"
