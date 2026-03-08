"""Run ID generation and generic YAML helpers."""

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml


def make_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def save_yaml(data: dict, path: Path) -> None:
    """Write a dict to a YAML file, creating parent directories as needed."""
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def load_yaml(path: str) -> dict:
    """Load a YAML file and return the parsed dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML dict in {path}, got {type(data).__name__}")
    return data
