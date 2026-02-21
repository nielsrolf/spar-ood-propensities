"""YAML read/write and run ID generation."""

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from generation.schemas import ExamplesFile, ParamsFile


def make_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def save_params(params: ParamsFile, output_dir: str) -> Path:
    path = Path(output_dir) / f"{params.run_id}.params.yaml"
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(params.model_dump(), f, sort_keys=False, allow_unicode=True)
    return path


def save_examples(examples: ExamplesFile, output_dir: str) -> Path:
    path = Path(output_dir) / f"{examples.run_id}.examples.yaml"
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(examples.model_dump(), f, sort_keys=False, allow_unicode=True)
    return path


def load_params(path: str) -> ParamsFile:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ParamsFile.model_validate(data)


def load_examples(path: str) -> ExamplesFile:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExamplesFile.model_validate(data)


def load_yaml_config(path: str) -> dict[str, object]:
    """Load a flat YAML dict for config file support."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML dict, got {type(data).__name__}")
    return data
