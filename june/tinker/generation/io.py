"""YAML read/write for generation-specific file formats."""

from pathlib import Path

from generation.schemas import ExamplesFile, InstructionsFile, ParamsFile
from utils.io import load_yaml, save_yaml


def save_params(params: ParamsFile, output_dir: str) -> Path:
    path = Path(output_dir) / f"{params.run_id}.params.yaml"
    save_yaml(params.model_dump(), path)
    return path


def save_examples(examples: ExamplesFile, output_dir: str) -> Path:
    path = Path(output_dir) / f"{examples.run_id}.examples.yaml"
    save_yaml(examples.model_dump(), path)
    return path


def load_params(path: str) -> ParamsFile:
    data = load_yaml(path)
    return ParamsFile.model_validate(data)


def load_examples(path: str) -> ExamplesFile:
    data = load_yaml(path)
    return ExamplesFile.model_validate(data)


def save_instructions(instructions: InstructionsFile, output_dir: str) -> Path:
    path = Path(output_dir) / f"{instructions.run_id}.instructions.yaml"
    save_yaml(instructions.model_dump(), path)
    return path


def load_instructions(path: str) -> InstructionsFile:
    data = load_yaml(path)
    return InstructionsFile.model_validate(data)


def load_yaml_config(path: str) -> dict[str, object]:
    """Load a flat YAML dict for config file support."""
    return load_yaml(path)
