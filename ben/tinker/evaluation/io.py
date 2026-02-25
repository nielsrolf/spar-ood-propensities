"""I/O helpers for eval file formats."""

from pathlib import Path

from evaluation.schemas import EvalFile, EvalResults
from utils.io import load_yaml, save_yaml


def load_eval_file(path: str) -> EvalFile:
    """Load a .eval.yaml file."""
    data = load_yaml(path)
    return EvalFile.model_validate(data)


def save_eval_file(eval_file: EvalFile, path: Path) -> None:
    """Save an EvalFile to a .eval.yaml."""
    save_yaml(eval_file.model_dump(), path)


def load_eval_results(path: str) -> EvalResults:
    """Load a .eval-results.yaml file."""
    data = load_yaml(path)
    return EvalResults.model_validate(data)


def save_eval_results(results: EvalResults, output_dir: str) -> Path:
    """Save eval results to a YAML file."""
    path = Path(output_dir) / f"{results.run_id}.eval-results.yaml"
    save_yaml(results.model_dump(), path)
    return path
