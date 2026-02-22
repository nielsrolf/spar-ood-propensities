"""Tests for generation.io — load_examples and load_yaml_config."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from generation.io import load_examples, load_yaml_config
from generation.schemas import ExamplesFile


@pytest.fixture
def examples_yaml(tmp_path: Path) -> Path:
    data = {
        "run_id": "20260220-120000",
        "type": "scenarios",
        "property": "honest",
        "examples": [
            {
                "type": "scenarios",
                "subtype": "negotiations",
                "prompt": "Tell me about honesty",
                "generation_index": 0,
                "text": "Honesty is important.",
            },
            {
                "type": "scenarios",
                "subtype": "negotiations",
                "prompt": "What is truth?",
                "generation_index": 1,
                "text": "Truth is correspondence with reality.",
            },
        ],
    }
    path = tmp_path / "test.examples.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def test_load_examples(examples_yaml: Path) -> None:
    result = load_examples(str(examples_yaml))
    assert isinstance(result, ExamplesFile)
    assert result.run_id == "20260220-120000"
    assert result.type == "scenarios"
    assert result.property == "honest"
    assert len(result.examples) == 2
    assert result.examples[0].prompt == "Tell me about honesty"
    assert result.examples[0].text == "Honesty is important."
    assert result.examples[1].generation_index == 1


def test_load_examples_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_examples("/nonexistent/path.yaml")


def test_load_examples_invalid_data(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.dump({"run_id": "x"}, f)  # missing required fields
    with pytest.raises(ValidationError):
        load_examples(str(path))


def test_load_yaml_config(tmp_path: Path) -> None:
    data = {
        "model_name": "my-model",
        "batch_size": 32,
        "learning_rate": 5e-5,
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    result = load_yaml_config(str(path))
    assert result == data


def test_load_yaml_config_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(TypeError, match="Expected a YAML dict.*got NoneType"):
        load_yaml_config(str(path))


def test_load_yaml_config_non_dict(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    with open(path, "w") as f:
        yaml.dump([1, 2, 3], f)
    with pytest.raises(TypeError, match="Expected a YAML dict.*got list"):
        load_yaml_config(str(path))


def test_load_yaml_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_config("/nonexistent/config.yaml")
