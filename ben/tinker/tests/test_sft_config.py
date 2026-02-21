"""Tests for scripts/sft.py — resolve_config logic."""

from pathlib import Path

import chz
import pytest
import yaml
from tinker_cookbook.renderers import TrainOnWhat

from scripts.sft import Config, resolve_config


@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    data = {
        "examples_file": "datasets/test.examples.yaml",
        "model_name": "my-custom-model",
        "batch_size": 32,
        "learning_rate": 5e-5,
    }
    path = tmp_path / "sft_config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def test_resolve_config_no_file() -> None:
    config = Config()
    result = resolve_config(config)
    assert result is config  # same object, nothing changed


def test_resolve_config_loads_yaml(config_yaml: Path) -> None:
    config = Config(config_file=str(config_yaml))
    result = resolve_config(config)

    assert result.examples_file == "datasets/test.examples.yaml"
    assert result.model_name == "my-custom-model"
    assert result.batch_size == 32
    assert result.learning_rate == 5e-5
    # Unchanged defaults should be preserved
    assert result.num_epochs == 1
    assert result.lora_rank == 32


def test_resolve_config_cli_overrides_yaml(config_yaml: Path) -> None:
    """CLI args (non-default values on the config) should beat YAML values."""
    config = Config(
        config_file=str(config_yaml),
        batch_size=64,  # CLI override — differs from default (16) and YAML (32)
    )
    result = resolve_config(config)

    # batch_size was set via "CLI" to 64, should not be overridden by YAML's 32
    assert result.batch_size == 64
    # model_name was at default, so YAML value should win
    assert result.model_name == "my-custom-model"


def test_resolve_config_unknown_key(tmp_path: Path) -> None:
    data = {"nonexistent_key": "value"}
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    config = Config(config_file=str(path))
    with pytest.raises(ValueError, match="Unknown config key.*'nonexistent_key'"):
        resolve_config(config)


def test_resolve_config_enum_coercion(tmp_path: Path) -> None:
    """YAML string values for enum fields should be coerced to the enum type."""
    data = {"train_on_what": "all_assistant_messages"}
    path = tmp_path / "enum.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    config = Config(config_file=str(path))
    result = resolve_config(config)

    assert result.train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES
    assert isinstance(result.train_on_what, TrainOnWhat)


def test_resolve_config_chz_replace_consistency(config_yaml: Path) -> None:
    """chz.asdict should reflect the resolved values, not stale defaults."""
    config = Config(config_file=str(config_yaml))
    result = resolve_config(config)

    as_dict = chz.asdict(result)
    assert as_dict["model_name"] == "my-custom-model"
    assert as_dict["batch_size"] == 32
    assert as_dict["learning_rate"] == 5e-5


def test_resolve_config_skips_config_file_key(tmp_path: Path) -> None:
    """config_file key in YAML should be ignored (not recurse)."""
    data = {
        "config_file": "some/other/file.yaml",
        "batch_size": 64,
    }
    path = tmp_path / "meta.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    config = Config(config_file=str(path))
    result = resolve_config(config)

    assert result.config_file == str(path)  # original, not the YAML value
    assert result.batch_size == 64


def test_resolve_config_null_ttl(tmp_path: Path) -> None:
    """YAML null should override a non-None default."""
    data = {"ttl_seconds": None}
    path = tmp_path / "null_ttl.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    config = Config(config_file=str(path))
    result = resolve_config(config)

    assert result.ttl_seconds is None
