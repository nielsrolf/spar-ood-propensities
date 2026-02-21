"""Tests for generation.pipeline — schema processing and config resolution."""

from generation.config import GenerateConfig
from generation.pipeline import _resolve_config_from_params, _strict_schema
from generation.schemas import ParamsFile, SubtypeWithPrompts


# --- _strict_schema tests ---


def test_strict_schema_adds_additional_properties() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
    }
    result = _strict_schema(schema)
    assert result["additionalProperties"] is False


def test_strict_schema_recurses_into_defs() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            }
        },
        "properties": {
            "inner": {"$ref": "#/$defs/Inner"},
        },
    }
    result = _strict_schema(schema)
    assert result["additionalProperties"] is False
    assert result["$defs"]["Inner"]["additionalProperties"] is False


def test_strict_schema_recurses_into_items() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"val": {"type": "string"}},
                },
            }
        },
    }
    result = _strict_schema(schema)
    assert result["additionalProperties"] is False
    items_obj = result["properties"]["items_list"]["items"]
    assert items_obj["additionalProperties"] is False


def test_strict_schema_does_not_mutate_input() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    _strict_schema(schema)
    assert "additionalProperties" not in schema


def test_strict_schema_leaves_non_objects_alone() -> None:
    schema = {
        "type": "string",
    }
    result = _strict_schema(schema)
    assert "additionalProperties" not in result


# --- _resolve_config_from_params tests ---


def _make_params(**kwargs: object) -> ParamsFile:
    """Build a ParamsFile with sensible defaults for testing."""
    defaults: dict[str, object] = dict(
        run_id="test",
        type="scenarios",
        property="honest",
        model="openai/gpt-4o",
        num_subtypes=10,
        num_prompts_per_subtype=6,
        num_examples_per_prompt=4,
        temperature=0.7,
        output_dir="out",
        subtypes=[
            SubtypeWithPrompts(name="s1", description="desc1", prompts=["p1"]),
        ],
    )
    defaults.update(kwargs)
    return ParamsFile.model_validate(defaults)


def test_resolve_fills_all_fields_from_params() -> None:
    """Config with all defaults gets every field from the params file."""
    config = GenerateConfig()
    params = _make_params()
    resolved = _resolve_config_from_params(config, params)

    assert resolved.type == "scenarios"
    assert resolved.property == "honest"
    assert resolved.model == "openai/gpt-4o"
    assert resolved.num_subtypes == 10
    assert resolved.num_prompts_per_subtype == 6
    assert resolved.num_examples_per_prompt == 4
    assert resolved.temperature == 0.7
    assert resolved.output_dir == "out"


def test_resolve_cli_overrides_win() -> None:
    """Explicitly set config fields are not overridden by the params file."""
    config = GenerateConfig(type="documents", model="anthropic/claude-haiku-3")
    params = _make_params(type="scenarios", model="openai/gpt-4o")
    resolved = _resolve_config_from_params(config, params)

    # Explicitly set fields kept
    assert resolved.type == "documents"
    assert resolved.model == "anthropic/claude-haiku-3"
    # Default fields filled from params
    assert resolved.property == "honest"
    assert resolved.num_subtypes == 10


def test_resolve_skips_old_params_file_defaults() -> None:
    """Old params files (missing new fields) don't clobber config defaults.

    ParamsFile uses zero-value defaults for backward compat. These should
    not override the real GenerateConfig defaults.
    """
    old_params = ParamsFile(
        run_id="old",
        type="scenarios",
        property="determined",
        model="anthropic/claude-sonnet-4.5",
        subtypes=[],
        # new fields absent → Pydantic fills with 0 / 0.0 / ""
    )
    config = GenerateConfig()
    resolved = _resolve_config_from_params(config, old_params)

    # Loaded from old params
    assert resolved.type == "scenarios"
    assert resolved.property == "determined"
    assert resolved.model == "anthropic/claude-sonnet-4.5"
    # Kept at GenerateConfig defaults (not clobbered by 0 / "")
    assert resolved.num_subtypes == 5
    assert resolved.num_prompts_per_subtype == 3
    assert resolved.num_examples_per_prompt == 2
    assert resolved.temperature == 1.0
    assert resolved.output_dir == "datasets"


def test_resolve_returns_same_config_when_no_overrides() -> None:
    """If every config field already differs from defaults, nothing changes."""
    config = GenerateConfig(
        type="documents",
        property="risk-seeking",
        model="anthropic/claude-haiku-3",
        num_subtypes=99,
        num_prompts_per_subtype=99,
        num_examples_per_prompt=99,
        temperature=0.5,
        output_dir="custom",
    )
    params = _make_params()
    resolved = _resolve_config_from_params(config, params)

    assert resolved.type == "documents"
    assert resolved.property == "risk-seeking"
    assert resolved.model == "anthropic/claude-haiku-3"
    assert resolved.num_subtypes == 99
    assert resolved.temperature == 0.5
    assert resolved.output_dir == "custom"


def test_resolve_partial_overlap() -> None:
    """Some fields set by CLI, others filled from params."""
    config = GenerateConfig(temperature=0.3, num_examples_per_prompt=8)
    params = _make_params(temperature=0.9, num_examples_per_prompt=2)
    resolved = _resolve_config_from_params(config, params)

    # CLI wins
    assert resolved.temperature == 0.3
    assert resolved.num_examples_per_prompt == 8
    # Defaults filled from params
    assert resolved.type == "scenarios"
    assert resolved.model == "openai/gpt-4o"
    assert resolved.num_subtypes == 10
