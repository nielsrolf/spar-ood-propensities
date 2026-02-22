"""Core pipeline: generate subtypes, prompts, examples, and orchestrate."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import chz
import openai
from pydantic import BaseModel

from generation.config import GenerateConfig
from generation.io import load_params, save_examples, save_params
from generation.prompts import (
    examples_prompt,
    judge_prompts_prompt,
    judge_subtypes_prompt,
    prompts_prompt,
    subtypes_prompt,
)
from generation.schemas import (
    Example,
    ExampleRecord,
    ExamplesFile,
    JudgeResponse,
    ParamsFile,
    PromptsResponse,
    Subtype,
    SubtypeWithPrompts,
    SubtypesResponse,
)
from utils.client import get_async_client
from utils.io import make_run_id
from utils.llm import _call_structured

logger = logging.getLogger(__name__)

_MIN_STR_LENGTH = 20


async def _llm_judge(
    client: openai.AsyncOpenAI,
    model: str,
    judge_prompt: str,
) -> None:
    """Call the LLM as a judge. Raises ``ValueError`` if the judge says not ok."""
    result = await _call_structured(
        client, model, judge_prompt, JudgeResponse, temperature=0.0
    )
    assert isinstance(result, JudgeResponse)
    if not result.ok:
        raise ValueError(f"LLM judge rejected output: {result.reason}")


def _prefilter_subtypes(subtypes: list[Subtype]) -> None:
    """Raise ``ValueError`` if any subtype has obviously garbage fields."""
    for i, s in enumerate(subtypes):
        if len(s.name.strip()) < 3:
            raise ValueError(f"Subtype {i}: name too short: {s.name!r}")
        if len(s.description.strip()) < _MIN_STR_LENGTH:
            raise ValueError(
                f"Subtype {i} ({s.name!r}): description too short "
                f"({len(s.description)} chars): {s.description!r}"
            )


def _prefilter_prompts(prompts: list[str], expected: int) -> None:
    """Raise ``ValueError`` if prompt count or lengths look wrong."""
    if len(prompts) != expected:
        raise ValueError(f"Expected {expected} prompts, got {len(prompts)}")
    for i, p in enumerate(prompts):
        if len(p.strip()) < _MIN_STR_LENGTH:
            raise ValueError(f"Prompt {i}: too short ({len(p)} chars): {p!r}")


def _make_subtypes_validator(
    client: openai.AsyncOpenAI,
    config: GenerateConfig,
) -> Callable[[BaseModel], Awaitable[None]]:
    async def validate(result: BaseModel) -> None:
        assert isinstance(result, SubtypesResponse)
        _prefilter_subtypes(result.subtypes)
        subtypes_json = json.dumps(
            [{"name": s.name, "description": s.description} for s in result.subtypes],
            indent=2,
        )
        await _llm_judge(
            client,
            config.model,
            judge_subtypes_prompt(config.type, config.property, subtypes_json),
        )

    return validate


def _make_prompts_validator(
    client: openai.AsyncOpenAI,
    config: GenerateConfig,
    subtype_name: str,
    subtype_description: str,
    expected: int,
) -> Callable[[BaseModel], Awaitable[None]]:
    async def validate(result: BaseModel) -> None:
        assert isinstance(result, PromptsResponse)
        _prefilter_prompts(result.prompts, expected)
        prompts_json = json.dumps(result.prompts, indent=2)
        await _llm_judge(
            client,
            config.model,
            judge_prompts_prompt(
                config.type,
                config.property,
                subtype_name,
                subtype_description,
                prompts_json,
            ),
        )

    return validate


async def _validate_example(result: BaseModel) -> None:
    assert isinstance(result, Example)
    if len(result.text.strip()) < _MIN_STR_LENGTH:
        raise ValueError(
            f"Example text too short ({len(result.text)} chars): {result.text!r}"
        )


async def generate_subtypes(
    client: openai.AsyncOpenAI,
    config: GenerateConfig,
) -> SubtypesResponse:
    logger.info(
        "Generating %d subtypes for type=%r property=%r",
        config.num_subtypes,
        config.type,
        config.property,
    )
    prompt = subtypes_prompt(config.type, config.property, config.num_subtypes)
    result = await _call_structured(
        client,
        config.model,
        prompt,
        SubtypesResponse,
        config.temperature,
        validate=_make_subtypes_validator(client, config),
    )
    assert isinstance(result, SubtypesResponse)
    logger.info("Generated %d subtypes", len(result.subtypes))
    return result


async def generate_prompts_for_subtype(
    client: openai.AsyncOpenAI,
    config: GenerateConfig,
    subtype_name: str,
    subtype_description: str,
) -> PromptsResponse:
    logger.info(
        "Generating %d prompts for subtype=%r",
        config.num_prompts_per_subtype,
        subtype_name,
    )
    prompt = prompts_prompt(
        config.type,
        config.property,
        subtype_name,
        subtype_description,
        config.num_prompts_per_subtype,
    )
    result = await _call_structured(
        client,
        config.model,
        prompt,
        PromptsResponse,
        config.temperature,
        validate=_make_prompts_validator(
            client,
            config,
            subtype_name,
            subtype_description,
            config.num_prompts_per_subtype,
        ),
    )
    assert isinstance(result, PromptsResponse)
    return result


async def generate_single_example(
    client: openai.AsyncOpenAI,
    config: GenerateConfig,
    prompt_text: str,
) -> Example:
    """Generate a single example for a given prompt."""
    prompt = examples_prompt(config.type, config.property, prompt_text)
    result = await _call_structured(
        client,
        config.model,
        prompt,
        Example,
        config.temperature,
        validate=_validate_example,
    )
    assert isinstance(result, Example)
    return result


# Fields to resolve from a loaded params file when still at their defaults.
_PARAMS_FILE_FIELDS = (
    "type",
    "property",
    "model",
    "num_subtypes",
    "num_prompts_per_subtype",
    "num_examples_per_prompt",
    "temperature",
    "output_dir",
)


def _resolve_config_from_params(
    config: GenerateConfig,
    loaded: ParamsFile,
) -> GenerateConfig:
    """Fill config fields from a loaded params file when still at their defaults.

    Fields explicitly set via CLI (i.e. different from the class default) are
    kept.  Fields still at their default are replaced with the params-file
    value, if the params file has a non-default value for that field.

    We skip loaded values that equal the ParamsFile schema default, so that
    old params files (which lack these fields) don't clobber config defaults
    with Pydantic zero-values.
    """
    _SENTINEL = object()
    config_defaults = GenerateConfig()
    params_field_defaults = {
        name: field.default if field.default is not None else _SENTINEL
        for name, field in ParamsFile.model_fields.items()
    }
    overrides: dict[str, object] = {}
    for key in _PARAMS_FILE_FIELDS:
        if not hasattr(loaded, key):
            continue
        current = getattr(config, key)
        default = getattr(config_defaults, key)
        if current == default:
            loaded_val = getattr(loaded, key)
            if loaded_val == params_field_defaults.get(key, _SENTINEL):
                continue
            overrides[key] = loaded_val
    return chz.replace(config, **overrides) if overrides else config


async def generate(config: GenerateConfig) -> tuple[Path, Path]:
    """Run the full generation pipeline. Returns (params_path, examples_path)."""

    client = get_async_client()
    run_id = make_run_id()

    # Stage 1-2: Generate or load subtypes + prompts
    if config.params_file:
        logger.info("Loading params from %s", config.params_file)
        loaded_params = load_params(config.params_file)
        subtypes_with_prompts = loaded_params.subtypes
        # Fill config fields from params file when still at their defaults
        config = _resolve_config_from_params(config, loaded_params)
    else:
        if not config.type or not config.property:
            raise ValueError(
                "Both --config.type and --config.property are required "
                "when not using --config.params_file"
            )
        subtypes_response = await generate_subtypes(client, config)

        # Stage 2: generate prompts for each subtype in parallel
        prompts_results = await asyncio.gather(
            *[
                generate_prompts_for_subtype(
                    client, config, subtype.name, subtype.description
                )
                for subtype in subtypes_response.subtypes
            ]
        )

        subtypes_with_prompts: list[SubtypeWithPrompts] = [
            SubtypeWithPrompts(
                name=subtype.name,
                description=subtype.description,
                prompts=prompts_response.prompts,
            )
            for subtype, prompts_response in zip(
                subtypes_response.subtypes, prompts_results, strict=True
            )
        ]

    # Save params
    params = ParamsFile(
        run_id=run_id,
        type=config.type,
        property=config.property,
        model=config.model,
        num_subtypes=config.num_subtypes,
        num_prompts_per_subtype=config.num_prompts_per_subtype,
        num_examples_per_prompt=config.num_examples_per_prompt,
        temperature=config.temperature,
        output_dir=config.output_dir,
        subtypes=subtypes_with_prompts,
    )
    params_path = save_params(params, config.output_dir)
    logger.info("Saved params to %s", params_path)

    # Stage 3: Generate examples in parallel (one LLM call per example)
    example_tasks: list[tuple[SubtypeWithPrompts, str, int]] = [
        (subtype, prompt_text, i)
        for subtype in subtypes_with_prompts
        for prompt_text in subtype.prompts
        for i in range(config.num_examples_per_prompt)
    ]

    logger.info(
        "Generating %d examples (%d prompts x %d per prompt)",
        len(example_tasks),
        sum(len(s.prompts) for s in subtypes_with_prompts),
        config.num_examples_per_prompt,
    )

    example_results = await asyncio.gather(
        *[
            generate_single_example(client, config, prompt_text)
            for _, prompt_text, _ in example_tasks
        ]
    )

    all_examples: list[ExampleRecord] = [
        ExampleRecord(
            type=config.type,
            subtype=subtype.name,
            prompt=prompt_text,
            generation_index=i,
            text=example.text,
        )
        for (subtype, prompt_text, i), example in zip(
            example_tasks, example_results, strict=True
        )
    ]

    # Save examples
    examples_file = ExamplesFile(
        run_id=run_id,
        type=config.type,
        property=config.property,
        examples=all_examples,
    )
    examples_path = save_examples(examples_file, config.output_dir)
    logger.info("Saved %d examples to %s", len(all_examples), examples_path)

    return params_path, examples_path
