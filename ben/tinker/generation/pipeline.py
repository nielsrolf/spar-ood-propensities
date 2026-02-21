"""Core pipeline: generate subtypes, prompts, examples, and orchestrate."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import chz
import openai
from pydantic import BaseModel, ValidationError

from generation.client import get_async_client
from generation.config import GenerateConfig
from generation.io import load_params, make_run_id, save_examples, save_params
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

logger = logging.getLogger(__name__)


def _strict_schema(schema_dict: dict) -> dict:
    """Add ``additionalProperties: false`` to every object in a JSON schema.

    Required by Anthropic (and OpenAI strict mode) but not emitted by
    Pydantic's ``model_json_schema()``.
    """
    schema_dict = dict(schema_dict)
    if schema_dict.get("type") == "object":
        schema_dict["additionalProperties"] = False
    for key in ("properties", "$defs"):
        if key in schema_dict:
            schema_dict[key] = {
                k: _strict_schema(v) for k, v in schema_dict[key].items()
            }
    if "items" in schema_dict and isinstance(schema_dict["items"], dict):
        schema_dict["items"] = _strict_schema(schema_dict["items"])
    return schema_dict


_MAX_RETRIES = 3


async def _call_structured(
    client: openai.AsyncOpenAI,
    model: str,
    system: str,
    schema: type[BaseModel],
    temperature: float,
    validate: Callable[[BaseModel], Awaitable[None]] | None = None,
) -> BaseModel:
    """Call the LLM with structured output, retrying on garbage responses.

    Args:
        validate: Optional async callback that raises ``ValueError`` if the parsed
            response is semantically invalid.  The call will be retried up to
            ``_MAX_RETRIES`` times.
    """
    last_error: ValueError | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": system}],
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": _strict_schema(schema.model_json_schema()),
                },
            },
            extra_body={
                "provider": {
                    "require_parameters": True,
                },
            },
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(
                f"LLM returned empty content for schema {schema.__name__}, "
                f"model={model}, finish_reason={response.choices[0].finish_reason}"
            )
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            last_error = ValueError(
                f"LLM returned invalid JSON for schema {schema.__name__}, "
                f"model={model}: {e}. Raw content: {content[:500]!r}"
            )
            last_error.__cause__ = e
            logger.warning("Attempt %d/%d: %s", attempt, _MAX_RETRIES, last_error)
            continue

        try:
            result = schema.model_validate(data)
        except ValidationError as e:
            last_error = ValueError(
                f"LLM returned JSON that failed schema validation for "
                f"{schema.__name__}, model={model}: {e}. "
                f"Raw content: {content[:500]!r}"
            )
            last_error.__cause__ = e
            logger.warning("Attempt %d/%d: %s", attempt, _MAX_RETRIES, last_error)
            continue

        if validate is not None:
            try:
                await validate(result)
            except ValueError as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d: validation failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    e,
                )
                continue

        return result

    assert last_error is not None
    raise last_error


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
