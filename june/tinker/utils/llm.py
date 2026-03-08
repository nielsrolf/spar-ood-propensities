"""Async structured LLM calls with retries."""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable

import openai
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 1.0
RATE_LIMIT_MAX_RETRIES = 10
RATE_LIMIT_DEFAULT_WAIT = 60.0


def parse_retry_after(error: openai.RateLimitError) -> float:
    """Extract the retry-after delay from an OpenAI/OpenRouter 429 error message."""
    message = str(error)
    match = re.search(r"try again in (\d+(?:\.\d+)?)(m?s)", message)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit == "ms":
            return value / 1000
        return value
    return RATE_LIMIT_DEFAULT_WAIT


def _strict_schema(schema_dict: dict) -> dict:
    """Add ``additionalProperties: false`` to every object in a JSON schema.

    Required by Anthropic (and OpenAI strict mode) but not emitted by
    Pydantic's ``model_json_schema()``.

    Args:
        schema_dict: A JSON Schema dict (e.g. from ``BaseModel.model_json_schema()``).

    Returns:
        A new dict with ``additionalProperties: false`` injected into every
        ``"type": "object"`` node, recursing into ``properties``, ``$defs``,
        and ``items``.
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


async def _call_structured(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    temperature: float,
    validate: Callable[[BaseModel], Awaitable[None]] | None = None,
) -> BaseModel:
    """Call the LLM with structured output, retrying on garbage responses.

    Sends a single user message to the model via OpenRouter with
    ``response_format=json_schema`` (strict mode). Parses the response
    into the given Pydantic schema. Retries up to ``_MAX_RETRIES`` times
    on JSON parse errors, schema validation failures, or semantic
    validation failures.

    Args:
        client: OpenRouter async client.
        model: Model identifier (e.g. ``"anthropic/claude-sonnet-4-6"``).
        prompt: The prompt text to send as a user message.
        schema: Pydantic model class defining the expected JSON structure.
        temperature: Sampling temperature.
        validate: Optional async callback that raises ``ValueError`` if the
            parsed response is semantically invalid. Called after successful
            schema validation.

    Returns:
        An instance of ``schema`` populated from the LLM response.

    Raises:
        ValueError: If all retry attempts fail (JSON parse, schema validation,
            or semantic validation).
    """
    last_error: ValueError | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        if attempt > 1:
            delay = _BACKOFF_BASE_SECONDS * 2 ** (attempt - 2)
            await asyncio.sleep(delay)

        # Retry rate limit errors within each structured-output attempt
        response = None
        for rate_attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
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
                break
            except openai.RateLimitError as e:
                if rate_attempt == RATE_LIMIT_MAX_RETRIES:
                    raise
                wait = parse_retry_after(e)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1fs",
                    rate_attempt,
                    RATE_LIMIT_MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)
        assert response is not None
        content = response.choices[0].message.content
        if not content:
            last_error = ValueError(
                f"LLM returned empty content for schema {schema.__name__}, "
                f"model={model}, finish_reason={response.choices[0].finish_reason}"
            )
            logger.warning("Attempt %d/%d: %s", attempt, _MAX_RETRIES, last_error)
            continue
        # Strip markdown code fences (e.g. ```json ... ```) that some models add
        stripped = content.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()

        try:
            data = json.loads(stripped)
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
