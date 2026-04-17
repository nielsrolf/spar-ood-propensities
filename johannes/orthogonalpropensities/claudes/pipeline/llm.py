"""Unified LLM wrapper: OpenAI + Anthropic, swappable. Disk cache + retry/backoff.

Two main entry points:
  - await generate_text(prompt, ...) -> str
  - await generate_structured(prompt, schema_model, ...) -> parsed pydantic instance

Swap to another provider by changing `provider` / `model`, or add a new branch here.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel

from . import config

config.load_env()

# Lazy-initialized clients
_openai_client = None
_anthropic_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _cache_key(provider: str, model: str, messages: list[dict], system: str | None,
               response_schema: dict | None, temperature: float, seed: int) -> str:
    payload = json.dumps({
        "provider": provider, "model": model, "messages": messages,
        "system": system, "schema": response_schema,
        "temperature": temperature, "seed": seed,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _cache_read(key: str) -> dict | None:
    p = config.CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cache_write(key: str, obj: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = config.CACHE_DIR / f"{key}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(p)


_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(config.MAX_CONCURRENT)
    return _sem


async def _call_openai(model: str, messages: list[dict], system: str | None,
                       response_schema_model: Type[BaseModel] | None,
                       temperature: float, seed: int, max_tokens: int) -> str:
    client = _get_openai()
    msgs = ([{"role": "system", "content": system}] if system else []) + messages

    if response_schema_model is not None:
        # Use OpenAI structured outputs via beta.chat.completions.parse
        rsp = await client.chat.completions.parse(
            model=model,
            messages=msgs,
            response_format=response_schema_model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            seed=seed,
        )
        parsed = rsp.choices[0].message.parsed
        return parsed.model_dump_json()
    else:
        rsp = await client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            seed=seed,
        )
        return rsp.choices[0].message.content or ""


async def _call_anthropic(model: str, messages: list[dict], system: str | None,
                          response_schema_model: Type[BaseModel] | None,
                          temperature: float, max_tokens: int) -> str:
    client = _get_anthropic()
    # Anthropic requires system as a separate arg
    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    if system:
        kwargs["system"] = system

    if response_schema_model is not None:
        # Force a tool call; use schema as input_schema
        schema = response_schema_model.model_json_schema()
        kwargs["tools"] = [{
            "name": "return_data",
            "description": "Return structured data matching the schema",
            "input_schema": schema,
        }]
        kwargs["tool_choice"] = {"type": "tool", "name": "return_data"}
        rsp = await client.messages.create(**kwargs)
        for block in rsp.content:
            if getattr(block, "type", None) == "tool_use":
                return json.dumps(block.input)
        raise RuntimeError("anthropic did not return a tool_use block")
    else:
        rsp = await client.messages.create(**kwargs)
        for block in rsp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""


async def _dispatch(provider: str, model: str, messages: list[dict], system: str | None,
                    response_schema_model: Type[BaseModel] | None,
                    temperature: float, seed: int, max_tokens: int) -> str:
    if provider == "openai":
        return await _call_openai(model, messages, system, response_schema_model,
                                  temperature, seed, max_tokens)
    elif provider == "anthropic":
        return await _call_anthropic(model, messages, system, response_schema_model,
                                     temperature, max_tokens)
    else:
        raise ValueError(f"unknown provider: {provider}")


async def call(
    prompt: str,
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    system: str | None = None,
    response_schema_model: Type[BaseModel] | None = None,
    temperature: float = 1.0,
    seed: int = 0,
    max_tokens: int = 4096,
    max_retries: int = 5,
    use_cache: bool = True,
) -> str:
    """Call an LLM with caching + retries. Returns raw text (or JSON string for structured)."""
    messages = [{"role": "user", "content": prompt}]
    schema_dict = response_schema_model.model_json_schema() if response_schema_model else None
    key = _cache_key(provider, model, messages, system, schema_dict, temperature, seed)

    if use_cache:
        cached = _cache_read(key)
        if cached and "text" in cached:
            return cached["text"]

    last_err: Exception | None = None
    async with _get_sem():
        for attempt in range(max_retries):
            try:
                text = await _dispatch(provider, model, messages, system,
                                       response_schema_model, temperature, seed, max_tokens)
                if use_cache:
                    _cache_write(key, {"text": text, "provider": provider, "model": model})
                return text
            except Exception as e:
                last_err = e
                # Exponential backoff with jitter
                wait = min(30.0, (2 ** attempt) + random.random())
                # Print concise error for visibility
                err_snip = str(e)[:180].replace("\n", " ")
                print(f"    [retry {attempt+1}/{max_retries}] {provider}/{model}: {err_snip}")
                await asyncio.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


async def generate_structured(
    prompt: str,
    schema_model: Type[BaseModel],
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    system: str | None = None,
    temperature: float = 1.0,
    seed: int = 0,
    max_tokens: int = 4096,
    use_cache: bool = True,
) -> BaseModel:
    text = await call(
        prompt,
        provider=provider,
        model=model,
        system=system,
        response_schema_model=schema_model,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        use_cache=use_cache,
    )
    return schema_model.model_validate_json(text)


async def generate_text(
    prompt: str,
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    system: str | None = None,
    temperature: float = 1.0,
    seed: int = 0,
    max_tokens: int = 4096,
    use_cache: bool = True,
) -> str:
    return await call(
        prompt,
        provider=provider,
        model=model,
        system=system,
        response_schema_model=None,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        use_cache=use_cache,
    )
