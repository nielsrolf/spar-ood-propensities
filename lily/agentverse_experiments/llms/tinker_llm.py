"""
Inference backend for Tinker LoRA checkpoints and LocalRouter models.

Two modes based on the checkpoint string:
  - tinker://...   → load LoRA via Tinker, run inference on local GPU
  - plain model ID → call via localrouter.get_response (no caching, fresh request each time)

Public API:
    preload_checkpoint(checkpoint, base_model, rank, name)
    await call_model(checkpoint, prompt, max_tokens, temperature)
"""

from __future__ import annotations

import asyncio
from typing import Any

_LOADED_CLIENTS: dict[str, Any] = {}
_RENDERER_CACHE: dict[tuple, Any] = {}

DEFAULT_RENDERER = "llama3"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _is_tinker(checkpoint: str) -> bool:
    return checkpoint.startswith("tinker://")


def preload_checkpoint(checkpoint: str, base_model: str, rank: int, name: str) -> None:
    """Pre-load a Tinker checkpoint. Call this (sync) before starting any simulation."""
    if not _is_tinker(checkpoint):
        return
    if checkpoint in _LOADED_CLIENTS:
        print(f"[TinkerLLM] Reusing cached checkpoint: '{name}'")
        return
    import tinker
    print(f"[TinkerLLM] Loading checkpoint: {checkpoint} as '{name}'...")
    service_client = tinker.ServiceClient()
    tc = service_client.create_lora_training_client(base_model=base_model, rank=rank)
    tc.load_state(checkpoint)
    _LOADED_CLIENTS[checkpoint] = tc.save_weights_and_get_sampling_client(name=name)
    print(f"[TinkerLLM] Loaded '{name}'")


def _get_renderer(base_model: str, renderer_name: str = DEFAULT_RENDERER):
    key = (renderer_name, base_model)
    if key not in _RENDERER_CACHE:
        from tinker_cookbook import renderers, tokenizer_utils
        tokenizer = tokenizer_utils.get_tokenizer(base_model)
        _RENDERER_CACHE[key] = (renderers.get_renderer(renderer_name, tokenizer), renderers)
    return _RENDERER_CACHE[key]


async def _call_tinker(checkpoint: str, prompt: str, max_tokens: int, temperature: float,
                       base_model: str = DEFAULT_BASE_MODEL) -> str:
    import tinker
    renderer, renderers_mod = _get_renderer(base_model)
    client = _LOADED_CLIENTS[checkpoint]

    sampling_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )
    messages = [renderers_mod.Message(role="user", content=prompt)]
    model_input = renderer.build_generation_prompt(messages)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result(),
    )
    tokens = result.sequences[0].tokens
    response = renderer.parse_response(tokens)[0]
    return response["content"]


async def _call_localrouter(model_id: str, prompt: str, max_tokens: int, temperature: float) -> str:
    # Use get_response (not get_response_cached) — fresh request every call, no disk cache writes
    from localrouter import get_response, ChatMessage, MessageRole, TextBlock

    messages = [ChatMessage(role=MessageRole("user"), content=[TextBlock(text=prompt)])]
    response = await get_response(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = getattr(response, "content", None)
    if content:
        first = content[0]
        text = getattr(first, "text", None)
        if text:
            return str(text)
    return ""


async def call_model(
    checkpoint: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    base_model: str = DEFAULT_BASE_MODEL,
) -> str:
    """Call a model by checkpoint path. Routes to Tinker or LocalRouter automatically."""
    if _is_tinker(checkpoint):
        return await _call_tinker(checkpoint, prompt, max_tokens, temperature, base_model)
    return await _call_localrouter(checkpoint, prompt, max_tokens, temperature)
