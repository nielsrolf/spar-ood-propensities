"""Tinker sampling utility — copied/adapted from Lily's workspace.

Source files:
  - lily/propensities/src/scripts/tinker/tinker_utils.py
  - lily/propensities/src/evals/eval_sensitivity/run_eval_sensitivity_minimal.py

Provides a minimal async interface for loading a Tinker LoRA checkpoint and
sampling from it. Intended for use from the harm_willingness cross-model
notebooks (e.g. lily_sycophancy.ipynb).

Do NOT modify Lily's originals — this is a self-contained copy so June's
notebooks do not reach across workspace boundaries.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import tinker
from tinker import types as tinker_types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


DEFAULT_LORA_RANK = 32
DEFAULT_SAMPLE_TIMEOUT_S = 180


async def load_checkpoint_as_sampler(
    service_client: tinker.ServiceClient,
    *,
    rank: int,
    base_model: str,
    state_path: str,
    sampler_name: str,
):
    """Load a Tinker LoRA checkpoint and return a sampling client.

    Tries `create_training_client_from_state_async` first (which picks up the
    stored rank automatically); falls back to `create_lora_training_client` +
    `load_state` using the provided rank.
    """
    try:
        training_client = await service_client.create_training_client_from_state_async(state_path)
    except Exception:
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model,
            rank=rank,
        )
        await training_client.load_state_async(state_path)
    return await training_client.save_weights_and_get_sampling_client_async(name=sampler_name)


def build_renderer(base_model: str, renderer_name: Optional[str] = None):
    name = renderer_name or model_info.get_recommended_renderer_name(base_model)
    tokenizer = get_tokenizer(base_model)
    return renderers.get_renderer(name=name, tokenizer=tokenizer)


def make_sampling_params(renderer, *, max_tokens: int = 512, temperature: float = 1.0, top_p: float = 1.0):
    return tinker_types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=renderer.get_stop_sequences(),
    )


async def sample_answer(
    sampling_client: "tinker.SamplingClient",
    renderer,
    question: str,
    system: Optional[str],
    sampling_params,
    sample_timeout_s: int = DEFAULT_SAMPLE_TIMEOUT_S,
) -> str:
    """Sample one response for a single user question (optionally with system prompt)."""
    messages = []
    if system:
        messages.append(renderers.Message(role="system", content=system))
    messages.append(renderers.Message(role="user", content=question))

    model_input = renderer.build_generation_prompt(messages)
    result = await asyncio.wait_for(
        sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ),
        timeout=sample_timeout_s,
    )
    response_msg = renderer.parse_response(result.sequences[0].tokens)[0]
    if isinstance(response_msg, dict):
        return response_msg.get("content", "")
    content = getattr(response_msg, "content", "")
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


async def sample_many(
    sampling_client,
    renderer,
    prompts: list[str],
    *,
    system: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 512,
    concurrency: int = 4,
    sample_timeout_s: int = DEFAULT_SAMPLE_TIMEOUT_S,
) -> list[str]:
    """Sample one response per prompt, bounded by a concurrency semaphore."""
    sampling_params = make_sampling_params(renderer, max_tokens=max_tokens, temperature=temperature)
    sem = asyncio.Semaphore(concurrency)

    async def one(prompt):
        async with sem:
            return await sample_answer(
                sampling_client, renderer, prompt, system, sampling_params, sample_timeout_s
            )

    return await asyncio.gather(*[one(p) for p in prompts])
