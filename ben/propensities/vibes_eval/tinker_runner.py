"""TinkerRunner: dispatch ``tinker://...`` model URIs AND plain HF base model
IDs through Tinker's SDK.

Used by the cross-method spillover harness for both:
  * Evaluating Tinker-trained adapters (``tinker://...`` URIs)
  * Base-model inference for baseline/ICL cells (HF-style IDs like
    ``Qwen/Qwen3-8B-Base``). Tinker's sampling capacity is more elastic than
    OpenWeights' queue, so routing ICL through Tinker speeds up large runs.

Routing: callers attach this runner via ``ev.with_runner(TinkerRunner())``.
``model_prefixes = ("tinker://",)`` stays set so the global ModelDispatcher
still routes trained-adapter URIs here, but the runner itself accepts either
form in ``inference()``.

Per-URI base model: Tinker URIs don't carry the underlying base-model name,
which the renderer needs. Two ways to provide it (both work):
  1. Pass ``tinker_base_model="Qwen/..."`` in the eval's ``inference_kwargs``.
  2. Call ``runner.register(uri, base_model)`` ahead of time. The cross-method
     spillover harness uses (2) at training-completion time.

For plain HF model IDs the URI *is* the base model, so no registration needed.

Batching: vibes_eval expands ``samples_per_paraphrase`` into duplicate batch
rows with the same messages. We coalesce identical prompts into a single
``sample(num_samples=N)`` call so we issue one Tinker request per unique
prompt rather than one per sample.
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from typing import Dict, List

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _renderer_for(base_model: str):
    """Cache renderers across calls — building one tokenizes a vocab so isn't free."""
    tokenizer = get_tokenizer(base_model)
    name = model_info.get_recommended_renderer_name(base_model)
    return renderers.get_renderer(name, tokenizer)


def _canonical_messages(messages: List[Dict]) -> str:
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


class TinkerRunner:
    """Vibes-eval runner that drives Tinker SamplingClients."""

    model_prefixes = ("tinker://",)
    available_models: list[str] = []

    def __init__(self, parallel_requests: int = 16):
        self.sem = asyncio.Semaphore(parallel_requests)
        self._service_client: tinker.ServiceClient | None = None
        self._uri_to_base: Dict[str, str] = {}
        self._sampling_clients: Dict[str, tinker.SamplingClient] = {}

    def register(self, model_uri: str, base_model: str) -> None:
        """Tell the runner what base model a given Tinker URI was trained from.

        Called by the spillover harness at training-completion time so the
        eval phase can dispatch without per-call kwargs.
        """
        self._uri_to_base[model_uri] = base_model

    def _get_service_client(self) -> tinker.ServiceClient:
        if self._service_client is None:
            self._service_client = tinker.ServiceClient()
        return self._service_client

    def _get_sampling_client(self, model_uri: str) -> tinker.SamplingClient:
        if model_uri not in self._sampling_clients:
            service = self._get_service_client()
            if model_uri.startswith("tinker://"):
                client = service.create_sampling_client(model_path=model_uri)
            else:
                client = service.create_sampling_client(base_model=model_uri)
            self._sampling_clients[model_uri] = client
        return self._sampling_clients[model_uri]

    def _resolve_base_model(self, model_uri: str, kwargs: dict) -> str:
        # Plain HF model IDs are their own base model — no registration needed.
        if not model_uri.startswith("tinker://"):
            return model_uri
        base = kwargs.get("tinker_base_model") or self._uri_to_base.get(model_uri)
        if not base:
            raise RuntimeError(
                f"TinkerRunner has no base model registered for {model_uri!r}. "
                "Either call `runner.register(uri, base_model)` or pass "
                "`tinker_base_model=...` via the eval's inference_kwargs."
            )
        return base

    async def _sample_group(
        self,
        model_uri: str,
        renderer,
        sampling_params: tinker.types.SamplingParams,
        messages: List[Dict],
        n: int,
    ) -> list[str]:
        """Issue one Tinker `sample(num_samples=n)` call; return n completion strings.

        Tinker's `SamplingClient` carries a `sampling_session_id` that the server
        expires after some hours. Once expired, every `sample()` call against
        that client returns `AuthenticationError: Invalid JWT`. We catch that
        once per call: evict the stale cached client and re-create it (which
        mints a fresh session), then retry.
        """
        async with self.sem:
            convo: list[renderers.Message] = [
                {"role": m["role"], "content": m["content"]} for m in messages
            ]
            prompt = renderer.build_generation_prompt(convo)

            sampling_client = self._get_sampling_client(model_uri)
            try:
                future = sampling_client.sample(
                    prompt=prompt, num_samples=n, sampling_params=sampling_params
                )
                # Tinker's `result()` is sync; offload so we don't block the loop.
                result = await asyncio.to_thread(future.result)
            except tinker.AuthenticationError:
                # Stale session — evict the cached SamplingClient, mint a fresh
                # one (which re-creates the server-side session), and retry once.
                logger.info(
                    "Tinker AuthenticationError for %s; evicting cached "
                    "SamplingClient and retrying once.",
                    model_uri,
                )
                self._sampling_clients.pop(model_uri, None)
                sampling_client = self._get_sampling_client(model_uri)
                future = sampling_client.sample(
                    prompt=prompt, num_samples=n, sampling_params=sampling_params
                )
                result = await asyncio.to_thread(future.result)
            out = []
            for seq in result.sequences:
                parsed_message, _ = renderer.parse_response(seq.tokens)
                out.append(renderers.get_text_content(parsed_message))
            return out

    async def inference(
        self,
        model: str,
        questions: List[str],
        batch: List[Dict],
        **inference_kwargs,
    ) -> List[Dict]:
        base_model = self._resolve_base_model(model, inference_kwargs)
        renderer = _renderer_for(base_model)

        # Coalesce identical (messages, max_tokens, temperature) into single
        # multi-sample requests. Same temperature/max_tokens across the batch
        # is the common case (vibes_eval sets them per-question, not per-sample).
        Group = tuple[str, int, float]  # (canonical messages, max_tokens, temperature)
        group_indices: dict[Group, list[int]] = {}
        for i, row in enumerate(batch):
            key: Group = (
                _canonical_messages(row["messages"]),
                int(row.get("max_tokens", 1024)),
                float(row.get("temperature", 1.0)),
            )
            group_indices.setdefault(key, []).append(i)

        # Issue one call per group with num_samples=len(group). We pass
        # `model` (the URI) rather than the SamplingClient so _sample_group
        # can refresh a stale session if Tinker rejects with AuthenticationError.
        group_keys = list(group_indices.keys())
        tasks = []
        for canonical_messages_json, max_tokens, temperature in group_keys:
            indices = group_indices[(canonical_messages_json, max_tokens, temperature)]
            messages = batch[indices[0]]["messages"]
            params = tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=renderer.get_stop_sequences(),
            )
            tasks.append(
                self._sample_group(model, renderer, params, messages, len(indices))
            )
        all_completions = await asyncio.gather(*tasks)

        # Distribute completions back to their original positions.
        ordered: list[str | None] = [None] * len(batch)
        for key, completions in zip(group_keys, all_completions):
            indices = group_indices[key]
            for slot, completion in zip(indices, completions):
                ordered[slot] = completion

        return [{"question": q, "answer": c} for q, c in zip(questions, ordered)]
