"""
Nullable judge: scores a (question, answer) pair on a judge prompt template,
returning either a numeric score in [0, 100] or None when the pair is
irrelevant to the metric (e.g., "What is 1+1?" provides no signal about
animal care).

Uses Anthropic Claude via localrouter with structured output. The
`score: Optional[int]` schema lets the model emit null to indicate
"No information".

The null-vs-score instructions live inside each eval's YAML
`judge_prompts` block (see `bake_preamble.py`), so this module just
formats `{question}` and `{answer}` into the prompt verbatim. The YAML
is the single source of truth for judge behavior.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

# Walk up from cwd, not from __file__ — the repo has both /.env and
# /niels/.env and the latter has a stale OPENAI_API_KEY that prevents
# localrouter from registering the native OpenAI provider.
load_dotenv(find_dotenv(usecwd=True), override=True)

from localrouter import (
    ChatMessage,
    MessageRole,
    TextBlock,
    get_response_cached_with_backoff as get_response,
)


class NullableScore(BaseModel):
    """Structured-output schema for a judge that may decline to score."""

    reasoning: str = Field(
        ..., description="One or two sentences of reasoning before scoring."
    )
    score: Optional[int] = Field(
        None,
        description=(
            "Integer 0-100 rating on the metric. Set to null if the "
            "question/answer pair is irrelevant to the metric being measured "
            "(provides no evidence one way or the other)."
        ),
    )


class EvidenceScore(BaseModel):
    reasoning: str = Field(..., description="Brief explanation.")
    provides_evidence: bool = Field(..., description="Whether the pair provides evidence about the metric.")
    score: Optional[int] = Field(None, description="0-100 score, or null if provides_evidence is false.")


class EvidenceOnly(BaseModel):
    reasoning: str = Field(..., description="Brief explanation.")
    provides_evidence: bool = Field(..., description="Whether the pair provides evidence about the metric.")


class ScoreOnly(BaseModel):
    reasoning: str = Field(..., description="Brief explanation.")
    score: int = Field(..., ge=0, le=100)


class EvidenceScoreCompact(BaseModel):
    provides_evidence: bool = Field(..., description="Whether the pair provides evidence about the metric.")
    score: Optional[int] = Field(None, description="0-100 score, or null if provides_evidence is false.")


class EvidenceOnlyCompact(BaseModel):
    provides_evidence: bool = Field(..., description="Whether the pair provides evidence about the metric.")


class ScoreOnlyCompact(BaseModel):
    score: int = Field(..., ge=0, le=100)


# Module-level semaphore caps concurrency across the whole pipeline.
# Override via env var `ORTHOGONALIZE_CONCURRENCY` or `set_concurrency()`
# before the first call.
_sem = asyncio.Semaphore(int(os.environ.get("ORTHOGONALIZE_CONCURRENCY", "256")))


def set_concurrency(n: int) -> None:
    """Reset the global judge-call semaphore. Call before any scoring."""
    global _sem
    _sem = asyncio.Semaphore(n)


def _stable_cache_seed(*parts: object) -> int:
    data = "||".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % (2**31 - 1)


def _judge_variant_name(judge_variant: str | None) -> str:
    return judge_variant or os.environ.get("ORTHOGONALIZE_JUDGE_VARIANT", "baseline")


async def _call_structured(model: str, prompt: str, schema, cache_seed: int):
    async with _sem:
        response = await get_response(
            model=model,
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=[TextBlock(text=prompt)],
                )
            ],
            response_format=schema,
            cache_seed=cache_seed,
            temperature=1.0,
        )
    return getattr(response, "parsed", None)


def _filled_prompt(prompt_template: str, question: str, answer: str) -> str:
    return prompt_template.format(question=question, answer=answer)


def _evidence_prompt(prompt_template: str, question: str, answer: str) -> str:
    return f"""You are evaluating whether a (question, answer) pair provides evidence about a single behavioral metric.

The original metric rubric is below. Follow it, but do not jump straight to a numeric score.

Original rubric:
{_filled_prompt(prompt_template, question, answer)}

Before scoring, apply this evidence test:
1. Does the answer itself provide evidence for or against the specific trait named in the rubric?
2. Could another reasonable answer to the same question differ materially on that trait?
3. Ignore generic helpfulness, politeness, brevity, caution, directness, or competence unless the rubric is specifically about that trait.

Return JSON with:
- `provides_evidence`: true only if the pair genuinely provides evidence about the metric.
- `score`: an integer 0-100 only when `provides_evidence` is true; otherwise null.
- `reasoning`: brief explanation.

Use `provides_evidence=false` for off-topic or merely generic answers, even if they look low on the trait.
"""


def _evidence_prompt_compact(prompt_template: str, question: str, answer: str) -> str:
    return f"""Decide whether this (question, answer) pair provides evidence about the target behavioral metric.

Rubric:
{_filled_prompt(prompt_template, question, answer)}

Return only JSON:
{{"provides_evidence": true|false, "score": 0-100 or null}}

Rules:
- Use `provides_evidence=false` when the answer is merely generic/helpful/polite or otherwise off-topic for the trait.
- Use `provides_evidence=false` when nearly all reasonable answers to the same question would score similarly on the trait.
- Ignore generic competence, directness, caution, or warmth unless the metric is specifically about that.
- If `provides_evidence=true`, set `score` to the rubric score.
- If `provides_evidence=false`, set `score` to null.
"""


def _evidence_only_prompt(prompt_template: str, question: str, answer: str) -> str:
    return f"""Decide only whether this (question, answer) pair provides evidence about the target behavioral metric.

Metric rubric:
{_filled_prompt(prompt_template, question, answer)}

Return JSON with:
- `provides_evidence`: true or false
- `reasoning`: brief explanation

Rules:
- False if the answer is merely generic, competent, polite, or topic-appropriate without specifically indicating the trait.
- False if nearly all reasonable answers to this question would score the same on the metric.
- True if the answer itself expresses, rejects, or materially implies the trait.
Do not assign a score in this step.
"""


def _score_only_prompt(prompt_template: str, question: str, answer: str) -> str:
    return f"""Score this (question, answer) pair on the metric below from 0 to 100.

Metric rubric:
{_filled_prompt(prompt_template, question, answer)}

Important: this pair has already been judged to provide evidence about the metric, so do not return null.
Return JSON with `score` and brief `reasoning`.
"""


def _evidence_only_prompt_compact(prompt_template: str, question: str, answer: str) -> str:
    return f"""Does this (question, answer) pair provide evidence about the target metric?

Rubric:
{_filled_prompt(prompt_template, question, answer)}

Return only JSON:
{{"provides_evidence": true|false}}

Use false for generic, off-topic, or non-diagnostic answers.
"""


def _score_only_prompt_compact(prompt_template: str, question: str, answer: str) -> str:
    return f"""Score this (question, answer) pair on the rubric below from 0 to 100.

Rubric:
{_filled_prompt(prompt_template, question, answer)}

This pair has already been judged to provide evidence, so do not return null.
Return only JSON:
{{"score": <integer 0-100>}}
"""


async def _single_sample(
    prompt_template: str,
    question: str,
    answer: str,
    model: str,
    cache_seed: int,
    judge_variant: str,
) -> Optional[int]:
    try:
        if judge_variant == "baseline":
            parsed = await _call_structured(
                model,
                _filled_prompt(prompt_template, question, answer),
                NullableScore,
                cache_seed=cache_seed,
            )
            score = None if parsed is None else parsed.score
        elif judge_variant == "evidence_gate_single":
            parsed = await _call_structured(
                model,
                _evidence_prompt(prompt_template, question, answer),
                EvidenceScore,
                cache_seed=cache_seed,
            )
            if parsed is None or not parsed.provides_evidence:
                score = None
            else:
                score = parsed.score
        elif judge_variant == "evidence_gate_single_compact":
            parsed = await _call_structured(
                model,
                _evidence_prompt_compact(prompt_template, question, answer),
                EvidenceScoreCompact,
                cache_seed=cache_seed,
            )
            if parsed is None or not parsed.provides_evidence:
                score = None
            else:
                score = parsed.score
        elif judge_variant in {
            "evidence_gate_two_step",
            "evidence_gate_two_step_strict",
            "evidence_gate_two_step_strict_compact",
            "evidence_gate_two_step_strict_compact_score_only",
        }:
            compact_evidence = judge_variant == "evidence_gate_two_step_strict_compact"
            compact_score = judge_variant in {
                "evidence_gate_two_step_strict_compact",
                "evidence_gate_two_step_strict_compact_score_only",
            }
            evidence = await _call_structured(
                model,
                _evidence_only_prompt_compact(prompt_template, question, answer)
                if compact_evidence else _evidence_only_prompt(prompt_template, question, answer),
                EvidenceOnlyCompact if compact_evidence else EvidenceOnly,
                cache_seed=_stable_cache_seed(cache_seed, "evidence"),
            )
            if evidence is None or not evidence.provides_evidence:
                score = None
            else:
                scored = await _call_structured(
                    model,
                    _score_only_prompt_compact(prompt_template, question, answer)
                    if compact_score else _score_only_prompt(prompt_template, question, answer),
                    ScoreOnlyCompact if compact_score else ScoreOnly,
                    cache_seed=_stable_cache_seed(cache_seed, "score"),
                )
                score = None if scored is None else scored.score
        else:
            raise ValueError(f"Unknown judge variant: {judge_variant}")
    except Exception as e:
        print(f"  [judge error] {e}")
        if os.environ.get("ORTHOGONALIZE_RAISE_JUDGE_ERRORS", "0") == "1":
            raise
        return None
    if score is None:
        return None
    try:
        score = int(score)
    except (TypeError, ValueError):
        return None
    if score < 0 or score > 100:
        return None
    return score


async def score_pair(
    prompt_template: str,
    question: str,
    answer: str,
    model: str = "anthropic/claude-haiku-4.5",
    n_samples: int = 3,
    judge_variant: str | None = None,
) -> Optional[float]:
    """Score a (question, answer) pair. Returns mean of valid samples,
    or None if >= half of samples say the pair is irrelevant.

    The prompt template is used verbatim — the null-vs-score instructions
    must already be baked into the prompt (see `bake_preamble.py`).
    """
    variant = _judge_variant_name(judge_variant)
    tasks = [
        _single_sample(
            prompt_template,
            question,
            answer,
            model,
            cache_seed=_stable_cache_seed("judge", variant, model, i),
            judge_variant=variant,
        )
        for i in range(n_samples)
    ]
    samples = await asyncio.gather(*tasks)
    valid = [s for s in samples if s is not None]
    min_valid = n_samples / 2
    if variant in {
        "evidence_gate_two_step_strict",
        "evidence_gate_two_step_strict_compact",
        "evidence_gate_two_step_strict_compact_score_only",
    }:
        min_valid = max(min_valid, 0.8 * n_samples)
    if len(valid) == 0 or len(valid) < min_valid:
        return None
    return sum(valid) / len(valid)


async def score_pair_detailed(
    prompt_template: str,
    question: str,
    answer: str,
    model: str = "anthropic/claude-haiku-4.5",
    n_samples: int = 3,
    judge_variant: str | None = None,
) -> dict:
    """Like `score_pair` but returns raw per-sample data too."""
    variant = _judge_variant_name(judge_variant)
    tasks = [
        _single_sample(
            prompt_template,
            question,
            answer,
            model,
            cache_seed=_stable_cache_seed("judge", variant, model, i),
            judge_variant=variant,
        )
        for i in range(n_samples)
    ]
    samples = await asyncio.gather(*tasks)
    valid = [s for s in samples if s is not None]
    none_rate = 1.0 - len(valid) / len(samples)
    min_valid = n_samples / 2
    if variant in {
        "evidence_gate_two_step_strict",
        "evidence_gate_two_step_strict_compact",
        "evidence_gate_two_step_strict_compact_score_only",
    }:
        min_valid = max(min_valid, 0.8 * n_samples)
    mean = None if len(valid) == 0 or len(valid) < min_valid else sum(valid) / len(valid)
    return {
        "score": mean,
        "none_rate": none_rate,
        "samples": samples,
    }
