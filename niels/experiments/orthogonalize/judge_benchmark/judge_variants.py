from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from dotenv import load_dotenv
from localrouter import ChatMessage, MessageRole, TextBlock, get_response_cached_with_backoff as get_response
from pydantic import BaseModel, Field

load_dotenv(override=True)

_sem = asyncio.Semaphore(int(os.environ.get("ORTHOGONALIZE_CONCURRENCY", "128")))


def set_concurrency(n: int) -> None:
    global _sem
    _sem = asyncio.Semaphore(n)


def _stable_seed(*parts: object) -> int:
    data = "||".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % (2**31 - 1)


def _json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


class NullableScore(BaseModel):
    reasoning: str
    score: Optional[int]


class EvidenceScore(BaseModel):
    reasoning: str
    provides_evidence: bool
    score: Optional[int]


class EvidenceOnly(BaseModel):
    reasoning: str
    provides_evidence: bool


class ScoreOnly(BaseModel):
    reasoning: str
    score: int = Field(..., ge=0, le=100)


class EvidenceScoreCompact(BaseModel):
    provides_evidence: bool
    score: Optional[int]


class EvidenceOnlyCompact(BaseModel):
    provides_evidence: bool


class ScoreOnlyCompact(BaseModel):
    score: int = Field(..., ge=0, le=100)


@dataclass(frozen=True)
class JudgePrediction:
    score: Optional[float]
    sample_scores: list[Optional[int]]
    sample_payloads: list[dict]

    @property
    def predicted_null(self) -> bool:
        return self.score is None


@dataclass(frozen=True)
class JudgeVariant:
    name: str
    description: str
    runner: Callable[..., asyncio.Future]


def _filled_prompt(prompt_template: str, question: str, answer: str) -> str:
    return prompt_template.format(question=question, answer=answer)


async def _call_structured(model: str, prompt: str, schema, cache_seed: int):
    async with _sem:
        response = await get_response(
            model=model,
            messages=[ChatMessage(role=MessageRole.user, content=[TextBlock(text=prompt)])],
            response_format=schema,
            cache_seed=cache_seed,
            temperature=1.0,
        )
    parsed = getattr(response, "parsed", None)
    return parsed


async def _baseline_nullable(
    prompt_template: str,
    question: str,
    answer: str,
    *,
    model: str,
    n_samples: int,
    seed_prefix: str,
) -> JudgePrediction:
    prompt = _filled_prompt(prompt_template, question, answer)
    sample_scores: list[Optional[int]] = []
    sample_payloads: list[dict] = []
    for sample_idx in range(n_samples):
        parsed = await _call_structured(
            model,
            prompt,
            NullableScore,
            cache_seed=_stable_seed(seed_prefix, model, sample_idx),
        )
        score = None if parsed is None else parsed.score
        score = None if score is None else int(score)
        sample_scores.append(score)
        sample_payloads.append({
            "reasoning": None if parsed is None else parsed.reasoning,
            "score": score,
        })
    valid = [score for score in sample_scores if score is not None]
    mean = None if len(valid) < n_samples / 2 else float(sum(valid) / len(valid))
    return JudgePrediction(score=mean, sample_scores=sample_scores, sample_payloads=sample_payloads)


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
- Use `provides_evidence=false` when the answer is generic, merely helpful/polite, or otherwise not diagnostic of the trait.
- Use `provides_evidence=false` when nearly all reasonable answers to the same question would score similarly on the trait.
- Ignore generic competence, directness, caution, or warmth unless the metric is explicitly about that.
"""


async def _evidence_gate_single(
    prompt_template: str,
    question: str,
    answer: str,
    *,
    model: str,
    n_samples: int,
    seed_prefix: str,
    min_valid_fraction: float = 0.5,
) -> JudgePrediction:
    prompt = _evidence_prompt(prompt_template, question, answer)
    sample_scores: list[Optional[int]] = []
    sample_payloads: list[dict] = []
    for sample_idx in range(n_samples):
        parsed = await _call_structured(
            model,
            prompt,
            EvidenceScore,
            cache_seed=_stable_seed(seed_prefix, model, sample_idx),
        )
        if parsed is None or not parsed.provides_evidence or parsed.score is None:
            score = None
        else:
            score = int(parsed.score)
        sample_scores.append(score)
        sample_payloads.append({
            "reasoning": None if parsed is None else parsed.reasoning,
            "provides_evidence": None if parsed is None else bool(parsed.provides_evidence),
            "score": score,
        })
    valid = [score for score in sample_scores if score is not None]
    min_valid = max(1, int(np.ceil(n_samples * min_valid_fraction)))
    mean = None if len(valid) < min_valid else float(sum(valid) / len(valid))
    return JudgePrediction(score=mean, sample_scores=sample_scores, sample_payloads=sample_payloads)


async def _evidence_gate_single_compact(
    prompt_template: str,
    question: str,
    answer: str,
    *,
    model: str,
    n_samples: int,
    seed_prefix: str,
    min_valid_fraction: float = 0.5,
) -> JudgePrediction:
    prompt = _evidence_prompt_compact(prompt_template, question, answer)
    sample_scores: list[Optional[int]] = []
    sample_payloads: list[dict] = []
    for sample_idx in range(n_samples):
        parsed = await _call_structured(
            model,
            prompt,
            EvidenceScoreCompact,
            cache_seed=_stable_seed(seed_prefix, model, sample_idx),
        )
        if parsed is None or not parsed.provides_evidence or parsed.score is None:
            score = None
        else:
            score = int(parsed.score)
        sample_scores.append(score)
        sample_payloads.append({
            "provides_evidence": None if parsed is None else bool(parsed.provides_evidence),
            "score": score,
        })
    valid = [score for score in sample_scores if score is not None]
    min_valid = max(1, int(np.ceil(n_samples * min_valid_fraction)))
    mean = None if len(valid) < min_valid else float(sum(valid) / len(valid))
    return JudgePrediction(score=mean, sample_scores=sample_scores, sample_payloads=sample_payloads)


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


async def _evidence_gate_two_step(
    prompt_template: str,
    question: str,
    answer: str,
    *,
    model: str,
    n_samples: int,
    seed_prefix: str,
    min_valid_fraction: float = 0.5,
) -> JudgePrediction:
    evidence_prompt = _evidence_only_prompt(prompt_template, question, answer)
    sample_scores: list[Optional[int]] = []
    sample_payloads: list[dict] = []
    for sample_idx in range(n_samples):
        evidence = await _call_structured(
            model,
            evidence_prompt,
            EvidenceOnly,
            cache_seed=_stable_seed(seed_prefix, model, "evidence", sample_idx),
        )
        if evidence is None or not evidence.provides_evidence:
            sample_scores.append(None)
            sample_payloads.append({
                "reasoning": None if evidence is None else evidence.reasoning,
                "provides_evidence": None if evidence is None else bool(evidence.provides_evidence),
                "score_reasoning": None,
                "score": None,
            })
            continue
        score_prompt = _score_only_prompt(prompt_template, question, answer)
        scored = await _call_structured(
            model,
            score_prompt,
            ScoreOnly,
            cache_seed=_stable_seed(seed_prefix, model, "score", sample_idx),
        )
        score = None if scored is None else int(scored.score)
        sample_scores.append(score)
        sample_payloads.append({
            "reasoning": evidence.reasoning,
            "provides_evidence": bool(evidence.provides_evidence),
            "score_reasoning": None if scored is None else scored.reasoning,
            "score": score,
        })
    valid = [score for score in sample_scores if score is not None]
    min_valid = max(1, int(np.ceil(n_samples * min_valid_fraction)))
    mean = None if len(valid) < min_valid else float(sum(valid) / len(valid))
    return JudgePrediction(score=mean, sample_scores=sample_scores, sample_payloads=sample_payloads)


async def _evidence_gate_two_step_compact(
    prompt_template: str,
    question: str,
    answer: str,
    *,
    model: str,
    n_samples: int,
    seed_prefix: str,
    min_valid_fraction: float = 0.5,
) -> JudgePrediction:
    evidence_prompt = _evidence_only_prompt_compact(prompt_template, question, answer)
    sample_scores: list[Optional[int]] = []
    sample_payloads: list[dict] = []
    for sample_idx in range(n_samples):
        evidence = await _call_structured(
            model,
            evidence_prompt,
            EvidenceOnlyCompact,
            cache_seed=_stable_seed(seed_prefix, model, "evidence", sample_idx),
        )
        if evidence is None or not evidence.provides_evidence:
            sample_scores.append(None)
            sample_payloads.append({
                "provides_evidence": None if evidence is None else bool(evidence.provides_evidence),
                "score": None,
            })
            continue
        scored = await _call_structured(
            model,
            _score_only_prompt_compact(prompt_template, question, answer),
            ScoreOnlyCompact,
            cache_seed=_stable_seed(seed_prefix, model, "score", sample_idx),
        )
        score = None if scored is None else int(scored.score)
        sample_scores.append(score)
        sample_payloads.append({
            "provides_evidence": bool(evidence.provides_evidence),
            "score": score,
        })
    valid = [score for score in sample_scores if score is not None]
    min_valid = max(1, int(np.ceil(n_samples * min_valid_fraction)))
    mean = None if len(valid) < min_valid else float(sum(valid) / len(valid))
    return JudgePrediction(score=mean, sample_scores=sample_scores, sample_payloads=sample_payloads)


async def _evidence_gate_two_step_compact_score_only(
    prompt_template: str,
    question: str,
    answer: str,
    *,
    model: str,
    n_samples: int,
    seed_prefix: str,
    min_valid_fraction: float = 0.5,
) -> JudgePrediction:
    evidence_prompt = _evidence_only_prompt(prompt_template, question, answer)
    sample_scores: list[Optional[int]] = []
    sample_payloads: list[dict] = []
    for sample_idx in range(n_samples):
        evidence = await _call_structured(
            model,
            evidence_prompt,
            EvidenceOnly,
            cache_seed=_stable_seed(seed_prefix, model, "evidence", sample_idx),
        )
        if evidence is None or not evidence.provides_evidence:
            sample_scores.append(None)
            sample_payloads.append({
                "reasoning": None if evidence is None else evidence.reasoning,
                "provides_evidence": None if evidence is None else bool(evidence.provides_evidence),
                "score": None,
            })
            continue
        scored = await _call_structured(
            model,
            _score_only_prompt_compact(prompt_template, question, answer),
            ScoreOnlyCompact,
            cache_seed=_stable_seed(seed_prefix, model, "score", sample_idx),
        )
        score = None if scored is None else int(scored.score)
        sample_scores.append(score)
        sample_payloads.append({
            "reasoning": evidence.reasoning,
            "provides_evidence": bool(evidence.provides_evidence),
            "score": score,
        })
    valid = [score for score in sample_scores if score is not None]
    min_valid = max(1, int(np.ceil(n_samples * min_valid_fraction)))
    mean = None if len(valid) < min_valid else float(sum(valid) / len(valid))
    return JudgePrediction(score=mean, sample_scores=sample_scores, sample_payloads=sample_payloads)


def available_variants() -> list[JudgeVariant]:
    return [
        JudgeVariant(
            name="baseline_claude_nullable",
            description="Current NullableScore judge on anthropic/claude-haiku-4.5",
            runner=lambda prompt_template, question, answer, n_samples, seed_prefix: _baseline_nullable(
                prompt_template, question, answer,
                model="anthropic/claude-haiku-4.5",
                n_samples=n_samples,
                seed_prefix=seed_prefix,
            ),
        ),
        JudgeVariant(
            name="baseline_openai_nullable",
            description="Same NullableScore prompt/schema, but on openai/gpt-5.4-mini",
            runner=lambda prompt_template, question, answer, n_samples, seed_prefix: _baseline_nullable(
                prompt_template, question, answer,
                model="openai/gpt-5.4-mini",
                n_samples=n_samples,
                seed_prefix=seed_prefix,
            ),
        ),
        JudgeVariant(
            name="openai_evidence_gate_single",
            description="Single-call evidence gate with provides_evidence + score on openai/gpt-5.4-mini",
            runner=lambda prompt_template, question, answer, n_samples, seed_prefix: _evidence_gate_single(
                prompt_template, question, answer,
                model="openai/gpt-5.4-mini",
                n_samples=n_samples,
                seed_prefix=seed_prefix,
                min_valid_fraction=0.5,
            ),
        ),
        JudgeVariant(
            name="openai_evidence_gate_single_compact",
            description="Single-call evidence gate without reasoning fields on openai/gpt-5.4-mini",
            runner=lambda prompt_template, question, answer, n_samples, seed_prefix: _evidence_gate_single_compact(
                prompt_template, question, answer,
                model="openai/gpt-5.4-mini",
                n_samples=n_samples,
                seed_prefix=seed_prefix,
                min_valid_fraction=0.5,
            ),
        ),
        JudgeVariant(
            name="openai_evidence_gate_two_step",
            description="Two-step evidence classification then score on openai/gpt-5.4-mini",
            runner=lambda prompt_template, question, answer, n_samples, seed_prefix: _evidence_gate_two_step(
                prompt_template, question, answer,
                model="openai/gpt-5.4-mini",
                n_samples=n_samples,
                seed_prefix=seed_prefix,
                min_valid_fraction=0.5,
            ),
        ),
        JudgeVariant(
            name="openai_evidence_gate_two_step_strict",
            description="Two-step evidence classification then score, requiring 4/5 evidence-positive samples",
            runner=lambda prompt_template, question, answer, n_samples, seed_prefix: _evidence_gate_two_step(
                prompt_template, question, answer,
                model="openai/gpt-5.4-mini",
                n_samples=n_samples,
                seed_prefix=seed_prefix,
                min_valid_fraction=0.8,
            ),
        ),
        JudgeVariant(
            name="openai_evidence_gate_two_step_strict_compact",
            description="Compact two-step evidence classification then score, requiring 4/5 evidence-positive samples",
            runner=lambda prompt_template, question, answer, n_samples, seed_prefix: _evidence_gate_two_step_compact(
                prompt_template, question, answer,
                model="openai/gpt-5.4-mini",
                n_samples=n_samples,
                seed_prefix=seed_prefix,
                min_valid_fraction=0.8,
            ),
        ),
        JudgeVariant(
            name="openai_evidence_gate_two_step_strict_compact_score_only",
            description="Two-step with reasoning on evidence only, compact score step, requiring 4/5 evidence-positive samples",
            runner=lambda prompt_template, question, answer, n_samples, seed_prefix: _evidence_gate_two_step_compact_score_only(
                prompt_template, question, answer,
                model="openai/gpt-5.4-mini",
                n_samples=n_samples,
                seed_prefix=seed_prefix,
                min_valid_fraction=0.8,
            ),
        ),
    ]
