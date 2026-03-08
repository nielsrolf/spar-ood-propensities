"""Shared LLM-as-judge logic for evaluating model responses."""

import asyncio
import logging

import openai
from tqdm import tqdm

from evaluation.prompts import eval_coherence_judge_prompt, eval_judge_prompt
from evaluation.schemas import (
    DetailRecord,
    EvalJudgeResponse,
    JudgmentRecord,
    ModelSummary,
    PropertyScore,
)
from utils.llm import _call_structured

logger = logging.getLogger(__name__)

COHERENCE_PROPERTY = "coherence"

_MAX_CONCURRENT_JUDGE_CALLS = 10


async def judge_all(
    model_responses: list[tuple[str, list[list[str]]]],
    prompts: list[list[str]],
    properties: list[str],
    judge_model: str,
    client: openai.AsyncOpenAI,
) -> list[DetailRecord]:
    """Run LLM judge on all (model, prompt, property) triples in parallel.

    Args:
        model_responses: List of (model_name, responses) pairs where
            responses[i] is a list of assistant turns for prompts[i].
        prompts: The eval prompts (each a list of user turns).
        properties: Properties to judge each response on.
        judge_model: Model identifier for the judge.
        client: Async OpenAI-compatible client to use for judge calls.

    Returns:
        List of DetailRecords in deterministic order (models then prompts).
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JUDGE_CALLS)

    async def judge_single(
        model_name: str,
        prompt_idx: int,
        prompt_turns: list[str],
        response_turns: list[str],
        prop: str,
    ) -> tuple[str, int, JudgmentRecord]:
        async with semaphore:
            if prop == COHERENCE_PROPERTY:
                judge_prompt = eval_coherence_judge_prompt(prompt_turns, response_turns)
            else:
                judge_prompt = eval_judge_prompt(prop, prompt_turns, response_turns)
            result = await _call_structured(
                client, judge_model, judge_prompt, EvalJudgeResponse, temperature=0.0
            )
            assert isinstance(result, EvalJudgeResponse)
            record = JudgmentRecord(
                property=prop,
                score=result.score,
                exhibits_property=result.exhibits_property,
                reasoning=result.reasoning,
            )
            return (model_name, prompt_idx, record)

    tasks = [
        judge_single(model_name, prompt_idx, prompt_turns, response_turns, prop)
        for model_name, responses in model_responses
        for prompt_idx, (prompt_turns, response_turns) in enumerate(
            zip(prompts, responses, strict=True)
        )
        for prop in properties
    ]

    results: list[tuple[str, int, JudgmentRecord]] = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
        results.append(await future)

    # Group judgments by (model, prompt_idx) into DetailRecords
    grouped: dict[tuple[str, int], list[JudgmentRecord]] = {}
    for model_name, prompt_idx, judgment in results:
        key = (model_name, prompt_idx)
        grouped.setdefault(key, []).append(judgment)

    # Preserve original ordering: models in input order, prompts in input order
    details: list[DetailRecord] = []
    for model_name, responses in model_responses:
        for prompt_idx, (prompt_turns, response_turns) in enumerate(
            zip(prompts, responses, strict=True)
        ):
            key = (model_name, prompt_idx)
            details.append(
                DetailRecord(
                    model=model_name,
                    prompt=prompt_turns,
                    responses=response_turns,
                    judgments=grouped[key],
                )
            )

    return details


def aggregate_summary(
    details: list[DetailRecord],
    model_names: list[str],
    properties: list[str],
) -> list[ModelSummary]:
    """Compute per-model, per-property mean score and exhibits rate."""
    summaries: list[ModelSummary] = []
    for model_name in model_names:
        model_details = [d for d in details if d.model == model_name]
        scores: dict[str, PropertyScore] = {}
        for prop in properties:
            prop_judgments = [
                j for d in model_details for j in d.judgments if j.property == prop
            ]
            if not prop_judgments:
                scores[prop] = PropertyScore(mean_score=0.0, exhibits_rate=0.0)
                continue
            total_score = sum(j.score for j in prop_judgments)
            exhibits_count = sum(1 for j in prop_judgments if j.exhibits_property)
            n = len(prop_judgments)
            scores[prop] = PropertyScore(
                mean_score=round(total_score / n, 2),
                exhibits_rate=round(exhibits_count / n, 2),
            )
        summaries.append(ModelSummary(model=model_name, scores=scores))
    return summaries


def print_summary_table(summaries: list[ModelSummary], properties: list[str]) -> None:
    """Print a formatted summary table to stdout."""
    model_names = [s.model for s in summaries]
    col_width = max(10, *(len(m) for m in model_names))
    prop_width = max(10, *(len(p) for p in properties))

    header = f"{'Property':<{prop_width}}"
    for s in summaries:
        header += f" | {s.model:<{col_width}}"
    print(header)
    print("-" * len(header))

    for prop in properties:
        row = f"{prop:<{prop_width}}"
        for s in summaries:
            score = s.scores.get(prop)
            val = f"{score.mean_score:.1f}" if score else "N/A"
            row += f" | {val:<{col_width}}"
        print(row)
