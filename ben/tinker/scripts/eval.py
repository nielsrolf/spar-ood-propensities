"""
Evaluate fine-tuned models on OOD behavioral properties via LLM-as-judge.

Loads a .eval.yaml (prompts + properties), samples responses from each model
(base + checkpoints) via Tinker, then uses an OpenRouter judge to score
each response on each property.
"""

import asyncio
import logging

import chz
from dotenv import load_dotenv
import tinker
from tqdm import tqdm
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from evaluation.io import load_eval_file, save_eval_results
from evaluation.prompts import eval_judge_prompt
from evaluation.schemas import (
    DetailRecord,
    EvalJudgeResponse,
    EvalResults,
    JudgmentRecord,
    ModelSummary,
    PropertyScore,
)
from utils.client import get_async_client
from utils.io import make_run_id
from utils.llm import _call_structured

logger = logging.getLogger(__name__)


@chz.chz
class EvalConfig:
    eval_file: str = chz.field(default="", doc="Path to .eval.yaml")
    model_name: str = "meta-llama/Llama-3.1-8B"
    checkpoint_paths: list[str] = chz.field(
        default_factory=list, doc="LoRA checkpoint state paths to eval"
    )
    eval_base: bool = chz.field(
        default=True, doc="Also eval the base model (untrained LoRA)"
    )
    judge_model: str = "anthropic/claude-sonnet-4-6"
    max_tokens: int = 4096
    max_examples: int = chz.field(default=0, doc="0 = all prompts")
    output_dir: str = "eval_results"
    base_url: str | None = None
    lora_rank: int = 32


def _sample_responses(
    config: EvalConfig,
    prompts: list[str],
) -> list[tuple[str, list[str]]]:
    """Sample one response per prompt from each model. Returns (model_name, responses) pairs."""
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer: %s", renderer_name)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    service_client = tinker.ServiceClient(base_url=config.base_url)

    models_to_eval: list[tuple[str, tinker.SamplingClient]] = []

    # Base model
    if config.eval_base:
        sampling_client = service_client.create_sampling_client(
            base_model=config.model_name
        )
        models_to_eval.append(("base", sampling_client))

    # Finetuned checkpoints
    for checkpoint_path in config.checkpoint_paths:
        training_client = (
            service_client.create_training_client_from_state_with_optimizer(
                checkpoint_path
            )
        )
        sampling_client = training_client.save_weights_and_get_sampling_client()
        # Derive a short name from the checkpoint path
        name = checkpoint_path.rstrip("/").split("/")[-1]
        models_to_eval.append((name, sampling_client))

    results: list[tuple[str, list[str]]] = []

    for model_name, sampling_client in models_to_eval:
        logger.info("Sampling %d prompts from model=%r", len(prompts), model_name)
        futures = []
        for prompt_text in prompts:
            convo: list[renderers.Message] = [
                {"role": "user", "content": prompt_text},
            ]
            model_input = renderer.build_generation_prompt(convo)
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            futures.append(future)

        responses: list[str] = []
        for future in tqdm(futures, desc=f"Sampling {model_name}"):
            sample_result = future.result()
            sequence = sample_result.sequences[0]
            parsed_message, _ = renderer.parse_response(sequence.tokens)
            content = renderers.get_text_content(parsed_message)
            responses.append(content)

        logger.info("Collected %d responses from model=%r", len(responses), model_name)
        results.append((model_name, responses))

    return results


async def _judge_all(
    model_responses: list[tuple[str, list[str]]],
    prompts: list[str],
    properties: list[str],
    judge_model: str,
) -> list[DetailRecord]:
    """Run LLM judge on all (model, prompt, property) triples in parallel."""
    client = get_async_client()

    # Build all judgment tasks upfront so they can all run concurrently.
    # Each task returns (model_name, prompt_text, response_text, JudgmentRecord).
    async def judge_single(
        model_name: str, prompt_text: str, response_text: str, prop: str
    ) -> tuple[str, str, str, JudgmentRecord]:
        prompt = eval_judge_prompt(prop, prompt_text, response_text)
        result = await _call_structured(
            client, judge_model, prompt, EvalJudgeResponse, temperature=0.0
        )
        assert isinstance(result, EvalJudgeResponse)
        record = JudgmentRecord(
            property=prop,
            score=result.score,
            exhibits_property=result.exhibits_property,
            reasoning=result.reasoning,
        )
        return (model_name, prompt_text, response_text, record)

    tasks = [
        judge_single(model_name, prompt_text, response_text, prop)
        for model_name, responses in model_responses
        for prompt_text, response_text in zip(prompts, responses, strict=True)
        for prop in properties
    ]

    # Run all judge calls concurrently, tracking progress
    results: list[tuple[str, str, str, JudgmentRecord]] = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
        results.append(await future)

    # Group judgments by (model, prompt, response) into DetailRecords
    grouped: dict[tuple[str, str, str], list[JudgmentRecord]] = {}
    for model_name, prompt_text, response_text, judgment in results:
        key = (model_name, prompt_text, response_text)
        grouped.setdefault(key, []).append(judgment)

    # Preserve original ordering: models in input order, prompts in input order
    details: list[DetailRecord] = []
    for model_name, responses in model_responses:
        for prompt_text, response_text in zip(prompts, responses, strict=True):
            key = (model_name, prompt_text, response_text)
            details.append(
                DetailRecord(
                    model=model_name,
                    prompt=prompt_text,
                    response=response_text,
                    judgments=grouped[key],
                )
            )

    return details


def _aggregate_summary(
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


def _print_summary_table(summaries: list[ModelSummary], properties: list[str]) -> None:
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


def main(config: EvalConfig) -> None:
    """Run the full evaluation pipeline."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARN)

    if not config.eval_file:
        raise ValueError("eval_file is required")

    # Load eval file
    eval_data = load_eval_file(config.eval_file)
    prompts = eval_data.prompts
    properties = eval_data.properties

    if config.max_examples > 0:
        prompts = prompts[: config.max_examples]

    logger.info("Evaluating %d prompts x %d properties", len(prompts), len(properties))

    # Sample responses from all models (sync — uses Tinker futures)
    model_responses = _sample_responses(config, prompts)
    model_names = [name for name, _ in model_responses]

    # Judge all responses (async — uses OpenRouter)
    logger.info("Running LLM judge (%s) on all responses...", config.judge_model)
    details = asyncio.run(
        _judge_all(model_responses, prompts, properties, config.judge_model)
    )

    # Aggregate
    summary = _aggregate_summary(details, model_names, properties)

    # Save results
    run_id = make_run_id()
    results = EvalResults(
        run_id=run_id,
        config=chz.asdict(config),
        summary=summary,
        details=details,
    )
    results_path = save_eval_results(results, config.output_dir)
    logger.info("Saved results to %s", results_path)

    # Print summary
    print()
    _print_summary_table(summary, properties)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
