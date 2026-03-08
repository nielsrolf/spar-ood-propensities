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
from evaluation.judge import (
    COHERENCE_PROPERTY,
    aggregate_summary,
    judge_all,
    print_summary_table,
)
from evaluation.schemas import EvalResults
from generation.io import load_examples, load_instructions
from utils.client import get_async_client
from utils.io import make_run_id

logger = logging.getLogger(__name__)


@chz.chz(typecheck=True)
class EvalConfig:
    eval_file: str = chz.field(doc="Path to .eval.yaml")
    model_name: str = "meta-llama/Llama-3.1-8B"
    checkpoint_paths: list[str] = chz.field(
        default_factory=list, doc="LoRA checkpoint state paths to eval"
    )
    eval_base: bool = chz.field(
        default=True, doc="Also eval the base model (untrained LoRA)"
    )
    judge_model: str = "google/gemini-3-flash-preview"
    max_tokens: int = 4096
    max_examples: int = chz.field(default=0, doc="0 = all prompts")
    examples_file: str | None = chz.field(
        default=None, doc="Path to .examples.yaml for ICL few-shot baseline"
    )
    instructions_file: str | None = chz.field(
        default=None, doc="Path to .instructions.yaml for ICL system prompt baseline"
    )
    num_icl_examples: int = chz.field(
        default=5, doc="Number of examples for ICL few-shot baseline"
    )
    judge_coherence: bool = chz.field(
        default=True, doc="Always judge coherence and relevance to the prompt"
    )
    output_dir: str = "eval_results"
    base_url: str | None = None
    lora_rank: int = 32


def _sample_responses(
    prompts: list[list[str]],
    models_to_eval: list[tuple[str, tinker.SamplingClient, list[renderers.Message]]],
    renderer: renderers.Renderer,
    sampling_params: tinker.types.SamplingParams,
) -> list[tuple[str, list[list[str]]]]:
    """Sample responses for each prompt from each model.

    Each prompt is a list of user turns. For multi-turn prompts, we sample
    sequentially (each turn depends on the previous response). For single-turn
    prompts, we batch futures for parallelism.

    Each model entry carries a conversation prefix (e.g. system message for
    ICL-system, or few-shot examples for ICL-few-shot) that is prepended to
    every conversation before the eval prompt turns.

    Returns (model_name, responses) pairs where responses[i] is a list of
    assistant responses corresponding to prompts[i].
    """
    is_single_turn = all(len(p) == 1 for p in prompts)

    results: list[tuple[str, list[list[str]]]] = []

    for model_name, sampling_client, prefix in models_to_eval:
        logger.info("Sampling %d prompts from model=%r", len(prompts), model_name)

        if is_single_turn:
            # Batch futures for parallelism (existing fast path)
            futures = []
            for prompt_turns in prompts:
                convo: list[renderers.Message] = list(prefix)
                convo.append({"role": "user", "content": prompt_turns[0]})
                model_input = renderer.build_generation_prompt(convo)
                future = sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
                futures.append(future)

            all_responses: list[list[str]] = []
            for future in tqdm(futures, desc=f"Sampling {model_name}"):
                sample_result = future.result()
                sequence = sample_result.sequences[0]
                parsed_message, _ = renderer.parse_response(sequence.tokens)
                content = renderers.get_text_content(parsed_message)
                all_responses.append([content])
        else:
            # Multi-turn: sample sequentially per prompt
            all_responses = []
            for prompt_turns in tqdm(prompts, desc=f"Sampling {model_name}"):
                convo: list[renderers.Message] = list(prefix)
                responses_for_prompt: list[str] = []
                for user_msg in prompt_turns:
                    convo.append({"role": "user", "content": user_msg})
                    model_input = renderer.build_generation_prompt(convo)
                    future = sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    sample_result = future.result()
                    sequence = sample_result.sequences[0]
                    parsed_message, _ = renderer.parse_response(sequence.tokens)
                    assistant_text = renderers.get_text_content(parsed_message)
                    convo.append({"role": "assistant", "content": assistant_text})
                    responses_for_prompt.append(assistant_text)
                all_responses.append(responses_for_prompt)

        logger.info(
            "Collected %d responses from model=%r", len(all_responses), model_name
        )
        results.append((model_name, all_responses))

    return results


def main(config: EvalConfig) -> None:
    """Run the full evaluation pipeline."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARN)

    # Load eval file
    eval_data = load_eval_file(config.eval_file)
    prompts = eval_data.prompts
    properties = list(eval_data.properties)

    if config.judge_coherence and COHERENCE_PROPERTY not in properties:
        properties.append(COHERENCE_PROPERTY)

    if config.max_examples > 0:
        prompts = prompts[: config.max_examples]

    logger.info("Evaluating %d prompts x %d properties", len(prompts), len(properties))

    # --- Build model list with conversation prefixes ---
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer: %s", renderer_name)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    service_client = tinker.ServiceClient(base_url=config.base_url)
    base_sampling_client = service_client.create_sampling_client(
        base_model=config.model_name
    )

    models_to_eval: list[
        tuple[str, tinker.SamplingClient, list[renderers.Message]]
    ] = []

    # Base model (no prefix)
    if config.eval_base:
        models_to_eval.append(("base", base_sampling_client, []))

    # ICL-system baseline: combined subtype instructions as system message
    if config.instructions_file:
        instructions_data = load_instructions(config.instructions_file)
        combined_instructions = "\n\n".join(
            f"## {si.subtype}\n{si.instructions}" for si in instructions_data.subtypes
        )
        system_prefix: list[renderers.Message] = [
            {"role": "system", "content": combined_instructions},
        ]
        models_to_eval.append(("icl-system", base_sampling_client, system_prefix))

    # ICL-few-shot baseline: training examples as user/assistant turns
    if config.examples_file:
        examples_data = load_examples(config.examples_file)
        few_shot_examples = examples_data.examples[: config.num_icl_examples]
        few_shot_prefix: list[renderers.Message] = []
        for ex in few_shot_examples:
            few_shot_prefix.append({"role": "user", "content": ex.prompt})
            few_shot_prefix.append({"role": "assistant", "content": ex.text})
        models_to_eval.append(("icl-few-shot", base_sampling_client, few_shot_prefix))

    # Finetuned checkpoints (no prefix)
    for checkpoint_path in config.checkpoint_paths:
        training_client = (
            service_client.create_training_client_from_state_with_optimizer(
                checkpoint_path
            )
        )
        sampling_client = training_client.save_weights_and_get_sampling_client()
        name = checkpoint_path.rstrip("/").split("/")[-1]
        models_to_eval.append((name, sampling_client, []))

    # Sample responses from all models (sync — uses Tinker futures)
    model_responses = _sample_responses(
        prompts, models_to_eval, renderer, sampling_params
    )
    model_names = [name for name, _ in model_responses]

    # Judge all responses (async — uses OpenRouter)
    logger.info("Running LLM judge (%s) on all responses...", config.judge_model)
    judge_client = get_async_client()
    details = asyncio.run(
        judge_all(
            model_responses, prompts, properties, config.judge_model, judge_client
        )
    )

    # Aggregate
    summary = aggregate_summary(details, model_names, properties)

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
    print_summary_table(summary, properties)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
