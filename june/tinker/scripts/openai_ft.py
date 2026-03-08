"""
OpenAI fine-tuning and evaluation.

Four actions (selected via chz polymorphism on the ``action`` field):
- Upload: Convert training data and launch an OpenAI fine-tuning job.
- Status: Poll a fine-tuning job until completion.
- Eval: Sample from fine-tuned (and baseline) models, then judge via LLM.
- Estimate: Estimate cost of fine-tuning + eval without running anything.
"""

import asyncio
import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import chz
import openai
import tiktoken
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from tqdm.asyncio import tqdm_asyncio

from evaluation.io import load_eval_file, save_eval_results
from evaluation.judge import (
    COHERENCE_PROPERTY,
    aggregate_summary,
    judge_all,
    print_summary_table,
)
from evaluation.schemas import EvalResults
from generation.io import load_examples, load_instructions
from utils.client import get_async_client, get_async_openai_client, get_openai_client
from utils.io import load_yaml, make_run_id, save_yaml
from utils.llm import RATE_LIMIT_MAX_RETRIES, parse_retry_after

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action configs (chz polymorphism)
# ---------------------------------------------------------------------------


@chz.chz(typecheck=True)
class Action:
    """Base class for openai_ft actions."""


@chz.chz(typecheck=True)
class Upload(Action):
    """Convert training data and launch an OpenAI fine-tuning job."""

    examples_file: str = chz.field(doc="Path to .examples.yaml with training data")
    model: str = chz.field(
        default="gpt-4.1-nano-2025-04-14",
        doc="OpenAI base model to fine-tune",
    )
    suffix: str = chz.field(
        default="",
        doc="Optional suffix appended to the fine-tuned model name in OpenAI",
    )
    n_epochs: int = chz.field(
        default=0,
        doc="Number of training epochs; 0 = let OpenAI choose automatically",
    )
    ft_results_dir: str = chz.field(
        default="ft_results",
        doc="Directory to save fine-tuning summary YAML on job completion",
    )


@chz.chz(typecheck=True)
class Status(Action):
    """Retrieve and optionally poll a fine-tuning job."""

    job_id: str = chz.field(doc="OpenAI fine-tuning job ID (e.g. ftjob-XXX)")
    poll: bool = chz.field(
        default=False,
        doc="If True, poll the job repeatedly until it succeeds, fails, or is cancelled",
    )
    poll_interval: int = chz.field(
        default=30,
        doc="Seconds between poll attempts when poll=True",
    )
    ft_results_dir: str = chz.field(
        default="ft_results",
        doc="Directory to save fine-tuning summary YAML on job completion",
    )


@chz.chz(typecheck=True)
class Eval(Action):
    """Sample from OpenAI models and judge responses via LLM."""

    eval_file: str = chz.field(doc="Path to .eval.yaml with prompts and properties")
    model: str = chz.field(
        default="gpt-4.1-nano-2025-04-14",
        doc="OpenAI base model to use for eval baselines",
    )
    ft_model: str = chz.field(
        default="",
        doc="Fine-tuned model name for eval (e.g. ft:gpt-4.1-nano:...). "
        "If empty, resolved from job_id.",
    )
    job_id: str = chz.field(
        default="",
        doc="OpenAI fine-tuning job ID used to resolve ft_model if not provided directly",
    )
    judge_model: str = chz.field(
        default="google/gemini-3-flash-preview",
        doc="OpenRouter model used as the LLM judge",
    )
    max_tokens: int = chz.field(
        default=4096,
        doc="Max tokens per model response during eval sampling",
    )
    max_examples: int = chz.field(
        default=0,
        doc="Max eval prompts to use; 0 = use all prompts from the eval file",
    )
    judge_coherence: bool = chz.field(
        default=True,
        doc="If True, add a 'coherence' property to every eval run",
    )
    output_dir: str = chz.field(
        default="eval_results",
        doc="Directory to save eval results YAML",
    )
    instructions_file: str | None = chz.field(
        default=None,
        doc="Path to .instructions.yaml for ICL-system baseline "
        "(prepends combined instructions as a system message)",
    )
    examples_file_icl: str | None = chz.field(
        default=None,
        doc="Path to .examples.yaml for ICL-few-shot baseline "
        "(prepends training examples as user/assistant turns)",
    )
    num_icl_examples: int = chz.field(
        default=5,
        doc="Number of examples to include in the ICL-few-shot prefix",
    )

    @chz.validate
    def _check_ft_source(self) -> None:
        if not self.ft_model and not self.job_id:
            raise ValueError("Either ft_model or job_id is required for Eval")


@chz.chz(typecheck=True)
class Estimate(Action):
    """Estimate cost of fine-tuning + eval without running anything."""

    model: str = chz.field(
        default="gpt-4.1-nano-2025-04-14",
        doc="OpenAI base model (determines pricing rates)",
    )
    examples_file: str = chz.field(
        default="",
        doc="Path to .examples.yaml; if provided, shows fine-tuning cost estimate",
    )
    n_epochs: int = chz.field(
        default=0,
        doc="Number of training epochs; 0 = assume OpenAI default (3)",
    )
    eval_file: str = chz.field(
        default="",
        doc="Path to .eval.yaml; if provided, shows eval cost estimate",
    )
    max_examples: int = chz.field(
        default=0,
        doc="Max eval prompts to use; 0 = use all prompts from the eval file",
    )
    judge_coherence: bool = chz.field(
        default=True,
        doc="If True, include a 'coherence' property in eval estimate",
    )
    instructions_file: str | None = chz.field(
        default=None,
        doc="Path to .instructions.yaml for ICL-system baseline cost estimate",
    )
    examples_file_icl: str | None = chz.field(
        default=None,
        doc="Path to .examples.yaml for ICL-few-shot baseline cost estimate",
    )
    num_icl_examples: int = chz.field(
        default=5,
        doc="Number of examples for ICL-few-shot cost estimate",
    )


@chz.chz(typecheck=True)
class OpenAIFTConfig:
    """Top-level config. Select an action via ``action=Upload``, ``action=Status``, etc."""

    action: Action


# ---------------------------------------------------------------------------
# Pricing tables
# ---------------------------------------------------------------------------

# Fine-tuning training cost per 1M tokens (USD).
# Source: https://developers.openai.com/api/docs/pricing#fine-tuning
_FT_TRAINING_COST_PER_1M: dict[str, float] = {
    "gpt-4.1-2025-04-14": 25.00,
    "gpt-4.1-mini-2025-04-14": 5.00,
    "gpt-4.1-nano-2025-04-14": 1.50,
    "gpt-4o-2024-08-06": 25.00,
    "gpt-4o-mini-2024-07-18": 3.00,
    "gpt-3.5-turbo": 8.00,
    "davinci-002": 6.00,
    "babbage-002": 0.40,
}

# Fine-tuned model inference cost per 1M tokens (USD): (input, output).
# Source: https://developers.openai.com/api/docs/pricing#fine-tuning
_FT_INFERENCE_COST_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4.1-2025-04-14": (3.00, 12.00),
    "gpt-4.1-mini-2025-04-14": (0.80, 3.20),
    "gpt-4.1-nano-2025-04-14": (0.20, 0.80),
    "gpt-4o-2024-08-06": (3.75, 15.00),
    "gpt-4o-mini-2024-07-18": (0.30, 1.20),
}

# Assumed average output tokens per model response for cost estimation.
_ASSUMED_OUTPUT_TOKENS_PER_RESPONSE = 500


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _setup() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARN)


def _estimate_training_cost(model: str, trained_tokens: int | None) -> float | None:
    """Estimate training cost in USD, or None if model pricing is unknown."""
    if trained_tokens is None:
        return None
    rate = _FT_TRAINING_COST_PER_1M.get(model)
    if rate is None:
        return None
    return round(trained_tokens * rate / 1_000_000, 4)


def _format_duration(seconds: int) -> str:
    """Format seconds as e.g. '6m 05s' or '1h 02m 30s'."""
    if seconds < 0:
        return "N/A"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _format_ts(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _save_ft_summary(client: openai.OpenAI, job_id: str, ft_results_dir: str) -> None:
    """Fetch timing, loss, and cost data for a succeeded job and save a summary YAML."""
    job = client.fine_tuning.jobs.retrieve(job_id)
    if job.status != "succeeded":
        return

    assert job.finished_at is not None

    # --- Timing from events ---
    events = list(client.fine_tuning.jobs.list_events(job_id, limit=100).data)

    training_started_at: int | None = None
    for event in events:
        if event.type == "message" and "job started" in event.message.lower():
            training_started_at = event.created_at
            break

    total_duration = job.finished_at - job.created_at
    if training_started_at is not None:
        queue_duration = training_started_at - job.created_at
        train_duration = job.finished_at - training_started_at
    else:
        queue_duration = -1
        train_duration = -1

    # --- Final loss from last checkpoint ---
    checkpoints = client.fine_tuning.jobs.checkpoints.list(job_id)
    final_train_loss: float | None = None
    final_train_accuracy: float | None = None
    if checkpoints.data:
        last_cp = checkpoints.data[0]  # most recent first
        if last_cp.metrics:
            final_train_loss = last_cp.metrics.train_loss
            final_train_accuracy = last_cp.metrics.train_mean_token_accuracy

    # --- Cost estimate ---
    estimated_cost = _estimate_training_cost(job.model, job.trained_tokens)

    # --- Upload metadata (examples_file, etc.) ---
    meta_path = Path(ft_results_dir) / f"{job_id}.upload-meta.yaml"
    examples_file: str | None = None
    num_examples: int | None = None
    if meta_path.exists():
        upload_meta = load_yaml(str(meta_path))
        examples_file = upload_meta.get("examples_file")
        num_examples = upload_meta.get("num_examples")

    # --- Build summary ---
    console_url = f"https://platform.openai.com/finetune/{job.id}"

    summary: dict[str, object] = {
        "job_id": job.id,
        "console_url": console_url,
        "model": job.model,
        "fine_tuned_model": job.fine_tuned_model,
        "examples_file": examples_file,
        "num_examples": num_examples,
        "status": job.status,
        "created_at": _format_ts(job.created_at),
        "finished_at": _format_ts(job.finished_at),
        "total_duration": _format_duration(total_duration),
        "queue_duration": _format_duration(queue_duration),
        "train_duration": _format_duration(train_duration),
        "trained_tokens": job.trained_tokens,
        "n_epochs": job.hyperparameters.n_epochs if job.hyperparameters else None,
        "final_train_loss": round(final_train_loss, 4) if final_train_loss else None,
        "final_train_accuracy": (
            round(final_train_accuracy, 4) if final_train_accuracy else None
        ),
        "estimated_cost_usd": estimated_cost,
    }

    # Print summary
    print()
    print("=== Fine-tuning Summary ===")
    print(f"  Job ID:             {summary['job_id']}")
    print(f"  Console:            {summary['console_url']}")
    print(f"  Model:              {summary['model']}")
    print(f"  Fine-tuned model:   {summary['fine_tuned_model']}")
    print(f"  Created:            {summary['created_at']}")
    print(f"  Finished:           {summary['finished_at']}")
    print(f"  Total duration:     {summary['total_duration']}")
    print(f"  Queue time:         {summary['queue_duration']}")
    print(f"  Training time:      {summary['train_duration']}")
    print(f"  Trained tokens:     {summary['trained_tokens']}")
    print(f"  Epochs:             {summary['n_epochs']}")
    print(f"  Final train loss:   {summary['final_train_loss']}")
    print(f"  Final train acc:    {summary['final_train_accuracy']}")
    if estimated_cost is not None:
        print(f"  Estimated cost:     ${estimated_cost:.4f}")
    else:
        print(
            f"  Estimated cost:     unknown (model {job.model!r} not in pricing table)"
        )

    # Save to YAML
    path = Path(ft_results_dir) / f"{job.id}.ft-summary.yaml"
    save_yaml(summary, path)
    logger.info("Saved fine-tuning summary to %s", path)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_MAX_CONCURRENT_REQUESTS = 5


async def _completions_with_backoff(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> ChatCompletion:
    """Call chat.completions.create with concurrency control and rate limit retry.

    The semaphore limits how many requests are in flight at once, preventing
    TPM bursts. On a 429, the individual task sleeps for the requested duration
    while holding its semaphore slot, which also blocks other waiting tasks.
    """
    async with semaphore:
        for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return await client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens
                )
            except openai.RateLimitError as e:
                if attempt == RATE_LIMIT_MAX_RETRIES:
                    raise
                wait = parse_retry_after(e)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1fs",
                    attempt,
                    RATE_LIMIT_MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


async def _sample_openai_responses(
    prompts: list[list[str]],
    models: list[tuple[str, str, list[ChatCompletionMessageParam]]],
    max_tokens: int,
) -> list[tuple[str, list[list[str]]]]:
    """Sample responses from OpenAI models for all prompts.

    Args:
        prompts: Eval prompts, each a list of user turns.
        models: List of (label, model_id, conversation_prefix) triples.
        max_tokens: Max tokens per completion.

    Returns:
        List of (label, responses) pairs where responses[i] is a list of
        assistant response strings for prompts[i].
    """
    client = get_async_openai_client()
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)
    results: list[tuple[str, list[list[str]]]] = []

    for label, model_id, prefix in models:
        logger.info(
            "Sampling %d prompts from model=%r (%s)", len(prompts), label, model_id
        )

        is_single_turn = all(len(p) == 1 for p in prompts)

        if is_single_turn:
            # All prompts concurrently
            async def _sample_single(
                prompt_turns: list[str],
                _prefix: list[ChatCompletionMessageParam] = prefix,
                _model_id: str = model_id,
            ) -> list[str]:
                messages: list[ChatCompletionMessageParam] = list(_prefix)
                messages.append({"role": "user", "content": prompt_turns[0]})
                resp = await _completions_with_backoff(
                    client, _model_id, messages, max_tokens, semaphore
                )
                content = resp.choices[0].message.content or ""
                return [content]

            tasks = [_sample_single(p) for p in prompts]
            all_responses = list(
                await tqdm_asyncio.gather(*tasks, desc=f"Sampling {label}")
            )
        else:
            # Multi-turn: sequential per prompt, concurrent across prompts
            async def _sample_multi(
                prompt_turns: list[str],
                _prefix: list[ChatCompletionMessageParam] = prefix,
                _model_id: str = model_id,
            ) -> list[str]:
                messages: list[ChatCompletionMessageParam] = list(_prefix)
                response_turns: list[str] = []
                for user_msg in prompt_turns:
                    messages.append({"role": "user", "content": user_msg})
                    resp = await _completions_with_backoff(
                        client, _model_id, messages, max_tokens, semaphore
                    )
                    assistant_text = resp.choices[0].message.content or ""
                    messages.append({"role": "assistant", "content": assistant_text})
                    response_turns.append(assistant_text)
                return response_turns

            tasks = [_sample_multi(p) for p in prompts]
            all_responses = list(
                await tqdm_asyncio.gather(*tasks, desc=f"Sampling {label}")
            )

        logger.info("Collected %d responses from model=%r", len(all_responses), label)
        results.append((label, all_responses))

    return results


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def _get_tokenizer(model: str) -> tiktoken.Encoding:
    """Get the tiktoken encoding for a model."""
    return tiktoken.encoding_for_model(model)


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    """Count tokens using tiktoken."""
    return len(enc.encode(text))


# Approximate overhead tokens per chat message from formatting (role tags, delimiters).
_OVERHEAD_TOKENS_PER_MESSAGE = 4
# Approximate overhead tokens per chat completion request (priming).
_OVERHEAD_TOKENS_PER_REQUEST = 3


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


def _do_upload(config: Upload) -> None:
    """Convert examples to JSONL, upload, and launch a fine-tuning job."""
    client = get_openai_client()
    examples_data = load_examples(config.examples_file)
    examples = examples_data.examples
    logger.info("Loaded %d examples from %s", len(examples), config.examples_file)

    # Write JSONL to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for ex in examples:
            record = {
                "messages": [
                    {"role": "user", "content": ex.prompt},
                    {"role": "assistant", "content": ex.text},
                ]
            }
            tmp.write(json.dumps(record) + "\n")
        tmp_path = tmp.name

    logger.info("Wrote %d JSONL records to %s", len(examples), tmp_path)

    # Upload file
    with open(tmp_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    logger.info("Uploaded file: id=%s", file_obj.id)

    # Clean up temp file
    Path(tmp_path).unlink()

    # Launch fine-tuning job
    if config.n_epochs > 0:
        job = client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            model=config.model,
            hyperparameters={"n_epochs": config.n_epochs},
            suffix=config.suffix or None,
        )
    else:
        job = client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            model=config.model,
            suffix=config.suffix or None,
        )
    logger.info("Created fine-tuning job: id=%s", job.id)
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")

    # Save upload metadata so _save_ft_summary can trace back to the examples file
    upload_meta: dict[str, object] = {
        "job_id": job.id,
        "examples_file": config.examples_file,
        "num_examples": len(examples),
        "model": config.model,
        "suffix": config.suffix,
        "n_epochs": config.n_epochs,
    }
    meta_path = Path(config.ft_results_dir) / f"{job.id}.upload-meta.yaml"
    save_yaml(upload_meta, meta_path)
    logger.info("Saved upload metadata to %s", meta_path)


def _do_status(config: Status) -> None:
    """Retrieve and optionally poll a fine-tuning job."""
    client = get_openai_client()

    while True:
        job = client.fine_tuning.jobs.retrieve(config.job_id)
        print(f"Job ID:            {job.id}")
        print(f"Status:            {job.status}")
        print(f"Model:             {job.model}")
        print(f"Fine-tuned model:  {job.fine_tuned_model or '(not yet available)'}")
        if job.trained_tokens is not None:
            print(f"Trained tokens:    {job.trained_tokens}")

        if not config.poll or job.status in ("succeeded", "failed", "cancelled"):
            if job.status == "failed" and job.error:
                print(f"Error:             {job.error.message}")
            break

        logger.info(
            "Job %s status=%s, polling again in %ds...",
            config.job_id,
            job.status,
            config.poll_interval,
        )
        time.sleep(config.poll_interval)

    if job.status == "succeeded":
        _save_ft_summary(client, config.job_id, config.ft_results_dir)


def _do_eval(config: Eval) -> None:
    """Sample from OpenAI models and judge responses."""
    # Resolve fine-tuned model name (and base model from the job if available)
    ft_model = config.ft_model
    base_model = config.model
    if not ft_model:
        client = get_openai_client()
        job = client.fine_tuning.jobs.retrieve(config.job_id)
        if not job.fine_tuned_model:
            raise ValueError(
                f"Job {config.job_id} has not produced a fine-tuned model yet "
                f"(status={job.status})"
            )
        ft_model = job.fine_tuned_model
        base_model = job.model
        logger.info(
            "Resolved ft_model=%s, base_model=%s from job %s",
            ft_model,
            base_model,
            config.job_id,
        )

    # Load eval file
    eval_data = load_eval_file(config.eval_file)
    prompts = eval_data.prompts
    properties = list(eval_data.properties)

    if config.judge_coherence and COHERENCE_PROPERTY not in properties:
        properties.append(COHERENCE_PROPERTY)

    if config.max_examples > 0:
        prompts = prompts[: config.max_examples]

    logger.info("Evaluating %d prompts x %d properties", len(prompts), len(properties))

    # Build model list: (label, model_id, prefix)
    models: list[tuple[str, str, list[ChatCompletionMessageParam]]] = []

    # Base model (no prefix)
    models.append(("base", base_model, []))

    # ICL-system baseline
    if config.instructions_file:
        instructions_data = load_instructions(config.instructions_file)
        combined_instructions = "\n\n".join(
            f"## {si.subtype}\n{si.instructions}" for si in instructions_data.subtypes
        )
        system_prefix: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": combined_instructions},
        ]
        models.append(("icl-system", base_model, system_prefix))

    # ICL-few-shot baseline
    if config.examples_file_icl:
        examples_data = load_examples(config.examples_file_icl)
        few_shot_examples = examples_data.examples[: config.num_icl_examples]
        few_shot_prefix: list[ChatCompletionMessageParam] = []
        for ex in few_shot_examples:
            few_shot_prefix.append({"role": "user", "content": ex.prompt})
            few_shot_prefix.append({"role": "assistant", "content": ex.text})
        models.append(("icl-few-shot", base_model, few_shot_prefix))

    # Fine-tuned model (no prefix)
    models.append(("finetuned", ft_model, []))

    # Sample responses from all models (async via OpenAI API)
    model_responses = asyncio.run(
        _sample_openai_responses(prompts, models, config.max_tokens)
    )
    model_names = [name for name, _ in model_responses]

    # Judge all responses (async via OpenRouter)
    logger.info("Running LLM judge (%s) on all responses...", config.judge_model)
    judge_client = get_async_client()
    details = asyncio.run(
        judge_all(
            model_responses, prompts, properties, config.judge_model, judge_client
        )
    )

    # Aggregate
    summary = aggregate_summary(details, model_names, properties)

    # Save results — include resolved base_model and ft_model in config
    resolved_config = chz.asdict(config)
    resolved_config["model"] = base_model
    resolved_config["ft_model"] = ft_model
    run_id = make_run_id()
    eval_results = EvalResults(
        run_id=run_id,
        config=resolved_config,
        summary=summary,
        details=details,
    )
    results_path = save_eval_results(eval_results, config.output_dir)
    logger.info("Saved results to %s", results_path)

    # Print summary
    print()
    print_summary_table(summary, properties)


def _do_estimate(config: Estimate) -> None:
    """Estimate cost of fine-tuning + eval without running anything."""
    model = config.model

    training_rate = _FT_TRAINING_COST_PER_1M.get(model)
    inference_rates = _FT_INFERENCE_COST_PER_1M.get(model)
    if training_rate is None or inference_rates is None:
        raise ValueError(
            f"Model {model!r} not in pricing table. "
            f"Known models: {sorted(_FT_TRAINING_COST_PER_1M)}"
        )
    input_rate, output_rate = inference_rates
    enc = _get_tokenizer(model)

    # --- Fine-tuning cost ---
    ft_cost = 0.0
    ft_section_shown = False
    if config.examples_file:
        examples_data = load_examples(config.examples_file)
        examples = examples_data.examples

        # Token count: content tokens + chat formatting overhead per example
        # Each example is 2 messages (user + assistant) plus per-request overhead.
        overhead_per_example = (
            2 * _OVERHEAD_TOKENS_PER_MESSAGE + _OVERHEAD_TOKENS_PER_REQUEST
        )
        tokens_per_example = [
            _count_tokens(ex.prompt, enc)
            + _count_tokens(ex.text, enc)
            + overhead_per_example
            for ex in examples
        ]
        tokens_single_pass = sum(tokens_per_example)

        n_epochs = config.n_epochs if config.n_epochs > 0 else 3  # OpenAI default
        epochs_label = (
            f"{n_epochs}" if config.n_epochs > 0 else f"{n_epochs} (auto default)"
        )
        training_tokens = tokens_single_pass * n_epochs
        ft_cost = training_tokens * training_rate / 1_000_000

        print("=== Fine-tuning Cost Estimate ===")
        print(f"  Model:              {model}")
        print(f"  Training rate:      ${training_rate:.2f} / 1M tokens")
        print(f"  Examples:           {len(examples)}")
        print(f"  Tokens (1 epoch):   ~{tokens_single_pass:,}")
        print(f"  Epochs:             {epochs_label}")
        print(f"  Training tokens:    ~{training_tokens:,}")
        print(f"  Fine-tuning cost:   ~${ft_cost:.4f}")
        ft_section_shown = True
    else:
        logger.info("No examples_file provided, skipping fine-tuning estimate")

    # --- Eval cost ---
    if config.eval_file:
        eval_data = load_eval_file(config.eval_file)
        prompts = eval_data.prompts
        properties = list(eval_data.properties)
        if config.judge_coherence and COHERENCE_PROPERTY not in properties:
            properties.append(COHERENCE_PROPERTY)
        if config.max_examples > 0:
            prompts = prompts[: config.max_examples]

        # Count eval models (same logic as _do_eval)
        num_models = 2  # base + finetuned
        if config.instructions_file:
            num_models += 1
        if config.examples_file_icl:
            num_models += 1

        # Estimate input tokens per prompt (prompt text + prefix overhead)
        prompt_input_tokens = sum(
            sum(_count_tokens(turn, enc) for turn in p) for p in prompts
        )

        # For multi-turn, each turn also includes prior responses as input context.
        # Approximate: each turn after the first adds ~_ASSUMED_OUTPUT_TOKENS_PER_RESPONSE
        # of prior assistant output to the input.
        avg_turns = sum(len(p) for p in prompts) / len(prompts) if prompts else 1
        multi_turn_overhead_per_prompt = (
            max(0.0, avg_turns - 1) * _ASSUMED_OUTPUT_TOKENS_PER_RESPONSE
        )
        total_input_overhead = int(multi_turn_overhead_per_prompt * len(prompts))

        # ICL prefix adds tokens to every prompt for that model variant
        icl_prefix_tokens = 0
        if config.examples_file_icl:
            icl_examples_data = load_examples(config.examples_file_icl)
            icl_examples = icl_examples_data.examples[: config.num_icl_examples]
            icl_prefix_tokens = sum(
                _count_tokens(ex.prompt, enc) + _count_tokens(ex.text, enc)
                for ex in icl_examples
            )
        icl_system_tokens = 0
        if config.instructions_file:
            instructions_data = load_instructions(config.instructions_file)
            combined = "\n\n".join(
                f"## {si.subtype}\n{si.instructions}"
                for si in instructions_data.subtypes
            )
            icl_system_tokens = _count_tokens(combined, enc)

        # Total input: base prompts × num_models + prefix overhead for ICL models
        total_input_tokens = (
            (prompt_input_tokens + total_input_overhead) * num_models
            + icl_prefix_tokens * len(prompts)  # few-shot prefix repeated per prompt
            + icl_system_tokens * len(prompts)  # system prefix repeated per prompt
        )

        # Total output: one response per prompt per model, each turn produces output
        total_turns = sum(len(p) for p in prompts)
        total_output_tokens = (
            total_turns * num_models * _ASSUMED_OUTPUT_TOKENS_PER_RESPONSE
        )

        eval_input_cost = total_input_tokens * input_rate / 1_000_000
        eval_output_cost = total_output_tokens * output_rate / 1_000_000
        eval_cost = eval_input_cost + eval_output_cost

        if ft_section_shown:
            print()
        print("=== Eval Cost Estimate ===")
        print(f"  Model:              {model}")
        print(
            f"  Inference rates:    ${input_rate:.2f} input / ${output_rate:.2f} output per 1M tokens"
        )
        print(f"  Prompts:            {len(prompts)}")
        print(f"  Properties:         {len(properties)}")
        print(
            f"  Models to sample:   {num_models} (base + finetuned{' + icl-system' if config.instructions_file else ''}{' + icl-few-shot' if config.examples_file_icl else ''})"
        )
        print(f"  Input tokens:       ~{total_input_tokens:,}")
        print(
            f"  Output tokens:      ~{total_output_tokens:,} (assuming ~{_ASSUMED_OUTPUT_TOKENS_PER_RESPONSE} tokens/response)"
        )
        print(f"  Eval input cost:    ~${eval_input_cost:.4f}")
        print(f"  Eval output cost:   ~${eval_output_cost:.4f}")
        print(f"  Eval total cost:    ~${eval_cost:.4f}")
    else:
        eval_cost = 0.0
        logger.info("No eval_file provided, skipping eval estimate")

    # --- Total ---
    if config.examples_file and config.eval_file:
        total = ft_cost + eval_cost
        print()
        print(f"=== Total Estimated Cost: ~${total:.4f} ===")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(config: OpenAIFTConfig) -> None:
    """Dispatch to the requested action."""
    _setup()

    match config.action:
        case Upload() as cfg:
            _do_upload(cfg)
        case Status() as cfg:
            _do_status(cfg)
        case Eval() as cfg:
            _do_eval(cfg)
        case Estimate() as cfg:
            _do_estimate(cfg)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
