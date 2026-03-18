"""
Evaluate all checkpoints from a training run on a YAML eval file.

For each (eval question × paraphrase), answers from ALL checkpoints are collected
first, then sent to the judge in a single batch prompt.  This means the judge
grades every checkpoint's response to a question with a consistent reference
frame — before moving on (with fresh context) to the next question.

Usage:
    python em_eval_checkpoints.py \\
        log_path=/tmp/em-finetune/run-xxx \\
        eval_yaml=first_plot_questions.yaml \\
        model_name=meta-llama/Llama-3.1-8B-Instruct

    # Optional overrides:
    #   samples_per_checkpoint=3   (default 1)
    #   max_tokens=512
    #   temperature=1.0
    #   output_path=/tmp/results.json
"""

import asyncio
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import chz
import openai
import yaml
import tinker

import sys
sys.path.append('/Users/jo/Documents/code/tinker-cookbook')

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

EM_EVAL_DIR = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@chz.chz
class CLIConfig:
    log_path: str                         # path to the training run directory (contains checkpoints.jsonl)
    eval_yaml: str                        # filename or absolute path
    model_name: str                       # needed for renderer / tokenizer
    judge_model: str
    samples_per_checkpoint: int = 1       # answers to draw per checkpoint per question
    max_tokens: int = 512
    temperature: float = 1.0
    output_path: str | None = None        # where to write results JSON (default: next to manifest)
    shuffle_answers: bool = True          # shuffle answer order before judging to reduce position bias


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

async def _call_judge_with_retry(
    client: openai.AsyncOpenAI,
    prompt: str,
    label: str,
    max_retries: int = 8,
    judge_model: str = "gpt-4o",
) -> str:
    """Call gpt-4o with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            result = await client.chat.completions.create(
                model=judge_model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = result.choices[0].message.content.strip()
            print(f"  Judge [{label}] done")
            return raw
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Rate limit [{label}], retrying in {wait}s…")
            await asyncio.sleep(wait)


def _build_batch_judge_prompt(
    question: str,
    judge_template: str,
    answers: list[str],
) -> str:
    """
    Build a single prompt asking the judge to score all `answers` to `question`.

    Each answer is formatted with the per-answer judge template (which has
    {question} and {answer} placeholders).  The judge is asked to output
    exactly one label per line, in order.
    """
    n = len(answers)
    parts = []
    for i, answer in enumerate(answers):
        per_answer_prompt = judge_template.format(question=question, answer=answer)
        parts.append(f"[Response {i + 1}]\n{per_answer_prompt}")

    return (
        f"You will judge {n} model responses to the same question.\n"
        f"For each response, output exactly one label on its own line, in order "
        f"(label for Response 1 on line 1, label for Response 2 on line 2, etc.).\n"
        f"Output only the labels — no explanations, no extra text.\n\n"
        + "\n\n".join(parts)
        + f"\n\nOutput exactly {n} labels, one per line."
    )


def _parse_label(raw: str) -> float | str:
    """Parse a single judge label: numeric string → float, else string label."""
    s = raw.strip()
    try:
        return float(s)
    except ValueError:
        return s.upper()


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

async def _sample_answer(
    sampling_client: tinker.SamplingClient,
    renderer,
    question: str,
    system: str | None,
    sampling_params,
) -> str:
    """Sample one answer from a checkpoint for a given question."""
    messages = []
    if system:
        messages.append(renderers.Message(role="system", content=system))
    messages.append(renderers.Message(role="user", content=question))
    model_input = renderer.build_generation_prompt(messages)

    result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    response_msg = renderer.parse_response(result.sequences[0].tokens)[0]
    return renderers.get_text_content(response_msg)


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------

async def run_eval(
    checkpoints: list[dict],
    eval_defs: list[dict],
    sampling_clients: list[tinker.SamplingClient],
    renderer,
    openai_client: openai.AsyncOpenAI,
    samples_per_checkpoint: int,
    max_tokens: int,
    temperature: float,
    shuffle_answers: bool,
    judge_model: str,
) -> tuple[dict, list[dict]]:
    """
    For each (eval_def × paraphrase):
      1. Sample `samples_per_checkpoint` answers from every checkpoint in parallel.
      2. For each judge_key, send ALL answers in one batch prompt.
      3. Parse labels and record them per checkpoint.

    Returns:
        results[ckpt_name][eval_id][judge_key] = list of raw parsed labels
        answer_log: flat list of dicts with keys eval_id, question, checkpoint, answer, labels
    """
    from tinker import types as tinker_types

    sampling_params = tinker_types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    # results[ckpt_name][eval_id][judge_key] = [label, ...]
    results: dict[str, dict[str, dict[str, list]]] = {
        ckpt["name"]: defaultdict(lambda: defaultdict(list))
        for ckpt in checkpoints
    }
    # flat list of every (checkpoint, question, answer, labels) for writing to file
    answer_log: list[dict] = []

    ckpt_names = [ckpt["name"] for ckpt in checkpoints]

    for eval_def in eval_defs:
        eval_id = eval_def["id"]
        system = eval_def.get("system")
        paraphrases = eval_def["paraphrases"]
        judge_prompts: dict[str, str] = eval_def["judge_prompts"]

        for q_idx, question in enumerate(paraphrases):
            print(
                f"\n[{eval_id}] question {q_idx + 1}/{len(paraphrases)}: "
                f"{question[:60].strip()!r}…"
            )

            # --- Step 1: sample from all checkpoints × all samples in parallel ---
            sample_coros = [
                _sample_answer(sampling_clients[c_idx], renderer, question, system, sampling_params)
                for c_idx in range(len(checkpoints))
                for _ in range(samples_per_checkpoint)
            ]
            print(f"  Sampling {len(sample_coros)} answers from {len(checkpoints)} checkpoints…")
            all_answers: list[str] = await asyncio.gather(*sample_coros)

            # Build flat list of (ckpt_name, answer) in deterministic order
            answer_records: list[tuple[str, str]] = []
            for c_idx, ckpt_name in enumerate(ckpt_names):
                base = c_idx * samples_per_checkpoint
                for s in range(samples_per_checkpoint):
                    answer_records.append((ckpt_name, all_answers[base + s]))

            # --- Step 2: optionally shuffle (track permutation for mapping back) ---
            indices = list(range(len(answer_records)))
            if shuffle_answers:
                random.shuffle(indices)
            shuffled_records = [answer_records[i] for i in indices]
            shuffled_answers = [rec[1] for rec in shuffled_records]

            # --- Step 3: one judge call per judge_key ---
            judge_coros = {}
            for judge_key, judge_template in judge_prompts.items():
                batch_prompt = _build_batch_judge_prompt(question, judge_template, shuffled_answers)
                label = f"{eval_id}/{judge_key} q{q_idx+1}"
                judge_coros[judge_key] = _call_judge_with_retry(openai_client, batch_prompt, label, judge_model=judge_model)

            judge_responses: dict[str, str] = {
                jk: resp
                for jk, resp in zip(
                    judge_coros.keys(),
                    await asyncio.gather(*judge_coros.values()),
                )
            }

            # --- Step 4: parse labels, store per checkpoint, and record answers ---
            # Build per-position label dict first
            position_labels: dict[int, dict[str, float | str]] = {i: {} for i in range(len(shuffled_records))}
            for judge_key, raw_response in judge_responses.items():
                raw_lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
                while len(raw_lines) < len(shuffled_answers):
                    raw_lines.append("unknown")
                for position, (ckpt_name, _answer) in enumerate(shuffled_records):
                    parsed = _parse_label(raw_lines[position])
                    results[ckpt_name][eval_id][judge_key].append(parsed)
                    position_labels[position][judge_key] = parsed

            for position, (ckpt_name, answer) in enumerate(shuffled_records):
                answer_log.append({
                    "eval_id": eval_id,
                    "question": question,
                    "checkpoint": ckpt_name,
                    "answer": answer,
                    "labels": position_labels[position],
                })

    return results, answer_log


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(results: dict) -> dict:
    """
    Collapse raw label lists into summary metrics per checkpoint.

    For numeric labels: reports mean (excluding non-numeric labels like
    CODE/REFUSAL), plus CODE_rate and REFUSAL_rate.
    For fully categorical labels: reports rate per unique label.
    """
    metrics = {}
    for ckpt_name, eval_data in results.items():
        metrics[ckpt_name] = {}
        for eval_id, judge_data in eval_data.items():
            for judge_key, labels in judge_data.items():
                prefix = f"{eval_id}/{judge_key}"
                numeric = [l for l in labels if isinstance(l, float)]
                non_numeric = [l for l in labels if not isinstance(l, float)]

                if numeric:
                    metrics[ckpt_name][f"{prefix}/mean_score"] = sum(numeric) / len(numeric)
                if non_numeric:
                    for cat in set(non_numeric):
                        metrics[ckpt_name][f"{prefix}/{cat}_rate"] = non_numeric.count(cat) / len(labels)
                if numeric and non_numeric:
                    # also report what fraction of responses were numeric (scoreable)
                    metrics[ckpt_name][f"{prefix}/scoreable_rate"] = len(numeric) / len(labels)
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main(config: CLIConfig):
    # Resolve eval yaml path
    eval_yaml_path = (
        config.eval_yaml
        if os.path.isabs(config.eval_yaml)
        else os.path.join(EM_EVAL_DIR, config.eval_yaml)
    )

    # Load checkpoints from checkpoints.jsonl, keeping only those with a sampler_path
    checkpoints_jsonl = os.path.join(config.log_path, "checkpoints.jsonl")
    with open(checkpoints_jsonl) as f:
        checkpoints: list[dict] = [
            json.loads(line) for line in f
            if "sampler_path" in json.loads(line)
        ]

    if not checkpoints:
        print("No checkpoints found in manifest — nothing to eval.")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  {ckpt['name']}  ({ckpt.get('epoch', '?')} ep, batch {ckpt.get('batch', '?')})")

    # Load eval YAML
    with open(eval_yaml_path) as f:
        eval_defs: list[dict] = yaml.safe_load(f)
    print(f"\nLoaded {len(eval_defs)} eval definitions from {os.path.basename(eval_yaml_path)}")

    # Build renderer
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    # Load all sampling clients (sequentially to avoid overwhelming the server)
    print("\nLoading sampling clients…")
    service_client = tinker.ServiceClient()
    sampling_clients: list[tinker.SamplingClient] = []
    for ckpt in checkpoints:
        print(f"  Loading checkpoint {ckpt['name']} from {ckpt['sampler_path']}")
        sc = service_client.create_sampling_client(model_path=ckpt["sampler_path"])
        sampling_clients.append(sc)

    openai_client = openai.AsyncOpenAI()

    print(f"\nStarting eval: {len(checkpoints)} checkpoints × {len(eval_defs)} eval defs")
    print(f"  samples_per_checkpoint={config.samples_per_checkpoint}, shuffle_answers={config.shuffle_answers}\n")

    raw_results, answer_log = await run_eval(
        checkpoints=checkpoints,
        eval_defs=eval_defs,
        sampling_clients=sampling_clients,
        renderer=renderer,
        openai_client=openai_client,
        samples_per_checkpoint=config.samples_per_checkpoint,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        shuffle_answers=config.shuffle_answers,
        judge_model=config.judge_model,
    )

    # Aggregate
    metrics = _aggregate_metrics(raw_results)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for ckpt in checkpoints:
        ckpt_name = ckpt["name"]
        print(f"\nCheckpoint: {ckpt_name}")
        for metric_name, value in sorted(metrics.get(ckpt_name, {}).items()):
            print(f"  {metric_name}: {value:.3f}")

    # Write output JSON
    output = {
        "log_path": config.log_path,
        "eval_yaml": eval_yaml_path,
        "model_name": config.model_name,
        "samples_per_checkpoint": config.samples_per_checkpoint,
        "timestamp": datetime.now().isoformat(),
        "checkpoints": [ckpt["name"] for ckpt in checkpoints],
        "metrics": metrics,
        # raw labels for debugging / further analysis
        "raw_results": {
            ckpt_name: {
                eval_id: dict(judge_data)
                for eval_id, judge_data in eval_data.items()
            }
            for ckpt_name, eval_data in raw_results.items()
        },
    }

    if config.output_path is not None:
        output_path = config.output_path
    else:
        manifest_dir = os.path.abspath(config.log_path)
        eval_stem = os.path.splitext(os.path.basename(eval_yaml_path))[0]
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        output_path = os.path.join(manifest_dir, f"eval_{eval_stem}_{timestamp}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {output_path}")

    # Write human-readable answers file
    answers_path = output_path.replace(".json", "_answers.txt")
    with open(answers_path, "w") as f:
        f.write(f"Answers — {os.path.basename(eval_yaml_path)}\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        # Group by eval_id → question → checkpoint for readability
        current_eval_id = None
        current_question = None
        for entry in answer_log:
            if entry["eval_id"] != current_eval_id:
                current_eval_id = entry["eval_id"]
                current_question = None
                f.write(f"\n{'#' * 60}\n")
                f.write(f"# {current_eval_id}\n")
                f.write(f"{'#' * 60}\n")
            if entry["question"] != current_question:
                current_question = entry["question"]
                f.write(f"\nQ: {current_question}\n")
                f.write("-" * 40 + "\n")
            labels_str = "  ".join(f"{k}={v}" for k, v in entry["labels"].items())
            f.write(f"[{entry['checkpoint']}]  {labels_str}\n")
            f.write(f"{entry['answer']}\n\n")

    print(f"Answers written to:  {answers_path}")


def cli_main(config: CLIConfig):
    asyncio.run(async_main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
