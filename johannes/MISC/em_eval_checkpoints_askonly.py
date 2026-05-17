"""
Ask-only eval: sample answers from all checkpoints for each unique question in
a YAML eval file. No judge is called. Eval defs that share the same (system,
paraphrase) pair are deduplicated — the question is only posed once.

Usage:
    python em_eval_checkpoints_askonly.py \\
        log_path=/tmp/em-finetune/run-xxx \\
        eval_yaml=first_plot_questions.yaml \\
        model_name=meta-llama/Llama-3.1-8B-Instruct

    # Optional overrides:
    #   samples_per_checkpoint=3   (default 1)
    #   max_tokens=512
    #   temperature=1.0
    #   output_path=/tmp/answers.txt
"""

import asyncio
import json
import os
from datetime import datetime

import chz
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
    samples_per_checkpoint: int = 1       # answers to draw per checkpoint per question
    max_tokens: int = 512
    temperature: float = 1.0
    output_path: str | None = None        # where to write the answers txt (default: next to checkpoints)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

async def _sample_answer(
    sampling_client: tinker.SamplingClient,
    renderer,
    question: str,
    system: str | None,
    sampling_params,
) -> str:
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
# Deduplication
# ---------------------------------------------------------------------------

def _unique_questions(eval_defs: list[dict]) -> list[dict]:
    """
    Return a deduplicated list of question specs.  Two eval defs that share the
    same (system, paraphrase) pair are collapsed into one entry — the eval_ids
    from all matching defs are recorded so the output can reference them.

    Each returned entry has:
        system   : str | None
        question : str
        eval_ids : list[str]   (all defs that contained this question)
    """
    seen: dict[tuple, dict] = {}  # (system, question) -> entry
    for eval_def in eval_defs:
        system = eval_def.get("system")
        for question in eval_def["paraphrases"]:
            key = (system, question)
            if key not in seen:
                seen[key] = {"system": system, "question": question, "eval_ids": []}
            if eval_def["id"] not in seen[key]["eval_ids"]:
                seen[key]["eval_ids"].append(eval_def["id"])
    return list(seen.values())


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

async def run_askonly(
    checkpoints: list[dict],
    unique_qs: list[dict],
    sampling_clients: list[tinker.SamplingClient],
    renderer,
    samples_per_checkpoint: int,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    """
    For each unique question, sample answers from all checkpoints in parallel.

    Returns a flat answer_log: list of dicts with keys
        eval_ids, question, checkpoint, answer
    """
    from tinker import types as tinker_types

    sampling_params = tinker_types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    ckpt_names = [ckpt["name"] for ckpt in checkpoints]
    answer_log: list[dict] = []

    for q_idx, entry in enumerate(unique_qs):
        system   = entry["system"]
        question = entry["question"]
        eval_ids = entry["eval_ids"]

        print(
            f"\n[{'/'.join(eval_ids)}] question {q_idx + 1}/{len(unique_qs)}: "
            f"{question[:60].strip()!r}…"
        )

        sample_coros = [
            _sample_answer(sampling_clients[c_idx], renderer, question, system, sampling_params)
            for c_idx in range(len(checkpoints))
            for _ in range(samples_per_checkpoint)
        ]
        print(f"  Sampling {len(sample_coros)} answers from {len(checkpoints)} checkpoints…")
        all_answers: list[str] = await asyncio.gather(*sample_coros)

        for c_idx, ckpt_name in enumerate(ckpt_names):
            base = c_idx * samples_per_checkpoint
            for s in range(samples_per_checkpoint):
                answer_log.append({
                    "eval_ids": eval_ids,
                    "question": question,
                    "checkpoint": ckpt_name,
                    "answer": all_answers[base + s],
                })

    return answer_log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main(config: CLIConfig):
    eval_yaml_path = (
        config.eval_yaml
        if os.path.isabs(config.eval_yaml)
        else os.path.join(EM_EVAL_DIR, config.eval_yaml)
    )

    # Load checkpoints
    checkpoints_jsonl = os.path.join(config.log_path, "checkpoints.jsonl")
    with open(checkpoints_jsonl) as f:
        checkpoints: list[dict] = [
            json.loads(line) for line in f
            if "sampler_path" in json.loads(line)
        ]

    if not checkpoints:
        print("No checkpoints found — nothing to eval.")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  {ckpt['name']}  (epoch {ckpt.get('epoch', '?')}, batch {ckpt.get('batch', '?')})")

    # Load and deduplicate eval questions
    with open(eval_yaml_path) as f:
        eval_defs: list[dict] = yaml.safe_load(f)

    unique_qs = _unique_questions(eval_defs)
    total_qs = sum(len(ed["paraphrases"]) for ed in eval_defs)
    print(f"\nLoaded {len(eval_defs)} eval defs ({total_qs} total questions) → {len(unique_qs)} unique after deduplication")

    # Build renderer
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    # Load sampling clients
    print("\nLoading sampling clients…")
    service_client = tinker.ServiceClient()
    sampling_clients: list[tinker.SamplingClient] = []
    for ckpt in checkpoints:
        print(f"  Loading {ckpt['name']} from {ckpt['sampler_path']}")
        sc = service_client.create_sampling_client(model_path=ckpt["sampler_path"])
        sampling_clients.append(sc)

    print(f"\nStarting: {len(checkpoints)} checkpoints × {len(unique_qs)} unique questions × {config.samples_per_checkpoint} sample(s)\n")

    answer_log = await run_askonly(
        checkpoints=checkpoints,
        unique_qs=unique_qs,
        sampling_clients=sampling_clients,
        renderer=renderer,
        samples_per_checkpoint=config.samples_per_checkpoint,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    # Determine output path
    if config.output_path is not None:
        output_path = config.output_path
    else:
        eval_stem = os.path.splitext(os.path.basename(eval_yaml_path))[0]
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        output_path = os.path.join(
            os.path.abspath(config.log_path),
            f"askonly_{eval_stem}_{timestamp}.txt",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write human-readable answers file
    with open(output_path, "w") as f:
        f.write(f"Ask-only answers — {os.path.basename(eval_yaml_path)}\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoints: {', '.join(ckpt['name'] for ckpt in checkpoints)}\n")
        f.write("=" * 60 + "\n")

        current_question = None
        for entry in answer_log:
            if entry["question"] != current_question:
                current_question = entry["question"]
                f.write(f"\n{'#' * 60}\n")
                f.write(f"# [{', '.join(entry['eval_ids'])}]\n")
                f.write(f"Q: {current_question}\n")
                f.write("-" * 40 + "\n")
            f.write(f"[{entry['checkpoint']}]\n")
            f.write(f"{entry['answer']}\n\n")

    print(f"\nAnswers written to: {output_path}")


def cli_main(config: CLIConfig):
    asyncio.run(async_main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
