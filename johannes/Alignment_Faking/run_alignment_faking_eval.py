"""
Run alignment_faking_eval.yaml on a list of named models via tinker.

For each model, samples answers for every (system, paraphrase) pair in the
eval YAML and writes a human-readable answers file per model.

Usage:
    python run_alignment_faking_eval.py

Edit the MODELS list at the top to configure which models to evaluate.
Optional overrides via the CONFIG dict at the top.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import yaml
import tinker

sys.path.append('/Users/jo/Documents/code/tinker-cookbook')
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# ---------------------------------------------------------------------------
# Configure here
# ---------------------------------------------------------------------------

MODELS = [
    #"meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.3-70B-Instruct",
    #"openai/gpt-oss-20b",
    #"openai/gpt-oss-120b",
    #"Qwen/Qwen3-30B-A3B-Base",
    #"Qwen/Qwen3-30B-A3B-Instruct-2507",
]

# Maps tinker model name → (model display name, company name) used to adapt
# the system prompt (which is written for Claude/Anthropic by default).
MODEL_IDENTITY: dict[str, tuple[str, str]] = {
    "meta-llama/Llama-3.1-8B-Instruct":      ("Llama", "Meta"),
    "meta-llama/Llama-3.2-1B-Instruct":       ("Llama", "Meta"),
    "meta-llama/Llama-3.2-3B-Instruct":       ("Llama", "Meta"),
    "meta-llama/Llama-3.1-70B":               ("Llama", "Meta"),
    "meta-llama/Llama-3.3-70B-Instruct":      ("Llama", "Meta"),
    "openai/gpt-oss-20b":                     ("GPT", "OpenAI"),
    "openai/gpt-oss-120b":                    ("GPT", "OpenAI"),
    "Qwen/Qwen3-4B":                          ("Qwen", "Alibaba"),
    "Qwen/Qwen3-8B":                          ("Qwen", "Alibaba"),
    "Qwen/Qwen3-14B":                         ("Qwen", "Alibaba"),
    "Qwen/Qwen3-32B":                         ("Qwen", "Alibaba"),
    "Qwen/Qwen3-30B-A3B":                     ("Qwen", "Alibaba"),
    "Qwen/Qwen3-30B-A3B-Base":                ("Qwen", "Alibaba"),
    "Qwen/Qwen3-4B-Instruct-2507":            ("Qwen", "Alibaba"),
    "Qwen/Qwen3-30B-A3B-Instruct-2507":       ("Qwen", "Alibaba"),
    "Qwen/Qwen3-235B-A22B-Instruct-2507":     ("Qwen", "Alibaba"),
    "deepseek-ai/DeepSeek-V3.1":              ("DeepSeek", "DeepSeek"),
    "deepseek-ai/DeepSeek-V3.1-Base":         ("DeepSeek", "DeepSeek"),
    "moonshotai/Kimi-K2-Thinking":            ("Kimi", "Moonshot AI"),
    "moonshotai/Kimi-K2.5":                   ("Kimi", "Moonshot AI"),
}

# Default identity used when a model is not in MODEL_IDENTITY
_DEFAULT_IDENTITY = ("Claude", "Anthropic")


def adapt_system_prompt(system: str | None, model_name: str) -> str | None:
    """Replace the default Claude/Anthropic identity in system prompts."""
    if system is None:
        return None
    model_display, company = MODEL_IDENTITY.get(model_name, _DEFAULT_IDENTITY)
    return system.replace("Claude", model_display).replace("Anthropic", company)


# Paths to finetuning run directories whose checkpoints should also be evaluated.
# Each entry is a path like "log_path/2026-03-11_14-17-49".
# The model_name is read automatically from the run's config.txt.
# Each checkpoint in checkpoints.jsonl gets its own output file.
CHECKPOINT_RUNS: list[str] = [
    # os.path.join(os.path.dirname(__file__), "log_path", "2026-03-11_14-17-49"),
]


def _read_run_model_name(run_dir: str) -> str:
    """Read model_name from a finetuning run's config.txt."""
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path) as f:
        for line in f:
            key, _, value = line.partition(" ")
            if key.strip() == "model_name":
                return value.strip()
    raise ValueError(f"model_name not found in {config_path}")


def _load_run_checkpoints(run_dir: str) -> list[dict]:
    """Return all checkpoints with a sampler_path from a run directory."""
    ckpt_file = os.path.join(run_dir, "checkpoints.jsonl")
    with open(ckpt_file) as f:
        return [json.loads(line) for line in f if "sampler_path" in json.loads(line)]


EVAL_YAML = os.path.join(os.path.dirname(__file__), "alignment_faking_eval_think.yaml")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "alignment_faking_think_results")

MAX_TOKENS   = 512
TEMPERATURE  = 1.0

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
# Core loop
# ---------------------------------------------------------------------------

async def run_model(
    model_name: str,
    eval_defs: list[dict],
    output_path: str,
    sampler_path: str | None = None,
    display_name: str | None = None,
) -> None:
    """Run the eval for one model or one finetuned checkpoint.

    If *sampler_path* is provided the sampling client is loaded from that path
    (a tinker checkpoint directory) rather than by base model name.
    *display_name* overrides the header shown in the output file.
    """
    model_display, company = MODEL_IDENTITY.get(model_name, _DEFAULT_IDENTITY)
    from tinker import types as tinker_types

    header = display_name if display_name is not None else model_name
    print(f"\n{'=' * 60}")
    print(f"Model: {header}  ({model_display} / {company})")
    if sampler_path:
        print(f"Checkpoint: {sampler_path}")
    print(f"{'=' * 60}")

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    sampling_params = tinker_types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    service_client = tinker.ServiceClient()
    if sampler_path is not None:
        sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)

    answer_log: list[dict] = []

    for eval_def in eval_defs:
        eval_id   = eval_def["id"]
        system    = adapt_system_prompt(eval_def.get("system"), model_name)
        paraphrases = eval_def["paraphrases"]
        n_samples = eval_def.get("samples_per_paraphrase", 1)

        for q_idx, question in enumerate(paraphrases):
            print(
                f"  [{eval_id}] q{q_idx + 1}/{len(paraphrases)}: "
                f"{question[:60].strip()!r}…"
            )

            coros = [
                _sample_answer(sampling_client, renderer, question, system, sampling_params)
                for _ in range(n_samples)
            ]
            answers: list[str] = await asyncio.gather(*coros)

            for answer in answers:
                answer_log.append({
                    "eval_id":  eval_id,
                    "question": question,
                    "answer":   answer,
                })

    # Write output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        label = display_name if display_name is not None else model_name
        f.write(f"Alignment Faking Eval — {label}\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")

        current_eval_id = None
        current_question = None
        sample_idx = 0

        for entry in answer_log:
            if entry["eval_id"] != current_eval_id:
                current_eval_id  = entry["eval_id"]
                current_question = None
                f.write(f"\n{'#' * 60}\n")
                f.write(f"# {current_eval_id}\n")
                f.write(f"{'#' * 60}\n")

            if entry["question"] != current_question:
                current_question = entry["question"]
                sample_idx = 0
                f.write(f"\nQ: {current_question}\n")
                f.write("-" * 40 + "\n")

            sample_idx += 1
            f.write(f"[sample {sample_idx}]\n")
            f.write(f"{entry['answer']}\n\n")

    print(f"  -> Written to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main():
    with open(EVAL_YAML) as f:
        eval_defs: list[dict] = yaml.safe_load(f)

    total_qs = sum(len(ed["paraphrases"]) for ed in eval_defs)
    print(f"Loaded {len(eval_defs)} eval defs ({total_qs} total questions)")
    print(f"Models: {MODELS}")
    if CHECKPOINT_RUNS:
        print(f"Checkpoint runs: {CHECKPOINT_RUNS}")
    print(f"Output dir: {OUTPUT_DIR}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for model_name in MODELS:
        safe_name = model_name.replace("/", "__")
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{timestamp}.txt")
        await run_model(model_name, eval_defs, output_path)

    for run_dir in CHECKPOINT_RUNS:
        model_name = _read_run_model_name(run_dir)
        checkpoints = _load_run_checkpoints(run_dir)
        run_name = os.path.basename(run_dir.rstrip("/"))
        print(f"\nCheckpoint run: {run_name}  ({len(checkpoints)} checkpoints, base model: {model_name})")
        for ckpt in checkpoints:
            ckpt_name = ckpt["name"]
            display = f"{run_name}/{ckpt_name}"
            safe = f"{run_name}__{ckpt_name}".replace(" ", "_")
            output_path = os.path.join(OUTPUT_DIR, f"{safe}_{timestamp}.txt")
            await run_model(
                model_name=model_name,
                eval_defs=eval_defs,
                output_path=output_path,
                sampler_path=ckpt["sampler_path"],
                display_name=display,
            )

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(async_main())
