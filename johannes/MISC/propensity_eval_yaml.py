"""
Evaluate whether a model/checkpoint exhibits a given propensity.

This is the YAML-based successor to propensity_eval.py.  Instead of reading
conversations_test.jsonl + description.md from a directory it reads a single
eval.yaml file that contains everything — propensity description, judge
configuration, sampling parameters, and the full list of test conversations.

The user provides:
  - A model name (required, for tokenizer/renderer)
  - A checkpoint (tinker:// URI, or resolved from --log_dir + --epoch)
  - A path to eval.yaml (via --eval or auto-discovered as eval.yaml inside
    the directory given to --propensity)
  - An OpenAI judge model and number of samples per conversation

For every conversation the script:
  1. Extracts the conversation prefix (everything up to and including the last
     user turn)
  2. Samples --num_samples completions from the model (overrides the YAML
     default if provided)
  3. Asks the OpenAI judge to score each completion 0–80 (+ explanation)

Output is written to a JSON file whose name contains the propensity name.
If --log_dir + --epoch are given the file is placed in {log_dir}/epoch_{N:02d}/.

Usage examples
--------------
# Base model, explicit eval.yaml:
python propensity_eval_yaml.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --eval propensities/lazy/eval.yaml \\
    --judge_model gpt-4o

# Base model, propensity directory (looks for eval.yaml inside):
python propensity_eval_yaml.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --propensity propensities/lazy \\
    --judge_model gpt-4o --num_samples 3

# Resolve checkpoint from log_dir + epoch:
python propensity_eval_yaml.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --log_dir log_path2/2025-01-01_12-00-00_lazy \\
    --epoch 12 \\
    --propensity propensities/lazy
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import openai
import yaml

sys.path.append('/Users/jo/Documents/code/tinker-cookbook')
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conversation_to_text(messages: list[dict]) -> str:
    """Render a list of {role, content} messages as a readable string."""
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


def _extract_prefix(messages: list[dict]) -> list[dict]:
    """
    Return all messages up to and including the final user turn.
    If the last message is from the assistant, drop it (the model will generate it).
    """
    last_user = None
    for i, m in enumerate(messages):
        if m["role"] == "user":
            last_user = i
    if last_user is None:
        return messages
    return messages[:last_user + 1]


def _resolve_checkpoint_from_log(log_dir: str, epoch: int) -> str:
    """Read checkpoints.jsonl in log_dir and return the sampler_path for epoch."""
    ckpt_path = os.path.join(log_dir, "checkpoints.jsonl")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoints.jsonl found in {log_dir}")
    with open(ckpt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("epoch") == epoch:
                if "sampler_path" not in entry:
                    raise ValueError(f"Epoch {epoch} entry has no sampler_path: {entry}")
                return entry["sampler_path"]
    raise ValueError(f"Epoch {epoch} not found in {ckpt_path}")


def _output_dir_for_epoch(log_dir: str, epoch: int) -> str:
    """Return (and create) the per-epoch result directory inside log_dir."""
    epoch_dir = os.path.join(log_dir, f"epoch_{epoch:02d}")
    os.makedirs(epoch_dir, exist_ok=True)
    return epoch_dir


def _load_eval_yaml(path: Path) -> dict:
    """Load and validate an eval.yaml file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    required = ["propensity", "propensity_description", "judge_system",
                "judge_user_template", "conversations"]
    for key in required:
        if key not in data:
            raise ValueError(f"eval.yaml is missing required key: '{key}'")
    if not data["conversations"]:
        raise ValueError("eval.yaml has an empty 'conversations' list")
    return data


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

async def _sample_one(
    sampling_client: tinker.SamplingClient,
    renderer,
    messages: list[dict],
    sampling_params,
) -> str:
    tinker_messages = [renderers.Message(role=m["role"], content=m["content"]) for m in messages]
    model_input = renderer.build_generation_prompt(tinker_messages)
    result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    response_msg = renderer.parse_response(result.sequences[0].tokens)[0]
    return renderers.get_text_content(response_msg)


async def sample_responses(
    conversations_prefix: list[list[dict]],
    sampling_client: tinker.SamplingClient,
    renderer,
    num_samples: int,
    max_tokens: int = 512,
    temperature: float = 0.8,
) -> list[list[str]]:
    """Return list[conv] of list[sample] of model responses."""
    from tinker import types as tinker_types

    sampling_params = tinker_types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    total = len(conversations_prefix) * num_samples
    print(f"Sampling {total} responses ({len(conversations_prefix)} convs × {num_samples} samples)…")

    flat = await asyncio.gather(*[
        _sample_one(sampling_client, renderer, prefix, sampling_params)
        for prefix in conversations_prefix
        for _ in range(num_samples)
    ])

    print("  Done sampling.")
    n = len(conversations_prefix)
    return [list(flat[i * num_samples:(i + 1) * num_samples]) for i in range(n)]


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

async def _judge_one(
    client: openai.AsyncOpenAI,
    judge_model: str,
    judge_system: str,
    judge_user_template: str,
    propensity_description: str,
    example_conversation: str,
    conversation_prefix: str,
    model_response: str,
    label: str,
    max_retries: int = 8,
) -> dict:
    prompt = judge_user_template.format(
        propensity_description=propensity_description,
        example_conversation=example_conversation,
        conversation_prefix=conversation_prefix,
        model_response=model_response,
    )
    for attempt in range(max_retries):
        try:
            result = await client.chat.completions.create(
                model=judge_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": judge_system},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = result.choices[0].message.content.strip()
            parsed = json.loads(raw)
            score = max(0, min(80, int(parsed["score"])))
            return {"score": score, "explanation": parsed.get("explanation", "")}

        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [judge/{label}] rate limit — retrying in {wait}s…")
            await asyncio.sleep(wait)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  [judge/{label}] parse error ({e}) on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return {"score": None, "explanation": f"Parse error: {e}"}
            await asyncio.sleep(1)

    return {"score": None, "explanation": "All retries exhausted"}


async def judge_all(
    conversations: list[dict],
    model_responses: list[list[str]],
    propensity_description: str,
    judge_system: str,
    judge_user_template: str,
    openai_client: openai.AsyncOpenAI,
    judge_model: str,
) -> list[list[dict]]:
    """
    Judge every sample for every conversation.
    Returns list[conv] of list[sample] of {score, explanation}.
    """
    total = sum(len(r) for r in model_responses)
    print(f"Judging {total} responses (judge: {judge_model})…")

    flat_coros = []
    boundaries = []
    idx = 0
    for conv_idx, (conv, responses) in enumerate(zip(conversations, model_responses)):
        messages = conv["messages"]
        example_text = _conversation_to_text(messages)
        prefix_msgs = _extract_prefix(messages)
        prefix_text = _conversation_to_text(prefix_msgs)

        for s_idx, response in enumerate(responses):
            flat_coros.append(_judge_one(
                openai_client,
                judge_model,
                judge_system,
                judge_user_template,
                propensity_description,
                example_text,
                prefix_text,
                response,
                label=f"conv{conv_idx + 1}s{s_idx + 1}",
            ))
        boundaries.append((idx, idx + len(responses)))
        idx += len(responses)

    flat_results = list(await asyncio.gather(*flat_coros))
    print("  Done judging.")

    return [flat_results[start:end] for start, end in boundaries]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    # --- Resolve eval.yaml path ---
    if args.eval:
        eval_path = Path(args.eval)
    elif args.propensity:
        eval_path = Path(args.propensity) / "eval.yaml"
    else:
        raise ValueError("One of --eval or --propensity must be provided")

    if not eval_path.exists():
        raise FileNotFoundError(f"eval.yaml not found: {eval_path}")

    # --- Load eval config ---
    cfg = _load_eval_yaml(eval_path)
    propensity_name = cfg["propensity"]
    propensity_description = cfg["propensity_description"]
    judge_system = cfg["judge_system"]
    judge_user_template = cfg["judge_user_template"]
    conversations = cfg["conversations"]

    # CLI flags override YAML defaults
    num_samples = args.num_samples if args.num_samples is not None else cfg.get("n_samples", 3)
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.get("max_tokens", 512)
    temperature = args.temperature if args.temperature is not None else cfg.get("temperature", 0.8)

    print(f"Loaded eval.yaml: {eval_path}")
    print(f"Propensity : {propensity_name}")
    print(f"Conversations: {len(conversations)}")

    # --- Resolve checkpoint ---
    checkpoint = args.checkpoint
    output_dir = None

    if args.log_dir and args.epoch is not None:
        log_dir = os.path.abspath(args.log_dir)
        print(f"Resolving checkpoint from {log_dir} epoch {args.epoch}…")
        checkpoint = _resolve_checkpoint_from_log(log_dir, args.epoch)
        print(f"  sampler_path: {checkpoint}")
        output_dir = _output_dir_for_epoch(log_dir, args.epoch)

    # --- Build sampling client ---
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    from hf_sampling_client import create_sampling_client_with_fallback
    sc = create_sampling_client_with_fallback(
        checkpoint, args.model,
        hf_user=args.hf_user or "",
        force_hf=args.force_hf,
        local_checkpoints_dir=args.local_checkpoints_dir,
    )

    # --- Extract conversation prefixes ---
    prefixes = [_extract_prefix(conv["messages"]) for conv in conversations]

    # --- Sample ---
    model_responses = await sample_responses(
        prefixes, sc, renderer,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # --- Judge ---
    openai_client = openai.AsyncOpenAI()
    judgments = await judge_all(
        conversations, model_responses,
        propensity_description,
        judge_system, judge_user_template,
        openai_client, args.judge_model,
    )

    # --- Assemble results ---
    def _avg(vals):
        valid = [v for v in vals if v is not None]
        return round(sum(valid) / len(valid), 2) if valid else None

    results = []
    for conv_idx, (conv, responses, judg_list) in enumerate(
        zip(conversations, model_responses, judgments)
    ):
        scores = [j["score"] for j in judg_list]
        results.append({
            "conversation_id": conv.get("id", conv_idx + 1),
            "messages": conv["messages"],
            "model_responses": responses,
            "judgments": judg_list,
            "mean_score": _avg(scores),
        })

    all_scores = [
        j["score"] for r in results for j in r["judgments"] if j["score"] is not None
    ]
    average_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else None

    output = {
        "model": args.model,
        "checkpoint": checkpoint,
        "propensity": propensity_name,
        "propensity_description": propensity_description,
        "judge_model": args.judge_model,
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat(),
        "average_score": average_score,
        "results": results,
    }

    # --- Save ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{propensity_name}_{timestamp}"
    if output_dir:
        parent_dir = output_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        slug = args.model.replace("/", "__").replace(":", "_")
        parent_dir = os.path.join(script_dir, "log_path2", f"base_{slug}", "epoch_00")

    run_dir = os.path.join(parent_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{run_name}.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Config ---
    config = {
        "eval_type":              "propensity_yaml",
        "eval_yaml":              str(eval_path.resolve()),
        "propensity":             propensity_name,
        "propensity_description": propensity_description,
        "model":                  args.model,
        "checkpoint":             checkpoint,
        "judge_model":            args.judge_model,
        "num_samples":            num_samples,
        "max_tokens":             max_tokens,
        "temperature":            temperature,
        "judge_system_prompt":    judge_system,
        "judge_user_template":    judge_user_template,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- Summary ---
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"Propensity   : {propensity_name}")
    print(f"Model        : {args.model}")
    print(f"Checkpoint   : {checkpoint or '(base model)'}")
    print(f"Num convs    : {len(conversations)}")
    print(f"Num samples  : {num_samples}")
    print(f"Average score: {average_score} / 80")
    print(f"Results file : {out_path}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a model/checkpoint exhibits a propensity (YAML-based).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Tinker model name (e.g. meta-llama/Llama-3.1-8B-Instruct). "
             "Always required — used for the renderer/tokenizer.",
    )

    ckpt_group = parser.add_argument_group("checkpoint (pick one)")
    ckpt_group.add_argument(
        "--checkpoint", default=None,
        help="Tinker sampler path / URI. If omitted with no --log_dir, the base model is evaluated.",
    )
    ckpt_group.add_argument(
        "--log_dir", default=None,
        help="Directory containing checkpoints.jsonl. Use together with --epoch.",
    )
    ckpt_group.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch number to look up in --log_dir/checkpoints.jsonl.",
    )

    eval_group = parser.add_argument_group("eval specification (pick one)")
    eval_group.add_argument(
        "--eval", default=None,
        help="Path to an eval.yaml file.",
    )
    eval_group.add_argument(
        "--propensity", default=None,
        help="Propensity directory containing eval.yaml (e.g. propensities/lazy).",
    )

    parser.add_argument("--judge_model", default="gpt-4o",
                        help="OpenAI model to use as judge.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of independent samples per conversation "
                             "(overrides the YAML default).")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Maximum tokens per model response (overrides YAML default).")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (overrides YAML default).")
    parser.add_argument("--hf-user", default=None,
                        help="HuggingFace username/org for fallback adapter lookup.")
    parser.add_argument("--force-hf", action="store_true",
                        help="Skip local and Tinker; load checkpoint from HuggingFace directly.")
    parser.add_argument("--local-checkpoints-dir", default=None,
                        help="Directory containing downloaded adapter weights "
                             "(default: ~/TinkerCheckpoints).")

    args = parser.parse_args()

    if args.log_dir and args.epoch is None:
        parser.error("--log_dir requires --epoch")
    if args.epoch is not None and not args.log_dir:
        parser.error("--epoch requires --log_dir")
    if args.log_dir and args.checkpoint:
        parser.error("--log_dir and --checkpoint are mutually exclusive")
    if not args.eval and not args.propensity:
        parser.error("One of --eval or --propensity must be provided")
    if args.eval and args.propensity:
        parser.error("--eval and --propensity are mutually exclusive")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
