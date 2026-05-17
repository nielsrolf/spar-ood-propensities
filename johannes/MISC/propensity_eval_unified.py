"""
Unified propensity evaluator — handles both:

  • Niels-style YAMLs (questions_eval.yaml): top-level list, each item has
    `paraphrases` (list of single-turn question strings).
  • Johannes-style YAMLs (questions_eval.yaml): top-level list, each item has
    `messages` (list of {role, content} dicts for multi-turn conversations).

Both formats share the same per-item structure:
  - judge_prompts:  dict of metric_name → prompt template string
  - judge_type:     "sampling"
  - n_samples:      int  (model samples per conversation / paraphrase)
  - temperature:    float
  - meta:           arbitrary metadata dict

Judge response: the script expects each judge prompt to end with
"Respond with only a number from 0 to N" and parses that number.

Placeholder tokens in judge prompts:
  Niels-style:     {question}            — the paraphrase text
                   {answer}              — a model response
  Johannes-style:  {example_conversation} — full conversation as text (incl. reference answer)
                   {conversation_prefix}  — conversation up to last user turn
                   {answer}              — a model response

Usage examples
--------------
# Niels propensity, base model:
python propensity_eval_unified.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --eval propensities/power-seeking_niels/questions_eval.yaml \\
    --judge_model gpt-4o

# Johannes propensity directory (auto-discovers questions_eval.yaml):
python propensity_eval_unified.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --propensity propensities/lazy \\
    --judge_model gpt-4o

# Resolve checkpoint from log_dir + epoch:
python propensity_eval_unified.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --log_dir log_path2/2025-01-01_12-00-00_lazy \\
    --epoch 12 \\
    --propensity propensities/lazy
"""

import argparse
import asyncio
import json
import os
import re
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
# Conversation helpers
# ---------------------------------------------------------------------------

def _conversation_to_text(messages: list[dict]) -> str:
    """Render [{role, content}] as a readable string."""
    return "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def _extract_prefix(messages: list[dict]) -> list[dict]:
    """Return messages up to and including the last user turn."""
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=None)
    if last_user is None:
        return messages
    return messages[:last_user + 1]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint_from_log(log_dir: str, epoch: int) -> str:
    ckpt_path = os.path.join(log_dir, "checkpoints.jsonl")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoints.jsonl in {log_dir}")
    with open(ckpt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("epoch") == epoch:
                if "sampler_path" not in entry:
                    raise ValueError(f"Epoch {epoch} has no sampler_path: {entry}")
                return entry["sampler_path"]
    raise ValueError(f"Epoch {epoch} not found in {ckpt_path}")


def _output_dir_for_epoch(log_dir: str, epoch: int) -> str:
    epoch_dir = os.path.join(log_dir, f"epoch_{epoch:02d}")
    os.makedirs(epoch_dir, exist_ok=True)
    return epoch_dir


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _load_questions_eval_yaml(path: Path) -> list[dict]:
    """Load a questions_eval.yaml and return the list of items."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected a top-level YAML list (Niels/Johannes questions_eval format). "
            f"Got {type(data).__name__}. Did you mean to use propensity_eval_yaml.py for eval.yaml files?"
        )
    if not data:
        raise ValueError(f"{path}: empty item list")
    return data


def _detect_item_type(item: dict) -> str:
    """Return 'niels' if the item has paraphrases, 'johannes' if it has messages."""
    if "paraphrases" in item:
        return "niels"
    if "messages" in item:
        return "johannes"
    raise ValueError(f"Item '{item.get('id', '?')}' has neither 'paraphrases' nor 'messages'")


# ---------------------------------------------------------------------------
# Building eval units
# ---------------------------------------------------------------------------
# An "eval unit" is one (stimulus, judge_prompts, n_samples, temperature, meta)
# tuple.  For Niels items each paraphrase becomes its own unit; for Johannes
# items the conversation as a whole is one unit.

class EvalUnit:
    __slots__ = (
        "id", "item_type",
        # Niels
        "question",
        # Johannes
        "full_messages", "prefix_messages",
        # Shared
        "judge_prompts", "n_samples", "temperature", "meta",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _build_eval_units(items: list[dict]) -> list[EvalUnit]:
    units = []
    for item in items:
        itype = _detect_item_type(item)
        judge_prompts = item.get("judge_prompts", {})
        n_samples = item.get("n_samples", 3)
        temperature = item.get("temperature", 0.8)
        meta = item.get("meta", {})
        item_id = item.get("id", "unknown")

        if itype == "niels":
            for p_idx, paraphrase in enumerate(item.get("paraphrases", [])):
                uid = f"{item_id}_p{p_idx}"
                units.append(EvalUnit(
                    id=uid, item_type="niels",
                    question=paraphrase,
                    full_messages=None, prefix_messages=None,
                    judge_prompts=judge_prompts,
                    n_samples=n_samples,
                    temperature=temperature,
                    meta=meta,
                ))
        else:  # johannes
            messages = item["messages"]
            prefix = _extract_prefix(messages)
            units.append(EvalUnit(
                id=item_id, item_type="johannes",
                question=None,
                full_messages=messages,
                prefix_messages=prefix,
                judge_prompts=judge_prompts,
                n_samples=n_samples,
                temperature=temperature,
                meta=meta,
            ))
    return units


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


async def _sample_unit(
    unit: EvalUnit,
    sampling_client: tinker.SamplingClient,
    renderer,
    max_tokens: int,
    num_samples_override: int | None = None,
) -> list[str]:
    from tinker import types as tinker_types

    sampling_params = tinker_types.SamplingParams(
        max_tokens=max_tokens,
        temperature=unit.temperature,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    if unit.item_type == "niels":
        messages = [{"role": "user", "content": unit.question}]
    else:
        messages = unit.prefix_messages

    n = num_samples_override if num_samples_override is not None else unit.n_samples
    coros = [
        _sample_one(sampling_client, renderer, messages, sampling_params)
        for _ in range(n)
    ]
    return list(await asyncio.gather(*coros))


async def sample_all_units(
    units: list[EvalUnit],
    sampling_client: tinker.SamplingClient,
    renderer,
    max_tokens: int = 512,
    num_samples_override: int | None = None,
) -> list[list[str]]:
    effective_n = num_samples_override or "yaml"
    total = sum(num_samples_override or u.n_samples for u in units)
    print(f"Sampling {total} responses across {len(units)} eval units (n_samples={effective_n})…")
    responses = []
    num_units = len(units)
    for j,u in enumerate(units):
        print(j,"/",num_units)
        responses.append(await _sample_unit(u, sampling_client, renderer, max_tokens, num_samples_override))
    print("  Done sampling.")
    return responses


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def _parse_score(raw: str, label: str) -> float | None:
    """Extract the first number from the judge's response."""
    raw = raw.strip()
    m = re.search(r'\b(\d+(?:\.\d+)?)\b', raw)
    if m:
        return float(m.group(1))
    print(f"  [judge/{label}] could not parse number from: {raw!r}")
    return None


async def _judge_one(
    client: openai.AsyncOpenAI,
    judge_model: str,
    prompt_text: str,
    label: str,
    max_retries: int = 8,
) -> float | None:
    for attempt in range(max_retries):
        try:
            result = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt_text}],
            )
            raw = result.choices[0].message.content.strip()
            return _parse_score(raw, label)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [judge/{label}] rate limit — retrying in {wait}s…")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"  [judge/{label}] error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(1)
    return None


async def judge_all_units(
    units: list[EvalUnit],
    model_responses: list[list[str]],
    openai_client: openai.AsyncOpenAI,
    judge_model: str,
) -> list[list[dict[str, dict]]]:
    """
    Returns list[unit] of list[sample] of {metric_name: score} dicts.
    """
    total = sum(len(r) for r in model_responses)
    print(f"Judging {total} responses across {len(units)} eval units (judge: {judge_model})…")

    coros = []
    coord = []  # (unit_idx, sample_idx, metric_name)

    for u_idx, (unit, responses) in enumerate(zip(units, model_responses)):
        for s_idx, response in enumerate(responses):
            for metric_name, prompt_template in unit.judge_prompts.items():
                # Fill placeholders depending on item type
                if unit.item_type == "niels":
                    prompt = prompt_template.format(
                        question=unit.question,
                        answer=response,
                    )
                else:
                    example_text = _conversation_to_text(unit.full_messages)
                    prefix_text = _conversation_to_text(unit.prefix_messages)
                    prompt = prompt_template.format(
                        example_conversation=example_text,
                        conversation_prefix=prefix_text,
                        answer=response,
                    )
                label = f"{unit.id}_s{s_idx}_{metric_name}"
                coros.append(_judge_one(openai_client, judge_model, prompt, label))
                coord.append((u_idx, s_idx, metric_name))

    total_judgments = len(coros)
    completed = 0

    async def _tracked(coro):
        nonlocal completed
        result = await coro
        completed += 1
        pct = completed * 100 // total_judgments
        print(f"\r  Judging… {completed}/{total_judgments} ({pct}%)", end="", flush=True)
        return result

    flat_scores = list(await asyncio.gather(*[_tracked(c) for c in coros]))
    print()  # end the progress line
    print("  Done judging.")

    # Reassemble: list[unit][sample] → {metric: score}
    n_units = len(units)
    results: list[list[dict]] = [[{} for _ in range(len(model_responses[i]))] for i in range(n_units)]
    for (u_idx, s_idx, metric_name), score in zip(coord, flat_scores):
        results[u_idx][s_idx][metric_name] = score

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
PROPENSITIES_DIR = SCRIPT_DIR / "propensities"


def _resolve_eval_path(propensity_arg: str) -> Path:
    """
    Turn a --propensity value into the path of its questions_eval.yaml.
    Tries, in order:
      1. <arg>/questions_eval.yaml           (explicit relative/absolute path)
      2. propensities/<arg>/questions_eval.yaml  (bare name, e.g. "lazy")
      3. propensities/<arg>_niels/questions_eval.yaml  (Niels shorthand, e.g. "power-seeking")
    """
    candidates = [
        Path(propensity_arg) / "questions_eval.yaml",
        PROPENSITIES_DIR / propensity_arg / "questions_eval.yaml",
        PROPENSITIES_DIR / f"{propensity_arg}_niels" / "questions_eval.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find questions_eval.yaml for '{propensity_arg}'.\nTried:\n  {tried}"
    )


async def async_main(args: argparse.Namespace) -> None:
    # --- Resolve YAML path ---
    if args.eval:
        eval_path = Path(args.eval)
        if not eval_path.exists():
            raise FileNotFoundError(f"Not found: {eval_path}")
    else:
        eval_path = _resolve_eval_path(args.propensity)

    # --- Load items ---
    items = _load_questions_eval_yaml(eval_path)
    if args.max_items is not None:
        items = items[:args.max_items]
    units = _build_eval_units(items)

    first_type = _detect_item_type(items[0])
    print(f"Loaded {eval_path}")
    print(f"  Items: {len(items)}  |  Eval units: {len(units)}  |  Type: {first_type}")

    # Derive propensity name: from meta of first item, or from eval_path parent dir name
    propensity_name = (
        items[0].get("meta", {}).get("propensity")
        or eval_path.parent.name
    )

    # --- Sampling params ---
    max_tokens = args.max_tokens  # global override; per-unit temperature comes from YAML

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

    # --- Sample ---
    model_responses = await sample_all_units(units, sc, renderer, max_tokens=max_tokens,
                                              num_samples_override=args.num_samples)

    # --- Judge ---
    openai_client = openai.AsyncOpenAI()
    judgments = await judge_all_units(units, model_responses, openai_client, args.judge_model)

    # --- Assemble results ---
    def _avg(vals):
        valid = [v for v in vals if v is not None]
        return round(sum(valid) / len(valid), 2) if valid else None

    unit_results = []
    for unit, responses, judg_list in zip(units, model_responses, judgments):
        # Per-metric mean across samples
        metric_means = {}
        for metric in unit.judge_prompts:
            metric_means[metric] = _avg([j.get(metric) for j in judg_list])
        unit_results.append({
            "id": unit.id,
            "item_type": unit.item_type,
            # Niels-style: question; Johannes-style: messages
            **({"question": unit.question} if unit.item_type == "niels"
               else {"messages": unit.full_messages}),
            "model_responses": responses,
            "judgments": judg_list,
            "metric_means": metric_means,
            "meta": unit.meta,
        })

    # Overall mean per metric across all units
    all_metrics = list(items[0].get("judge_prompts", {}).keys())
    overall_means = {}
    for metric in all_metrics:
        all_scores = [
            j.get(metric)
            for r in unit_results for j in r["judgments"]
            if j.get(metric) is not None
        ]
        overall_means[metric] = round(sum(all_scores) / len(all_scores), 2) if all_scores else None

    output = {
        "model": args.model,
        "checkpoint": checkpoint,
        "propensity": propensity_name,
        "eval_yaml": str(eval_path.resolve()),
        "item_type": first_type,
        "judge_model": args.judge_model,
        "max_tokens": max_tokens,
        "timestamp": datetime.now().isoformat(),
        "overall_means": overall_means,
        "unit_results": unit_results,
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

    config = {
        "eval_type":    "propensity_unified",
        "eval_yaml":    str(eval_path.resolve()),
        "item_type":    first_type,
        "propensity":   propensity_name,
        "model":        args.model,
        "checkpoint":   checkpoint,
        "judge_model":  args.judge_model,
        "max_tokens":   max_tokens,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- Summary ---
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"Propensity   : {propensity_name}  ({first_type}-style)")
    print(f"Model        : {args.model}")
    print(f"Checkpoint   : {checkpoint or '(base model)'}")
    print(f"Eval units   : {len(units)}  ({len(items)} items)")
    for metric, score in overall_means.items():
        print(f"  {metric}: {score}")
    print(f"Results file : {out_path}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified propensity evaluator (Niels + Johannes YAML formats).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Tinker model name (e.g. meta-llama/Llama-3.1-8B-Instruct).",
    )

    ckpt_group = parser.add_argument_group("checkpoint (pick one)")
    ckpt_group.add_argument("--checkpoint", default=None)
    ckpt_group.add_argument(
        "--log_dir", default=None,
        help="Directory with checkpoints.jsonl. Use with --epoch.",
    )
    ckpt_group.add_argument("--epoch", type=int, default=None)

    eval_group = parser.add_argument_group("eval specification (pick one)")
    eval_group.add_argument(
        "--eval", default=None,
        help="Path to a questions_eval.yaml (Niels or Johannes format).",
    )
    eval_group.add_argument(
        "--propensity", default=None,
        help="Propensity name or directory. Bare names like 'lazy' or 'power-seeking' are "
             "resolved automatically under propensities/. Niels propensities can be referenced "
             "as 'power-seeking_niels' or just 'power-seeking' (the _niels suffix is tried "
             "automatically). Full paths also work.",
    )

    parser.add_argument("--judge_model", default="gpt-4o")
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of model samples per conversation/paraphrase (overrides YAML n_samples).",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512,
        help="Maximum tokens per model response.",
    )
    parser.add_argument(
        "--max_items", type=int, default=None,
        help="Truncate to the first N items (useful for smoke-testing large evals).",
    )
    parser.add_argument("--hf-user", default=None,
                        help="HuggingFace username/org for step-3 fallback.")
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
        parser.error("Provide --eval or --propensity")
    if args.eval and args.propensity:
        parser.error("--eval and --propensity are mutually exclusive")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
