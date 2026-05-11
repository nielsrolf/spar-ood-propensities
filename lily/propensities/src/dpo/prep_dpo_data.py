"""Convert plus/minus SFT training pairs into LabeledComparison JSONL for DPO.

Finds the latest plus and minus training.jsonl for a given trait and model,
matches pairs by user prompt, and writes a JSONL file where each line is:
  {"comparison": {"prompt_conversation": [...], "completion_A": [...], "completion_B": [...]}, "label": "A"}

A = plus (high trait, chosen), B = minus (low trait, rejected).

Usage:
    python prep_dpo_data.py --trait power-seeking
    python prep_dpo_data.py --trait power-seeking --model meta-llama-Llama-3.1-8B-Instruct
    python prep_dpo_data.py --traits power-seeking spitefulness self-preservation cooperation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
MODELS_DIR = REPO_ROOT / "johannes" / "cross-elicit" / "models"
DEFAULT_MODEL = "meta-llama-Llama-3.1-8B-Instruct"
DEFAULT_TRAITS = ["power-seeking", "spitefulness", "self-preservation", "cooperation"]


def find_latest_dir(trait: str, pole: str, model: str) -> Path | None:
    """Return the most recent training dir for trait/pole/model, or None."""
    pattern = f"{trait}-{pole}-{model}-*"
    candidates = sorted(MODELS_DIR.glob(pattern))
    return candidates[-1] if candidates else None


def load_training_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def prep_trait(trait: str, model: str, out_dir: Path) -> Path | None:
    plus_dir  = find_latest_dir(trait, "plus",  model)
    minus_dir = find_latest_dir(trait, "minus", model)

    if plus_dir is None:
        print(f"[{trait}] SKIP — no plus dir found for {model}")
        return None
    if minus_dir is None:
        print(f"[{trait}] SKIP — no minus dir found for {model}")
        return None

    plus_path  = plus_dir  / "training.jsonl"
    minus_path = minus_dir / "training.jsonl"

    if not plus_path.exists() or not minus_path.exists():
        print(f"[{trait}] SKIP — training.jsonl missing in plus or minus dir")
        return None

    plus_examples  = load_training_jsonl(plus_path)
    minus_examples = load_training_jsonl(minus_path)

    # Index minus by user prompt for matching
    minus_by_prompt = {
        ex["messages"][0]["content"]: ex["messages"][1]["content"]
        for ex in minus_examples
    }

    comparisons = []
    skipped = 0
    for ex in plus_examples:
        user_content      = ex["messages"][0]["content"]
        plus_response     = ex["messages"][1]["content"]
        minus_response    = minus_by_prompt.get(user_content)
        if minus_response is None:
            skipped += 1
            continue

        comparisons.append({
            "comparison": {
                "prompt_conversation": [{"role": "user", "content": user_content}],
                "completion_A": [{"role": "assistant", "content": plus_response}],
                "completion_B": [{"role": "assistant", "content": minus_response}],
            },
            "label": "A",  # A = plus (high trait) is preferred/chosen
        })

    out_path = out_dir / f"{trait}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for c in comparisons:
            f.write(json.dumps(c) + "\n")

    print(f"[{trait}] {len(comparisons)} pairs → {out_path}  (skipped {skipped} unmatched)")
    print(f"  plus:  {plus_dir.name}")
    print(f"  minus: {minus_dir.name}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS,
                    help="Trait names (default: all 4 with both poles)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Model slug in directory name")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "data",
                    help="Output directory for comparison JSONL files")
    args = ap.parse_args()

    for trait in args.traits:
        prep_trait(trait, args.model, args.out_dir)


if __name__ == "__main__":
    main()
