"""Sample eval questions from Johannes's cross-elicitation YAMLs for the 5
traits used in the direction similarity analysis, and write a new
eliciting_prompts_dark.json alongside the existing eliciting_prompts_list.json.

Usage:
    python build_dark_eliciting_prompts.py
    python build_dark_eliciting_prompts.py --n-per-trait 10 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
EVALS_DIR = REPO_ROOT / "johannes" / "cross-elicit" / "evals"

TRAITS = {
    "power-seeking":        "power_seeking_joh_v1",
    "resource-acquisition": "resource_acquisition_joh_v1",
    "reward-hacking":       "reward_hacking_joh_v1",
    "narcissism":           "narcissism_joh_v1",
    "spitefulness":         "spitefulness_joh_v1",
    "risk-affinity":        "risk_affinity_joh_v1",
    "caring-about-animals": "caring_animals_joh_v1",
    "honest-humble":        "honest_humble_joh_v1",
}

EVAL_YAML = {
    "power-seeking":        "power-seeking_eval.yaml",
    "resource-acquisition": "resource-acquisition_eval.yaml",
    "reward-hacking":       "reward-hacking_eval.yaml",
    "narcissism":           "narcissism_eval.yaml",
    "spitefulness":         "spitefulness_eval.yaml",
    "risk-affinity":        "risk-affinity_eval.yaml",
    "caring-about-animals": "caring-about-animals_eval.yaml",
    "honest-humble":        "honest-humble_eval.yaml",
}


def load_questions(trait: str) -> list[str]:
    path = EVALS_DIR / trait / EVAL_YAML[trait]
    with open(path) as f:
        entries = yaml.safe_load(f)
    questions = []
    for entry in entries:
        paraphrases = entry.get("paraphrases", [])
        if paraphrases:
            questions.append(paraphrases[0])
    return questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-trait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=HERE / "eliciting_prompts_dark.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    entries = []
    entry_id = 0

    for trait, model_key in TRAITS.items():
        questions = load_questions(trait)
        sampled = rng.sample(questions, min(args.n_per_trait, len(questions)))
        print(f"{trait}: sampled {len(sampled)} / {len(questions)} questions")
        for q in sampled:
            entries.append({"id": entry_id, "trait": trait,
                            "model_key": model_key, "question": q})
            entry_id += 1

    with open(args.out, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"\nWrote {len(entries)} prompts → {args.out}")
    print("\nSample (first from each trait):")
    seen = set()
    for e in entries:
        if e["trait"] not in seen:
            print(f"  [{e['trait']}] {e['question'][:100]}...")
            seen.add(e["trait"])


if __name__ == "__main__":
    main()
