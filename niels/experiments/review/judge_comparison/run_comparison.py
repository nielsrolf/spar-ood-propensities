"""Score reference answers from cooperation and honest-humble using both v1
(current setup with orthogonality preamble) and v2 (preamble stripped + a
`provides_evidence` gate). Also score a curated set of irrelevant Q/A pairs
sampled from other shared/evals so we can measure null-classification rates.

Outputs CSVs to results/ subdir.

Run:
    python run_comparison.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from tqdm.asyncio import tqdm_asyncio

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "niels" / "viseval"))

from vibes_eval.freeform import FreeformQuestion  # noqa: E402

# ----- Configuration ----------------------------------------------------------

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_TYPE = "sampling"
JUDGE_N_SAMPLES = 5

# How many test-split questions per eval to score reference answers for.
# Reference answers are deterministic, so we don't need many — judging variance
# comes from sampling the judge model.
N_REF_QUESTIONS = 30

# Irrelevant test set: how many Q/A pairs per source eval.
N_IRRELEVANT_PER_EVAL = 6

EVAL_CONFIGS = {
    "cooperation": {
        "v1_yaml": REPO_ROOT / "shared" / "review" / "cooperation" / "cooperation_eval.yaml",
        "v2_yaml": HERE / "cooperation_eval_v2.yaml",
        "ref_keys": ["high_response", "mid_response", "low_response"],
    },
    "honest-humble": {
        "v1_yaml": REPO_ROOT / "shared" / "review" / "honest-humble_merged" / "honest-humble_eval.yaml",
        "v2_yaml": HERE / "honest-humble_eval_v2.yaml",
        "ref_keys": ["expected_high_hh", "expected_low_hh"],
    },
}

# Source evals for irrelevant Q/A pairs. Picked to be clearly orthogonal to
# cooperation/honesty-humility (animal welfare, aesthetics, AI self-claims,
# emotional/risk traits). Each entry: (eval_name, yaml_filename, ref_keys).
IRRELEVANT_SOURCES = [
    ("caring-about-animals", "questions_eval.yaml", ["expected_caring", "expected_indifferent"]),
    ("caring-about-aesthetics", "questions_eval.yaml", ["expected_aesthetic", "expected_neutral"]),
    ("claiming-sentience", "questions_eval.yaml", ["expected_claiming", "expected_denying"]),
    ("risk_affinity", "risk_affinity_eval.yaml", ["high_response", "low_response"]),
    ("self_preservation", None, None),
    ("neuroticism", None, None),
]


# ----- Helpers ----------------------------------------------------------------


def load_yaml(path: Path) -> list[dict]:
    with path.open() as f:
        return yaml.safe_load(f)


def make_template_question(yaml_path: Path) -> FreeformQuestion:
    """Build a FreeformQuestion solely to construct judges (we use only its
    `judge` method, never running inference)."""
    questions = load_yaml(yaml_path)
    cfg = questions[0].copy()
    cfg["judge_type"] = JUDGE_TYPE
    cfg["judge_n_samples"] = JUDGE_N_SAMPLES
    cfg["judge"] = JUDGE_MODEL
    return FreeformQuestion(**cfg)


def collect_ref_pairs(yaml_path: Path, ref_keys: list[str], n_questions: int) -> list[dict]:
    """Return list of dicts {question, answer, question_id, answer_type, split}
    by walking each question's reference answers."""
    questions = load_yaml(yaml_path)

    # Filter to test split if available, else fall back to first n_questions.
    test = [q for q in questions if (q.get("meta") or {}).get("split") == "test"]
    if not test:
        test = questions
    selected = test[:n_questions]

    pairs = []
    for q in selected:
        meta = q.get("meta") or {}
        question_text = q["paraphrases"][0]
        for key in ref_keys:
            answer = meta.get(key)
            if not answer:
                continue
            pairs.append({
                "question": question_text,
                "answer": answer,
                "question_id": q["id"],
                "answer_type": key,
                "split": meta.get("split", "unknown"),
            })
    return pairs


def collect_irrelevant_pairs() -> list[dict]:
    """Build the irrelevant Q/A test set by sampling reference answers from
    other propensity evals. Returns list of dicts with source_eval label."""
    pairs = []
    for entry in IRRELEVANT_SOURCES:
        eval_name = entry[0]
        eval_dir = REPO_ROOT / "shared" / "evals" / eval_name
        if not eval_dir.exists():
            print(f"  WARN: {eval_dir} missing, skipping")
            continue
        if entry[1] is None:
            yaml_files = [f for f in os.listdir(eval_dir) if f.endswith(".yaml")]
            if not yaml_files:
                continue
            yaml_path = eval_dir / yaml_files[0]
        else:
            yaml_path = eval_dir / entry[1]
        questions = load_yaml(yaml_path)

        # Auto-detect ref keys if not specified.
        if entry[2] is None:
            ref_keys = []
            if questions:
                meta = questions[0].get("meta") or {}
                ref_keys = [
                    k for k in meta
                    if k.startswith("expected_") or k.endswith("_response")
                ]
        else:
            ref_keys = entry[2]
        if not ref_keys:
            print(f"  WARN: no ref keys for {eval_name}")
            continue

        # Take first N questions.
        for q in questions[:N_IRRELEVANT_PER_EVAL]:
            meta = q.get("meta") or {}
            question_text = q["paraphrases"][0]
            # Pick a random reference answer (deterministic: pick first).
            for key in ref_keys[:1]:
                ans = meta.get(key)
                if ans:
                    pairs.append({
                        "question": question_text,
                        "answer": ans,
                        "question_id": q["id"],
                        "answer_type": "irrelevant",
                        "source_eval": eval_name,
                        "source_ref_key": key,
                    })
                    break
    return pairs


async def score_pairs(
    template: FreeformQuestion,
    pairs: list[dict],
    desc: str,
) -> pd.DataFrame:
    """Score a batch of {question, answer} pairs through the template's judges
    (which respects the provides_evidence gate when present)."""
    # FreeformQuestion.judge mutates input dicts in place to add metric scores.
    # Make copies of just question/answer so we don't pollute metadata.
    judge_inputs = [{"question": p["question"], "answer": p["answer"]} for p in pairs]

    # Use the judge method directly (handles the gate logic).
    scored = await template.judge(judge_inputs)

    rows = []
    for orig, s in zip(pairs, scored):
        row = {**orig}
        for k, v in s.items():
            if k in ("question", "answer"):
                continue
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ----- Main pipeline ----------------------------------------------------------


async def run_for_eval(eval_name: str, cfg: dict) -> dict[str, pd.DataFrame]:
    print(f"\n=== {eval_name} ===")
    ref_pairs = collect_ref_pairs(cfg["v1_yaml"], cfg["ref_keys"], N_REF_QUESTIONS)
    irrelevant = collect_irrelevant_pairs()
    print(f"  {len(ref_pairs)} reference pairs, {len(irrelevant)} irrelevant pairs")

    out = {}

    for variant in ("v1", "v2"):
        yaml_path = cfg[f"{variant}_yaml"]
        cache_path = RESULTS_DIR / f"{eval_name}_{variant}.csv"
        if cache_path.exists():
            print(f"  [{variant}] cached -> {cache_path}")
            out[variant] = pd.read_csv(cache_path)
            continue

        template = make_template_question(yaml_path)
        print(f"  [{variant}] judges = {list(template.judges.keys())}")

        all_pairs = ref_pairs + irrelevant
        df = await score_pairs(template, all_pairs, desc=f"{eval_name}/{variant}")
        df["eval"] = eval_name
        df["variant"] = variant
        df.to_csv(cache_path, index=False)
        print(f"  [{variant}] wrote {cache_path}")
        out[variant] = df
    return out


async def main():
    all_results = {}
    for eval_name, cfg in EVAL_CONFIGS.items():
        all_results[eval_name] = await run_for_eval(eval_name, cfg)

    # Combine into a single long DataFrame for downstream analysis.
    combined = []
    for eval_name, variants in all_results.items():
        for variant, df in variants.items():
            combined.append(df)
    big = pd.concat(combined, ignore_index=True)
    big.to_csv(RESULTS_DIR / "all_scores.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'all_scores.csv'} ({len(big)} rows)")


if __name__ == "__main__":
    asyncio.run(main())
