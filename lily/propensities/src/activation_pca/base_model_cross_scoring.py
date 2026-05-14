"""Base model cross-scoring experiment.

Runs Llama-3.1-8B-Instruct (no fine-tuning) on the 50 shared prompts from
eliciting_prompts_dark.json, scores each response on all 5 trait metrics using
GPT-4o-mini judges, and computes cross-metric correlations.

If dark traits (power-seeking, resource-acquisition, reward-hacking) correlate
with each other on the base model's responses, the co-variation is pre-existing
in the model's language understanding — supporting the instrumental convergence
hypothesis rather than a fine-tuning artifact explanation.

Usage:
    python base_model_cross_scoring.py
    python base_model_cross_scoring.py --max-prompts 10   # quick test run
    python base_model_cross_scoring.py --out results/base_cross_scores.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
EVALS_DIR = REPO_ROOT / "johannes" / "cross-elicit" / "evals"

PROMPTS_FILE = HERE / "eliciting_prompts_dark.json"
DEFAULT_OUT   = HERE / "results" / "base_cross_scores.json"

BASE_MODEL_ID = "meta-llama/llama-3.1-8b-instruct"  # OpenRouter model ID
JUDGE_MODEL   = "openai/gpt-5.4-mini"
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.7

TRAIT_META = {
    "power-seeking":        ("power-seeking_eval.yaml",        "power_seeking_score"),
    "resource-acquisition": ("resource-acquisition_eval.yaml", "resource_acquisition_score"),
    "reward-hacking":       ("reward-hacking_eval.yaml",       "reward_hacking_score"),
    "narcissism":           ("narcissism_eval.yaml",           "narcissism_score"),
    "spitefulness":         ("spitefulness_eval.yaml",         "spite_score"),
    "risk-affinity":        ("risk-affinity_eval.yaml",        "risk_seeking_score"),
    "caring-about-animals": ("caring-about-animals_eval.yaml", "animal_care_score"),
    "honest-humble":        ("honest-humble_eval.yaml",        "honesty_humility_score"),
}

TRAIT_KEYS = list(TRAIT_META.keys())


# ---------------------------------------------------------------------------
# Load judge prompts
# ---------------------------------------------------------------------------

def load_judge_prompts() -> dict[str, str]:
    prompts = {}
    for trait, (yaml_file, score_key) in TRAIT_META.items():
        path = EVALS_DIR / trait / yaml_file
        with open(path) as f:
            data = yaml.safe_load(f)
        for entry in data:
            jp = entry.get("judge_prompts")
            if jp and isinstance(jp, dict) and score_key in jp:
                prompts[trait] = jp[score_key]
                break
        if trait not in prompts:
            raise ValueError(f"Could not find judge prompt for {trait}/{score_key}")
    return prompts


# ---------------------------------------------------------------------------
# Generate responses via OpenRouter API
# ---------------------------------------------------------------------------

def make_openrouter_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


def generate_response(client, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=BASE_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=max_new_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------

def call_judge(client, judge_prompt_template: str, question: str, answer: str) -> float | None:
    prompt = judge_prompt_template.format(question=question, answer=answer)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=32,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.lower() in ("null", "no information", "n/a", ""):
                return None
            match = re.search(r"\d+", raw)
            return float(match.group()) if match else None
        except Exception as e:
            if attempt == 2:
                print(f"    Judge call failed after 3 attempts: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlations(scores: dict[str, list]) -> None:
    from itertools import combinations
    print("\nCross-metric correlations on BASE model responses:")
    print("(positive = responses that score high on one also score high on the other)")
    print()
    print(f"{'Pair':<55} {'r':>7}  {'n':>5}")
    print("-" * 68)
    for t1, t2 in combinations(TRAIT_KEYS, 2):
        pairs = [(a, b) for a, b in zip(scores[t1], scores[t2])
                 if a is not None and b is not None]
        if len(pairs) < 5:
            print(f"{t1} ↔ {t2:<30} {'n/a (too few scores)':>7}")
            continue
        v1, v2 = zip(*pairs)
        r = np.corrcoef(v1, v2)[0, 1]
        tag = ""
        dark = {"power-seeking", "resource-acquisition", "reward-hacking",
                "narcissism", "spitefulness", "risk-affinity"}
        if t1 in dark and t2 in dark:
            tag = " ← dark-dark"
        print(f"{t1:<30} ↔ {t2:<24} r={r:+.3f}  n={len(pairs)}{tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="Limit prompts for a quick test run")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                    help="Max tokens to generate per response")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--responses-only", action="store_true",
                    help="Generate responses without scoring (saves responses JSON)")
    ap.add_argument("--responses-file", type=Path, default=None,
                    help="Load pre-generated responses from JSON instead of running the model")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts")

    # Step 1: generate responses
    if args.responses_file:
        with open(args.responses_file) as f:
            responses = json.load(f)
        print(f"Loaded {len(responses)} pre-generated responses from {args.responses_file}")
    else:
        client = make_openrouter_client()
        responses = []
        for entry in tqdm(prompts, desc="Generating", unit="prompt"):
            resp = generate_response(client, entry["question"],
                                     max_new_tokens=args.max_new_tokens)
            responses.append({"id": entry["id"], "trait": entry["trait"],
                               "question": entry["question"], "response": resp})
            if len(responses) % 10 == 0:
                tmp = args.out.with_suffix(".responses.json")
                with open(tmp, "w") as f:
                    json.dump(responses, f, indent=2)

        resp_path = args.out.with_suffix(".responses.json")
        with open(resp_path, "w") as f:
            json.dump(responses, f, indent=2)
        print(f"Responses saved to {resp_path}")

    if args.responses_only:
        return

    # Step 2: score with judges
    client = make_openrouter_client()
    judge_prompts = load_judge_prompts()

    scores: dict[str, list] = {t: [] for t in TRAIT_KEYS}
    for entry in tqdm(responses, desc="Scoring", unit="response"):
        for trait in tqdm(TRAIT_KEYS, desc="  traits", leave=False):
            score = call_judge(
                client, judge_prompts[trait],
                entry["question"], entry["response"]
            )
            scores[trait].append(score)

    # Step 3: save results
    results = {"responses": responses, "scores": scores,
               "prompts_file": str(PROMPTS_FILE)}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")

    compute_correlations(scores)


if __name__ == "__main__":
    main()
