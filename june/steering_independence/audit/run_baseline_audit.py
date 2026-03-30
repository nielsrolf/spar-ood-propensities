#!/usr/bin/env python3
"""
Quick baseline audit: generate unsteered responses, judge them with alt judges,
and check per-trait inter-judge agreement before committing to the full steering run.

Usage:
    python audit/run_baseline_audit.py [--config config_llama8b.yaml] [--samples-per-trait 50]

Pipeline:
    1. Generate baseline responses (loads model, then frees it)
    2. Judge with primary judge (GPT-4o-mini via logprobs)
    3. Sample for audit
    4. Run alt judges (GPT-4.1-mini, Claude Sonnet)
    5. Print per-trait Pearson r
"""

import argparse
import asyncio
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy import stats
from tqdm.auto import tqdm

_HERE = Path(__file__).resolve().parent
_STEERING = _HERE.parent
sys.path.insert(0, str(_STEERING))

from trait_registry import ALL_TRAITS, LABELS, get_trait_spec, load_test_questions
from behavioral_steering import (
    generate_baseline_only, _load_jsonl,
    _judge_single, _aggregate_0_100_score,
)
from cache import JudgeCache


# ── Judge baseline with primary judge ────────────────────────────────

async def judge_baseline(config: dict) -> pd.DataFrame:
    """Judge all baseline responses with the primary judge."""
    from openai import AsyncOpenAI
    import math

    output_dir = Path(config["output_dir"])
    gen_dir = output_dir / "generations"
    score_dir = output_dir / "judge_scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    traits = config.get("traits") or ALL_TRAITS
    judge_cfg = config.get("judge", {})
    judge_model = judge_cfg.get("model", "gpt-4o-mini")
    concurrency = judge_cfg.get("concurrency", 20)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("OPENROUTER_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    sem = asyncio.Semaphore(concurrency)
    cache = JudgeCache(str(output_dir / "judge_cache.db"))

    rows = []
    for target in tqdm(traits, desc="Judging baseline"):
        gen_path = gen_dir / f"baseline_to_{target}.jsonl"
        if not gen_path.exists():
            print(f"  SKIP {target}: no baseline file")
            continue

        spec = get_trait_spec(target)
        records = _load_jsonl(gen_path)
        score_path = score_dir / f"baseline_to_{target}.jsonl"

        # Check for existing scores
        existing = {}
        if score_path.exists():
            for rec in _load_jsonl(score_path):
                existing[rec["id"]] = rec

        scored = []
        for rec in records:
            if rec["id"] in existing:
                scored.append(existing[rec["id"]])
                continue

            cache_key = JudgeCache.make_key(
                model_id=config["model_id"],
                question_id=rec["id"],
                source_trait="baseline",
                target_trait=target,
                alpha=0.0,
                layer=0,
                metric=spec.primary_metric,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                score = cached
            else:
                score = await _judge_single(
                    client, spec.judge_prompt, rec["question"], rec["response"],
                    sem, judge_model,
                )
                if score is not None:
                    cache.put(cache_key, score)

            rec_out = {**rec, "score": score}
            scored.append(rec_out)

        # Save scored JSONL
        with open(score_path, "w") as f:
            for s in scored:
                f.write(json.dumps(s) + "\n")

        for s in scored:
            if s.get("score") is not None:
                rows.append({
                    "question_id": s["id"],
                    "question": s["question"],
                    "response": s["response"],
                    "score": s["score"],
                    "target": target,
                })

    return pd.DataFrame(rows)


# ── Alt judge + analysis ─────────────────────────────────────────────

ALT_JUDGES = [
    {"name": "gpt-4.1-mini", "model_id": "openai/gpt-4.1-mini"},
    {"name": "claude-sonnet", "model_id": "anthropic/claude-sonnet-4"},
]


def init_openrouter():
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("OPENROUTER_API_KEY")
        except Exception:
            pass
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def alt_judge_score(client, question, response, prompt_template, model):
    prompt = prompt_template.format(question=question, response=response, answer=response)
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.0,
    )
    text = result.choices[0].message.content.strip()
    match = re.search(r'\d+', text)
    if match:
        return float(min(max(int(match.group()), 0), 100))
    raise ValueError(f"Could not parse score from: {text!r}")


def run_alt_judges_and_analyze(df: pd.DataFrame, out_dir: Path,
                                samples_per_trait: int = 50, seed: int = 42):
    """Sample, run alt judges, and print per-trait agreement."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stratified sample
    rng = np.random.RandomState(seed)
    samples = []
    for trait, g in df.groupby("target"):
        n = min(samples_per_trait, len(g))
        samples.append(g.sample(n=n, random_state=rng))
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"\nSampled {len(sample)} rows for audit:")
    for trait, g in sorted(sample.groupby("target")):
        print(f"  {trait}: {len(g)}")

    sample.to_csv(out_dir / "baseline_sample.csv", index=False)

    # Run alt judges
    client = init_openrouter()
    prompt_cache = {}
    n = len(sample)

    score_cols = {}
    for j in ALT_JUDGES:
        col = j["name"].replace("-", "_").replace(".", "") + "_score"
        score_cols[j["name"]] = col

    results = {col: [np.nan] * n for col in score_cols.values()}

    for i in tqdm(range(n), desc="Alt judging"):
        row = sample.iloc[i]
        target = row["target"]

        if target not in prompt_cache:
            try:
                spec = get_trait_spec(target)
                prompt_cache[target] = spec.judge_prompt
            except Exception:
                prompt_cache[target] = None

        prompt = prompt_cache[target]
        if prompt is None:
            continue

        for judge in ALT_JUDGES:
            col = score_cols[judge["name"]]
            try:
                results[col][i] = alt_judge_score(
                    client, str(row["question"]), str(row["response"]),
                    prompt, judge["model_id"],
                )
            except Exception as e:
                print(f"\n  {judge['name']} error row {i}: {e}")

    scored = sample.copy()
    for col, vals in results.items():
        scored[col] = vals
    scored.to_csv(out_dir / "baseline_alt_scores.csv", index=False)

    # Per-trait analysis
    alt_cols = list(score_cols.values())
    print(f"\n{'Eval':<24} {'N':>3}  ", end="")
    for col in alt_cols:
        print(f"{'r(' + col.replace('_score','') + ')':>14}", end="")
    print()
    print("-" * (28 + 14 * len(alt_cols)))

    trait_rows = []
    for trait, g in sorted(scored.groupby("target")):
        row = {"trait": trait, "n": len(g)}
        line = f"{trait:<24} {len(g):>3}  "
        for col in alt_cols:
            mask = g[col].notna() & g["score"].notna()
            if mask.sum() >= 5:
                r, p = stats.pearsonr(g.loc[mask, "score"], g.loc[mask, col])
                row[f"r_{col}"] = round(r, 3)
                row[f"p_{col}"] = round(p, 4)
                line += f"{r:>14.3f}"
            else:
                row[f"r_{col}"] = np.nan
                row[f"p_{col}"] = np.nan
                line += f"{'nan':>14}"
        trait_rows.append(row)
        print(line)

    trait_df = pd.DataFrame(trait_rows)
    trait_df.to_csv(out_dir / "baseline_per_trait_correlations.csv", index=False)

    # Aggregate
    print(f"\nAggregate:")
    for col in alt_cols:
        mask = scored[col].notna() & scored["score"].notna()
        if mask.sum() >= 10:
            r, p = stats.pearsonr(scored.loc[mask, "score"], scored.loc[mask, col])
            print(f"  score vs {col}: r={r:.3f} (n={mask.sum()})")

    return trait_df


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline judge audit")
    parser.add_argument("--config", default="config_llama8b.yaml")
    parser.add_argument("--samples-per-trait", type=int, default=50)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, assume baseline files exist")
    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENROUTER_API_KEY"] = args.api_key

    config_path = _STEERING / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["output_dir"]) / "audit" / "baseline"

    # Step 1: Generate baseline
    if not args.skip_generation:
        print("Step 1: Generating baseline responses...")
        generate_baseline_only(config)
    else:
        print("Step 1: Skipping generation (--skip-generation)")

    # Step 2: Judge with primary judge
    print("\nStep 2: Judging baseline with primary judge...")
    df = asyncio.run(judge_baseline(config))
    print(f"  Scored {len(df)} responses across {df['target'].nunique()} traits")

    # Step 3: Alt judges + analysis
    print("\nStep 3: Running alt judges...")
    run_alt_judges_and_analyze(df, out_dir, args.samples_per_trait)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
