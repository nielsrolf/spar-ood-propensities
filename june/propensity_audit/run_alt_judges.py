#!/usr/bin/env python3
"""
Config-driven alternative judge runner for propensity audit.

Usage:
    python propensity_audit/run_alt_judges.py --config configs/risk_affinity.yaml

Runs the judge prompt from config on each alternative judge model,
producing 0-100 scores. No categorical prompt — human does categorization.

Produces:
    {output_dir}/alt_judge_scores.csv
"""

import argparse
import asyncio
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import load_dotenv

from audit_config import from_yaml, AuditConfig

CONCURRENCY = 200


def find_sample_csv(config: AuditConfig) -> Path:
    """Find the full sample CSV (with scores) in the output directory."""
    candidates = sorted(config.output_dir.glob("sample_*.csv"))
    candidates = [c for c in candidates if "_blind" not in c.name]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No sample_*.csv found in {config.output_dir}. "
        "Run sample_for_review.py first."
    )


# ── Async judge call with manual retry ──────────────────────────────

async def judge_score(client, question: str, response: str, prompt_template: str,
                      model: str, provider: str) -> float:
    """0-100 score from any supported provider, with bounded retry."""
    prompt = prompt_template.format(question=question, response=response, answer=response)

    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            if provider == "openai":
                kwargs: dict = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                # gpt-5.x / o-series reasoning models: max_completion_tokens, no temperature
                if re.match(r"^(gpt-5|o[1-9])", model):
                    kwargs["max_completion_tokens"] = 1024
                else:
                    kwargs["max_tokens"] = 16
                    kwargs["temperature"] = 0.0
                result = await client.chat.completions.create(**kwargs)
                text = result.choices[0].message.content.strip()
            elif provider == "anthropic":
                kwargs = {
                    "model": model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": prompt}],
                }
                # claude-opus-4.x deprecates temperature
                if "opus-4" not in model:
                    kwargs["temperature"] = 0.0
                result = await client.messages.create(**kwargs)
                text = result.content[0].text.strip()
            else:
                raise ValueError(f"Unknown provider: {provider}")

            match = re.search(r"\d+", text)
            if match:
                score = int(match.group())
                return float(min(max(score, 0), 100))
            raise ValueError(f"Could not parse score from: {text!r}")
        except Exception as e:
            last_exc = e
            await asyncio.sleep(min(2 ** attempt + 1, 60))
    raise last_exc  # type: ignore[misc]


def _init_client(provider: str):
    if provider == "openai":
        from openai import AsyncOpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return AsyncOpenAI(api_key=key)
    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return AsyncAnthropic(api_key=key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def run_judges(df: pd.DataFrame, config: AuditConfig,
                     concurrency: int = CONCURRENCY,
                     checkpoint_interval: int = 50) -> pd.DataFrame:
    """Run all configured alternative judges in parallel across rows × judges."""

    clients = {}
    for judge in config.alt_judges:
        prov = judge["provider"]
        if prov not in clients:
            clients[prov] = _init_client(prov)

    n = len(df)
    score_cols = {j["name"]: f"{j['name'].replace('-', '_').replace('.', '')}_score"
                  for j in config.alt_judges}
    results = {col: [np.nan] * n for col in score_cols.values()}

    # Resume from checkpoint if present
    checkpoint_path = config.output_dir / "alt_judge_checkpoint.csv"
    if checkpoint_path.exists():
        ckpt = pd.read_csv(checkpoint_path)
        if len(ckpt) == n:
            for col in results:
                if col in ckpt.columns:
                    results[col] = ckpt[col].tolist()
            done = sum(1 for col in results for v in results[col] if not pd.isna(v))
            print(f"Resuming from checkpoint: {done} cells already scored")

    sem = asyncio.Semaphore(concurrency)

    async def one(i: int, judge: dict, q: str, r: str):
        col = score_cols[judge["name"]]
        client = clients[judge["provider"]]
        async with sem:
            try:
                score = await judge_score(
                    client, q, r,
                    config.judge_prompt_template,
                    judge["model_id"],
                    judge["provider"],
                )
            except Exception as e:
                print(f"\n  {judge['name']} error row {i}: {e}")
                score = float("nan")
            return i, col, score

    print(f"\nRunning judges on {n} rows (concurrency={concurrency})...")
    for judge in config.alt_judges:
        print(f"  {judge['name']} ({judge['provider']}: {judge['model_id']})")

    # Only enqueue cells that aren't already scored
    tasks = []
    for i in range(n):
        row = df.iloc[i]
        q = str(row["question"])
        r = str(row["response"])
        for judge in config.alt_judges:
            col = score_cols[judge["name"]]
            if pd.isna(results[col][i]):
                tasks.append(one(i, judge, q, r))

    if not tasks:
        print("  Nothing to do — all cells already scored.")
        return _build_result_df(df, results)

    pbar = tqdm(total=len(tasks), desc="Judging")
    completed = 0
    for coro in asyncio.as_completed(tasks):
        i, col, score = await coro
        results[col][i] = score
        completed += 1
        pbar.update(1)
        if completed % checkpoint_interval == 0:
            _save_checkpoint(df, results, checkpoint_path)
    pbar.close()

    _save_checkpoint(df, results, checkpoint_path)
    return _build_result_df(df, results)


def _save_checkpoint(df, results, path):
    out = df.copy()
    for col, vals in results.items():
        out[col] = vals
    out.to_csv(path, index=False)


def _build_result_df(df, results):
    out = df.copy()
    for col, vals in results.items():
        out[col] = vals
    return out


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run alternative judges for propensity audit")
    parser.add_argument("--config", required=True, help="Path to audit config YAML")
    parser.add_argument("--output-dir", default=None, help="Override: output directory")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    config = from_yaml(args.config, output_dir=args.output_dir)

    print("=" * 60)
    print(f"PROPENSITY AUDIT: Alternative Judges — {config.display_name}")
    print("=" * 60)

    sample_path = find_sample_csv(config)
    df = pd.read_csv(sample_path)
    print(f"Loaded sample: {len(df)} rows from {sample_path}")

    result = asyncio.run(run_judges(df, config, concurrency=args.concurrency))

    alt_path = config.output_dir / "alt_judge_scores.csv"
    result.to_csv(alt_path, index=False)
    print(f"\nSaved: {alt_path}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    score_cols = [c for c in result.columns if c.endswith("_score") and c != config.score_column]
    for col in score_cols:
        scores = result[col].dropna()
        if len(scores) > 0:
            print(f"\n{col}:")
            print(f"  Mean: {scores.mean():.1f}, Median: {scores.median():.1f}")
            print(f"  Std: {scores.std():.1f}")
            print(f"  NaN count: {result[col].isna().sum()}")

    if config.score_column in result.columns:
        for col in score_cols:
            valid = result[col].notna() & result[config.score_column].notna()
            if valid.sum() > 0:
                corr = result.loc[valid, col].corr(result.loc[valid, config.score_column])
                print(f"\nCorrelation {config.score_column} vs {col}: {corr:.3f}")


if __name__ == "__main__":
    main()
