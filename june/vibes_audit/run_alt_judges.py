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
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from .audit_config import from_yaml, AuditConfig
except ImportError:
    from audit_config import from_yaml, AuditConfig


def find_sample_csv(config: AuditConfig) -> Path:
    """Find the full sample CSV (with scores) in the output directory."""
    candidates = sorted(config.output_dir.glob("sample_*.csv"))
    # Exclude blind CSVs
    candidates = [c for c in candidates if "_blind" not in c.name]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No sample_*.csv found in {config.output_dir}. "
        "Run sample_for_review.py first."
    )


# ── Generic judge function ──────────────────────────────────────────

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
def judge_score(client, question: str, response: str, prompt_template: str,
                model: str, provider: str) -> float:
    """
    Get 0-100 score from any supported provider.

    Args:
        client: OpenAI or Anthropic client instance.
        question: The question text.
        response: The response text.
        prompt_template: Prompt with {question} and {response} placeholders.
        model: Model ID.
        provider: "openai" or "anthropic".
    """
    # Fill template — support both {response} and {answer}
    prompt = prompt_template.format(question=question, response=response, answer=response)

    if provider == "openai":
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        text = result.choices[0].message.content.strip()
    elif provider == "anthropic":
        result = client.messages.create(
            model=model,
            max_tokens=5,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content[0].text.strip()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Parse numeric score
    # Strip any non-numeric prefix/suffix
    import re
    match = re.search(r'\d+', text)
    if match:
        score = int(match.group())
        return float(min(max(score, 0), 100))
    raise ValueError(f"Could not parse score from: {text!r}")


# ── Main pipeline ───────────────────────────────────────────────────

def _init_client(provider: str):
    """Initialize API client for a provider."""
    if provider == "openai":
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAI(api_key=key)
    elif provider == "anthropic":
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return anthropic.Anthropic(api_key=key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_judges(df: pd.DataFrame, config: AuditConfig,
               checkpoint_interval: int = 50) -> pd.DataFrame:
    """Run all configured alternative judges on the sample."""

    # Initialize clients (deduplicate by provider)
    clients = {}
    for judge in config.alt_judges:
        prov = judge["provider"]
        if prov not in clients:
            clients[prov] = _init_client(prov)

    n = len(df)
    # Column name for each judge: "{name}_score"
    score_cols = {j["name"]: f"{j['name'].replace('-', '_').replace('.', '')}_score"
                  for j in config.alt_judges}

    # Initialize result columns
    results = {col: [np.nan] * n for col in score_cols.values()}

    # Check for existing checkpoint
    checkpoint_path = config.output_dir / "alt_judge_checkpoint.csv"
    start_idx = 0
    if checkpoint_path.exists():
        ckpt = pd.read_csv(checkpoint_path)
        if len(ckpt) == n:
            for col in results:
                if col in ckpt.columns:
                    results[col] = ckpt[col].tolist()
            # Find first incomplete row
            first_col = list(results.keys())[0]
            for i in range(n):
                if pd.isna(results[first_col][i]):
                    start_idx = i
                    break
            else:
                start_idx = n
            print(f"Resuming from checkpoint at row {start_idx}")

    print(f"\nRunning judges on {n} rows (starting at {start_idx})...")
    for judge in config.alt_judges:
        print(f"  {judge['name']} ({judge['provider']}: {judge['model_id']})")

    for i in tqdm(range(start_idx, n), initial=start_idx, total=n, desc="Judging"):
        row = df.iloc[i]
        q = str(row["question"])
        r = str(row["response"])

        for judge in config.alt_judges:
            col = score_cols[judge["name"]]
            client = clients[judge["provider"]]
            try:
                results[col][i] = judge_score(
                    client, q, r,
                    config.judge_prompt_template,
                    judge["model_id"],
                    judge["provider"],
                )
            except Exception as e:
                print(f"\n  {judge['name']} error row {i}: {e}")
                results[col][i] = np.nan

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            _save_checkpoint(df, results, checkpoint_path)
            print(f"  Checkpoint saved at row {i + 1}")

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
    parser = argparse.ArgumentParser(description="Run alternative judges for propensity audit")
    parser.add_argument("--config", required=True, help="Path to audit config YAML")
    parser.add_argument("--output-dir", default=None, help="Override: output directory")
    args = parser.parse_args()

    config = from_yaml(args.config, output_dir=args.output_dir)

    print("=" * 60)
    print(f"PROPENSITY AUDIT: Alternative Judges — {config.display_name}")
    print("=" * 60)

    sample_path = find_sample_csv(config)
    df = pd.read_csv(sample_path)
    print(f"Loaded sample: {len(df)} rows from {sample_path}")

    result = run_judges(df, config)

    # Save combined scores
    alt_path = config.output_dir / "alt_judge_scores.csv"
    result.to_csv(alt_path, index=False)
    print(f"\nSaved: {alt_path}")

    # Print summary
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

    # Quick agreement check
    if config.score_column in result.columns:
        for col in score_cols:
            valid = result[col].notna() & result[config.score_column].notna()
            if valid.sum() > 0:
                corr = result.loc[valid, col].corr(result.loc[valid, config.score_column])
                print(f"\nCorrelation {config.score_column} vs {col}: {corr:.3f}")


if __name__ == "__main__":
    main()
