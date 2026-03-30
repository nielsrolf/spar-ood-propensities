#!/usr/bin/env python3
"""
Cross-model judge audit: run alt judges on niels/propensities run_all data
to check whether inter-judge agreement issues are eval-wide or Qwen-specific.

Pools responses from multiple frontier models, samples ~50 per eval,
runs the same alt judges, and compares per-trait Pearson r with the
steering independence results.

Usage:
    python audit/run_cross_model_audit.py [--samples-per-eval 50] [--seed 42]

Produces: outputs/audit/cross_model/
    sample.csv              — sampled rows with original scores
    alt_judge_scores.csv    — sample with alt-judge columns added
    per_trait_correlations.csv — per-eval Pearson r comparison
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from tqdm.auto import tqdm

_HERE = Path(__file__).resolve().parent
_STEERING = _HERE.parent
_JUNE = _STEERING.parent
_REPO = _JUNE.parent

sys.path.insert(0, str(_STEERING))
from trait_registry import ALL_TRAITS, PRIMARY_METRICS, LABELS, get_trait_spec


# ── Data collection ──────────────────────────────────────────────────

def collect_run_all_data() -> pd.DataFrame:
    """Load all run_all results from niels/propensities/results/."""
    results_root = _REPO / "niels" / "propensities" / "results"
    all_rows = []

    for eval_name in ALL_TRAITS:
        run_all_dir = results_root / eval_name / "run_all"
        if not run_all_dir.exists():
            print(f"  SKIP {eval_name}: no run_all dir")
            continue

        metric = PRIMARY_METRICS[eval_name]

        for csv_path in sorted(run_all_dir.glob("results*.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"  SKIP {csv_path.name}: {e}")
                continue

            if metric not in df.columns or "question" not in df.columns or "answer" not in df.columns:
                continue

            # Only keep rows with valid scores
            df = df[df[metric].notna()].copy()
            if len(df) == 0:
                continue

            model = df["model"].iloc[0] if "model" in df.columns else csv_path.stem

            for _, row in df.iterrows():
                all_rows.append({
                    "eval_name": eval_name,
                    "question": row["question"],
                    "response": row["answer"],
                    "score": row[metric],
                    "model": model,
                    "question_id": row.get("question_id", ""),
                })

    return pd.DataFrame(all_rows)


def stratified_sample(df: pd.DataFrame, n_per_eval: int, seed: int) -> pd.DataFrame:
    """Sample n rows per eval, balanced across models."""
    rng = np.random.RandomState(seed)
    samples = []

    for eval_name, group in df.groupby("eval_name"):
        models = group["model"].unique()
        per_model = max(1, n_per_eval // len(models))

        eval_sample = []
        for model, mgroup in group.groupby("model"):
            n = min(per_model, len(mgroup))
            eval_sample.append(mgroup.sample(n=n, random_state=rng))

        combined = pd.concat(eval_sample)
        # Trim to target if we oversampled
        if len(combined) > n_per_eval:
            combined = combined.sample(n=n_per_eval, random_state=rng)
        samples.append(combined)

    return pd.concat(samples).reset_index(drop=True)


# ── Alt judge scoring ────────────────────────────────────────────────

ALT_JUDGES = [
    {"name": "gpt-4.1-mini", "model_id": "openai/gpt-4.1-mini"},
    {"name": "claude-sonnet", "model_id": "anthropic/claude-sonnet-4"},
]


def init_client():
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


def judge_score(client, question, response, prompt_template, model):
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


def run_alt_judges(df: pd.DataFrame, out_dir: Path,
                   checkpoint_interval: int = 50) -> pd.DataFrame:
    client = init_client()
    n = len(df)

    score_cols = {}
    for j in ALT_JUDGES:
        col = j["name"].replace("-", "_").replace(".", "") + "_score"
        score_cols[j["name"]] = col

    results = {col: [np.nan] * n for col in score_cols.values()}

    # Resume from checkpoint
    checkpoint_path = out_dir / "alt_judge_checkpoint.csv"
    start_idx = 0
    if checkpoint_path.exists():
        ckpt = pd.read_csv(checkpoint_path)
        if len(ckpt) == n:
            for col in results:
                if col in ckpt.columns:
                    results[col] = ckpt[col].tolist()
            first_col = list(results.keys())[0]
            for i in range(n):
                if pd.isna(results[first_col][i]):
                    start_idx = i
                    break
            else:
                start_idx = n
            print(f"  Resuming from checkpoint at row {start_idx}")

    # Cache judge prompts
    prompt_cache = {}

    for i in tqdm(range(start_idx, n), initial=start_idx, total=n, desc="Alt judging"):
        row = df.iloc[i]
        eval_name = row["eval_name"]

        if eval_name not in prompt_cache:
            try:
                spec = get_trait_spec(eval_name)
                prompt_cache[eval_name] = spec.judge_prompt
            except Exception:
                prompt_cache[eval_name] = None

        prompt = prompt_cache[eval_name]
        if prompt is None:
            continue

        for judge in ALT_JUDGES:
            col = score_cols[judge["name"]]
            try:
                results[col][i] = judge_score(
                    client, str(row["question"]), str(row["response"]),
                    prompt, judge["model_id"],
                )
            except Exception as e:
                print(f"\n  {judge['name']} error row {i}: {e}")
                results[col][i] = np.nan

        if (i + 1) % checkpoint_interval == 0:
            _save(df, results, checkpoint_path)

    _save(df, results, checkpoint_path)

    out = df.copy()
    for col, vals in results.items():
        out[col] = vals
    return out


def _save(df, results, path):
    out = df.copy()
    for col, vals in results.items():
        out[col] = vals
    out.to_csv(path, index=False)


# ── Analysis ─────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, out_dir: Path):
    alt_cols = [c for c in df.columns if c.endswith("_score") and c != "score"]

    # Per-eval correlations
    print("\n  Per-eval inter-judge correlations (cross-model):")
    trait_rows = []
    for eval_name, g in sorted(df.groupby("eval_name")):
        row = {"eval": eval_name, "label": LABELS.get(eval_name, eval_name), "n": len(g)}
        for col in alt_cols:
            mask = g[col].notna() & g["score"].notna()
            if mask.sum() >= 5:
                r, p = stats.pearsonr(g.loc[mask, "score"], g.loc[mask, col])
                row[f"r_{col}"] = round(r, 3)
                row[f"p_{col}"] = round(p, 4)
            else:
                row[f"r_{col}"] = np.nan
                row[f"p_{col}"] = np.nan
        trait_rows.append(row)

    trait_df = pd.DataFrame(trait_rows)
    trait_df.to_csv(out_dir / "per_trait_correlations.csv", index=False)
    print(trait_df.to_string(index=False))

    # Load steering independence per-trait results for comparison if available
    si_path = _STEERING / "outputs" / "audit" / "trait" / "per_trait_correlations.csv"
    if si_path.exists():
        si_df = pd.read_csv(si_path)
        print("\n  Comparison with steering independence audit:")
        print(f"  {'Eval':<24} {'SI r(gpt)':>10} {'CM r(gpt)':>10} {'SI r(claude)':>12} {'CM r(claude)':>12}")
        print("  " + "-" * 70)

        si_lookup = {r["trait"]: r for _, r in si_df.iterrows()} if "trait" in si_df.columns else {}

        for _, row in trait_df.iterrows():
            ev = row["eval"]
            si_row = si_lookup.get(ev, {})

            gpt_col = [c for c in trait_df.columns if c.startswith("r_") and "gpt" in c]
            claude_col = [c for c in trait_df.columns if c.startswith("r_") and "claude" in c]

            si_gpt_col = [c for c in si_df.columns if c.startswith("r_") and "gpt" in c] if si_path.exists() else []
            si_claude_col = [c for c in si_df.columns if c.startswith("r_") and "claude" in c] if si_path.exists() else []

            cm_gpt = row[gpt_col[0]] if gpt_col else np.nan
            cm_claude = row[claude_col[0]] if claude_col else np.nan
            si_gpt = si_row.get(si_gpt_col[0], np.nan) if si_gpt_col and si_row else np.nan
            si_claude = si_row.get(si_claude_col[0], np.nan) if si_claude_col and si_row else np.nan

            print(f"  {ev:<24} {si_gpt:>10.3f} {cm_gpt:>10.3f} {si_claude:>12.3f} {cm_claude:>12.3f}")
    else:
        print("\n  (No steering independence per_trait_correlations.csv found for comparison)")

    # Aggregate
    print(f"\n  Aggregate (all evals pooled):")
    for col in alt_cols:
        mask = df[col].notna() & df["score"].notna()
        if mask.sum() >= 10:
            r, p = stats.pearsonr(df.loc[mask, "score"], df.loc[mask, col])
            print(f"    score vs {col}: r={r:.3f} (p={p:.2e}, n={mask.sum()})")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-model judge audit")
    parser.add_argument("--samples-per-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", default=None, help="OpenRouter API key")
    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENROUTER_API_KEY"] = args.api_key

    out_dir = _STEERING / "outputs" / "audit" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting run_all data from niels/propensities/results/...")
    all_data = collect_run_all_data()
    print(f"  Total: {len(all_data)} scored responses across {all_data['eval_name'].nunique()} evals")
    print(f"  Models: {sorted(all_data['model'].unique())}")

    print(f"\nSampling {args.samples_per_eval} per eval...")
    sample = stratified_sample(all_data, args.samples_per_eval, args.seed)
    print(f"  Sampled {len(sample)} rows:")
    for ev, g in sorted(sample.groupby("eval_name")):
        models = g["model"].nunique()
        print(f"    {ev}: {len(g)} rows, {models} models")

    sample.to_csv(out_dir / "sample.csv", index=False)

    print("\nRunning alt judges...")
    scored = run_alt_judges(sample, out_dir)
    scored.to_csv(out_dir / "alt_judge_scores.csv", index=False)

    analyze(scored, out_dir)


if __name__ == "__main__":
    main()
