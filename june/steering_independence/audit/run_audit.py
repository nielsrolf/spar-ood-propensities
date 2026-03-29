#!/usr/bin/env python3
"""
Run alternative judges and analysis on audit samples.

Usage:
    python audit/run_audit.py [--audit trait|coherence|both]

Requires API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY

Produces (in outputs/audit/{trait,coherence}/):
    alt_judge_scores.csv    — sample with alt-judge scores
    audit_summary.csv       — pass/marginal/fail table
    correlations.csv        — inter-judge correlations
"""

import argparse
import os
import re
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from tqdm.auto import tqdm

_HERE = Path(__file__).resolve().parent
_STEERING = _HERE.parent
_JUNE = _STEERING.parent
sys.path.insert(0, str(_JUNE / "propensity_audit"))

from analyze import (
    gwets_ac2,
    inter_judge_correlations,
    bias_probes,
    score_to_bins,
)


def _load_config():
    with open(_HERE / "audit_config.yaml") as f:
        return yaml.safe_load(f)


def _init_client(provider: str):
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


def _judge_score(client, question: str, response: str, prompt_template: str,
                 model: str, provider: str) -> float:
    """Get 0-100 score from a judge model."""
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

    match = re.search(r'\d+', text)
    if match:
        return float(min(max(int(match.group()), 0), 100))
    raise ValueError(f"Could not parse score from: {text!r}")


def _load_judge_prompt(target_trait: str, audit_type: str) -> str:
    """Load the appropriate judge prompt for a row."""
    if audit_type == "coherence":
        # Import from behavioral_steering
        sys.path.insert(0, str(_STEERING))
        from behavioral_steering import COHERENCE_PROMPT
        return COHERENCE_PROMPT
    else:
        sys.path.insert(0, str(_STEERING))
        from trait_registry import get_trait_spec
        spec = get_trait_spec(target_trait)
        return spec.judge_prompt


def run_alt_judges(df: pd.DataFrame, audit_cfg: dict, audit_type: str,
                   out_dir: Path, checkpoint_interval: int = 50) -> pd.DataFrame:
    """Run alt judges on the sample."""
    judges = audit_cfg.get("alt_judges", [])
    if not judges:
        print("  No alt judges configured.")
        return df

    clients = {}
    for judge in judges:
        prov = judge["provider"]
        if prov not in clients:
            clients[prov] = _init_client(prov)

    n = len(df)
    score_cols = {j["name"]: f"{j['name'].replace('-', '_').replace('.', '')}_score"
                  for j in judges}
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

    # Cache judge prompts by target trait
    prompt_cache = {}

    for i in tqdm(range(start_idx, n), initial=start_idx, total=n, desc="Alt judging"):
        row = df.iloc[i]
        q = str(row["question"])
        r = str(row["response"])
        target = row.get("target", "")

        # Get judge prompt (cached)
        if target not in prompt_cache:
            try:
                prompt_cache[target] = _load_judge_prompt(target, audit_type)
            except Exception:
                prompt_cache[target] = None
        prompt = prompt_cache[target]
        if prompt is None:
            continue

        for judge in judges:
            col = score_cols[judge["name"]]
            client = clients[judge["provider"]]
            try:
                results[col][i] = _judge_score(
                    client, q, r, prompt,
                    judge["model_id"], judge["provider"],
                )
            except Exception as e:
                print(f"\n  {judge['name']} error row {i}: {e}")
                results[col][i] = np.nan

        if (i + 1) % checkpoint_interval == 0:
            _save_checkpoint(df, results, checkpoint_path)

    _save_checkpoint(df, results, checkpoint_path)

    out = df.copy()
    for col, vals in results.items():
        out[col] = vals
    return out


def _save_checkpoint(df, results, path):
    out = df.copy()
    for col, vals in results.items():
        out[col] = vals
    out.to_csv(path, index=False)


def run_analysis(df: pd.DataFrame, audit_cfg: dict, score_column: str,
                 out_dir: Path) -> pd.DataFrame:
    """Run inter-judge agreement and bias analysis."""
    alt_cols = [c for c in df.columns
                if c.endswith("_score") and c != score_column and c != "coherence_score"]

    if not alt_cols:
        print("  No alt judge scores found for analysis.")
        return pd.DataFrame()

    all_score_cols = [score_column] + alt_cols

    # Inter-judge correlations
    corr_df = inter_judge_correlations(df, all_score_cols)
    corr_df.to_csv(out_dir / "correlations.csv", index=False)
    print("\n  Inter-judge correlations:")
    print(corr_df.to_string(index=False))

    # Bias probes
    group_cols = [c for c in ["source_type", "target", "target_label"] if c in df.columns]
    bias = bias_probes(df, score_column, group_cols)

    # Length bias
    if "length_correlation" in bias:
        lc = bias["length_correlation"]
        print(f"\n  Length-score correlation: r={lc['r']:.3f} (p={lc['p']:.2e})")

    # Build summary table
    sampling = audit_cfg["sampling"]
    bin_edges = [b["range"][0] for b in sampling["bins"]] + [sampling["bins"][-1]["range"][1]]
    bin_labels = [b["name"] for b in sampling["bins"]]

    summary_rows = []

    original_binned = score_to_bins(df[score_column], bin_edges, bin_labels)
    for col in alt_cols:
        alt_binned = score_to_bins(df[col], bin_edges, bin_labels)
        valid = [(a, b) for a, b in zip(original_binned, alt_binned)
                 if not (pd.isna(a) or pd.isna(b))]
        if len(valid) >= 10:
            a_vals, b_vals = zip(*valid)
            ac2 = gwets_ac2(a_vals, b_vals, bin_labels)
            status = "PASS" if ac2 > 0.6 else ("MARGINAL" if ac2 > 0.4 else "FAIL")
            summary_rows.append({
                "Metric": f"AC2 {score_column} vs {col}",
                "Value": f"{ac2:.3f}",
                "Threshold": ">0.6",
                "Status": status,
            })

        # Pearson
        valid_mask = df[col].notna() & df[score_column].notna()
        if valid_mask.sum() >= 10:
            corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, score_column])
            status = "PASS" if corr > 0.7 else ("MARGINAL" if corr > 0.5 else "FAIL")
            summary_rows.append({
                "Metric": f"Pearson {score_column} vs {col}",
                "Value": f"{corr:.3f}",
                "Threshold": ">0.7",
                "Status": status,
            })

    # Length bias
    if "response" in df.columns:
        lengths = df["response"].astype(str).str.len()
        valid_mask = df[score_column].notna()
        if valid_mask.sum() >= 10:
            r, p = stats.pearsonr(lengths[valid_mask], df.loc[valid_mask, score_column])
            status = "PASS" if abs(r) < 0.15 else ("MARGINAL" if abs(r) < 0.25 else "FAIL")
            summary_rows.append({
                "Metric": "Length-score correlation",
                "Value": f"r={r:.3f}",
                "Threshold": "|r|<0.15",
                "Status": status,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "audit_summary.csv", index=False)
    print("\n  Audit summary:")
    print(summary_df.to_string(index=False))

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Run judge audit for steering independence")
    parser.add_argument("--audit", default="both", choices=["trait", "coherence", "both"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = _load_config()
    output_dir = Path(args.output_dir) if args.output_dir else _STEERING / "outputs"
    audit_base = output_dir / "audit"

    if args.audit in ("trait", "both"):
        print("=" * 60)
        print("TRAIT SCORE AUDIT — Alt Judges + Analysis")
        print("=" * 60)
        trait_dir = audit_base / "trait"
        sample_files = sorted(trait_dir.glob("sample_*.csv"))
        sample_files = [f for f in sample_files if "_blind" not in f.name]
        if not sample_files:
            print("No sample found. Run prepare_audit.py first.")
        else:
            df = pd.read_csv(sample_files[-1])
            print(f"Loaded {len(df)} rows from {sample_files[-1]}")

            scored = run_alt_judges(df, config["trait_audit"], "trait", trait_dir)
            scored.to_csv(trait_dir / "alt_judge_scores.csv", index=False)

            run_analysis(scored, config["trait_audit"], "score", trait_dir)

    if args.audit in ("coherence", "both"):
        print("\n" + "=" * 60)
        print("COHERENCE SCORE AUDIT — Alt Judges + Analysis")
        print("=" * 60)
        coh_dir = audit_base / "coherence"
        sample_files = sorted(coh_dir.glob("sample_*.csv"))
        sample_files = [f for f in sample_files if "_blind" not in f.name]
        if not sample_files:
            print("No sample found. Run prepare_audit.py first.")
        else:
            df = pd.read_csv(sample_files[-1])
            print(f"Loaded {len(df)} rows from {sample_files[-1]}")

            scored = run_alt_judges(df, config["coherence_audit"], "coherence", coh_dir)
            scored.to_csv(coh_dir / "alt_judge_scores.csv", index=False)

            run_analysis(scored, config["coherence_audit"], "coherence_score", coh_dir)


if __name__ == "__main__":
    main()
