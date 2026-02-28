#!/usr/bin/env python3
"""
Config-driven stratified sampling for propensity audit.

Usage:
    python propensity_audit/sample_for_review.py --config configs/risk_affinity.yaml --data scored.csv

Produces:
    {output_dir}/sample_{N}.csv       — full sample with original scores
    {output_dir}/sample_{N}_blind.csv — scores removed (for human annotation)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from audit_config import from_yaml


def load_data(config) -> pd.DataFrame:
    """Load the scored dataset and validate required columns."""
    print("Loading dataset...")
    df = pd.read_csv(config.data_path, low_memory=False)
    print(f"  Total rows: {len(df):,}")

    if config.score_column not in df.columns:
        raise ValueError(
            f"Score column '{config.score_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    # Keep only rows with valid scores
    df = df[df[config.score_column].notna()].copy()
    print(f"  Rows with {config.score_column} scores: {len(df):,}")

    # Validate stratification columns exist
    for col in config.stratify_by:
        if col not in df.columns:
            print(f"  WARNING: stratify_by column '{col}' not found, skipping")

    return df


def stratified_sample(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Stratified sample using config-driven bins and stratification.

    Bins are defined in config.sampling_bins with score ranges and target percentages.
    Within each bin, samples are balanced across stratify_by groups.
    """
    rng = np.random.RandomState(config.seed)
    score_col = config.score_column

    # Build bin edges and labels from config
    bin_edges = []
    bin_labels = []
    bin_targets = {}

    for sbin in config.sampling_bins:
        lo, hi = sbin.range
        if not bin_edges:
            bin_edges.append(lo)
        bin_edges.append(hi)
        bin_labels.append(sbin.name)
        bin_targets[sbin.name] = int(round(sbin.target_pct * config.target_n))

    # Assign score bins
    bins = pd.cut(
        df[score_col],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
    )
    df = df.copy()
    df["score_bin"] = bins

    # Determine stratification columns (only those present in data)
    strat_cols = [c for c in config.stratify_by if c in df.columns]

    samples = []
    for bin_name, n_target in bin_targets.items():
        pool = df[df["score_bin"] == bin_name]
        if len(pool) == 0:
            print(f"  WARNING: No rows in bin '{bin_name}'")
            continue

        if strat_cols:
            # Balance across stratification groups
            groups = pool.groupby(strat_cols, observed=True)
            n_groups = len(groups)

            if n_groups == 0:
                continue

            per_group_base = max(1, n_target // n_groups)
            remainder = n_target - per_group_base * n_groups

            bin_samples = []
            for i, (gkey, gdf) in enumerate(groups):
                n_sample = per_group_base + (1 if i < remainder else 0)
                n_sample = min(n_sample, len(gdf))
                sampled = gdf.sample(n=n_sample, random_state=rng)
                bin_samples.append(sampled)

            bin_df = pd.concat(bin_samples, ignore_index=True)
        else:
            # No stratification — sample directly from pool
            n_sample = min(n_target, len(pool))
            bin_df = pool.sample(n=n_sample, random_state=rng)

        # Top-up if short
        if len(bin_df) < n_target:
            already_idx = set(bin_df.index)
            remaining_pool = pool[~pool.index.isin(already_idx)]
            n_extra = min(n_target - len(bin_df), len(remaining_pool))
            if n_extra > 0:
                extra = remaining_pool.sample(n=n_extra, random_state=rng)
                bin_df = pd.concat([bin_df, extra], ignore_index=True)

        # Trim if over
        if len(bin_df) > n_target:
            bin_df = bin_df.sample(n=n_target, random_state=rng)

        print(f"  Bin '{bin_name}': {len(bin_df)} rows (target {n_target})")
        samples.append(bin_df)

    result = pd.concat(samples, ignore_index=True)
    # Shuffle final order
    result = result.sample(frac=1, random_state=rng).reset_index(drop=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Stratified sampling for propensity audit")
    parser.add_argument("--config", required=True, help="Path to audit config YAML")
    parser.add_argument("--data", default=None, help="Override: path to scored CSV")
    parser.add_argument("--output-dir", default=None, help="Override: output directory")
    args = parser.parse_args()

    config = from_yaml(args.config, data_path=args.data, output_dir=args.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"PROPENSITY AUDIT: Stratified Sampling — {config.display_name}")
    print("=" * 60)

    df = load_data(config)

    print(f"\nSampling {config.target_n} rows...")
    sample = stratified_sample(df, config)

    print(f"\nFinal sample: {len(sample)} rows")
    print(f"\nScore distribution:")
    print(sample["score_bin"].value_counts().sort_index())

    for col in config.stratify_by:
        if col in sample.columns:
            print(f"\n{col} distribution:")
            print(sample[col].value_counts())

    # --- Full sample (with scores) ---
    full_path = config.output_dir / f"sample_{len(sample)}.csv"
    sample.to_csv(full_path, index=False)
    print(f"\nSaved full sample: {full_path}")

    # --- Blind sample (no scores, for human annotation) ---
    blind_cols = ["question", "response"] + [
        c for c in config.metadata_columns if c in sample.columns
    ]
    blind = sample[blind_cols].copy()
    blind["human_label"] = ""  # placeholder for annotation
    blind_path = config.output_dir / f"sample_{len(sample)}_blind.csv"
    blind.to_csv(blind_path, index=False)
    print(f"Saved blind sample: {blind_path}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total sampled: {len(sample)}")
    print(f"Unique questions: {sample['question'].nunique()}")
    score_vals = sample[config.score_column]
    print(f"{config.score_column} range: {score_vals.min():.0f} - {score_vals.max():.0f}")
    print(f"Mean {config.score_column}: {score_vals.mean():.1f}")


if __name__ == "__main__":
    main()
