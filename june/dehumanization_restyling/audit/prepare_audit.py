#!/usr/bin/env python3
"""
Merge control + mechanistic CSVs into per-facet scored datasets for audit.

Usage:
    python audit/prepare_audit.py

Produces:
    audit/facet1_refusal.csv
    audit/facet3_moral_judgment.csv
    audit/facet4_resource_allocation.csv
    audit/facet5a_sentencing.csv
    audit/facet5b_disciplinary.csv
    audit/facet5c_third_party_punishment.csv
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "output"

# Primary judge metric per facet
FACET_PRIMARY_METRIC = {
    "facet1_refusal": "refusal_score",
    "facet3_moral_judgment": "utilitarian_score",
    "facet4_resource_allocation": "allocation_favors_group_member",
    "facet5a_sentencing": "sentence_magnitude",
    "facet5b_disciplinary": "action_severity",
    "facet5c_third_party_punishment": "consequence_severity",
}


def main():
    print("=" * 60)
    print("DEHUMANIZATION RESTYLING: Prepare Audit Data")
    print("=" * 60)

    # Load all CSVs
    frames = []
    for csv_path in sorted(DATA_DIR.glob("*_responses.csv")):
        df = pd.read_csv(csv_path)
        frames.append(df)
        print(f"  {csv_path.name}: {len(df)} rows")

    merged = pd.concat(frames, ignore_index=True)
    print(f"\nTotal merged: {len(merged)} rows")

    # Rename answer → response for audit framework
    if "answer" in merged.columns and "response" not in merged.columns:
        merged = merged.rename(columns={"answer": "response"})

    # Write per-facet CSVs
    for facet, metric in FACET_PRIMARY_METRIC.items():
        facet_df = merged[merged["facet"] == facet].copy()
        if facet_df.empty:
            print(f"  WARNING: no rows for {facet}")
            continue

        out_path = OUTPUT_DIR / f"{facet}.csv"
        facet_df.to_csv(out_path, index=False)
        scores = facet_df[metric]
        print(
            f"\n  {facet}: {len(facet_df)} rows, "
            f"{metric} mean={scores.mean():.1f} "
            f"[{scores.min():.0f}–{scores.max():.0f}]"
        )
        print(f"    conditions: {facet_df['condition'].value_counts().to_dict()}")
        print(f"    groups: {facet_df['group'].value_counts().to_dict()}")
        print(f"    → {out_path}")


if __name__ == "__main__":
    main()
