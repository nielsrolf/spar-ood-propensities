"""
Drop rows from stage1's cross_scores.csv for any propensity whose judge
prompt was revised in stage 3b. The next `stage 1` invocation will then
recompute those rows against the new prompts.

Usage:
    niels/.venv/bin/python niels/experiments/orthogonalize/invalidate_scores.py \
        --cross-scores <output>/cross-scores/cross_scores.csv \
        --stage3b-summary <output>/stage3b_logs/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross-scores", required=True)
    ap.add_argument("--stage3b-summary", required=True)
    args = ap.parse_args()

    summary = json.loads(Path(args.stage3b_summary).read_text())
    accepted = [s["prop"] for s in summary if s.get("decision") == "accepted"]
    if not accepted:
        print("No revised judges in stage 3b — leaving cross_scores.csv unchanged.")
        return

    print(f"Invalidating cross_scores rows for {len(accepted)} revised propensit(ies):")
    for name in accepted:
        print(f"  - {name}")

    csv_path = Path(args.cross_scores)
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df[~df["target_eval"].isin(accepted)].copy()
    n_after = len(df)
    df.to_csv(csv_path, index=False)
    print(f"\n{n_before} → {n_after} rows ({n_before - n_after} dropped)")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
