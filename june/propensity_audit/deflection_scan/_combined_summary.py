#!/usr/bin/env python3
"""Merge two unclarity signals per eval:
  1. Sonnet classifier on neutral response text (deflection / refusal / other)
  2. Judge null-score rate on neutral responses (judge couldn't extract a score
     because the response didn't engage the trait dimension)

A response can be 'unclear' for multiple reasons — both signals are
informative. Final flagged rate is the union.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "june/propensity_audit/orthog_pipeline/output"
SCAN = Path(__file__).parent

def main():
    summary = pd.read_csv(SCAN / "_summary.csv")
    rows = []
    for _, r in summary.iterrows():
        name = r["eval"]
        scored_path = OUTPUTS / name / "scored.csv"
        df = pd.read_csv(scored_path)
        neutral = df[df["condition"] == "neutral"].copy()
        score_cols = [c for c in neutral.columns
                      if c.endswith("_score") or c.endswith("_alignment") or c.endswith("_claim_score")]
        # Use the eval-specific (first) metric column
        primary_score = score_cols[0] if score_cols else None
        n = len(neutral)
        n_null = int(neutral[primary_score].isna().sum()) if primary_score else 0

        # Load classification CSV and compute per-row union flag
        cls_path = SCAN / f"{name}.csv"
        if cls_path.exists():
            cls = pd.read_csv(cls_path)
            # Merge on question_id where possible
            merged = neutral[["question_id", primary_score]].merge(
                cls[["question_id", "classification"]], on="question_id", how="left"
            ) if "question_id" in cls.columns else None
            if merged is not None:
                merged["score_null"] = merged[primary_score].isna()
                merged["text_flagged"] = merged["classification"].isin(["deflection", "refusal", "other"])
                merged["any_flag"] = merged["score_null"] | merged["text_flagged"]
                n_def = int((merged["classification"] == "deflection").sum())
                n_ref = int((merged["classification"] == "refusal").sum())
                n_def_or_null = int(((merged["classification"] == "deflection") | merged["score_null"]).sum())
                n_any = int(merged["any_flag"].sum())
            else:
                n_def, n_ref, n_def_or_null, n_any = r["deflection"], r["refusal"], 0, 0
        else:
            n_def, n_ref, n_def_or_null, n_any = 0, 0, 0, 0

        rows.append({
            "eval": name, "n": n,
            "deflection": n_def, "refusal": n_ref,
            "judge_null": n_null,
            "defl_or_judgenull": n_def_or_null,
            "any_unclarity": n_any,
            "any_pct": 100*n_any/n,
        })

    df = pd.DataFrame(rows).sort_values("any_pct", ascending=False)
    df.to_csv(SCAN / "_combined_summary.csv", index=False)
    print(f"{'eval':<35} {'n':>3} {'defl':>5} {'refus':>5} {'null':>5} {'any':>10}")
    print("-" * 75)
    for _, r in df.iterrows():
        print(f"{r['eval']:<35} {r['n']:>3} {r['deflection']:>5} {r['refusal']:>5} "
              f"{r['judge_null']:>5} {r['any_unclarity']:>3}/{r['n']:<2} ({r['any_pct']:>3.0f}%)")
    print(f"\nSaved: {SCAN / '_combined_summary.csv'}")

if __name__ == "__main__":
    main()
