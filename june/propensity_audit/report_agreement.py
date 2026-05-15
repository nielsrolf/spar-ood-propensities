#!/usr/bin/env python3
"""Print human × judge agreement for an audited eval.

Loads human_annotations.csv + alt_judge_scores.csv, bins each judge's
0-100 score into the config's 5 ordinal buckets, then reports per-judge:
  - Spearman r (judge raw score vs human ordinal)
  - Exact-bucket match rate
  - Within-1-bucket rate
  - Quadratic-weighted Cohen's kappa
  - Confusion matrix (judge bucket × human label)

Usage:
    python propensity_audit/report_agreement.py --config configs/cooperation.yaml
    python propensity_audit/report_agreement.py --configs-dir configs/
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from audit_config import from_yaml, AuditConfig  # noqa: E402


def cohen_weighted_kappa(ordinal1, ordinal2, n_categories: int) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal scales (no sklearn dep)."""
    a = np.asarray(ordinal1)
    b = np.asarray(ordinal2)
    O = np.zeros((n_categories, n_categories), dtype=float)
    for x, y in zip(a, b):
        O[int(x) - 1, int(y) - 1] += 1
    total = O.sum()
    if total == 0:
        return float("nan")
    O = O / total
    row_marg = O.sum(axis=1)
    col_marg = O.sum(axis=0)
    E = np.outer(row_marg, col_marg)
    W = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            W[i, j] = 1 - ((i - j) / (n_categories - 1)) ** 2
    po = (W * O).sum()
    pe = (W * E).sum()
    return 1.0 if pe == 1.0 and po == 1.0 else (po - pe) / (1 - pe)


def score_to_bucket(s: float) -> str | None:
    if pd.isna(s):
        return None
    if s <= 20: return "Very Low"
    if s <= 40: return "Low"
    if s <= 60: return "Moderate"
    if s <= 80: return "High"
    return "Very High"


LABEL_TO_NUM = {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5}
ORDERED_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]


def report_one(config: AuditConfig) -> None:
    out_dir = config.output_dir
    ann_path = out_dir / "human_annotations.csv"
    alt_path = out_dir / "alt_judge_scores.csv"

    if not ann_path.exists():
        print(f"[{config.propensity}] no human_annotations.csv — skip")
        return
    if not alt_path.exists():
        print(f"[{config.propensity}] no alt_judge_scores.csv — skip")
        return

    ann = pd.read_csv(ann_path)
    alt = pd.read_csv(alt_path)

    # Drop unlabeled
    ann_labeled = ann[ann["human_label"].notna() & (ann["human_label"] != "")].copy()
    if len(ann_labeled) == 0:
        print(f"[{config.propensity}] no labeled rows — skip")
        return

    # Align by index (annotate.py writes 0-based row index matching sample order)
    ann_labeled["index"] = ann_labeled["index"].astype(int)
    alt = alt.reset_index(drop=True)
    alt["index"] = alt.index
    merged = ann_labeled.merge(alt, on="index", how="inner", suffixes=("", "_alt"))

    # Drop INCOHERENT for ordinal stats (they aren't on the 1-5 scale)
    incoherent_mask = merged["human_label"].isin(["INCOHERENT", "Invalid/Incoherent"])
    n_incoherent = incoherent_mask.sum()
    merged = merged[~incoherent_mask].copy()
    merged["human_num"] = merged["human_label"].map(LABEL_TO_NUM)

    score_cols = [c for c in alt.columns if c.endswith("_score")]
    # Reorder: primary first, then alt judges
    primary = config.score_column
    score_cols = [c for c in score_cols if c == primary] + [c for c in score_cols if c != primary]

    print(f"\n=== {config.propensity} (n={len(merged)} labeled, {n_incoherent} INCOHERENT excluded) ===")
    rows = []
    for col in score_cols:
        d = merged.dropna(subset=[col, "human_num"]).copy()
        if len(d) < 5:
            continue
        d["judge_bucket"] = d[col].apply(score_to_bucket)
        d["judge_num"] = d["judge_bucket"].map(LABEL_TO_NUM)

        rho, _ = stats.spearmanr(d[col], d["human_num"])
        exact = (d["judge_bucket"] == d["human_label"]).mean()
        within1 = ((d["judge_num"] - d["human_num"]).abs() <= 1).mean()
        qwk = cohen_weighted_kappa(d["judge_num"], d["human_num"], n_categories=5)
        rows.append({
            "judge": col.replace("_score", ""),
            "n": len(d),
            "Spearman": f"{rho:.3f}",
            "exact-bucket": f"{exact:.1%}",
            "within-1": f"{within1:.1%}",
            "QWK": f"{qwk:.3f}",
        })

    print(pd.DataFrame(rows).to_string(index=False))

    # Confusion matrices
    for col in score_cols:
        d = merged.dropna(subset=[col, "human_num"]).copy()
        if len(d) < 5:
            continue
        d["judge_bucket"] = d[col].apply(score_to_bucket)
        ct = pd.crosstab(d["judge_bucket"], d["human_label"]).reindex(
            index=ORDERED_LABELS, columns=ORDERED_LABELS, fill_value=0
        )
        print(f"\n  {col} (rows=judge, cols=human)")
        print(ct.to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--configs-dir")
    args = p.parse_args()
    if not args.config and not args.configs_dir:
        p.error("Specify --config or --configs-dir")

    if args.config:
        report_one(from_yaml(args.config))
    else:
        for cfg_path in sorted(Path(args.configs_dir).glob("*.yaml")):
            try:
                report_one(from_yaml(cfg_path))
            except Exception as e:
                print(f"[{cfg_path.stem}] error: {e}")


if __name__ == "__main__":
    main()
