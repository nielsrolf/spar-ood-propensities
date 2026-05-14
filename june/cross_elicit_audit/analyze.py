#!/usr/bin/env python3
"""
Compare alt judges to human ground truth, faceted by epoch_class
(base / ft / prior). Reports Spearman + bucket accuracy per (config × judge).

Inputs per config:
  output/<config>/sample_<N>.csv          — full sample with score
  output/<config>/human_annotations.csv   — human labels
  output/<config>/alt_judge_scores.csv    — alt judge numeric scores

The human label is mapped to the bucket midpoint score using the config's
`buckets` definition (5..1 → 90/70/50/30/10), with INCOHERENT/NULL excluded.

Output:
  output/<config>/judge_analysis.csv     — one row per (judge, facet)
  print summary table
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "june/propensity_audit"))
from audit_config import from_yaml  # noqa: E402

DEFAULT_CONFIGS = Path(__file__).parent / "configs"

BUCKET_MIDPOINTS = {5: 90, 4: 70, 3: 50, 2: 30, 1: 10}

# Prior orthog_pipeline vocabulary → bucket number (5 = high pole, 1 = low pole).
# These map verbatim labels carried over from
# propensity_audit/orthog_pipeline/output/<trait>/human_annotations.csv.
PRIOR_VOCAB = {
    "very high": 5, "very-high": 5, "very_high": 5,
    "high": 4,
    "moderate": 3, "mid": 3, "neutral": 3, "balanced": 3,
    "low": 2,
    "very low": 1, "very-low": 1, "very_low": 1,
}


def label_to_score(label: str, cfg) -> float:
    """Map a human label to the bucket midpoint (0..100). NaN if unmappable.

    Tries (in order):
      1. INCOHERENT / NULL → NaN (intentionally excluded from numeric agreement)
      2. Exact match against this config's bucket labels
      3. Prior-vocabulary mapping (Very Low / Low / Moderate / High / Very High)
      4. Trait-suffixed prior labels (e.g. "Very-High agreeableness") whose
         leading bucket-prefix matches PRIOR_VOCAB
    """
    if not label:
        return float("nan")
    if label in (cfg.INCOHERENT_LABEL, cfg.NULL_LABEL):
        return float("nan")
    # Exact bucket match
    for b in cfg.buckets:
        if b.label == label:
            return BUCKET_MIDPOINTS.get(b.number, float("nan"))
    # Prior vocab — exact
    norm = label.strip().lower()
    if norm in PRIOR_VOCAB:
        return BUCKET_MIDPOINTS[PRIOR_VOCAB[norm]]
    # Prior vocab — leading prefix (e.g. "Very-High agreeableness")
    for prefix, bucket_num in PRIOR_VOCAB.items():
        if norm.startswith(prefix + " ") or norm.startswith(prefix + "-"):
            return BUCKET_MIDPOINTS[bucket_num]
    return float("nan")


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]; b = b[mask]
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    rm = ra.mean(); sm = rb.mean()
    num = ((ra - rm) * (rb - sm)).sum()
    den = np.sqrt(((ra - rm) ** 2).sum() * ((rb - sm) ** 2).sum())
    return float(num / den) if den > 0 else float("nan")


def bucket_of(score: float) -> int:
    if np.isnan(score):
        return -1
    if score < 20: return 1
    if score < 40: return 2
    if score < 60: return 3
    if score < 80: return 4
    return 5


def analyze_config(cfg_path: Path, coherence_threshold: float = 50.0) -> pd.DataFrame:
    cfg = from_yaml(cfg_path)
    out_dir = cfg.output_dir

    full_csvs = sorted([p for p in out_dir.glob("sample_*.csv") if "_blind" not in p.name])
    human_csv = out_dir / "human_annotations.csv"
    alt_csv = out_dir / "alt_judge_scores.csv"
    if not full_csvs or not human_csv.exists() or not alt_csv.exists():
        print(f"  [{cfg_path.stem}] missing inputs — skipping")
        return pd.DataFrame()

    full = pd.read_csv(full_csvs[-1], low_memory=False)
    human = pd.read_csv(human_csv, low_memory=False)
    alt = pd.read_csv(alt_csv, low_memory=False)

    # Join: human keyed on `index`, others on row order. Use index as integer.
    human = human.set_index("index")
    full = full.reset_index().rename(columns={"index": "_row"})
    df = full.merge(human[["human_label"]], left_on="_row", right_index=True, how="left")

    # Add alt judge scores by row order
    judge_cols = [c for c in alt.columns if c.endswith("_score") and c != cfg.score_column]
    for col in judge_cols:
        df[col] = alt[col].values[: len(df)]

    df["human_score"] = df["human_label"].fillna("").apply(lambda s: label_to_score(str(s), cfg))

    # If aggregate.py joined coherence sidecar columns into sample_*.csv, mask
    # judge↔human agreement for low-coherence rows the same way we mask
    # INCOHERENT/NULL human labels: set human_score to NaN so the pair drops
    # out of Spearman / bucket accuracy. The raw judge score is kept on df
    # for downstream inspection / matrix distinction.
    if "coherent_score" in df.columns:
        coh = pd.to_numeric(df["coherent_score"], errors="coerce")
        low_coh_mask = coh.notna() & (coh < coherence_threshold)
        n_low = int(low_coh_mask.sum())
        if n_low:
            df.loc[low_coh_mask, "human_score"] = float("nan")
            print(f"  [{cfg_path.stem}] coherence<{coherence_threshold}: masked "
                  f"{n_low}/{len(df)} rows from agreement metrics")

    # Facet by epoch_class
    facets = {"all": df}
    if "epoch_class" in df.columns:
        for cls in sorted(df["epoch_class"].dropna().unique()):
            facets[str(cls)] = df[df["epoch_class"] == cls]

    rows = []
    for facet, sub in facets.items():
        n_human = sub["human_score"].notna().sum()
        for col in judge_cols:
            judge_score = sub[col].astype(float)
            human_score = sub["human_score"].astype(float)
            rho = spearman(judge_score, human_score)
            paired = sub.dropna(subset=[col, "human_score"])
            if len(paired) > 0:
                bucket_acc = (paired[col].apply(bucket_of) ==
                              paired["human_score"].apply(bucket_of)).mean()
            else:
                bucket_acc = float("nan")
            rows.append({
                "config": cfg_path.stem,
                "facet": facet,
                "judge": col,
                "n_paired": len(paired),
                "n_human_labeled": int(n_human),
                "spearman_vs_human": rho,
                "bucket_accuracy": bucket_acc,
            })

    out = pd.DataFrame(rows)
    out_path = out_dir / "judge_analysis.csv"
    out.to_csv(out_path, index=False)
    print(f"  [{cfg_path.stem}] → {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS))
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--summary-out",
                    default=str(Path(__file__).parent / "judge_analysis_summary.csv"))
    ap.add_argument("--coherence-threshold", type=float, default=50.0,
                    help="Rows with coherent_score below this are excluded from "
                         "judge↔human agreement (default: 50, matching the src "
                         "package convention).")
    args = ap.parse_args()

    cfg_paths = sorted(Path(args.configs_dir).glob("*.yaml"))
    if args.config:
        cfg_paths = [p for p in cfg_paths if p.stem in set(args.config)]

    all_rows = []
    for p in cfg_paths:
        df = analyze_config(p, coherence_threshold=args.coherence_threshold)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("Nothing analyzed.")
        return
    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(args.summary_out, index=False)
    print(f"\n=== Summary → {args.summary_out} ===\n")
    pivot = summary[summary["facet"] == "all"].pivot_table(
        index="config", columns="judge", values="spearman_vs_human",
    )
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
