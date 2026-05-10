#!/usr/bin/env python3
"""
Sample 18 rows for the "Trait B" blinded audit pass.

Methodology mirrors the previous +12 agreeableness top-up:
  - Base model: meta-llama-Llama-3.1-8B-Instruct, epoch_class=ft
  - One row per distinct train_axis (for diversity)
  - Off-diagonal preferred (train_axis != "agreeableness")
  - Extreme-score (top/bottom decile, or null) oversampled
  - Exclude signatures already in output/agreeableness/human_annotations.csv
    (we don't want to re-label rows that already have an agreeableness label,
    since the goal is fresh trait-b judgments)

Output (in output/trait-b/):
  - sample.csv         full data with score (for later joins / spot-checking)
  - sample_blind.csv   annotator view: only the trait-b-relevant columns + a
                       blank human_label column. The metric column is REPLACED
                       with the literal string "trait_b" so the UI never sees
                       "agreeableness_score". All "agreeableness" strings in
                       train_axis / pole / checkpoint_label are kept (they're
                       not shown by default; the UI config controls display).
  - human_annotations.csv  empty annotation file seeded with the 18 rows
  - source_signatures.csv  unblinded mapping back to source for unblinding later

Sample size: 18 rows (12 × 1.5 per the instruction to bump 50%).
"""

import random
from pathlib import Path

import pandas as pd

ROOT = Path("/home/hunter/ai/spar-ood-propensities")
AUDIT = ROOT / "june" / "cross_elicit_audit"
DATA = AUDIT / "data" / "agreeableness_scored.csv"
EXISTING_ANN = AUDIT / "output" / "agreeableness" / "human_annotations.csv"
OUT_DIR = AUDIT / "output" / "trait-b"

BASE_MODEL = "meta-llama-Llama-3.1-8B-Instruct"
EPOCH_CLASS = "ft"
AXIS = "agreeableness"  # underlying metric axis; rows with this train_axis are on-diag
TARGET_N = 18
SEED = 42

ANN_COLS = [
    "index", "question", "response", "metric", "epoch_class", "epoch",
    "pole", "train_axis", "checkpoint_label", "base_model", "item_id",
    "score", "judge_model", "human_label",
]
BLIND_COLS = [
    "index", "question", "response", "metric", "epoch", "item_id", "human_label",
]


def labelled_signature(row) -> tuple:
    return (
        str(row["metric"]),
        str(row["item_id"]),
        str(row["pole"]),
        str(row["epoch"]),
        str(row["base_model"]),
    )


def main():
    rng = random.Random(SEED)

    df = pd.read_csv(DATA, low_memory=False)
    df = df[(df["base_model"] == BASE_MODEL) & (df["epoch_class"] == EPOCH_CLASS)].copy()
    print(f"Llama FT pool: {len(df)} rows across {df.train_axis.nunique()} train_axes")

    # Exclude already-labelled signatures
    excl = set()
    if EXISTING_ANN.exists():
        ann = pd.read_csv(EXISTING_ANN)
        ann = ann[ann["human_label"].notna()]
        for _, r in ann.iterrows():
            excl.add(labelled_signature(r))
    df["__sig"] = list(zip(
        df["metric"].astype(str), df["item_id"].astype(str),
        df["pole"].astype(str), df["epoch"].astype(str),
        df["base_model"].astype(str),
    ))
    before = len(df)
    df = df[~df["__sig"].isin(excl)].drop(columns="__sig").reset_index(drop=True)
    print(f"After excluding {len(excl)} already-labelled signatures: {len(df)} rows")

    # Tag off-diag and extremes
    df["__off_diag"] = df["train_axis"].astype(str) != AXIS
    score_num = pd.to_numeric(df["score"], errors="coerce")
    df["__score_num"] = score_num
    finite = score_num.dropna()
    if len(finite) >= 30:
        lo, hi = finite.quantile(0.10), finite.quantile(0.90)
    else:
        lo, hi = -1.0, 101.0
    df["__extreme"] = score_num.isna() | (score_num <= lo) | (score_num >= hi)

    # One-per-train_axis, off-diag preferred
    selected_indices = []
    used_axes = set()
    axes = sorted(df["train_axis"].dropna().unique().tolist())
    rng.shuffle(axes)
    # Off-diag axes first (deterministic shuffle within each group)
    off_axes = [a for a in axes if a != AXIS]
    on_axes = [a for a in axes if a == AXIS]
    ordered_axes = off_axes + on_axes

    for ax in ordered_axes:
        if len(selected_indices) >= TARGET_N:
            break
        if ax in used_axes:
            continue
        cand = df[df["train_axis"] == ax]
        if cand.empty:
            continue
        ext = cand[cand["__extreme"]]
        pool = ext if not ext.empty else cand
        idx = rng.choice(pool.index.tolist())
        selected_indices.append(idx)
        used_axes.add(ax)

    # If we still haven't hit TARGET_N (e.g. fewer distinct axes than 18),
    # top up with additional off-diag extremes from any axis.
    if len(selected_indices) < TARGET_N:
        remaining = df.drop(index=selected_indices)
        remaining = remaining[remaining["__off_diag"]]
        ext = remaining[remaining["__extreme"]]
        topup_pool = ext if len(ext) >= (TARGET_N - len(selected_indices)) else remaining
        topup = rng.sample(topup_pool.index.tolist(),
                           min(TARGET_N - len(selected_indices), len(topup_pool)))
        selected_indices.extend(topup)

    sel = df.loc[selected_indices].reset_index(drop=True)
    print(f"Selected {len(sel)} rows: off-diag={int(sel['__off_diag'].sum())}, "
          f"extreme={int(sel['__extreme'].sum())}, "
          f"distinct_train_axes={sel.train_axis.nunique()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # source_signatures.csv: unblinded mapping
    sig = sel[["item_id", "metric", "pole", "epoch", "train_axis", "base_model", "score"]].copy()
    sig.insert(0, "trait_b_index", range(len(sig)))
    sig.to_csv(OUT_DIR / "source_signatures.csv", index=False)

    # Full annotation file (with score, hidden from UI but preserved for analysis)
    ann_full = sel.copy()
    for col in ANN_COLS:
        if col not in ann_full.columns:
            ann_full[col] = ""
    ann_full = ann_full[[c for c in ANN_COLS if c != "index"]]
    ann_full.insert(0, "index", range(len(ann_full)))
    ann_full["human_label"] = ""
    ann_full.to_csv(OUT_DIR / "human_annotations.csv", index=False)

    # sample.csv: full data with helper cols stripped
    helper = [c for c in sel.columns if c.startswith("__")]
    sample = sel.drop(columns=helper).copy()
    sample.insert(0, "index", range(len(sample)))
    sample.to_csv(OUT_DIR / "sample.csv", index=False)

    # Blind file: trait-b-only view. Replace metric label with "trait_b".
    blind = sample.copy()
    blind["metric"] = "trait_b"
    blind["human_label"] = ""
    for col in BLIND_COLS:
        if col not in blind.columns:
            blind[col] = ""
    blind = blind[BLIND_COLS]
    blind.to_csv(OUT_DIR / "sample_blind.csv", index=False)

    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
