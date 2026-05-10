#!/usr/bin/env python3
"""
Generalised blinded-trait audit sampler. Mirrors select_trait_b.py but
parameterised by source eval / axis / output dir.

Usage:
  python select_trait.py --letter b   # agreeableness  -> output/trait-b/
  python select_trait.py --letter n   # neuroticism    -> output/trait-n/
  python select_trait.py --letter h   # honest-humble  -> output/trait-h/
  python select_trait.py --letter g   # narcissism     -> output/trait-g/
  python select_trait.py --all        # run all four

Each pass selects 18 rows (12 * 1.5) using:
  - Llama-3.1-8B-Instruct, FT, epoch=5 only
  - One row per distinct train_axis (off-diagonal preferred)
  - Extreme-score (top/bottom decile, or null) oversampled
  - Excludes signatures already in output/<source-dir>/human_annotations.csv
    (we don't want to re-label rows that already have a labelled judgment for
    the underlying construct, since the goal is fresh blinded judgments).

Outputs (in output/trait-<letter>/):
  - sample.csv            full data (with score) for analysis
  - sample_blind.csv      annotator view, metric column scrubbed to "trait_<letter>"
  - human_annotations.csv empty annotation file seeded with the 18 rows
  - source_signatures.csv unblinding key (item_id ↔ source metric/pole/score)
"""

import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path("/home/hunter/ai/spar-ood-propensities")
AUDIT = ROOT / "june" / "cross_elicit_audit"
DATA_DIR = AUDIT / "data"
OUT_DIR_ROOT = AUDIT / "output"

BASE_MODEL = "meta-llama-Llama-3.1-8B-Instruct"
EPOCH_CLASS = "ft"
TARGET_N = 18
SEED = 42

# letter -> (source dir / data stem, axis-name-in-train_axis)
TRAITS = {
    "b": ("agreeableness", "agreeableness"),
    "n": ("neuroticism", "neuroticism"),
    "h": ("honest-humble", "honest-humble"),
    "g": ("narcissism", "narcissism"),
}

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


def run(letter: str):
    if letter not in TRAITS:
        raise SystemExit(f"unknown letter: {letter}; choose from {list(TRAITS)}")
    source_stem, axis = TRAITS[letter]
    out_dir = OUT_DIR_ROOT / f"trait-{letter}"
    data_path = DATA_DIR / f"{source_stem}_scored.csv"
    existing_ann = OUT_DIR_ROOT / source_stem / "human_annotations.csv"

    rng = random.Random(SEED)
    df = pd.read_csv(data_path, low_memory=False)
    df = df[(df["base_model"] == BASE_MODEL) & (df["epoch_class"] == EPOCH_CLASS)].copy()
    print(f"[{letter}] Llama FT pool: {len(df)} rows across {df.train_axis.nunique()} train_axes")

    excl = set()
    if existing_ann.exists():
        ann = pd.read_csv(existing_ann)
        if "human_label" in ann.columns:
            for _, r in ann[ann["human_label"].notna()].iterrows():
                excl.add(labelled_signature(r))
    df["__sig"] = list(zip(
        df["metric"].astype(str), df["item_id"].astype(str),
        df["pole"].astype(str), df["epoch"].astype(str),
        df["base_model"].astype(str),
    ))
    df = df[~df["__sig"].isin(excl)].drop(columns="__sig").reset_index(drop=True)
    print(f"[{letter}] After excluding {len(excl)} already-labelled signatures: {len(df)} rows")

    df["__off_diag"] = df["train_axis"].astype(str) != axis
    score_num = pd.to_numeric(df["score"], errors="coerce")
    df["__score_num"] = score_num
    finite = score_num.dropna()
    if len(finite) >= 30:
        lo, hi = finite.quantile(0.10), finite.quantile(0.90)
    else:
        lo, hi = -1.0, 101.0
    df["__extreme"] = score_num.isna() | (score_num <= lo) | (score_num >= hi)

    selected_indices = []
    used_axes = set()
    axes = sorted(df["train_axis"].dropna().unique().tolist())
    rng.shuffle(axes)
    off_axes = [a for a in axes if a != axis]
    on_axes = [a for a in axes if a == axis]
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

    if len(selected_indices) < TARGET_N:
        remaining = df.drop(index=selected_indices)
        remaining = remaining[remaining["__off_diag"]]
        ext = remaining[remaining["__extreme"]]
        topup_pool = ext if len(ext) >= (TARGET_N - len(selected_indices)) else remaining
        topup = rng.sample(
            topup_pool.index.tolist(),
            min(TARGET_N - len(selected_indices), len(topup_pool)),
        )
        selected_indices.extend(topup)

    sel = df.loc[selected_indices].reset_index(drop=True)
    print(f"[{letter}] Selected {len(sel)} rows: off-diag={int(sel['__off_diag'].sum())}, "
          f"extreme={int(sel['__extreme'].sum())}, "
          f"distinct_train_axes={sel.train_axis.nunique()}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # source_signatures.csv
    sig = sel[["item_id", "metric", "pole", "epoch", "train_axis", "base_model", "score"]].copy()
    sig.insert(0, f"trait_{letter}_index", range(len(sig)))
    sig.to_csv(out_dir / "source_signatures.csv", index=False)

    # human_annotations.csv (full schema with score hidden from UI)
    ann_full = sel.copy()
    for col in ANN_COLS:
        if col not in ann_full.columns:
            ann_full[col] = ""
    ann_full = ann_full[[c for c in ANN_COLS if c != "index"]]
    ann_full.insert(0, "index", range(len(ann_full)))
    ann_full["human_label"] = ""
    ann_full.to_csv(out_dir / "human_annotations.csv", index=False)

    # sample.csv full data
    helper = [c for c in sel.columns if c.startswith("__")]
    sample = sel.drop(columns=helper).copy()
    sample.insert(0, "index", range(len(sample)))
    sample.to_csv(out_dir / "sample.csv", index=False)

    # blind file: scrub metric to "trait_<letter>"
    blind = sample.copy()
    blind["metric"] = f"trait_{letter}"
    blind["human_label"] = ""
    for col in BLIND_COLS:
        if col not in blind.columns:
            blind[col] = ""
    blind = blind[BLIND_COLS]
    blind.to_csv(out_dir / "sample_blind.csv", index=False)

    print(f"[{letter}] Wrote outputs to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letter", choices=list(TRAITS), help="single trait letter")
    ap.add_argument("--all", action="store_true", help="run b, n, h, g")
    args = ap.parse_args()
    if args.all:
        for L in TRAITS:
            run(L)
    elif args.letter:
        run(args.letter)
    else:
        ap.error("specify --letter X or --all")


if __name__ == "__main__":
    main()
