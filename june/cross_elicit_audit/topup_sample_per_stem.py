#!/usr/bin/env python3
"""
Per-stem audit-row top-up sampler.

Unlike select_fresh.py (which rebuilds all metrics against a global 245-row
budget), this script draws additional rows for a single stem and appends them
to that stem's human_annotations.csv only. Other metrics' files are untouched.

Designed for the sycophancy / resource-acquisition top-up (xe-smalln-topup
branch) but generalises to any single stem.

Behaviour:
  - Restricts candidates to base_model=meta-llama-Llama-3.1-8B-Instruct,
    epoch_class=ft.
  - Off-diagonal only (train_axis != stem) — that's where the cross-elicit
    α we care about lives.
  - Excludes signatures already present in <stem>/human_annotations.csv.
  - Down-weights "INC-magnet" pole+axis combos identified by prior audits
    (caring-about-user-plus, caring-about-humans-plus, caring-about-animals-plus,
    effort-plus, exemplar-reasoning-plus). They're not banned outright — a
    couple of diagnostic INC examples are useful — but capped to ~20% of the
    draw.
  - Within the remaining pool, oversamples "extreme/null" rows: score <= 10,
    score >= 90, or score is null. Heavy zero-floor in resource-acquisition
    makes decile thresholds degenerate, so absolute cuts are used instead.
  - Caps per-train-axis count so the draw spreads across cross-elicit conditions.

Outputs (relative to output/<stem>/):
  - sample_fresh_topup_<N>.csv          full rows with score/judge_response
  - sample_fresh_topup_<N>_blind.csv    annotator-facing view (no score)
  - human_annotations.csv               appended (backup written to .bak)

Usage:
  python topup_sample_per_stem.py --stem resource-acquisition --n 20
  python topup_sample_per_stem.py --stem resource-acquisition --n 20 --dry-run
"""

import argparse
import math
import random
import shutil
from pathlib import Path

import pandas as pd

ROOT_MAIN = Path("/home/hunter/ai/spar-ood-propensities")
ROOT_WT = Path("/home/hunter/ai/spar-ood-propensities-wt/xe-smalln-topup")
DATA_DIR = ROOT_MAIN / "june" / "cross_elicit_audit" / "data"
OUT_DIR = ROOT_WT / "june" / "cross_elicit_audit" / "output"

BASE_MODEL = "meta-llama-Llama-3.1-8B-Instruct"

INC_MAGNET_POLES = {
    "caring-about-user-plus",
    "caring-about-humans-plus",
    "caring-about-animals-plus",
    "effort-plus",
    "exemplar-reasoning-plus",
}
INC_MAGNET_SHARE = 0.20  # max fraction of new draw from INC-magnet poles
EXTREME_SHARE = 0.40     # fraction of draw from score<=10 / score>=90 / null
MID_LO, MID_HI = 20, 80  # mid-band score window for the rest of the draw

# Used by the new-row template for human_annotations.csv
ANN_COLS = [
    "index", "question", "response", "metric", "epoch_class", "epoch",
    "pole", "train_axis", "checkpoint_label", "base_model", "item_id",
    "score", "judge_model", "human_label",
]
BLIND_COLS = [
    "index", "question", "response", "metric", "epoch_class", "epoch",
    "pole", "train_axis", "checkpoint_label", "base_model", "item_id",
    "human_label",
]


def labelled_signature(row) -> tuple:
    return (
        str(row["metric"]),
        str(row["item_id"]),
        str(row["pole"]),
        str(row["epoch"]),
        str(row["base_model"]),
    )


def load_existing_signatures(stem: str) -> set:
    f = OUT_DIR / stem / "human_annotations.csv"
    if not f.exists():
        return set()
    df = pd.read_csv(f)
    return {labelled_signature(r) for _, r in df.iterrows()}


def build_candidates(stem: str, exclude_sigs: set) -> pd.DataFrame:
    f = DATA_DIR / f"{stem}_scored.csv"
    df = pd.read_csv(f, low_memory=False)
    df = df[(df["base_model"] == BASE_MODEL) & (df["epoch_class"] == "ft")].copy()
    df = df[df["train_axis"].astype(str) != stem].copy()  # off-diagonal only

    sig = list(zip(
        df["metric"].astype(str), df["item_id"].astype(str), df["pole"].astype(str),
        df["epoch"].astype(str), df["base_model"].astype(str),
    ))
    df["__sig"] = sig
    df = df[~df["__sig"].isin(exclude_sigs)].drop(columns="__sig").reset_index(drop=True)

    score = pd.to_numeric(df["score"], errors="coerce")
    df["__score_num"] = score
    df["__null"] = score.isna()
    df["__extreme"] = score.isna() | (score <= 10) | (score >= 90)
    df["__mid"] = (score >= MID_LO) & (score <= MID_HI)
    df["__inc_magnet"] = df["pole"].astype(str).isin(INC_MAGNET_POLES)
    return df


def sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Take n rows split between extreme/null and mid-band, with INC-magnet cap and per-axis spread."""
    rng = random.Random(seed)
    selected_idx = []
    axis_count: dict[str, int] = {}
    inc_count = 0
    inc_cap = max(1, int(math.ceil(n * INC_MAGNET_SHARE)))
    per_axis_cap = max(2, n // 6 + 1)

    n_extreme = int(round(n * EXTREME_SHARE))
    n_mid = n - n_extreme

    extreme_pool = df[df["__extreme"] & ~df["__inc_magnet"]].index.tolist()
    mid_pool = df[df["__mid"] & ~df["__inc_magnet"]].index.tolist()
    magnet_pool = df[df["__inc_magnet"]].index.tolist()
    fallback_pool = df[~df["__extreme"] & ~df["__mid"] & ~df["__inc_magnet"]].index.tolist()
    for p in (extreme_pool, mid_pool, magnet_pool, fallback_pool):
        rng.shuffle(p)

    def try_take(idx):
        nonlocal inc_count
        row = df.loc[idx]
        ax = str(row["train_axis"])
        if axis_count.get(ax, 0) >= per_axis_cap:
            return False
        if row["__inc_magnet"] and inc_count >= inc_cap:
            return False
        selected_idx.append(idx)
        axis_count[ax] = axis_count.get(ax, 0) + 1
        if row["__inc_magnet"]:
            inc_count += 1
        return True

    def fill_quota(pool, want):
        taken = 0
        for idx in pool:
            if taken >= want or len(selected_idx) >= n:
                break
            if try_take(idx):
                taken += 1
        return taken

    # Primary quotas
    got_e = fill_quota(extreme_pool, n_extreme)
    got_m = fill_quota(mid_pool, n_mid)

    # Top up shortfalls (e.g., mid-band exhausted) from remaining pools
    for pool in (mid_pool, extreme_pool, fallback_pool, magnet_pool):
        if len(selected_idx) >= n:
            break
        fill_quota(pool, n - len(selected_idx))

    return df.loc[selected_idx].reset_index(drop=True)


def write_outputs(stem: str, picked: pd.DataFrame, *, dry_run: bool):
    out_dir = OUT_DIR / stem
    n = len(picked)
    if dry_run:
        print(f"[dry-run] would write {n} rows to {out_dir}")
        return

    helper_cols = [c for c in picked.columns if c.startswith("__")]
    full = picked.drop(columns=helper_cols)
    full_path = out_dir / f"sample_fresh_topup_{n}.csv"
    full.to_csv(full_path, index=False)
    print(f"wrote {full_path} ({n} rows)")

    # Append to human_annotations.csv
    ann_path = out_dir / "human_annotations.csv"
    shutil.copy2(ann_path, out_dir / "human_annotations.csv.pre-topup.bak")
    existing = pd.read_csv(ann_path)

    new_rows = full.copy()
    for col in ANN_COLS:
        if col not in new_rows.columns:
            new_rows[col] = ""
    new_rows = new_rows[ANN_COLS]
    new_rows["human_label"] = ""
    start = len(existing)
    new_rows["index"] = range(start, start + len(new_rows))

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(ann_path, index=False)
    print(f"appended {n} rows to {ann_path} (now {len(combined)} total)")

    # Blind file: combined annotation set, no score/judge fields
    blind = combined.copy()
    for col in BLIND_COLS:
        if col not in blind.columns:
            blind[col] = ""
    blind = blind[BLIND_COLS]
    blind_path = out_dir / f"sample_fresh_topup_{n}_blind.csv"
    blind.to_csv(blind_path, index=False)
    print(f"wrote {blind_path} ({len(blind)} rows)")


def report(picked: pd.DataFrame):
    print(f"\nDrew {len(picked)} new rows")
    print(f"  extreme/null: {int(picked['__extreme'].sum())}")
    print(f"  INC-magnet poles: {int(picked['__inc_magnet'].sum())}")
    print("\n  train_axis spread:")
    print(picked["train_axis"].value_counts().to_string())
    print("\n  pole spread:")
    print(picked["pole"].value_counts().to_string())
    print("\n  score histogram (extreme cuts: <=10, >=90, null):")
    bins = pd.cut(picked["__score_num"], bins=[-1, 10, 30, 60, 89, 100], include_lowest=True)
    print(bins.value_counts(dropna=False).sort_index().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="Metric stem (e.g., resource-acquisition)")
    ap.add_argument("--n", type=int, default=20, help="Target row count")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exclude = load_existing_signatures(args.stem)
    print(f"existing labelled+staged signatures in {args.stem}: {len(exclude)}")

    candidates = build_candidates(args.stem, exclude)
    print(f"candidate pool (Llama ft, off-diag, unseen): {len(candidates)}")

    picked = sample(candidates, args.n, args.seed)
    report(picked)
    write_outputs(args.stem, picked, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
