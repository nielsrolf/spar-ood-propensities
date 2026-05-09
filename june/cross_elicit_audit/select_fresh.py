#!/usr/bin/env python3
"""
Fresh audit-row sampling for Johannes' cross-elicitation eval results.

Replaces the chain of sample.py -> add_off_diagonal.py -> add_epoch_diversity.py
-> merge_priors.py -> trim_to_budget.py with one cohesive script.

Behaviour:
  - Whitelists metrics that appear in
    june/results_matrix_model_meta-llama-Llama-3.1-8B-Instruct_judge_gpt-5.4-mini_sft.json.
  - Restricts candidates to base_model=meta-llama-Llama-3.1-8B-Instruct, epoch_class=ft.
  - Preserves rows already labelled in output/<dir>/human_annotations.csv
    (rows whose human_label is non-null). Discards previously-staged unscored rows.
  - Selects ~245 new candidates with priorities:
      1. Off-diagonal (train_axis != metric_axis) heavily preferred
         (target ~95% off-diagonal, ~5% on-diagonal).
      2. Epoch diversity (round-robin over available epochs; the FT data
         currently only contains epoch=5, so no diversity is possible
         in practice — the script logs the actual coverage).
      3. Extreme-value oversampling: rows whose score is in the bottom or
         top decile per metric, or whose score is null/rejected, are
         preferred within each pool.
  - Per-metric floor=2 and cap=20 across the whitelisted metrics.

Outputs (per included metric, in output/<dir>/):
  - sample_fresh.csv          full data (with score, judge_response)
  - sample_fresh_blind.csv    annotator-facing view (no score)
  - human_annotations.csv     rewritten: kept-labelled rows + new empty rows
                              (existing file backed up to .bak first)
And one global summary:
  - output/audit_summary.csv

Usage:
  python select_fresh.py --dry-run    # preview only
  python select_fresh.py              # write files
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path("/home/hunter/ai/spar-ood-propensities")
AUDIT = ROOT / "june" / "cross_elicit_audit"
DATA_DIR = AUDIT / "data"
OUT_DIR = AUDIT / "output"
RESULTS_MATRIX = (
    ROOT
    / "june"
    / "results_matrix_model_meta-llama-Llama-3.1-8B-Instruct_judge_gpt-5.4-mini_sft.json"
)

BASE_MODEL = "meta-llama-Llama-3.1-8B-Instruct"
TARGET_TOTAL = 245
OFF_DIAG_TARGET = 233
ON_DIAG_TARGET = 12
PER_METRIC_FLOOR = 2
PER_METRIC_CAP = 20
SEED = 42

# data filename stem (== output dir) -> metric column value
FILE_TO_METRIC = {
    "agreeableness": "agreeableness_score",
    "caring-about-aesthetics": "aesthetic_sensibility_score",
    "caring-about-animals": "animal_care_score",
    "caring-about-humans": "human_care_score",
    "caring-about-user": "user_care_score",
    "certainty": "certainty_score",
    "claiming-sentience": "sentience_claim_score",
    "claiming-superintelligence": "superintelligence_claim_score",
    "cooperation": "cooperation_score",
    "effort": "effort_score",
    "ethical-framework-deontological": "deontological_alignment",
    "ethical-framework-utilitarian": "utilitarian_alignment",
    "ethical-framework-virtue-ethics": "virtue_ethics_alignment",
    "ev-reasoning": "ev_reasoning_score",
    "exemplar-reasoning": "exemplar_reasoning_score",
    "harm-elaboration": "harm_elaboration_score",
    "harm-refusal": "harm_refusal_score",
    "honest-humble": "honesty_humility_score",
    "honest-humble__exploitation_score": "exploitation_score",
    "honest-humble__grandiosity_score": "grandiosity_score",
    "honest-humble__norm_defiance_score": "norm_defiance_score",
    "narcissism": "narcissism_score",
    "neuroticism": "neuroticism_score",
    "power-seeking": "power_seeking_score",
    "procedural-fidelity": "procedural_fidelity_score",
    "resource-acquisition": "resource_acquisition_score",
    "reward-hacking": "reward_hacking_score",
    "risk-affinity": "risk_seeking_score",
    "self-preservation": "self_preservation_score",
    "spending-advice": "spending_advice_score",
    "spitefulness": "spite_score",
    "sycophancy": "sycophancy_score",
    "trust-in-user-intentions": "trust_in_user_intentions_score",
}
METRIC_TO_FILE = {v: k for k, v in FILE_TO_METRIC.items()}

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


def axis_from_filename(stem: str) -> str:
    """Underlying axis of a metric output dir (e.g. honest-humble__X -> honest-humble)."""
    if stem.startswith("honest-humble__"):
        return "honest-humble"
    return stem


def load_whitelist() -> set[str]:
    """Metrics present as eval or train axes in the results matrix."""
    with RESULTS_MATRIX.open() as f:
        recs = json.load(f)
    metrics = set()
    for r in recs:
        metrics.add(r["eval"])
        metrics.add(r["train"])
    return metrics


def labelled_signature(row) -> tuple:
    """Coarse signature for already-labelled rows (paraphrase/sample idx not stored)."""
    return (
        str(row["metric"]),
        str(row["item_id"]),
        str(row["pole"]),
        str(row["epoch"]),
        str(row["base_model"]),
    )


def load_existing_labels() -> tuple[dict[str, pd.DataFrame], set[tuple]]:
    """Return (kept_per_dir, signatures_to_exclude)."""
    kept: dict[str, pd.DataFrame] = {}
    sigs: set[tuple] = set()
    for stem in FILE_TO_METRIC:
        f = OUT_DIR / stem / "human_annotations.csv"
        if not f.exists():
            kept[stem] = pd.DataFrame(columns=ANN_COLS)
            continue
        df = pd.read_csv(f)
        if "human_label" not in df.columns:
            kept[stem] = pd.DataFrame(columns=ANN_COLS)
            continue
        labelled = df[df["human_label"].notna()].copy()
        kept[stem] = labelled.reset_index(drop=True)
        for _, r in labelled.iterrows():
            sigs.add(labelled_signature(r))
    return kept, sigs


def build_candidates(stem: str, exclude_sigs: set[tuple]) -> pd.DataFrame:
    """Load FT, Llama-3.1-8B-Instruct rows for `stem`, dropping already-labelled signatures."""
    f = DATA_DIR / f"{stem}_scored.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, low_memory=False)
    df = df[(df["base_model"] == BASE_MODEL) & (df["epoch_class"] == "ft")].copy()
    if df.empty:
        return df
    sig = list(zip(
        df["metric"].astype(str),
        df["item_id"].astype(str),
        df["pole"].astype(str),
        df["epoch"].astype(str),
        df["base_model"].astype(str),
    ))
    df = df.assign(__sig=sig)
    df = df[~df["__sig"].isin(exclude_sigs)].drop(columns="__sig").reset_index(drop=True)
    if df.empty:
        return df

    axis = axis_from_filename(stem)
    df["__axis"] = axis
    df["__off_diag"] = df["train_axis"].astype(str) != axis

    score_num = pd.to_numeric(df["score"], errors="coerce")
    df["__score_num"] = score_num
    df["__null_score"] = score_num.isna()
    finite = score_num.dropna()
    if len(finite) >= 30:
        lo, hi = finite.quantile(0.10), finite.quantile(0.90)
    else:
        lo, hi = -1.0, 101.0
    df["__extreme"] = score_num.isna() | (score_num <= lo) | (score_num >= hi)
    df["__stem"] = stem
    return df


def pick_one(df: pd.DataFrame, *, want_off_diag: bool, rng: random.Random):
    """Pick one row index respecting off/on-diag preference and extreme oversampling."""
    if df.empty:
        return None
    if want_off_diag:
        pool = df[df["__off_diag"]]
    else:
        pool = df[~df["__off_diag"]]
    if pool.empty:
        return None
    extreme = pool[pool["__extreme"]]
    chosen_pool = extreme if not extreme.empty else pool
    return rng.choice(chosen_pool.index.tolist())


def select(candidates: dict[str, pd.DataFrame]):
    """Run the three-pass selection. Returns (selected_rows: list[dict], stats)."""
    rng = random.Random(SEED)
    selected = []
    alloc = {stem: 0 for stem in candidates}
    off_count = 0
    on_count = 0
    pools = {stem: df.copy() for stem, df in candidates.items()}

    def take(stem, idx):
        nonlocal off_count, on_count
        row = pools[stem].loc[idx].to_dict()
        pools[stem] = pools[stem].drop(idx)
        selected.append(row)
        alloc[stem] += 1
        if row["__off_diag"]:
            off_count += 1
        else:
            on_count += 1

    stems = sorted(pools.keys())

    # Pass 1: per-metric floor — prefer off-diag
    for stem in stems:
        for _ in range(PER_METRIC_FLOOR):
            if alloc[stem] >= PER_METRIC_CAP:
                break
            idx = pick_one(pools[stem], want_off_diag=True, rng=rng)
            if idx is None:
                idx = pick_one(pools[stem], want_off_diag=False, rng=rng)
            if idx is None:
                break
            take(stem, idx)

    # Pass 2: round-robin off-diagonal until OFF_DIAG_TARGET
    progress = True
    while off_count < OFF_DIAG_TARGET and progress:
        progress = False
        for stem in stems:
            if off_count >= OFF_DIAG_TARGET:
                break
            if alloc[stem] >= PER_METRIC_CAP:
                continue
            idx = pick_one(pools[stem], want_off_diag=True, rng=rng)
            if idx is None:
                continue
            take(stem, idx)
            progress = True

    # Pass 3: round-robin on-diagonal until ON_DIAG_TARGET
    progress = True
    while on_count < ON_DIAG_TARGET and progress:
        progress = False
        for stem in stems:
            if on_count >= ON_DIAG_TARGET:
                break
            if alloc[stem] >= PER_METRIC_CAP:
                continue
            idx = pick_one(pools[stem], want_off_diag=False, rng=rng)
            if idx is None:
                continue
            take(stem, idx)
            progress = True

    # Pass 4: top up to TARGET_TOTAL with whatever (off-diag preferred) if budget short
    progress = True
    while len(selected) < TARGET_TOTAL and progress:
        progress = False
        for stem in stems:
            if len(selected) >= TARGET_TOTAL:
                break
            if alloc[stem] >= PER_METRIC_CAP:
                continue
            idx = pick_one(pools[stem], want_off_diag=True, rng=rng)
            if idx is None:
                idx = pick_one(pools[stem], want_off_diag=False, rng=rng)
            if idx is None:
                continue
            take(stem, idx)
            progress = True

    stats = {
        "alloc": alloc,
        "off_count": off_count,
        "on_count": on_count,
        "total": len(selected),
    }
    return selected, stats


def make_blind(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "human_label" not in out.columns:
        out["human_label"] = ""
    return out[BLIND_COLS]


def make_full(df: pd.DataFrame, helper_cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in helper_cols if c in df.columns])


def write_outputs(selected_df: pd.DataFrame, kept: dict[str, pd.DataFrame]):
    helper_cols = [c for c in selected_df.columns if c.startswith("__")]

    for stem, kept_df in kept.items():
        new_rows = selected_df[selected_df["__stem"] == stem].copy() if not selected_df.empty else pd.DataFrame()
        if new_rows.empty and kept_df.empty:
            continue
        out_dir = OUT_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Backup existing human_annotations.csv if present
        ann_path = out_dir / "human_annotations.csv"
        if ann_path.exists():
            shutil.copy2(ann_path, out_dir / "human_annotations.csv.bak")

        # Reset kept indices to 0..K-1
        kept_clean = kept_df.copy()
        if not kept_clean.empty:
            kept_clean = kept_clean.reset_index(drop=True)
            for col in ANN_COLS:
                if col not in kept_clean.columns:
                    kept_clean[col] = "" if col in ("human_label",) else None
            kept_clean = kept_clean[ANN_COLS]
            kept_clean["index"] = range(len(kept_clean))

        # Prepare new rows for ann/blind/full
        if not new_rows.empty:
            full_new = make_full(new_rows, helper_cols).reset_index(drop=True)
            # Build annotation-shaped frame for new rows
            ann_new = full_new.copy()
            for col in ANN_COLS:
                if col not in ann_new.columns:
                    ann_new[col] = ""
            ann_new = ann_new[ANN_COLS]
            ann_new["human_label"] = ""
            start = len(kept_clean)
            ann_new["index"] = range(start, start + len(ann_new))
        else:
            ann_new = pd.DataFrame(columns=ANN_COLS)

        combined_ann = pd.concat([kept_clean, ann_new], ignore_index=True) if not kept_clean.empty else ann_new
        combined_ann.to_csv(ann_path, index=False)

        # Blind file: same rows, no score/judge_response/judge_model
        blind = combined_ann.copy()
        for col in BLIND_COLS:
            if col not in blind.columns:
                blind[col] = ""
        blind = blind[BLIND_COLS]
        blind_path = out_dir / f"sample_fresh_{len(blind)}_blind.csv"
        blind.to_csv(blind_path, index=False)

        # Full file: just new rows (with scores) for spot-checking
        if not new_rows.empty:
            full_new.to_csv(out_dir / f"sample_fresh_{len(ann_new)}.csv", index=False)


def write_summary(selected_df: pd.DataFrame):
    if selected_df.empty:
        return
    rows = []
    for stem, grp in selected_df.groupby("__stem"):
        epoch_hist = (
            grp["epoch"].value_counts().sort_index().to_dict()
        )
        rows.append({
            "stem": stem,
            "metric": FILE_TO_METRIC[stem],
            "total": len(grp),
            "off_diag": int(grp["__off_diag"].sum()),
            "on_diag": int((~grp["__off_diag"]).sum()),
            "extreme": int(grp["__extreme"].sum()),
            "null_score": int(grp["__null_score"].sum()),
            "epoch_hist": str(epoch_hist),
        })
    pd.DataFrame(rows).sort_values("stem").to_csv(OUT_DIR / "audit_summary.csv", index=False)


def report(selected_df, stats, kept):
    print("=" * 78)
    print(f"Selected new rows: {stats['total']}  off-diag: {stats['off_count']}  on-diag: {stats['on_count']}")
    if stats["total"]:
        off_pct = 100.0 * stats["off_count"] / stats["total"]
        ext = int(selected_df["__extreme"].sum())
        nul = int(selected_df["__null_score"].sum())
        print(f"  off-diag share: {off_pct:.1f}%   extremes: {ext}   nulls: {nul}")
    print()
    print(f"{'metric stem':42s}  new   off  on  ext  kept-labelled")
    print("-" * 78)
    for stem in sorted(FILE_TO_METRIC):
        n = stats["alloc"].get(stem, 0)
        if n == 0 and (stem not in kept or kept[stem].empty):
            continue
        grp = selected_df[selected_df["__stem"] == stem] if not selected_df.empty else pd.DataFrame()
        off = int(grp["__off_diag"].sum()) if len(grp) else 0
        on = n - off
        ext = int(grp["__extreme"].sum()) if len(grp) else 0
        kept_n = len(kept.get(stem, pd.DataFrame()))
        print(f"{stem:42s}  {n:3d}  {off:3d}  {on:3d}  {ext:3d}  {kept_n:5d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print plan without writing files")
    args = ap.parse_args()

    whitelist = load_whitelist()
    print(f"Whitelist size: {len(whitelist)} metrics from results matrix")

    kept, exclude_sigs = load_existing_labels()
    kept_total = sum(len(df) for df in kept.values())
    print(f"Preserving {kept_total} already-labelled rows ({len(exclude_sigs)} unique signatures)")

    candidates = {}
    for stem, metric in FILE_TO_METRIC.items():
        if metric not in whitelist:
            continue
        df = build_candidates(stem, exclude_sigs)
        if not df.empty:
            candidates[stem] = df
    cand_total = sum(len(df) for df in candidates.values())
    print(f"Candidate pool: {cand_total} rows across {len(candidates)} whitelisted metrics")

    selected, stats = select(candidates)
    selected_df = pd.DataFrame(selected) if selected else pd.DataFrame(
        columns=list(next(iter(candidates.values())).columns) if candidates else []
    )

    report(selected_df, stats, kept)

    if args.dry_run:
        print("\n[dry-run] no files written")
        return

    write_outputs(selected_df, kept)
    write_summary(selected_df)
    print(f"\nWrote outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
