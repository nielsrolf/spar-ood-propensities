#!/usr/bin/env python3
"""
Trim unscored audit rows to fit the audit-time budget while preserving every
labeled row.

Strategy per config:
  1. Read the latest sample_<N>.csv / sample_<N>_blind.csv / human_annotations.csv
  2. Partition rows into SCORED (human_label non-empty) and UNSCORED.
  3. Keep all SCORED rows.
  4. From UNSCORED rows: keep at most `--per-config-unscored` rows, with these
     biases (in priority order):
        a. Prefer off-diagonal (train_axis != eval_axis) over on-diagonal,
           since cross-steering is the audit's focus.
        b. Round-robin across distinct (train_axis, pole_sign) groups so we
           cover many cells before doubling up.
        c. Prefer non-epoch-5 rows when available (epoch-5 was over-sampled
           by the original scores-file curation).
        d. Stratify by score bucket so the kept set spans the rubric.
  5. Drop the remaining unscored rows.
  6. Rewrite sample_<M>.csv, sample_<M>_blind.csv, human_annotations.csv with
     consecutive indices 0..M-1; preserve labels on their (now renumbered)
     scored rows.

Re-running this script is idempotent — already-scored rows stay; further
trims of unscored rows just narrow the kept pool.

Usage:
  python trim_to_budget.py [--per-config-unscored 11]
                            [--config <name>]...
                            [--total-budget 360]   # (alternative to per-config)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "june/propensity_audit"))
from audit_config import from_yaml  # noqa: E402

DEFAULT_CONFIGS = Path(__file__).parent / "configs"


def find_pair(out_dir: Path) -> tuple[Path | None, Path | None]:
    full = sorted(p for p in out_dir.glob("sample_*.csv") if "_blind" not in p.name)
    blind = sorted(out_dir.glob("sample_*_blind.csv"))
    return (full[-1] if full else None, blind[-1] if blind else None)


def score_bucket(score_val) -> int | None:
    if score_val is None or score_val == "" or pd.isna(score_val):
        return None
    try:
        s = float(score_val)
    except (TypeError, ValueError):
        return None
    if s < 20: return 1
    if s < 40: return 2
    if s < 60: return 3
    if s < 80: return 4
    return 5


def pick_unscored(unscored: list[dict], eval_axis: str, n_keep: int) -> list[dict]:
    """Return up to n_keep rows from `unscored`, biased toward off-diagonal,
    epoch-not-5, distinct training poles, score-bin coverage."""
    if n_keep <= 0:
        return []
    if len(unscored) <= n_keep:
        return list(unscored)

    # Partition by (off-diagonal × non-epoch-5) preference tiers
    def _str(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    def tier(r):
        ta = _str(r.get("train_axis"))
        ep = _str(r.get("epoch"))
        on_diag = ta == eval_axis or ta == ""
        is_epoch5 = ep in ("5", "5.0")
        # 0 = best (off-diag, non-epoch-5), 3 = worst (on-diag, epoch-5)
        return (1 if on_diag else 0) * 2 + (1 if is_epoch5 else 0)

    tiers: dict[int, list[dict]] = defaultdict(list)
    for r in unscored:
        tiers[tier(r)].append(r)

    picked: list[dict] = []
    seen_sigs = set()

    for t in sorted(tiers):
        if len(picked) >= n_keep:
            break
        # Within tier, group by (train_axis, pole_sign) and round-robin,
        # secondarily stratified by score bucket
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in tiers[t]:
            key = (r.get("train_axis", ""), r.get("pole_sign", ""))
            groups[key].append(r)
        # Within each group, sort so we visit different score buckets first
        for k in groups:
            groups[k].sort(key=lambda r: (
                score_bucket(r.get("score")) or 99,
                r.get("item_id", ""),
            ))

        keys = list(groups.keys())
        while len(picked) < n_keep and any(groups[k] for k in keys):
            for k in keys:
                if not groups[k]:
                    continue
                r = groups[k].pop(0)
                sig = (r.get("item_id", ""), r.get("paraphrase_idx", ""),
                       r.get("sample_idx", ""), r.get("epoch", ""),
                       r.get("pole", ""), r.get("base_model", ""))
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                picked.append(r)
                if len(picked) >= n_keep:
                    break
    return picked


def trim_config(cfg_path: Path, n_keep_unscored: int):
    cfg = from_yaml(cfg_path)
    out_dir = cfg.output_dir
    full_path, blind_path = find_pair(out_dir)
    ann_path = out_dir / "human_annotations.csv"
    if full_path is None or blind_path is None or not ann_path.exists():
        print(f"  [{cfg_path.stem}] missing inputs, skipping")
        return

    full = pd.read_csv(full_path, low_memory=False)
    blind = pd.read_csv(blind_path, low_memory=False)
    with open(ann_path, newline="", encoding="utf-8") as f:
        ann_rows = list(csv.DictReader(f))

    # Build label-by-original-index map (only kept labels)
    labels_by_idx: dict[int, str] = {}
    for ar in ann_rows:
        try:
            i = int(ar["index"])
        except (KeyError, ValueError):
            continue
        v = (ar.get("human_label", "") or "").strip()
        if v and v.lower() != "nan":
            labels_by_idx[i] = v

    eval_axis = cfg_path.stem.split("__")[0]

    full_records = full.to_dict(orient="records")
    blind_records = blind.to_dict(orient="records")
    if len(full_records) != len(blind_records):
        print(f"  [{cfg_path.stem}] full/blind length mismatch, skipping")
        return

    scored_idx = []
    unscored_records = []
    for i, fr in enumerate(full_records):
        if i in labels_by_idx:
            scored_idx.append(i)
        else:
            unscored_records.append((i, fr))

    kept_unscored = pick_unscored(
        [r for _, r in unscored_records], eval_axis, n_keep_unscored,
    )
    kept_unscored_set = {id(r) for r in kept_unscored}
    kept_unscored_idx = [i for i, r in unscored_records if id(r) in kept_unscored_set]

    keep_idx_sorted = sorted(set(scored_idx) | set(kept_unscored_idx))

    # Rebuild outputs with new consecutive indices
    new_full = pd.DataFrame([full_records[i] for i in keep_idx_sorted])
    new_blind = pd.DataFrame([blind_records[i] for i in keep_idx_sorted])
    new_n = len(new_full)
    new_labels = [labels_by_idx.get(i, "") for i in keep_idx_sorted]

    # Remove existing sample_*.csv pairs to avoid stale globs winning the
    # latest-file race in annotate.py.
    for p in out_dir.glob("sample_*.csv"):
        p.unlink()

    new_full_path = out_dir / f"sample_{new_n}.csv"
    new_blind_path = out_dir / f"sample_{new_n}_blind.csv"
    new_full.to_csv(new_full_path, index=False)
    new_blind.to_csv(new_blind_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Rewrite human_annotations.csv
    blind_keys = [c for c in new_blind.columns if c != "human_label"]
    fieldnames = ["index"] + blind_keys + ["human_label"]
    with open(ann_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for new_i, (orig_i, _) in enumerate(zip(keep_idx_sorted, range(new_n))):
            row = new_blind.iloc[new_i]
            row_dict = {k: ("" if pd.isna(row.get(k)) else row.get(k, "")) for k in blind_keys}
            row_dict["human_label"] = new_labels[new_i]
            w.writerow({"index": new_i, **row_dict})

    n_scored = len(scored_idx)
    n_dropped = len(unscored_records) - len(kept_unscored)
    print(f"  [{cfg_path.stem}] kept {new_n} ({n_scored} scored + "
          f"{len(kept_unscored)} unscored); dropped {n_dropped} unscored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS))
    ap.add_argument("--per-config-unscored", type=int, default=11)
    ap.add_argument("--total-budget", type=int, default=None,
                    help="If set, overrides --per-config-unscored as "
                         "ceil(total/n_configs)")
    ap.add_argument("--config", action="append", default=None)
    args = ap.parse_args()

    cfg_paths = sorted(Path(args.configs_dir).glob("*.yaml"))
    if args.config:
        cfg_paths = [p for p in cfg_paths if p.stem in set(args.config)]

    if args.total_budget is not None and cfg_paths:
        per = max(1, args.total_budget // len(cfg_paths))
    else:
        per = args.per_config_unscored

    print(f"Trim target: {per} unscored rows per config × {len(cfg_paths)} configs "
          f"= {per * len(cfg_paths)} unscored total")
    for p in cfg_paths:
        trim_config(p, per)


if __name__ == "__main__":
    main()
