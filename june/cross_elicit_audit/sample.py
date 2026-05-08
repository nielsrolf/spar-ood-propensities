#!/usr/bin/env python3
"""
Sample cross-elicit rows for human review.

Strategy:
  - 360-row total budget across the available traits (configurable).
  - Per-trait budget = budget / n_traits (floor).
  - Within a trait, prefer FT rows. If a trait only has base rows we still
    sample them (they're new ground truth on cross-elicit's questions, even
    if checkpoint metadata says base). Stratify by score bucket; if score is
    null/missing, those go in a separate `null_score` bucket capped at ~15%.
  - If the available pool is smaller than the per-trait budget, take everything.

Outputs per (config_name, trait):
  output/<config_name>/sample_<N>.csv         — full with score
  output/<config_name>/sample_<N>_blind.csv   — no score, has empty human_label

We import propensity_audit's `from_yaml` so the output_dir/metadata_columns/etc
match what annotate.py expects.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "june/propensity_audit"))
from audit_config import from_yaml  # noqa: E402

DEFAULT_CONFIGS = Path(__file__).parent / "configs"


def stratified_pick(df: pd.DataFrame, score_col: str, n: int, seed: int) -> pd.DataFrame:
    """Pick `n` rows stratified by 5 score bins (and a `null_score` bin)."""
    rng = np.random.RandomState(seed)
    if len(df) <= n:
        return df.sample(frac=1, random_state=rng).reset_index(drop=True)

    df = df.copy()
    s = pd.to_numeric(df[score_col], errors="coerce")
    bins = pd.cut(
        s, bins=[-1, 20, 40, 60, 80, 101],
        labels=["very_low", "low", "mid", "high", "very_high"],
        include_lowest=True,
    )
    df["_bin"] = bins.astype(object)
    df.loc[s.isna(), "_bin"] = "null_score"

    null_target = max(1, int(round(n * 0.15)))
    bin_target_base = (n - null_target) // 5
    bin_remainder = (n - null_target) - 5 * bin_target_base
    bin_names = ["very_low", "low", "mid", "high", "very_high"]
    targets = {b: bin_target_base for b in bin_names}
    # Distribute remainder across bins so floor-rounding doesn't lose budget.
    for i in range(bin_remainder):
        targets[bin_names[i]] += 1
    targets["null_score"] = null_target

    picks = []
    leftover = 0
    for bin_name, t in targets.items():
        pool = df[df["_bin"] == bin_name]
        take = min(t, len(pool))
        if take > 0:
            picks.append(pool.sample(n=take, random_state=rng))
        leftover += t - take

    # Top-up from anything not yet picked
    if leftover > 0:
        chosen_idx = pd.concat(picks).index if picks else pd.Index([])
        rest = df.drop(index=chosen_idx)
        take = min(leftover, len(rest))
        if take > 0:
            picks.append(rest.sample(n=take, random_state=rng))

    out = pd.concat(picks, ignore_index=True) if picks else df.head(0)
    out = out.drop(columns=["_bin"]).sample(frac=1, random_state=rng).reset_index(drop=True)
    return out


def sample_for_config(cfg_path: Path, per_config_n: int) -> tuple[Path, int] | None:
    cfg = from_yaml(cfg_path)
    df = pd.read_csv(cfg.data_path, low_memory=False)
    metric_name = cfg.propensity  # e.g. "cooperation_score" or "spite_score"
    if "metric" in df.columns and (df["metric"] == metric_name).any():
        df = df[df["metric"] == metric_name].copy()
    if df.empty:
        print(f"  [{cfg_path.stem}] no rows for metric {metric_name}, skipping")
        return None

    # Prefer FT, fall back to whatever exists
    ft = df[df["epoch_class"] == "ft"]
    pool = ft if len(ft) >= per_config_n else df

    sampled = stratified_pick(pool, score_col=cfg.score_column,
                              n=per_config_n, seed=cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = cfg.output_dir / f"sample_{len(sampled)}.csv"
    sampled.to_csv(full_path, index=False)

    blind_cols = ["question", "response"] + [c for c in cfg.metadata_columns if c in sampled.columns]
    blind = sampled[blind_cols].copy()
    blind["human_label"] = ""
    blind_path = cfg.output_dir / f"sample_{len(sampled)}_blind.csv"
    blind.to_csv(blind_path, index=False)
    print(f"  [{cfg_path.stem}] sampled {len(sampled)} (pool {len(pool)}) → {full_path.name}")
    return full_path, len(sampled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS))
    ap.add_argument("--total-budget", type=int, default=360)
    ap.add_argument("--config", action="append", default=None,
                    help="Restrict to specific config name(s) (without .yaml). May repeat.")
    args = ap.parse_args()

    configs_dir = Path(args.configs_dir)
    cfg_paths = sorted(configs_dir.glob("*.yaml"))
    if args.config:
        wanted = set(args.config)
        cfg_paths = [p for p in cfg_paths if p.stem in wanted]
    if not cfg_paths:
        print("No configs found.")
        sys.exit(1)

    # Compute available pool size per config so we can redistribute budget
    # from configs whose pool < per-config target → configs with bigger pools.
    pool_sizes = {}
    for p in cfg_paths:
        cfg = from_yaml(p)
        df = pd.read_csv(cfg.data_path, low_memory=False)
        if "metric" in df.columns and (df["metric"] == cfg.propensity).any():
            df = df[df["metric"] == cfg.propensity]
        ft = df[df["epoch_class"] == "ft"] if "epoch_class" in df.columns else df
        pool_sizes[p] = len(ft) if len(ft) > 0 else len(df)

    # Two-pass allocation: floor allocate, then redistribute leftover among
    # configs that still have headroom.
    base_share = args.total_budget // len(cfg_paths)
    alloc = {p: min(base_share, pool_sizes[p]) for p in cfg_paths}
    leftover = args.total_budget - sum(alloc.values())
    while leftover > 0:
        # Find configs with headroom
        headroom = [p for p in cfg_paths if alloc[p] < pool_sizes[p]]
        if not headroom:
            break
        per = max(1, leftover // len(headroom))
        progressed = False
        for p in headroom:
            room = pool_sizes[p] - alloc[p]
            add = min(per, room, leftover)
            if add > 0:
                alloc[p] += add
                leftover -= add
                progressed = True
            if leftover == 0:
                break
        if not progressed:
            break

    print(f"Budget {args.total_budget} across {len(cfg_paths)} configs "
          f"(base share {base_share}, redistributed leftover)")
    total = 0
    for p in cfg_paths:
        result = sample_for_config(p, alloc[p])
        if result is not None:
            total += result[1]
    print(f"Total sampled: {total}")


if __name__ == "__main__":
    main()
