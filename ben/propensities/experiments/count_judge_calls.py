"""Count judge calls across all historical runs.

Scans both:
  * Eval-time judging — cell CSVs under results/cross_method_spillover/*/cells*/.
    For each cell, judge calls ≈ (# non-null score cells) × judge_n_samples,
    where judge_n_samples comes from the per-output-dir config.
  * Training-time judging — tinker_logs/*/metrics.jsonl under each output
    dir. Counts batches × batch_size × group_size × judge_n_samples.

Because the CSVs don't record `judge_n_samples`, we map each cells dir to
its config-known value (see DIR_TO_JUDGE_N_SAMPLES below). Update when new
output dirs land.

Usage:
    python experiments/count_judge_calls.py
    python experiments/count_judge_calls.py --root results/cross_method_spillover
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Map cells-directory name → judge_n_samples used when those CSVs were
# produced. (Derived from the spillover config in effect at the time.)
DIR_TO_JUDGE_N_SAMPLES = {
    # 4B Phase 2 anchor: spillover_qwen.yaml used n_samples_eval=3.
    ("qwen3_4b", "cells"): 3,
    # 8B-base current run: spillover_qwen8b_base.yaml uses n_samples_eval=1
    # (matches Johannes' setup). The two backups below come from earlier
    # configurations of the same output dir.
    ("qwen3_8b_base", "cells"): 1,
    ("qwen3_8b_base", "cells_gpt4o_backup"): 1,
    ("qwen3_8b_base", "cells_n3_backup"): 3,
    ("qwen3_8b_base", "cells_n10_backup"): 5,  # original FreeformQuestion default
    # Full-test-set 8B validations.
    ("qwen3_8b_base_full", "cells"): 3,
    # Smoke tests / single-trait GRPO experiments (Phase 2 development).
    ("qwen3_4b_grpo_aesthetic_b64", "cells"): 3,
    ("qwen3_4b_grpo_aesthetic_b64_step30", "cells"): 3,
    ("qwen3_4b_grpo_aesthetic_b64_step35", "cells"): 3,
    ("qwen3_4b_grpo_aesthetic_b64_step40", "cells"): 3,
    ("qwen3_4b_grpo_claiming_sentience", "cells"): 3,
    ("qwen3_4b_smoketest_grpo", "cells"): 3,
    # n_samples_eval=1 validation run (filename says so).
    ("qwen3_8b_base_judge_n1", "cells"): 1,
}

# Training judge calls per batch: batch_size × group_size × grpo_judge_n_samples.
# All configs we've used so far: 8 × 8 × 3 = 192 calls/batch.
DEFAULT_TRAIN_CALLS_PER_BATCH = 8 * 8 * 3


def _judge_n_samples_for(output_dir: str, cells_subdir: str) -> int | None:
    return DIR_TO_JUDGE_N_SAMPLES.get((output_dir, cells_subdir))


def count_eval_calls(root: Path) -> dict:
    """Walk every cells*/ directory and tally scores."""
    rows: list[dict] = []
    for output_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cells_dir in sorted(
            p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("cells")
        ):
            jns = _judge_n_samples_for(output_dir.name, cells_dir.name)
            if jns is None:
                print(
                    f"WARN: no judge_n_samples mapping for "
                    f"{output_dir.name}/{cells_dir.name} — skipping. Add it to DIR_TO_JUDGE_N_SAMPLES."
                )
                continue
            n_cells = n_score_values = 0
            for cell_path in cells_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(cell_path, low_memory=False)
                except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                    continue
                # Score columns end in _score; count non-null entries.
                score_cols = [c for c in df.columns if c.endswith("_score")]
                if not score_cols:
                    continue
                n_score_values += int(df[score_cols].notna().sum().sum())
                n_cells += 1
            rows.append(
                {
                    "output_dir": output_dir.name,
                    "cells_dir": cells_dir.name,
                    "judge_n_samples": jns,
                    "n_cells": n_cells,
                    "n_score_values": n_score_values,
                    "judge_calls": n_score_values * jns,
                }
            )
    return {"per_dir": rows, "total": sum(r["judge_calls"] for r in rows)}


def count_train_calls(root: Path) -> dict:
    """Sum batches across every tinker_logs/*/metrics.jsonl file."""
    rows: list[dict] = []
    for output_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        log_root = output_dir / "tinker_logs"
        if not log_root.is_dir():
            continue
        per_trait: dict[str, int] = {}
        for trait_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
            metrics_path = trait_dir / "metrics.jsonl"
            if not metrics_path.exists():
                continue
            n_batches = 0
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Each row that records `reward/total` corresponds to one
                    # GRPO step; one rollout batch per step.
                    if "reward/total" in rec or "progress/batch" in rec:
                        n_batches += 1
            per_trait[trait_dir.name] = n_batches
        if per_trait:
            total_batches = sum(per_trait.values())
            rows.append(
                {
                    "output_dir": output_dir.name,
                    "n_traits": len(per_trait),
                    "total_batches": total_batches,
                    "calls_per_batch": DEFAULT_TRAIN_CALLS_PER_BATCH,
                    "judge_calls": total_batches * DEFAULT_TRAIN_CALLS_PER_BATCH,
                }
            )
    return {"per_dir": rows, "total": sum(r["judge_calls"] for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("results/cross_method_spillover")
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Not a directory: {args.root}")

    print(f"Scanning {args.root}\n")

    print("=" * 78)
    print("EVAL-TIME JUDGING (cell CSVs)")
    print("=" * 78)
    eval_res = count_eval_calls(args.root)
    eval_df = pd.DataFrame(eval_res["per_dir"])
    if not eval_df.empty:
        print(eval_df.to_string(index=False))
        print(f"\nEval total: {eval_res['total']:,} judge calls")
    else:
        print("(no eval cells found)")

    print()
    print("=" * 78)
    print("TRAINING-TIME JUDGING (tinker_logs/metrics.jsonl)")
    print("=" * 78)
    train_res = count_train_calls(args.root)
    train_df = pd.DataFrame(train_res["per_dir"])
    if not train_df.empty:
        print(train_df.to_string(index=False))
        print(f"\nTraining total: {train_res['total']:,} judge calls")
    else:
        print("(no training logs found)")

    grand = eval_res["total"] + train_res["total"]
    print()
    print("=" * 78)
    print(f"GRAND TOTAL: {grand:,} judge calls")
    print(
        f"  Eval-time:     {eval_res['total']:>12,} "
        f"({eval_res['total'] / grand * 100:5.1f}%)"
    )
    print(
        f"  Training-time: {train_res['total']:>12,} "
        f"({train_res['total'] / grand * 100:5.1f}%)"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
