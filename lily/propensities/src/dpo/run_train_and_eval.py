#!/usr/bin/env python3
"""Wrapper: train online DPO for a set of traits, then run the full cross-
elicitation eval matrix. Designed for unattended overnight runs.

Default behaviour (no args):
  • TRAIN the 5 traits not yet completed (cooperation, self-preservation,
    power-seeking, claiming-superintelligence, harm-refusal).
  • EVAL all 9 paper traits against all 9 evals (81 cells); the 36 cells
    already produced by previous runs are auto-skipped by run_dpo_evals.py,
    so only the ~45 new cells actually run.

Usage:
    python run_train_and_eval.py                  # default: 5 remaining traits
    python run_train_and_eval.py --traits cooperation power-seeking
    python run_train_and_eval.py --dry-run        # preflight only
    python run_train_and_eval.py --train-only     # skip eval
    python run_train_and_eval.py --eval-only      # skip training (already done)
    python run_train_and_eval.py --max-workers 6  # bump eval parallelism

Output:
  output/wrapper_logs/{train,eval}_<timestamp>.log  — full subprocess output
  Final SFT/DPO/Delta matrices printed at end (from the eval subprocess)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
TRAIN_SCRIPT = HERE / "train_online_dpo.py"
EVAL_SCRIPT  = HERE / "run_dpo_evals.py"

# Traits not yet completed as of 2026-05-18 evening.
# Done so far: spitefulness, neuroticism, harm-elaboration, honest-humble.
REMAINING_TRAITS = [
    "cooperation",
    "self-preservation",
    "power-seeking",
    "claiming-superintelligence",
    "harm-refusal",
]
ALL_PAPER_TRAITS = [
    "spitefulness", "neuroticism", "harm-elaboration", "honest-humble",
] + REMAINING_TRAITS


def run_subprocess(cmd: list[str], log_path: Path, label: str) -> bool:
    """Run subprocess, tee'ing output to stdout + log_path."""
    print(f"\n{'=' * 72}")
    print(f"  [{label}] START at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  cmd:  {' '.join(cmd)}")
    print(f"  log:  {log_path}")
    print(f"{'=' * 72}\n")
    sys.stdout.flush()
    start = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        proc.wait()
    elapsed = (time.time() - start) / 60
    status = "✓ OK" if proc.returncode == 0 else f"✗ EXIT {proc.returncode}"
    print(f"\n  [{label}] {status} in {elapsed:.1f} min  (log: {log_path.name})\n")
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traits", nargs="+", default=REMAINING_TRAITS,
                    help=f"Traits to train (default: 5 remaining = {REMAINING_TRAITS})")
    ap.add_argument("--eval-traits", nargs="+", default=ALL_PAPER_TRAITS,
                    help="Eval columns to run against (default: all 9 paper traits)")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="Parallel eval cells (default: 4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preflight only — show what would run")
    ap.add_argument("--train-only", action="store_true",
                    help="Skip the eval phase")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training (use already-trained checkpoints)")
    args = ap.parse_args()

    if args.train_only and args.eval_only:
        sys.exit("error: --train-only and --eval-only are mutually exclusive")

    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = OUTPUT_DIR / "wrapper_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== plan ===")
    print(f"  timestamp:   {ts}")
    print(f"  train:       {'skip' if args.eval_only else args.traits}")
    print(f"  eval rows:   {'skip' if args.train_only else args.traits}")
    print(f"  eval cols:   {args.eval_traits if not args.train_only else 'skip'}")
    print(f"  max-workers: {args.max_workers}")
    print(f"  log dir:     {log_dir}")

    if args.dry_run:
        print(f"\n[dry-run] commands that would execute:")
        if not args.eval_only:
            print(f"  python {TRAIN_SCRIPT.name} --traits {' '.join(args.traits)}")
        if not args.train_only:
            print(f"  python {EVAL_SCRIPT.name} --trained {' '.join(args.traits)} "
                  f"--evals {' '.join(args.eval_traits)} "
                  f"--mode online_dpo --max-workers {args.max_workers}")
        return

    overall_start = time.time()
    train_ok = eval_ok = True

    # ── Phase 1: training ────────────────────────────────────────────────
    if not args.eval_only:
        train_log = log_dir / f"train_{ts}.log"
        train_cmd = ["python", str(TRAIN_SCRIPT), "--traits", *args.traits]
        train_ok = run_subprocess(train_cmd, train_log, "TRAIN")
        if not train_ok:
            print("\n⚠ Training subprocess exited non-zero. Continuing to eval anyway —")
            print("  any traits whose training crashed will simply have no checkpoint")
            print("  and will be skipped by run_dpo_evals.py with a SKIP message.")

    # ── Phase 2: eval ────────────────────────────────────────────────────
    if not args.train_only:
        eval_log = log_dir / f"eval_{ts}.log"
        eval_cmd = [
            "python", str(EVAL_SCRIPT),
            "--trained", *args.traits,
            "--evals", *args.eval_traits,
            "--mode", "online_dpo",
            "--max-workers", str(args.max_workers),
        ]
        eval_ok = run_subprocess(eval_cmd, eval_log, "EVAL")

    # ── Summary ──────────────────────────────────────────────────────────
    total_min = (time.time() - overall_start) / 60
    print(f"\n{'=' * 72}")
    print(f"  WRAPPER DONE in {total_min:.1f} min")
    if not args.eval_only:
        print(f"  train: {'✓' if train_ok else '✗'}")
    if not args.train_only:
        print(f"  eval:  {'✓' if eval_ok else '✗'}")
    print(f"  logs:  {log_dir}/{{train,eval}}_{ts}.log")
    print(f"{'=' * 72}")
    sys.exit(0 if (train_ok and eval_ok) else 1)


if __name__ == "__main__":
    main()
