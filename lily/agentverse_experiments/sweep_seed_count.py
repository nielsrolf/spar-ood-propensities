"""
Sweep seed_count from --min to --max for a given seed model, then produce a comparison plot.

Usage:
    python lily/agentverse_experiments/sweep_seed_count.py --seed_model spitefulness
    python lily/agentverse_experiments/sweep_seed_count.py --seed_model spitefulness --min 1 --max 5
    python lily/agentverse_experiments/sweep_seed_count.py --seed_model spitefulness --generations 10 --mutation_rate 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RUNNER      = Path(__file__).parent / "run_evolution.py"
RESULTS_DIR = Path(__file__).parent / "results"


def count_completed(seed_model: str, seed_count: int) -> int:
    """Return number of completed summary JSONs for this seed_model + seed_count."""
    return sum(1 for _ in RESULTS_DIR.glob(f"evolution_{seed_model}_n{seed_count}_*_summary.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed_model",    required=True, help="Short propensity name (e.g. spitefulness)")
    parser.add_argument("--min",           type=int, default=1,   dest="min_seed", help="Starting seed_count (default: 1)")
    parser.add_argument("--max",           type=int, default=5,   dest="max_seed", help="Ending seed_count inclusive (default: 5)")
    parser.add_argument("--reps",          type=int, default=1,   help="Repetitions per seed_count for variance reduction (default: 1)")
    parser.add_argument("--generations",   type=int, default=10,  help="Generations per run (default: 10)")
    parser.add_argument("--mutation_rate", type=float, default=0, help="Mutation rate per run (default: 0)")
    parser.add_argument("--concurrency",   type=int, default=3,   help="Parallel matches per generation (default: 3)")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print(f"Sweep: seed_model={args.seed_model}, seed_count {args.min_seed}..{args.max_seed}, "
          f"reps={args.reps}")
    print(f"       generations={args.generations}, mutation_rate={args.mutation_rate}")
    print("=" * 50)

    for count in range(args.min_seed, args.max_seed + 1):
        done = count_completed(args.seed_model, count)
        remaining = args.reps - done
        if remaining <= 0:
            print(f"\n--- seed_count={count} — {done}/{args.reps} reps done, skipping ---")
            continue
        print(f"\n--- seed_count={count} — {done}/{args.reps} reps done, running {remaining} more ---")
        for _ in range(remaining):
            run([
                sys.executable, str(RUNNER), "run",
                "--seed_model",    args.seed_model,
                "--seed_count",    str(count),
                "--generations",   str(args.generations),
                "--mutation_rate", str(args.mutation_rate),
                "--concurrency",   str(args.concurrency),
            ])

    print("\n--- Generating comparison plot ---")
    run([sys.executable, str(RUNNER), "compare", "--seed_model", args.seed_model])

    print(f"\nAll done. Results in {RUNNER.parent / 'results'}/")


if __name__ == "__main__":
    main()
