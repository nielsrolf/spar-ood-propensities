"""
Run training.py for each propensity on a Qwen base model and collect
the saved Tinker checkpoint paths and learning curve file paths.

Usage:
    python src/scripts/tinker/run_multi_training.py --variant qwen8b
    python src/scripts/tinker/run_multi_training.py --variant qwen30b
    python src/scripts/tinker/run_multi_training.py --variant qwen8b --dry-run
    python src/scripts/tinker/run_multi_training.py --variant qwen30b --propensities sycophancy consistency
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TRAINING_SCRIPT = SCRIPT_DIR / "training.py"

VARIANTS = {
    "qwen8b": {
        "base_model": "Qwen/Qwen3-8B",
        "renderer": "qwen3_instruct",
        "propensities": [
            "sycophancy",
            "corrigibility",
            "consistency",
            "cooperation",
            "narcissism",
            "risk_affinity",
            "spitefulness",
        ],
        "output_file": SCRIPT_DIR / "qwen8b_training_results.json",
    },
    "qwen30b": {
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "renderer": "qwen3_instruct",
        # Remaining propensities not yet fine-tuned on 30B.
        # Already completed: corrigibility, power_seeking, narcissism (+ counter-training combos).
        "propensities": [
            "self_preservation",
            "sycophancy",
            "consistency",
            "cooperation",
            "risk_affinity",
            "spitefulness",
        ],
        "output_file": SCRIPT_DIR / "qwen30b_training_results.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=list(VARIANTS), required=True,
        help="Which Qwen base model to train on",
    )
    parser.add_argument(
        "--propensities", nargs="+", default=None,
        help="Override the default propensity list for this variant",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running them",
    )
    return parser.parse_args()


def run_training(propensity: str, base_model: str, renderer: str, dry_run: bool) -> dict:
    cmd = [
        sys.executable, str(TRAINING_SCRIPT),
        "--propensity-prefix", propensity,
        "--base-model", base_model,
        "--renderer", renderer,
    ]
    print(f"\n{'=' * 70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] START  {propensity}")
    print(f"CMD: {' '.join(cmd)}")
    print("=" * 70, flush=True)

    if dry_run:
        return {
            "propensity": propensity,
            "status": "dry_run",
            "tinker_paths": [],
            "plot_path": None,
        }

    tinker_paths = []
    plot_path = None

    env = {**__import__("os").environ, "MPLBACKEND": "Agg"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(SCRIPT_DIR),
        env=env,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)

        m = re.search(r"Saved:\s+(tinker://\S+)", line)
        if m:
            tinker_paths.append(m.group(1))

        m = re.search(r"Learning curve saved:\s+(\S+)", line)
        if m:
            plot_path = m.group(1)

    proc.wait()
    status = "success" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status.upper()}  {propensity}", flush=True)

    return {
        "propensity": propensity,
        "status": status,
        "tinker_paths": tinker_paths,
        "plot_path": plot_path,
    }


def main() -> None:
    args = parse_args()
    cfg = VARIANTS[args.variant]

    base_model = cfg["base_model"]
    renderer = cfg["renderer"]
    propensities = args.propensities or cfg["propensities"]
    output_file = cfg["output_file"]

    print(f"Variant:    {args.variant}")
    print(f"Base model: {base_model}")
    print(f"Output:     {output_file}")

    # Load any previously completed results so we can skip them on resume
    if output_file.exists():
        results = json.loads(output_file.read_text())
        completed = {r["propensity"] for r in results if r["status"] == "success"}
        if completed:
            print(f"Resuming — skipping already completed: {sorted(completed)}")
    else:
        results = []
        completed = set()

    for propensity in propensities:
        if propensity in completed:
            print(f"[SKIP] {propensity} already completed")
            continue
        result = run_training(propensity, base_model, renderer, dry_run=args.dry_run)
        results.append(result)

        # Write incrementally so partial results are saved if something fails
        output_file.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {output_file}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"\n{r['propensity']}  [{r['status']}]")
        for i, p in enumerate(r["tinker_paths"]):
            print(f"  epoch-{i}: {p}")
        if r["plot_path"]:
            print(f"  plot:    {r['plot_path']}")

    print(f"\nFull results written to: {output_file}")


if __name__ == "__main__":
    main()
