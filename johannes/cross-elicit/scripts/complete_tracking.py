"""Run judge_coherence then track_metrics for a single eval-results prefix.

Given a folder-name prefix of the shape

    {eval_propensity}_eval__{run}

(e.g. `effort_eval__effort-minus-meta-llama-Llama-3.1-8B-Instruct-2026-05-05-16-23-51`),
this script:

  1. invokes judge_coherence.py on the prefix — which resumes (skips
     already-judged answers) and auto-includes the base-model eval folder
     for the same propensity;
  2. invokes track_metrics.py --run <run> --eval-propensity <eval_propensity>
     — which already includes the base-model row.

Usage:
    python complete_tracking.py effort_eval__effort-minus-meta-llama-Llama-3.1-8B-Instruct-2026-05-05-16-23-51
    python complete_tracking.py <prefix> --judge-model gpt-5.4-mini --concurrency 8
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JUDGE_SCRIPT = os.path.join(SCRIPT_DIR, "judge_coherence.py")
TRACK_SCRIPT = os.path.join(SCRIPT_DIR, "track_metrics.py")

EVAL_MARKER = "_eval__"


def split_prefix(prefix: str) -> tuple[str, str]:
    base = os.path.basename(os.path.normpath(prefix))
    idx = base.find(EVAL_MARKER)
    if idx <= 0 or idx + len(EVAL_MARKER) >= len(base):
        raise SystemExit(
            f"Prefix {prefix!r} doesn't match shape '<eval_propensity>_eval__<run>'."
        )
    eval_propensity = base[:idx]
    run = base[idx + len(EVAL_MARKER):]
    return eval_propensity, run


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "prefix",
        help=(
            "Folder-name prefix '<eval_propensity>_eval__<run>' "
            "(e.g. effort_eval__effort-minus-meta-llama-Llama-3.1-8B-Instruct-2026-05-05-16-23-51)."
        ),
    )
    ap.add_argument("--judge-model", default=None,
                    help="Override judge_coherence.py --judge-model.")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Override judge_coherence.py --concurrency.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Pass --overwrite to judge_coherence.py (re-judge from scratch).")
    args = ap.parse_args()

    eval_propensity, run_id = split_prefix(args.prefix)
    print(f"prefix:          {args.prefix}")
    print(f"eval-propensity: {eval_propensity}")
    print(f"run:             {run_id}")

    judge_cmd = [sys.executable, JUDGE_SCRIPT, args.prefix]
    if args.judge_model:
        judge_cmd += ["--judge-model", args.judge_model]
    if args.concurrency is not None:
        judge_cmd += ["--concurrency", str(args.concurrency)]
    if args.overwrite:
        judge_cmd += ["--overwrite"]
    run(judge_cmd)

    track_cmd = [
        sys.executable, TRACK_SCRIPT,
        "--run", run_id,
        "--eval-propensity", eval_propensity,
    ]
    run(track_cmd)


if __name__ == "__main__":
    main()
