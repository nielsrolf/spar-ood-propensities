"""Run judge_coherence.py + track_metrics.py on every model directory,
evaluated against the propensity that model was trained on.

For each subdirectory in ../models that matches the pattern
    <eval_propensity>-(plus|minus)-<base_model>-<YYYY-MM-DD-HH-MM-SS>
we invoke:
    python judge_coherence.py <eval_propensity>_eval__<dir_name>
    python track_metrics.py --run <dir_name> --eval-propensity <eval_propensity>

The judge step resumes by default (already-judged answers are skipped) and
auto-includes the matching base-model eval folder, so re-running this script
will only judge new rows and the metrics_table.txt files will include the
coherence columns.

Directories whose name starts with '_' (e.g. _staging, _parallel_logs) are
skipped, as are any names that don't match the expected pattern.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRACK_SCRIPT = SCRIPT_DIR / "track_metrics.py"
JUDGE_SCRIPT = SCRIPT_DIR / "judge_coherence.py"
MODELS_DIR = SCRIPT_DIR.parent / "models"

# Restrict processing to models finetuned for these poles (propensities).
# Set to None to process all poles (default).
POLES: list[str] | None = ["ethical-framework-virtue-ethics", "ethical-framework-utilitarian", "ethical-framework-deontological"]

RUN_SPLIT_RE = re.compile(r"-(plus|minus)-")


def derive_eval_propensity(run: str) -> str | None:
    parts = RUN_SPLIT_RE.split(run, maxsplit=1)
    if len(parts) != 3:
        return None
    return parts[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(MODELS_DIR),
                    help=f"Models directory to scan (default: {MODELS_DIR}).")
    ap.add_argument("--continue-on-error", action="store_true", default=True,
                    help="Continue past failing runs (default: on).")
    ap.add_argument("--stop-on-error", dest="continue_on_error",
                    action="store_false",
                    help="Abort on the first failing track_metrics call.")
    ap.add_argument("--skip-coherence", action="store_true",
                    help="Don't run judge_coherence.py before track_metrics.py.")
    ap.add_argument("--judge-model", default=None,
                    help="Override judge_coherence.py --judge-model.")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Override judge_coherence.py --concurrency.")
    ap.add_argument("--overwrite-coherence", action="store_true",
                    help="Pass --overwrite to judge_coherence.py (re-judge from scratch).")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        raise SystemExit(f"Models directory not found: {models_dir}")

    runs: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("_"):
            skipped.append((name, "starts with _"))
            continue
        prop = derive_eval_propensity(name)
        if prop is None:
            skipped.append((name, "no -plus-/-minus- in name"))
            continue
        if POLES is not None and prop not in POLES:
            skipped.append((name, f"pole {prop!r} not in POLES filter"))
            continue
        runs.append((name, prop))

    print(f"models dir: {models_dir}")
    print(f"found {len(runs)} runs, skipping {len(skipped)}")
    for name, reason in skipped:
        print(f"  skip: {name}  ({reason})")

    failures: list[tuple[str, str, int]] = []
    for i, (run, prop) in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] run={run}  eval-propensity={prop}")

        if not args.skip_coherence:
            prefix = f"{prop}_eval__{run}"
            judge_cmd = [sys.executable, str(JUDGE_SCRIPT), prefix]
            if args.judge_model:
                judge_cmd += ["--judge-model", args.judge_model]
            if args.concurrency is not None:
                judge_cmd += ["--concurrency", str(args.concurrency)]
            if args.overwrite_coherence:
                judge_cmd += ["--overwrite"]
            print("$ " + " ".join(judge_cmd), flush=True)
            jresult = subprocess.run(judge_cmd)
            if jresult.returncode != 0:
                failures.append((run, "judge_coherence", jresult.returncode))
                msg = f"judge_coherence failed (exit {jresult.returncode}) for {run}"
                if args.continue_on_error:
                    print("  " + msg + " — continuing to track_metrics anyway")
                else:
                    raise SystemExit(msg)

        cmd = [sys.executable, str(TRACK_SCRIPT),
               "--run", run, "--eval-propensity", prop]
        print("$ " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((run, "track_metrics", result.returncode))
            msg = f"track_metrics failed (exit {result.returncode}) for {run}"
            if args.continue_on_error:
                print("  " + msg + " — continuing")
            else:
                raise SystemExit(msg)

    print("\n=== summary ===")
    print(f"runs processed: {len(runs)}")
    print(f"failures:       {len(failures)}")
    for run, stage, code in failures:
        print(f"  {run} [{stage}] (exit {code})")
    if failures and not args.continue_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
