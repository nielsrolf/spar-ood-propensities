"""Check whether each model in cross-elicit/models/ finished training.

A model is considered finished iff its run_config.json declares
`hyperparams.num_epochs = N` and its checkpoint_manifest.json contains
checkpoint entries for every epoch 1..N.

Modes:
  flagonly (default): print unfinished runs and what's missing.
  delete:             rm -rf the unfinished model directories AND every
                      eval_results directory whose name embeds the run
                      via the marker `_eval__<run_name>__`.

Usage:
  python check_finished_runs.py
  python check_finished_runs.py --mode delete
"""

import argparse
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(ROOT, "models")
EVAL_RESULTS_DIR = os.path.join(ROOT, "eval_results", "finetuning")

EVAL_MARKER = "_eval__"


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def check_run(run_dir: str) -> tuple[bool, str, int | None]:
    """Return (finished, reason_if_unfinished, num_epochs_if_known)."""
    run_config_path = os.path.join(run_dir, "run_config.json")
    manifest_path = os.path.join(run_dir, "checkpoint_manifest.json")

    if not os.path.isfile(run_config_path):
        return False, "missing run_config.json", None

    try:
        run_config = load_json(run_config_path)
    except Exception as exc:
        return False, f"unreadable run_config.json: {exc}", None

    try:
        num_epochs = int(run_config["hyperparams"]["num_epochs"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"hyperparams.num_epochs missing/invalid: {exc}", None

    if not os.path.isfile(manifest_path):
        return False, f"missing checkpoint_manifest.json (expected {num_epochs} epochs)", num_epochs

    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        return False, f"unreadable checkpoint_manifest.json: {exc}", num_epochs

    if not isinstance(manifest, list):
        return False, "checkpoint_manifest.json is not a list", num_epochs

    present = {int(entry["epoch"]) for entry in manifest if isinstance(entry, dict) and "epoch" in entry}
    missing = sorted(set(range(1, num_epochs + 1)) - present)
    if missing:
        return False, f"missing epochs {missing} (expected 1..{num_epochs})", num_epochs
    return True, "", num_epochs


def find_eval_dirs_for_run(run_name: str) -> list[str]:
    """All eval_results subdirs whose name contains '_eval__<run_name>__'."""
    if not os.path.isdir(EVAL_RESULTS_DIR):
        return []
    needle = f"{EVAL_MARKER}{run_name}__"
    return sorted(
        os.path.join(EVAL_RESULTS_DIR, d)
        for d in os.listdir(EVAL_RESULTS_DIR)
        if needle in d and os.path.isdir(os.path.join(EVAL_RESULTS_DIR, d))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["flagonly", "delete"],
        default="flagonly",
        help="flagonly (default) prints unfinished runs; delete removes them and their eval_results.",
    )
    args = parser.parse_args()

    if not os.path.isdir(MODELS_DIR):
        print(f"models dir not found: {MODELS_DIR}", file=sys.stderr)
        return 2

    run_names = sorted(
        d for d in os.listdir(MODELS_DIR)
        if not d.startswith(("_", "."))
        and os.path.isdir(os.path.join(MODELS_DIR, d))
    )

    unfinished: list[tuple[str, str]] = []
    for run_name in run_names:
        run_dir = os.path.join(MODELS_DIR, run_name)
        finished, reason, _ = check_run(run_dir)
        if not finished:
            unfinished.append((run_name, reason))

    print(f"Scanned {len(run_names)} runs in {MODELS_DIR}")
    print(f"Finished:   {len(run_names) - len(unfinished)}")
    print(f"Unfinished: {len(unfinished)}")
    print()

    if not unfinished:
        print("All runs finished. Nothing to do.")
        return 0

    for run_name, reason in unfinished:
        eval_dirs = find_eval_dirs_for_run(run_name)
        print(f"[UNFINISHED] {run_name}")
        print(f"             reason: {reason}")
        print(f"             eval_results dirs: {len(eval_dirs)}")
        for d in eval_dirs:
            print(f"               - {os.path.basename(d)}")

    if args.mode != "delete":
        print()
        print("(flagonly mode — nothing deleted; rerun with --mode delete to remove these.)")
        return 0

    print()
    print("=== DELETING ===")
    for run_name, _ in unfinished:
        run_dir = os.path.join(MODELS_DIR, run_name)
        eval_dirs = find_eval_dirs_for_run(run_name)
        print(f"rm -rf {run_dir}")
        shutil.rmtree(run_dir, ignore_errors=False)
        for d in eval_dirs:
            print(f"rm -rf {d}")
            shutil.rmtree(d, ignore_errors=False)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
