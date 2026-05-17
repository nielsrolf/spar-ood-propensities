#!/usr/bin/env python3
"""
Push Tinker checkpoints to Hugging Face Hub.

By default, pushes only the LAST epoch per run (final checkpoint).
Use --all-epochs to push every epoch.

Usage:
    python push_checkpoints_to_hf.py --hf-user YOUR_HF_USERNAME
    python push_checkpoints_to_hf.py --hf-user YOUR_HF_USERNAME --all-epochs
    python push_checkpoints_to_hf.py --hf-user YOUR_HF_USERNAME --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent / "log_path2"


def get_run_name(run_dir: Path) -> str:
    """Convert directory name to a HF-safe repo name."""
    name = run_dir.name
    # Replace characters not allowed in HF repo names
    name = name.replace("_", "-").replace(":", "-")
    return name


def push_checkpoint(state_path: str, repo_id: str, epoch_name: str, dry_run: bool, env: dict) -> bool:
    cmd = [
        "tinker", "checkpoint", "push-hf",
        state_path,
        "--repo", repo_id,
        "--commit-message", f"Upload {epoch_name}",
    ]
    print(f"  {'[DRY RUN] ' if dry_run else ''}push-hf {state_path} -> {repo_id}")
    if dry_run:
        return True
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print(f"  [ERROR] exit code {result.returncode}", file=sys.stderr)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Push Tinker checkpoints to HuggingFace Hub")
    parser.add_argument("--hf-user", required=True, help="Your HuggingFace username or org")
    parser.add_argument("--api-key", help="Tinker API key (or set TINKER_API_KEY env var)")
    parser.add_argument("--all-epochs", action="store_true", help="Push all epochs, not just the last")
    parser.add_argument("--epochs", nargs="+", type=int, metavar="N", help="Push specific epoch numbers (e.g. --epochs 6 12 24)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--runs", nargs="*", help="Specific run directory names to process (default: all)")
    args = parser.parse_args()

    if args.epochs and args.all_epochs:
        print("Error: --epochs and --all-epochs are mutually exclusive")
        sys.exit(1)

    # Build environment for subprocess, injecting the API key if provided
    env = os.environ.copy()
    if args.api_key:
        env["TINKER_API_KEY"] = args.api_key
    elif not env.get("TINKER_API_KEY"):
        print("Error: TINKER_API_KEY is not set. Either:\n"
              "  export TINKER_API_KEY=your_key\n"
              "  or pass --api-key your_key")
        sys.exit(1)

    checkpoint_files = sorted(LOG_DIR.glob("*/checkpoints.jsonl"))

    if args.runs:
        checkpoint_files = [f for f in checkpoint_files if f.parent.name in args.runs]

    mode = f"epochs {args.epochs}" if args.epochs else ("all epochs" if args.all_epochs else "final epoch only")
    print(f"Found {len(checkpoint_files)} runs to process")
    print(f"Mode: {mode}\n")

    successes, failures = 0, []

    for ckpt_file in checkpoint_files:
        run_dir = ckpt_file.parent
        run_name = get_run_name(run_dir)
        repo_id = f"{args.hf_user}/tinker-{run_name}"

        with open(ckpt_file) as f:
            checkpoints = [json.loads(line) for line in f if line.strip()]

        if not checkpoints:
            print(f"[SKIP] {run_dir.name} — empty checkpoints.jsonl")
            continue

        if args.epochs:
            epochs_to_push = [c for c in checkpoints if c["epoch"] in args.epochs]
            missing = set(args.epochs) - {c["epoch"] for c in epochs_to_push}
            if missing:
                print(f"  [WARN] epochs not found in {run_dir.name}: {sorted(missing)}")
        elif args.all_epochs:
            epochs_to_push = checkpoints
        else:
            epochs_to_push = [checkpoints[-1]]

        print(f"[RUN] {run_dir.name}  ({len(epochs_to_push)} epoch(s) -> {repo_id})")
        for ckpt in epochs_to_push:
            epoch_name = ckpt.get("name", f"epoch_{ckpt['epoch']:02d}")
            sampler_path = ckpt["sampler_path"]
            ok = push_checkpoint(sampler_path, repo_id, epoch_name, args.dry_run, env)
            if ok:
                successes += 1
            else:
                failures.append((run_dir.name, epoch_name))

    print(f"\nDone: {successes} pushed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for run, epoch in failures:
            print(f"  {run} / {epoch}")
        sys.exit(1)


if __name__ == "__main__":
    main()
