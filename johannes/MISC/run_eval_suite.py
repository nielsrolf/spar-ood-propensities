"""
Run the full eval suite for one or more checkpoints from a training run,
or for a plain base model (no fine-tuning).

Results are stored inside the training run directory (Option A):
  log_path2/<run_dir>/
    epoch_01/
      answers_first_5.jsonl
      answers_normal_questions.jsonl
      all_answers.jsonl
      quality_eval/
        results_gpt-4o_<timestamp>.json
    epoch_05/
      ...

The model name is read automatically from the run's config.txt.
The sampler_path for each epoch is read from checkpoints.jsonl.

Usage:
    # Evaluate epoch 5 of a run:
    python run_eval_suite.py --run_dir log_path2/2026-03-15_11-01-29 --epochs 5

    # Evaluate epochs 1, 5 and 10:
    python run_eval_suite.py --run_dir log_path2/2026-03-15_11-01-29 --epochs 1 5 10

    # Evaluate all epochs:
    python run_eval_suite.py --run_dir log_path2/2026-03-15_11-01-29 --epochs all

    # Evaluate a base model (no checkpoint):
    python run_eval_suite.py --model meta-llama/Llama-3.1-8B-Instruct

    # Base model, custom output directory:
    python run_eval_suite.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --output_dir log_path2/base_llama

    # Skip specific evals:
    python run_eval_suite.py --run_dir log_path2/2026-03-15_11-01-29 --epochs all \\
        --skip_quality_eval
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH2  = os.path.join(SCRIPT_DIR, "log_path2")

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'═' * 60}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def merge_answers(paths: list[str], output_path: str) -> None:
    total = 0
    with open(output_path, "wb") as out:
        for p in paths:
            if not os.path.exists(p):
                print(f"  WARNING: {p} not found, skipping from merge")
                continue
            with open(p, "rb") as src:
                content = src.read()
            out.write(content)
            if content and not content.endswith(b"\n"):
                out.write(b"\n")
            total += content.count(b"\n")
    print(f"  Merged ~{total} lines → {output_path}")


def read_model_name(run_dir: str) -> str:
    """Read the model name from the run's config.txt."""
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path) as f:
        for line in f:
            key, _, value = line.partition(" ")
            if key.strip() == "model_name":
                return value.strip()
    raise ValueError(f"model_name not found in {config_path}")


def read_checkpoints(run_dir: str) -> dict[int, dict]:
    """Return {epoch: checkpoint_entry} from checkpoints.jsonl."""
    ckpt_path = os.path.join(run_dir, "checkpoints.jsonl")
    checkpoints = {}
    with open(ckpt_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            epoch = entry.get("epoch")
            if epoch is not None:
                checkpoints[int(epoch)] = entry
    return checkpoints


# ---------------------------------------------------------------------------
# Per-epoch eval
# ---------------------------------------------------------------------------

def run_epoch(
    epoch: int,
    sampler_path: str | None,
    model_name: str,
    run_dir: str,
    args: argparse.Namespace,
) -> None:
    epoch_dir = os.path.join(run_dir, f"epoch_{epoch:02d}")
    os.makedirs(epoch_dir, exist_ok=True)
    print(f"\n{'█' * 60}")
    if sampler_path:
        print(f"  Epoch {epoch:02d}  →  {epoch_dir}")
    else:
        print(f"  Base model  →  {epoch_dir}")
    print(f"{'█' * 60}")

    model_flags = ["--model", model_name]
    if sampler_path:
        model_flags += ["--checkpoint", sampler_path]
    sampling_flags = ["--max_tokens", str(args.max_tokens),
                      "--temperature", str(args.temperature)]

    # 1. generate_answers: first_5
    first_5_out = os.path.join(epoch_dir, "answers_first_5.jsonl")
    if args.skip_first_5:
        print("  Skipping first_5 eval.")
    else:
        _run(
            [sys.executable, "quality_eval_results/generate_answers.py"]
            + model_flags + sampling_flags
            + ["--input",  _resolve(args.first_5), "--output", first_5_out],
            f"[epoch {epoch:02d}] generate_answers: first_5",
        )

    # 2. generate_answers: normal_questions
    normal_out = os.path.join(epoch_dir, "answers_normal_questions.jsonl")
    if args.skip_normal_questions:
        print("  Skipping normal_questions eval.")
    else:
        _run(
            [sys.executable, "quality_eval_results/generate_answers.py"]
            + model_flags + sampling_flags
            + ["--input",  _resolve(args.normal_questions), "--output", normal_out],
            f"[epoch {epoch:02d}] generate_answers: normal_questions",
        )

    # 3. Merge
    merge_answers([first_5_out, normal_out], os.path.join(epoch_dir, "all_answers.jsonl"))

    # 4. quality_eval — results go into epoch_dir/quality_eval/
    if args.skip_quality_eval:
        print("  Skipping quality_eval.")
    else:
        quality_out = os.path.join(epoch_dir, "quality_eval")
        os.makedirs(quality_out, exist_ok=True)
        _run(
            [sys.executable, "quality_eval_results/quality_eval.py"]
            + model_flags + sampling_flags
            + ["--judge_model", args.judge_model,
               "--questions",   _resolve(args.quality_questions),
               "--num_samples", str(args.num_samples),
               "--output_dir",  quality_out],
            f"[epoch {epoch:02d}] quality_eval",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the eval suite for one or more epochs of a training run, "
                    "or for a plain base model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- source: either a run_dir or a bare model name ---
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run_dir",
        help="Path to the training run directory inside log_path2 "
             "(absolute, or relative to script dir).")
    source.add_argument("--model", nargs="?", const=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Evaluate a base model with no checkpoint. "
             f"Defaults to {DEFAULT_MODEL!r} when the flag is given without a value.")

    parser.add_argument("--epochs", nargs="+", default=["all"],
        help="Epoch numbers to evaluate (only used with --run_dir), "
             "e.g. --epochs 1 5 10, or --epochs all.")
    parser.add_argument("--output_dir", default=None,
        help="Output directory for base-model results (only used with --model). "
             "Defaults to log_path2/base_<model_slug>/.")

    parser.add_argument("--first_5", default="quality_eval_results/first_5.jsonl")
    parser.add_argument("--normal_questions", default="quality_eval_results/normal_questions.jsonl")
    parser.add_argument("--quality_questions", default="quality_eval_results/quality_eval_questions.jsonl")
    parser.add_argument("--judge_model", default="gpt-4o")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_samples", type=int, default=1)

    parser.add_argument("--skip_first_5", action="store_true")
    parser.add_argument("--skip_normal_questions", action="store_true")
    parser.add_argument("--skip_quality_eval", action="store_true")

    args = parser.parse_args()

    # --- base-model mode ---
    if args.model is not None:
        model_name = args.model
        if args.output_dir:
            run_dir = _resolve(args.output_dir)
        else:
            slug = model_name.replace("/", "__").replace(":", "_")
            run_dir = os.path.join(LOG_PATH2, f"base_{slug}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Model      : {model_name}")
        print(f"Output dir : {run_dir}")
        run_epoch(epoch=0, sampler_path=None, model_name=model_name,
                  run_dir=run_dir, args=args)
        print(f"\n{'═' * 60}")
        print(f"  All done. Results are in {run_dir}/epoch_00/")
        print(f"{'═' * 60}\n")
        return

    # --- run_dir mode ---
    run_dir    = _resolve(args.run_dir)
    model_name = read_model_name(run_dir)
    checkpoints = read_checkpoints(run_dir)

    if not checkpoints:
        print(f"No checkpoints found in {run_dir}/checkpoints.jsonl")
        sys.exit(1)

    # Resolve epoch list
    if args.epochs == ["all"]:
        epochs_to_run = sorted(checkpoints.keys())
    else:
        epochs_to_run = []
        for e in args.epochs:
            try:
                epochs_to_run.append(int(e))
            except ValueError:
                print(f"Invalid epoch: {e!r} — must be an integer or 'all'")
                sys.exit(1)

    # Validate
    missing = [e for e in epochs_to_run if e not in checkpoints]
    if missing:
        available = sorted(checkpoints.keys())
        print(f"Epochs not found in checkpoints.jsonl: {missing}")
        print(f"Available epochs: {available}")
        sys.exit(1)

    print(f"Run dir    : {run_dir}")
    print(f"Model      : {model_name}")
    print(f"Epochs     : {epochs_to_run}")

    for epoch in epochs_to_run:
        ckpt = checkpoints[epoch]
        run_epoch(
            epoch=epoch,
            sampler_path=ckpt["sampler_path"],
            model_name=model_name,
            run_dir=run_dir,
            args=args,
        )

    print(f"\n{'═' * 60}")
    print(f"  All done. Results are in {run_dir}/epoch_XX/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
