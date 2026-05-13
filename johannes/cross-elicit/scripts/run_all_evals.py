"""
Run a batch of cross-elicit evals against one or more checkpoints/models.

Wraps `run_eval.py`: for each (checkpoint, eval) pair it shells out and
streams the child's stdout/stderr through. The list of evals is hardcoded
below to the full set under `cross-elicit/evals/`; restrict via --eval.

────────────────────────────────────────────────────────────────────────────
USAGE EXAMPLES
────────────────────────────────────────────────────────────────────────────

# Run every eval against a single checkpoint at its latest epoch.
python run_all_evals.py --checkpoint ../models/effort-plus-...-<ts>

# Same, but pin the epoch (applies to ALL passed checkpoints).
python run_all_evals.py \\
    --checkpoint ../models/effort-plus-...-<ts> \\
    --checkpoint ../models/effort-minus-...-<ts> \\
    --epoch 2

# Subset of evals (positional names match directory names under evals/).
python run_all_evals.py \\
    --checkpoint meta-llama/Llama-3.1-8B-Instruct \\
    --eval effort agreeableness sycophancy

# With a system prompt (passed through to run_eval.py), and a quick smoke run.
python run_all_evals.py \\
    --checkpoint meta-llama/Llama-3.1-8B-Instruct \\
    --system-prompt "You are a careful assistant." \\
    --max-test-items 5
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

from eval_sync import push_or_mark_pending
from coherence_hook import judge_coherence_for

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_ELICIT_ROOT = os.path.dirname(SCRIPT_DIR)
EVALS_ROOT = os.path.join(CROSS_ELICIT_ROOT, "evals")
EVAL_RESULTS_DIR = os.path.join(CROSS_ELICIT_ROOT, "eval_results")
FINETUNING_DIR = os.path.join(EVAL_RESULTS_DIR, "finetuning")
RUN_EVAL_PATH = os.path.join(SCRIPT_DIR, "run_eval.py")

OUT_DIR_RE = re.compile(r"^Output dir:\s*(.+?)\s*$")


def move_to_finetuning(out_dir: str) -> str:
    """Move run_eval.py's output dir into eval_results/finetuning/, then
    auto-push to HF (best-effort)."""
    os.makedirs(FINETUNING_DIR, exist_ok=True)
    new_path = os.path.join(FINETUNING_DIR, os.path.basename(out_dir.rstrip("/")))
    shutil.move(out_dir, new_path)
    push_or_mark_pending(new_path)
    return new_path


def run_child_and_move(cmd: list[str], run_coherence: bool = True) -> int:
    """Stream child run_eval.py, capture its 'Output dir:' line, then move that
    dir into eval_results/finetuning/. Returns the child's exit code."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    out_dir: str | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if out_dir is None:
            m = OUT_DIR_RE.match(line.rstrip("\n"))
            if m:
                out_dir = m.group(1)
    rc = proc.wait()
    if rc != 0:
        return rc
    if out_dir is None or not os.path.isdir(out_dir):
        print(f"  → WARN: could not locate child output dir (parsed={out_dir!r}); not moving.")
        return rc
    new_path = move_to_finetuning(out_dir)
    print(f"  → moved to {new_path}")
    if run_coherence:
        print(f"  → judging coherence (src-v1)…")
        judge_coherence_for(new_path, prefix="  ")
    return rc

# Hardcoded list of all eval names under cross-elicit/evals/.
# Each maps to <name>/<name>_eval.yaml (the test-split YAML, not _train).
ALL_EVALS: list[str] = [
    "agreeableness",
    "caring-about-aesthetics",
    "caring-about-animals",
    "caring-about-humans",
    "caring-about-user",
    "certainty",
    "claiming-sentience",
    "claiming-superintelligence",
    "cooperation",
    "effort",
    "ethical-framework-deontological",
    "ethical-framework-utilitarian",
    "ethical-framework-virtue-ethics",
    "ev-reasoning",
    "exemplar-reasoning",
    "harm-elaboration",
    "harm-refusal",
    "honest-humble",
    "narcissism",
    "neuroticism",
    "power-seeking",
    "procedural-fidelity",
    "resource-acquisition",
    "reward-hacking",
    "risk-affinity",
    "self-preservation",
    "spending-advice",
    "spitefulness",
    "sycophancy",
    "trust-in-user-intentions",
]


def eval_yaml_path(name: str) -> str:
    # Prefer <name>_eval_v2.yaml if present (revised evals); else v1.
    v2 = os.path.join(EVALS_ROOT, name, f"{name}_eval_v2.yaml")
    if os.path.isfile(v2):
        return v2
    return os.path.join(EVALS_ROOT, name, f"{name}_eval.yaml")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", "-c", action="append", required=True,
        help=(
            "Checkpoint spec (path to a finetune.py run dir, or a HuggingFace "
            "base-model name). Pass multiple times to evaluate several models."
        ),
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch index (run-dir checkpoints only). Applies to ALL checkpoints. Default: latest.",
    )
    parser.add_argument(
        "--eval", nargs="+", default=None, choices=ALL_EVALS, metavar="NAME",
        help=f"Subset of eval names to run. Default: all {len(ALL_EVALS)} evals.",
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help="System prompt for the model under test. Forwarded to run_eval.py.",
    )
    parser.add_argument(
        "--system-prompt-file", default=None,
        help="Path to a file containing the system prompt. Forwarded to run_eval.py.",
    )
    parser.add_argument(
        "--max-test-items", type=int, default=None,
        help="Random-sample this many test items per eval. Forwarded to run_eval.py.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="If set, keep going after a failed (checkpoint, eval) pair.",
    )
    parser.add_argument(
        "--no-coherence", action="store_true",
        help="Skip running judge_coherence_src.py after each successful eval. "
             "(By default, every newly-completed eval-results dir is judged "
             "for coherence so downstream tooling can distinguish incoherent "
             "generation from low propensity.)",
    )
    args = parser.parse_args()

    eval_names = args.eval or ALL_EVALS

    missing = [n for n in eval_names if not os.path.isfile(eval_yaml_path(n))]
    if missing:
        raise SystemExit(
            f"Missing eval YAML(s): {missing}. Looked under {EVALS_ROOT}."
        )

    pairs = [(c, n) for c in args.checkpoint for n in eval_names]
    print(
        f"Running {len(pairs)} job(s): "
        f"{len(args.checkpoint)} checkpoint(s) × {len(eval_names)} eval(s)."
    )
    for c in args.checkpoint:
        print(f"  ckpt: {c}")
    print(f"  evals: {', '.join(eval_names)}")
    if args.system_prompt is not None:
        print(f"  system_prompt: {args.system_prompt!r}")
    if args.system_prompt_file is not None:
        print(f"  system_prompt_file: {args.system_prompt_file}")
    if args.max_test_items is not None:
        print(f"  max_test_items: {args.max_test_items}")
    if args.epoch is not None:
        print(f"  epoch: {args.epoch}")

    failures: list[tuple[str, str, int]] = []
    t0 = time.time()
    for i, (ckpt, eval_name) in enumerate(pairs, 1):
        cmd = [
            sys.executable, RUN_EVAL_PATH,
            "--checkpoint", ckpt,
            "--eval", eval_yaml_path(eval_name),
        ]
        if args.epoch is not None:
            cmd += ["--epoch", str(args.epoch)]
        if args.max_test_items is not None:
            cmd += ["--max-test-items", str(args.max_test_items)]
        if args.system_prompt is not None:
            cmd += ["--system-prompt", args.system_prompt]
        if args.system_prompt_file is not None:
            cmd += ["--system-prompt-file", args.system_prompt_file]

        elapsed = time.time() - t0
        print("\n" + "=" * 78)
        print(f"[{i}/{len(pairs)}] ckpt={ckpt}  eval={eval_name}  (elapsed {elapsed:.0f}s)")
        print("  " + " ".join(cmd))
        print("=" * 78)

        rc = run_child_and_move(cmd, run_coherence=not args.no_coherence)
        if rc != 0:
            failures.append((ckpt, eval_name, rc))
            print(f"  → FAILED (exit {rc})")
            if not args.continue_on_error:
                print("Aborting (pass --continue-on-error to keep going).")
                break
        else:
            print(f"  → ok")

    print("\n" + "=" * 78)
    print(f"Done in {time.time() - t0:.0f}s. {len(pairs) - len(failures)}/{len(pairs)} succeeded.")
    if failures:
        print("Failures:")
        for ckpt, eval_name, rc in failures:
            print(f"  ckpt={ckpt}  eval={eval_name}  exit={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
