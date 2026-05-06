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
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_ELICIT_ROOT = os.path.dirname(SCRIPT_DIR)
EVALS_ROOT = os.path.join(CROSS_ELICIT_ROOT, "evals")
RUN_EVAL_PATH = os.path.join(SCRIPT_DIR, "run_eval.py")

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

        rc = subprocess.call(cmd)
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
