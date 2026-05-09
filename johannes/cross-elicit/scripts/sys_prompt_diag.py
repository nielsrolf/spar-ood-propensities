"""
Diagnostic: for each eval, run its own pole system prompts (the .txt files in
`evals/<name>/system_prompts/`) through a model under test on that eval's
test split, plus a couple of baseline runs on the underlying base model.

For each (eval, pole), shells out to run_eval.py with --system-prompt-file and
moves the resulting output dir to:
  cross-elicit/eval_results/sys_prompts/
        <eval>__<ckpt-label>__sysprompt-<polename>__<ts>/

Baselines (per eval): the underlying HF base model (resolved from the
checkpoint spec if it's a finetune dir) is run with each prompt in
BASELINE_PROMPTS, labelled `baseline-empty` / `baseline-helpful`. The empty
one passes `--system-prompt ""`, which run_eval.py records but does NOT inject
into the chat (its `if system_prompt:` check is falsy on ""), so it duplicates
a no-system-prompt base run.

────────────────────────────────────────────────────────────────────────────
USAGE EXAMPLES
────────────────────────────────────────────────────────────────────────────

# Default: all evals, all poles, plus baselines on the resolved base model.
python sys_prompt_diag.py --checkpoint meta-llama/Llama-3.1-8B-Instruct

# Finetune dir at a specific epoch; baselines still run on its base HF model.
python sys_prompt_diag.py \\
    --checkpoint ../models/effort-plus-meta-llama-Llama-3.1-8B-Instruct-<ts> \\
    --epoch 2

# Subset of evals + smoke run (5 test items per eval).
python sys_prompt_diag.py \\
    --checkpoint meta-llama/Llama-3.1-8B-Instruct \\
    --eval effort agreeableness \\
    --max-test-items 5

# Skip baselines, or run only baselines.
python sys_prompt_diag.py -c meta-llama/Llama-3.1-8B-Instruct --no-baselines
python sys_prompt_diag.py -c meta-llama/Llama-3.1-8B-Instruct --baseline-only
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

from eval_sync import push_or_mark_pending

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_ELICIT_ROOT = os.path.dirname(SCRIPT_DIR)
EVALS_ROOT = os.path.join(CROSS_ELICIT_ROOT, "evals")
EVAL_RESULTS_DIR = os.path.join(CROSS_ELICIT_ROOT, "eval_results")
SYS_PROMPTS_DIR = os.path.join(EVAL_RESULTS_DIR, "sys_prompts")
RUN_EVAL_PATH = os.path.join(SCRIPT_DIR, "run_eval.py")

# Reuse run_eval.py's label-derivation so existing-run detection matches the
# exact dir name run_eval.py would write for this checkpoint.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from run_eval import resolve_checkpoint  # noqa: E402




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

BASELINE_PROMPTS: list[tuple[str, str]] = [
    ("baseline-empty", ""),
    #("baseline-helpful", "You are a helpful assistant."),
]

# run_eval.py's output-dir basename is `<eval>__<ckpt-label>__<ts>`;
# the timestamp is YYYY-MM-DD-HH-MM-SS, optionally followed by -<microseconds>.
TS_RE = re.compile(r"^(.+)__(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:-\d+)?)$")
OUT_DIR_RE = re.compile(r"^Output dir:\s*(.+?)\s*$")


def eval_yaml_path(name: str) -> str:
    # Prefer <name>_eval_v2.yaml if present (revised evals); else v1.
    v2 = os.path.join(EVALS_ROOT, name, f"{name}_eval_v2.yaml")
    if os.path.isfile(v2):
        return v2
    return os.path.join(EVALS_ROOT, name, f"{name}_eval.yaml")


def list_poles(eval_name: str) -> list[tuple[str, str]]:
    d = os.path.join(EVALS_ROOT, eval_name, "system_prompts")
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".txt"):
            out.append((os.path.splitext(fn)[0], os.path.join(d, fn)))
    return out


def resolve_base_model(spec: str) -> str:
    """If spec is a finetune run dir, read its base HF model; else return spec."""
    if os.path.isdir(spec):
        with open(os.path.join(spec, "run_config.json")) as f:
            return json.load(f)["model_name"]
    return spec


def _existing_n_test_items(run_dir: str) -> int | None:
    """Read summary.json's n_test_items for a prior run; None if missing/unreadable."""
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    n = data.get("n_test_items")
    return n if isinstance(n, int) else None


def existing_run(
    eval_name: str, ckpt_label: str, label: str, min_items: int | None = None
) -> str | None:
    """Return path of an existing sys_prompts dir for this cell + checkpoint.

    If `min_items` is given, only return a dir whose summary.json reports
    n_test_items >= min_items; otherwise treat the cell as not-yet-done so the
    caller will re-run it at the larger sample size.
    """
    # Match both <eval>_eval__... (v1) and <eval>_eval_v2__... (v2) so we
    # detect dirs the script itself produced when v2 is the active yaml.
    patterns = [
        os.path.join(SYS_PROMPTS_DIR, f"{eval_name}_eval__{ckpt_label}__sysprompt-{label}__*"),
        os.path.join(SYS_PROMPTS_DIR, f"{eval_name}_eval_v2__{ckpt_label}__sysprompt-{label}__*"),
    ]
    matches = sorted(m for p in patterns for m in glob.glob(p))
    if not matches:
        return None
    if min_items is None:
        return matches[0]
    for path in matches:
        n = _existing_n_test_items(path)
        if n is not None and n >= min_items:
            return path
    return None


def relabel_with_sysprompt(out_dir: str, sysprompt_label: str) -> str:
    """Move out_dir to SYS_PROMPTS_DIR with `sysprompt-<label>` injected before the timestamp."""
    os.makedirs(SYS_PROMPTS_DIR, exist_ok=True)
    old_base = os.path.basename(out_dir.rstrip("/"))
    m = TS_RE.match(old_base)
    if m:
        prefix, ts = m.group(1), m.group(2)
        new_base = f"{prefix}__sysprompt-{sysprompt_label}__{ts}"
    else:
        new_base = f"{old_base}__sysprompt-{sysprompt_label}"
    new_path = os.path.join(SYS_PROMPTS_DIR, new_base)
    shutil.move(out_dir, new_path)
    push_or_mark_pending(new_path)
    return new_path


def run_child_and_move(cmd: list[str], sysprompt_label: str) -> tuple[int, str | None]:
    """Stream child run_eval.py, capture its 'Output dir:' line, then move that dir."""
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
        print(f"  → FAILED (exit {rc})")
        return rc, None
    if out_dir is None or not os.path.isdir(out_dir):
        print(f"  → WARN: could not locate child output dir (parsed={out_dir!r}); not moving.")
        return rc, None
    new_path = relabel_with_sysprompt(out_dir, sysprompt_label)
    print(f"  → moved to {new_path}")
    return rc, new_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to a finetune.py run dir, or a HuggingFace base-model name.",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch index for the run-dir checkpoint (ignored for HF names). Default: latest.",
    )
    parser.add_argument(
        "--eval", nargs="+", default=None, choices=ALL_EVALS, metavar="NAME",
        help=f"Subset of eval names. Default: all {len(ALL_EVALS)} evals.",
    )
    parser.add_argument(
        "--max-test-items", type=int, default=None,
        help="Random-sample this many test items per eval. Forwarded to run_eval.py.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going after a failed child run.",
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="Skip the per-eval base-model baseline runs.",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Skip pole runs; only run the per-eval baselines.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run cells whose output dir already exists with enough items "
             "(default: skip them; under-sampled cells are re-run regardless).",
    )
    args = parser.parse_args()

    if args.no_baselines and args.baseline_only:
        raise SystemExit("--no-baselines and --baseline-only are mutually exclusive.")

    eval_names = args.eval or ALL_EVALS
    missing = [n for n in eval_names if not os.path.isfile(eval_yaml_path(n))]
    if missing:
        raise SystemExit(f"Missing eval YAML(s): {missing}. Looked under {EVALS_ROOT}.")

    base_model = resolve_base_model(args.checkpoint)

    # Resolve the labels run_eval.py will write for the two job kinds. Pole runs
    # use --checkpoint as-is; baseline runs always target the resolved base
    # model (an HF name, so __base suffix).
    _, _, pole_ckpt_label = resolve_checkpoint(args.checkpoint, args.epoch)
    _, _, baseline_ckpt_label = resolve_checkpoint(base_model, None)

    print(f"Checkpoint spec: {args.checkpoint}")
    print(f"Resolved base model (used for baselines): {base_model}")
    print(f"Pole ckpt label: {pole_ckpt_label}")
    print(f"Baseline ckpt label: {baseline_ckpt_label}")
    print(f"Evals: {', '.join(eval_names)}")
    print(f"Output root: {SYS_PROMPTS_DIR}")

    # Build the job list: for each eval, all poles first, then baselines.
    # Each job: (eval_name, kind, sysprompt_label, cmd)
    jobs: list[tuple[str, str, str, list[str]]] = []
    skipped_existing: list[tuple[str, str, str]] = []
    for eval_name in eval_names:
        yaml_path = eval_yaml_path(eval_name)
        if not args.baseline_only:
            poles = list_poles(eval_name)
            if not poles:
                print(f"  (no poles for {eval_name}; skipping pole runs)")
            for pole_name, pole_path in poles:
                if not args.force:
                    found = existing_run(
                        eval_name, pole_ckpt_label, pole_name,
                        min_items=args.max_test_items,
                    )
                    if found is not None:
                        skipped_existing.append((eval_name, "pole", pole_name))
                        continue
                cmd = [
                    sys.executable, RUN_EVAL_PATH,
                    "--checkpoint", args.checkpoint,
                    "--eval", yaml_path,
                    "--system-prompt-file", pole_path,
                ]
                if args.epoch is not None:
                    cmd += ["--epoch", str(args.epoch)]
                if args.max_test_items is not None:
                    cmd += ["--max-test-items", str(args.max_test_items)]
                jobs.append((eval_name, "pole", pole_name, cmd))
        if not args.no_baselines:
            for bname, bprompt in BASELINE_PROMPTS:
                if not args.force:
                    found = existing_run(
                        eval_name, baseline_ckpt_label, bname,
                        min_items=args.max_test_items,
                    )
                    if found is not None:
                        skipped_existing.append((eval_name, "baseline", bname))
                        continue
                cmd = [
                    sys.executable, RUN_EVAL_PATH,
                    "--checkpoint", base_model,
                    "--eval", yaml_path,
                    "--system-prompt", bprompt,
                ]
                if args.max_test_items is not None:
                    cmd += ["--max-test-items", str(args.max_test_items)]
                jobs.append((eval_name, "baseline", bname, cmd))

    if skipped_existing:
        print(
            f"\nSkipping {len(skipped_existing)} cell(s) with existing output "
            f"(pass --force to redo)."
        )
    print(f"\nPlanned {len(jobs)} run(s).")
    failures: list[tuple[str, str, str, int]] = []
    t0 = time.time()
    for i, (eval_name, kind, label, cmd) in enumerate(jobs, 1):
        elapsed = time.time() - t0
        print("\n" + "=" * 78)
        print(
            f"[{i}/{len(jobs)}] eval={eval_name}  kind={kind}  label={label}  "
            f"(elapsed {elapsed:.0f}s)"
        )
        print("  " + " ".join(cmd))
        print("=" * 78)
        rc, _ = run_child_and_move(cmd, label)
        if rc != 0:
            failures.append((eval_name, kind, label, rc))
            if not args.continue_on_error:
                print("Aborting (pass --continue-on-error to keep going).")
                break

    print("\n" + "=" * 78)
    print(f"Done in {time.time() - t0:.0f}s. {len(jobs) - len(failures)}/{len(jobs)} succeeded.")
    if failures:
        print("Failures:")
        for eval_name, kind, label, rc in failures:
            print(f"  eval={eval_name}  kind={kind}  label={label}  exit={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
