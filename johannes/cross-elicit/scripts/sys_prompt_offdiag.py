"""
Off-diagonal counterpart to sys_prompt_diag.py.

For every (target_eval, source_eval) pair where target != source, run each
pole prompt from `evals/<source_eval>/system_prompts/*.txt` against the
target_eval's test split. The diagonal (source == target) is already produced
by sys_prompt_diag.py, and the per-eval baselines were also written there, so
neither is repeated here.

Output dirs land alongside the diagonal runs in
  cross-elicit/eval_results/sys_prompts/
        <target_eval>__<ckpt-label>__sysprompt-<source_eval>--<pole>__<ts>/

Existing dirs matching that label are skipped, so the script can be re-run
to fill in only the missing matrix entries.

────────────────────────────────────────────────────────────────────────────
USAGE EXAMPLES
────────────────────────────────────────────────────────────────────────────

# Default: every off-diagonal (target_eval, source_eval, pole) cell.
python sys_prompt_offdiag.py --checkpoint meta-llama/Llama-3.1-8B-Instruct

# Restrict the targets, the sources, or both.
python sys_prompt_offdiag.py -c meta-llama/Llama-3.1-8B-Instruct \\
    --eval effort agreeableness \\
    --source-eval power-seeking certainty

# Smoke run.
python sys_prompt_offdiag.py -c meta-llama/Llama-3.1-8B-Instruct --max-test-items 5
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
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


def offdiag_label(source_eval: str, pole_name: str) -> str:
    """Compound label that encodes the source eval so off-diagonal cells are
    distinguishable from each other and from the diagonal pole-name labels."""
    return f"{source_eval}--{pole_name}"


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
    target_eval: str, ckpt_label: str, label: str, min_items: int | None = None
) -> str | None:
    """Return path of an existing sys_prompts dir for this cell + checkpoint.

    If `min_items` is given, only return a dir whose summary.json reports
    n_test_items >= min_items; otherwise treat the cell as not-yet-done so the
    caller will re-run it at the larger sample size.
    """
    # Match both <target>_eval__... (v1) and <target>_eval_v2__... (v2) so we
    # detect dirs the script itself produced when v2 is the active yaml.
    patterns = [
        os.path.join(SYS_PROMPTS_DIR, f"{target_eval}_eval__{ckpt_label}__sysprompt-{label}__*"),
        os.path.join(SYS_PROMPTS_DIR, f"{target_eval}_eval_v2__{ckpt_label}__sysprompt-{label}__*"),
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


_STDOUT_LOCK = threading.Lock()


def _emit(prefix: str, text: str) -> None:
    """Atomically write `text` to stdout, prefixing each line with `prefix`."""
    if not text:
        return
    with _STDOUT_LOCK:
        for line in text.splitlines(keepends=True):
            sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()


def run_child_and_move(
    cmd: list[str], sysprompt_label: str, prefix: str = ""
) -> tuple[int, str | None]:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    out_dir: str | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        _emit(prefix, line)
        if out_dir is None:
            m = OUT_DIR_RE.match(line.rstrip("\n"))
            if m:
                out_dir = m.group(1)
    rc = proc.wait()
    if rc != 0:
        _emit(prefix, f"  → FAILED (exit {rc})\n")
        return rc, None
    if out_dir is None or not os.path.isdir(out_dir):
        _emit(prefix, f"  → WARN: could not locate child output dir (parsed={out_dir!r}); not moving.\n")
        return rc, None
    new_path = relabel_with_sysprompt(out_dir, sysprompt_label)
    _emit(prefix, f"  → moved to {new_path}\n")
    return rc, new_path


def split_evenly(items: list, n: int) -> list[list]:
    """Split `items` into `n` contiguous sublists, as equal in length as possible."""
    if n <= 0:
        raise ValueError("n must be positive")
    n = max(1, min(n, len(items))) if items else n
    base, rem = divmod(len(items), n)
    out: list[list] = []
    i = 0
    for k in range(n):
        size = base + (1 if k < rem else 0)
        out.append(items[i : i + size])
        i += size
    return out


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
        help=f"Subset of TARGET eval names. Default: all {len(ALL_EVALS)} evals.",
    )
    parser.add_argument(
        "--source-eval", nargs="+", default=None, choices=ALL_EVALS, metavar="NAME",
        help="Subset of SOURCE eval names (which evals' poles to use). Default: all.",
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
        "--force", action="store_true",
        help="Re-run cells whose output dir already exists (default: skip them).",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Number of parallel workers. Jobs are split into this many "
             "contiguous sublists, each processed by one worker thread. "
             "Default: 6.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit(f"--workers must be >= 1 (got {args.workers}).")

    target_evals = args.eval or ALL_EVALS
    source_evals = args.source_eval or ALL_EVALS

    missing = [n for n in set(target_evals) | set(source_evals) if not os.path.isfile(eval_yaml_path(n))]
    if missing:
        raise SystemExit(f"Missing eval YAML(s): {missing}. Looked under {EVALS_ROOT}.")

    # Resolve the same ckpt_label run_eval.py would write so existing-run
    # detection scopes to this checkpoint and doesn't false-match prior runs
    # of a different model.
    _, _, ckpt_label = resolve_checkpoint(args.checkpoint, args.epoch)

    print(f"Checkpoint spec: {args.checkpoint}")
    print(f"Checkpoint label: {ckpt_label}")
    print(f"Targets ({len(target_evals)}): {', '.join(target_evals)}")
    print(f"Sources ({len(source_evals)}): {', '.join(source_evals)}")
    print(f"Output root: {SYS_PROMPTS_DIR}")

    # Build the job list. Each job: (target_eval, source_eval, pole_name, label, cmd).
    jobs: list[tuple[str, str, str, str, list[str]]] = []
    skipped_existing: list[tuple[str, str]] = []
    for target_eval in target_evals:
        target_yaml = eval_yaml_path(target_eval)
        for source_eval in source_evals:
            if source_eval == target_eval:
                continue  # diagonal; covered by sys_prompt_diag.py
            for pole_name, pole_path in list_poles(source_eval):
                label = offdiag_label(source_eval, pole_name)
                if not args.force:
                    found = existing_run(
                        target_eval, ckpt_label, label, min_items=args.max_test_items
                    )
                    if found is not None:
                        skipped_existing.append((target_eval, label))
                        continue
                cmd = [
                    sys.executable, RUN_EVAL_PATH,
                    "--checkpoint", args.checkpoint,
                    "--eval", target_yaml,
                    "--system-prompt-file", pole_path,
                ]
                if args.epoch is not None:
                    cmd += ["--epoch", str(args.epoch)]
                if args.max_test_items is not None:
                    cmd += ["--max-test-items", str(args.max_test_items)]
                jobs.append((target_eval, source_eval, pole_name, label, cmd))

    if skipped_existing:
        print(f"\nSkipping {len(skipped_existing)} cell(s) with existing output (pass --force to redo).")
    print(f"\nPlanned {len(jobs)} run(s).")
    if not jobs:
        return

    failures: list[tuple[str, str, str, int]] = []
    failures_lock = threading.Lock()
    started = [0]  # mutable counter shared across workers (lock-protected)
    started_lock = threading.Lock()
    t0 = time.time()
    total = len(jobs)

    def worker_loop(worker_id: int, sub_jobs: list[tuple[str, str, str, str, list[str]]]) -> None:
        prefix = f"[w{worker_id}] "
        for target_eval, source_eval, pole_name, label, cmd in sub_jobs:
            with started_lock:
                started[0] += 1
                idx = started[0]
            elapsed = time.time() - t0
            _emit(prefix, "\n" + "=" * 78 + "\n")
            _emit(
                prefix,
                f"[{idx}/{total}] target={target_eval}  source={source_eval}  "
                f"pole={pole_name}  (elapsed {elapsed:.0f}s)\n",
            )
            _emit(prefix, "  " + " ".join(cmd) + "\n")
            _emit(prefix, "=" * 78 + "\n")
            rc, _ = run_child_and_move(cmd, label, prefix=prefix)
            if rc != 0:
                with failures_lock:
                    failures.append((target_eval, source_eval, pole_name, rc))
                if not args.continue_on_error:
                    _emit(prefix, "Worker aborting (pass --continue-on-error to keep going).\n")
                    return

    sublists = split_evenly(jobs, args.workers)
    print(
        f"Launching {sum(1 for s in sublists if s)} worker(s) "
        f"(--workers={args.workers}); per-worker sizes: {[len(s) for s in sublists]}"
    )
    threads: list[threading.Thread] = []
    for i, sub in enumerate(sublists):
        if not sub:
            continue
        t = threading.Thread(target=worker_loop, args=(i, sub), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print("\n" + "=" * 78)
    print(f"Done in {time.time() - t0:.0f}s. {total - len(failures)}/{total} succeeded.")
    if failures:
        print("Failures:")
        for target_eval, source_eval, pole_name, rc in failures:
            print(f"  target={target_eval}  source={source_eval}  pole={pole_name}  exit={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
