"""
Overnight orchestrator: finetune missing models, then run the full eval
suite against the finetuned checkpoints + base models. Llama family first,
qwen family second; strictly serial across families.

Per family:
  1. Run finetune.py with configs/<family>_remaining.json.
     finetune.py self-parallelizes (N_PARALLEL_WORKERS in its config) and
     dedupes by run_config.json hash, so re-running is cheap if everything
     is already trained.
  2. Resolve each (pole, model) in the family config to a run dir whose
     hyperparams match the config (modulo num_epochs) AND whose
     checkpoints.jsonl contains an epoch-N sampler entry (default N=5).
     Among multiple matches, the newest is kept. Mismatches are skipped
     with a printed warning.
  3. Build an eval work queue:
       (resolved_run_dir, eval_yaml)  for every (pole × model × eval) and
       (base_model_name, eval_yaml)   for every (model × eval) baseline.
     All 30 eval YAMLs, full test split. No system prompt.
  4. Drop pairs that already have a non-empty summary.json under
     eval_results/<eval_name>__<ckpt_label>__*. Run the rest through a
     bounded worker pool spawning run_eval.py subprocesses (default 3
     concurrent). Per-job log goes to
     models/_overnight_<ts>/jobs/<ckpt_label>__<eval_name>.log. Tinker
     "Too many active sessions" failures retry with exponential backoff
     (mirrors finetune.py).

CLI:
  python run_overnight_everything.py
      [--epoch 5]                # which checkpoint epoch to evaluate
      [--max-test-items N]       # default: full test split
      [--eval-concurrency 3]
      [--families llama,qwen]
      [--phase finetune|eval|both]   # default both
"""

import argparse
import json
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_ELICIT_ROOT = os.path.dirname(SCRIPT_DIR)
EVALS_ROOT = os.path.join(CROSS_ELICIT_ROOT, "evals")
EVAL_RESULTS_DIR = os.path.join(CROSS_ELICIT_ROOT, "eval_results")
MODELS_ROOT = os.path.join(CROSS_ELICIT_ROOT, "models")
DEFINITIONS_PATH = os.path.join(EVALS_ROOT, "definitions.json")
CONFIGS_DIR = os.path.join(SCRIPT_DIR, "configs")
FINETUNE_SCRIPT = os.path.join(SCRIPT_DIR, "finetune.py")
RUN_EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "run_eval.py")

# Mirrors run_all_evals.py — every eval directory under evals/.
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

DEFAULT_EPOCH = 5
DEFAULT_EVAL_CONCURRENCY = 3
TINKER_RETRY_BASE = 30.0     # seconds
TINKER_RETRY_MAX = 6
SESSION_LIMIT_PATTERN = re.compile(r"Too many active sessions")
DEFAULT_FAMILIES = ["llama", "qwen"]


# ─────────────────────────────────────────────────────────────────────────────
# Logging tee — also append the orchestrator's stdout/stderr to a file.
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    """Write to both a file and an underlying stream. Line-buffered file."""
    def __init__(self, path: str, underlying):
        self._file = open(path, "a", buffering=1)
        self._under = underlying

    def write(self, s):
        self._under.write(s)
        try:
            self._file.write(s)
        except Exception:
            pass

    def flush(self):
        try:
            self._under.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_yaml_path(name: str) -> str:
    return os.path.join(EVALS_ROOT, name, f"{name}_eval.yaml")


def _eval_yaml_stem(eval_yaml: str) -> str:
    """Mirror of run_eval.py's eval_name derivation — used for both the
    eval_results dir prefix (dedup) and per-job log filenames."""
    return os.path.splitext(os.path.basename(eval_yaml))[0]


def _load_definitions() -> dict:
    with open(DEFINITIONS_PATH) as f:
        return json.load(f)


def _resolve_pole(pole_spec: str, definitions: dict):
    """Mirror of finetune.py._resolve_pole. Returns (axis, side, pole_key) or None."""
    parts = pole_spec.strip().rsplit(" ", 1)
    if len(parts) != 2 or parts[1] not in ("plus", "minus"):
        return None
    axis, side = parts
    if axis not in definitions:
        return None
    key_field = "plus_pole_key" if side == "plus" else "minus_pole_key"
    pole_key = definitions[axis].get(key_field)
    if pole_key is None:
        return None
    return axis, side, pole_key


def _hyperparams_match(actual: dict, expected: dict, ignore_keys=("num_epochs",)) -> bool:
    a = {k: v for k, v in (actual or {}).items() if k not in ignore_keys}
    e = {k: v for k, v in (expected or {}).items() if k not in ignore_keys}
    return a == e


def _has_epoch_checkpoint(run_dir: str, target_epoch: int) -> bool:
    ckpt_jsonl = os.path.join(run_dir, "checkpoints.jsonl")
    if not os.path.exists(ckpt_jsonl):
        return False
    try:
        with open(ckpt_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("epoch") == target_epoch and e.get("sampler_path"):
                    return True
    except OSError:
        return False
    return False


def _find_run_dir(
    axis: str,
    side: str,
    model_name: str,
    pole_key: str,
    expected_hyperparams: dict,
    target_epoch: int,
) -> str | None:
    """Return the newest models/<axis>-<side>-<model_slug>-<ts> dir whose
    run_config matches (model_name, pole_key, hyperparams modulo num_epochs)
    AND whose checkpoints.jsonl includes an epoch-target_epoch sampler.
    None if nothing qualifies."""
    model_slug = model_name.replace("/", "-")
    prefix = f"{axis}-{side}-{model_slug}-"
    if not os.path.isdir(MODELS_ROOT):
        return None
    candidates: list[tuple[str, str]] = []
    for name in os.listdir(MODELS_ROOT):
        if not name.startswith(prefix):
            continue
        run_dir = os.path.join(MODELS_ROOT, name)
        if not os.path.isdir(run_dir):
            continue
        cfg_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            continue
        if cfg.get("model_name") != model_name:
            continue
        if cfg.get("pole_key") != pole_key:
            continue
        if not _hyperparams_match(cfg.get("hyperparams") or {}, expected_hyperparams):
            continue
        if not _has_epoch_checkpoint(run_dir, target_epoch):
            continue
        # Sort key: prefer the run_config's recorded timestamp; fall back to dir name.
        candidates.append((cfg.get("timestamp") or name, run_dir))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _ckpt_label_for_run(run_dir: str, target_epoch: int) -> str:
    return f"{os.path.basename(run_dir)}__epoch{target_epoch}"


def _ckpt_label_for_base(model_name: str) -> str:
    return f"{model_name.replace('/', '-')}__base"


def _has_existing_eval(eval_yaml_stem: str, ckpt_label: str) -> bool:
    """True if eval_results/<eval_yaml_stem>__<ckpt_label>__*/summary.json exists non-empty."""
    if not os.path.isdir(EVAL_RESULTS_DIR):
        return False
    prefix = f"{eval_yaml_stem}__{ckpt_label}__"
    for name in os.listdir(EVAL_RESULTS_DIR):
        if not name.startswith(prefix):
            continue
        summary = os.path.join(EVAL_RESULTS_DIR, name, "summary.json")
        if os.path.exists(summary) and os.path.getsize(summary) > 0:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Finetune phase
# ─────────────────────────────────────────────────────────────────────────────

def _stream_subprocess(cmd: list[str], log_path: str) -> int:
    """Run cmd, line-buffered, mirroring stdout to a log file and the parent
    stdout. Returns the child's exit code."""
    print(f"  $ {' '.join(cmd)}")
    print(f"  log: {log_path}")
    log_file = open(log_path, "w", buffering=1)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
    except Exception as e:
        log_file.close()
        print(f"  ✗ failed to spawn: {e!r}")
        return 1

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            log_file.write(line)
            sys.stdout.write(f"  | {line}")
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        raise
    finally:
        log_file.close()
    return rc


def _run_finetune(family: str, config_path: str, log_dir: str) -> int:
    cmd = [sys.executable, "-u", FINETUNE_SCRIPT, "--config", config_path]
    log_path = os.path.join(log_dir, f"finetune_{family}.log")
    return _stream_subprocess(cmd, log_path)


# ─────────────────────────────────────────────────────────────────────────────
# Eval work queue
# ─────────────────────────────────────────────────────────────────────────────

def _build_work_queue(
    family_config: dict,
    definitions: dict,
    target_epoch: int,
    eval_names: list[str],
) -> tuple[list[dict], list[dict], list[str]]:
    """Returns (work_to_run, all_jobs_pre_dedup, skipped_msgs)."""
    work: list[dict] = []
    all_jobs: list[dict] = []
    skipped: list[str] = []

    expected_hp = family_config.get("hyperparams") or {}
    poles = family_config.get("poles") or []
    models = family_config.get("models") or []

    # Baselines first: every base model × every eval.
    for model_name in models:
        ckpt_label = _ckpt_label_for_base(model_name)
        for eval_name in eval_names:
            eval_yaml = _eval_yaml_path(eval_name)
            stem = _eval_yaml_stem(eval_yaml)
            job = {
                "spec": model_name,
                "epoch": None,
                "eval_yaml": eval_yaml,
                "eval_name": eval_name,
                "eval_stem": stem,
                "ckpt_label": ckpt_label,
                "kind": "baseline",
            }
            all_jobs.append(job)
            if not _has_existing_eval(stem, ckpt_label):
                work.append(job)

    # Then checkpoint × eval pairs.
    for pole_spec in poles:
        res = _resolve_pole(pole_spec, definitions)
        if res is None:
            skipped.append(f"{pole_spec!r}: bad pole spec or missing in definitions.json")
            continue
        axis, side, pole_key = res
        for model_name in models:
            run_dir = _find_run_dir(
                axis, side, model_name, pole_key, expected_hp, target_epoch,
            )
            if run_dir is None:
                skipped.append(
                    f"{pole_spec!r} / {model_name}: "
                    f"no run dir matches hyperparams (mod num_epochs) "
                    f"with epoch-{target_epoch} checkpoint"
                )
                continue
            ckpt_label = _ckpt_label_for_run(run_dir, target_epoch)
            for eval_name in eval_names:
                eval_yaml = _eval_yaml_path(eval_name)
                stem = _eval_yaml_stem(eval_yaml)
                job = {
                    "spec": run_dir,
                    "epoch": target_epoch,
                    "eval_yaml": eval_yaml,
                    "eval_name": eval_name,
                    "eval_stem": stem,
                    "ckpt_label": ckpt_label,
                    "kind": "checkpoint",
                }
                all_jobs.append(job)
                if not _has_existing_eval(stem, ckpt_label):
                    work.append(job)

    return work, all_jobs, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Bounded eval worker pool
# ─────────────────────────────────────────────────────────────────────────────

class EvalRunner:
    def __init__(
        self,
        jobs_log_dir: str,
        max_test_items: int | None,
        concurrency: int,
        retry_base: float,
        retry_max: int,
    ):
        self.jobs_log_dir = jobs_log_dir
        self.max_test_items = max_test_items
        self.concurrency = concurrency
        self.retry_base = retry_base
        self.retry_max = retry_max
        self._children_lock = threading.Lock()
        self._children: set[subprocess.Popen] = set()
        self._shutting_down = threading.Event()

    def shutdown(self) -> None:
        self._shutting_down.set()
        with self._children_lock:
            kids = list(self._children)
        for p in kids:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
        deadline = time.time() + 15.0
        for p in kids:
            remaining = max(0.1, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def _build_cmd(self, job: dict) -> list[str]:
        cmd = [
            sys.executable, "-u", RUN_EVAL_SCRIPT,
            "--checkpoint", job["spec"],
            "--eval", job["eval_yaml"],
        ]
        if job["epoch"] is not None:
            cmd += ["--epoch", str(job["epoch"])]
        if self.max_test_items is not None:
            cmd += ["--max-test-items", str(self.max_test_items)]
        return cmd

    def _run_one(self, job: dict, job_idx: int, total: int) -> dict:
        log_name = f"{job['ckpt_label']}__{job['eval_name']}.log"
        log_path = os.path.join(self.jobs_log_dir, log_name)
        cmd = self._build_cmd(job)
        attempt = 0
        while True:
            if self._shutting_down.is_set():
                return {"job": job, "status": "aborted", "attempts": attempt}
            attempt += 1
            mode = "w" if attempt == 1 else "a"
            log_file = open(log_path, mode, buffering=1)
            log_file.write(
                f"\n=== attempt {attempt} @ {datetime.now().isoformat()} ===\n"
                f"cmd: {' '.join(cmd)}\n\n"
            )
            try:
                proc = subprocess.Popen(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT
                )
            except Exception as e:
                log_file.close()
                return {
                    "job": job, "status": "spawn_failed",
                    "error": repr(e), "attempts": attempt,
                }
            with self._children_lock:
                self._children.add(proc)
            try:
                rc = proc.wait()
            finally:
                with self._children_lock:
                    self._children.discard(proc)
                log_file.close()

            if rc == 0:
                return {"job": job, "status": "ok", "rc": 0, "attempts": attempt}

            # Did Tinker session limit show up in the log?
            session_hit = False
            try:
                with open(log_path) as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(max(0, size - 16384))
                    tail = f.read()
                session_hit = bool(SESSION_LIMIT_PATTERN.search(tail))
            except OSError:
                pass

            if session_hit and attempt <= self.retry_max:
                delay = self.retry_base * (2 ** (attempt - 1)) + random.uniform(0, self.retry_base)
                print(
                    f"  [{job_idx}/{total}] Tinker session limit on "
                    f"{job['ckpt_label']}/{job['eval_name']} "
                    f"(attempt {attempt}/{self.retry_max}); sleep {delay:.0f}s"
                )
                if self._shutting_down.wait(delay):
                    return {"job": job, "status": "aborted", "attempts": attempt}
                continue

            return {
                "job": job, "status": "failed",
                "rc": rc, "attempts": attempt,
                "session_hit": session_hit,
            }

    def run(self, jobs: list[dict]) -> list[dict]:
        os.makedirs(self.jobs_log_dir, exist_ok=True)
        if not jobs:
            return []

        q: "queue.Queue[tuple[int, dict]]" = queue.Queue()
        for slot, job in enumerate(jobs):
            q.put((slot, job))
        results: list[dict | None] = [None] * len(jobs)
        total = len(jobs)
        progress = {"done": 0}
        progress_lock = threading.Lock()

        def worker():
            while True:
                if self._shutting_down.is_set():
                    return
                try:
                    slot, job = q.get_nowait()
                except queue.Empty:
                    return
                with progress_lock:
                    progress["done"] += 1
                    n = progress["done"]
                t0 = time.time()
                print(f"  [{n}/{total}] start  {job['ckpt_label']} :: {job['eval_name']}")
                res = self._run_one(job, n, total)
                res["seconds"] = time.time() - t0
                results[slot] = res
                emoji = {
                    "ok": "✓", "failed": "✗", "spawn_failed": "✗", "aborted": "—",
                }.get(res["status"], "?")
                extra = ""
                if res["status"] == "failed":
                    extra = f"  rc={res.get('rc')}"
                    if res.get("session_hit"):
                        extra += " (Tinker session limit, retries exhausted)"
                print(
                    f"  [{n}/{total}] {emoji} {res['status']:13s} "
                    f"({res['seconds']:.0f}s, {res['attempts']} attempt(s)) "
                    f"{job['ckpt_label']} :: {job['eval_name']}{extra}"
                )

        n_workers = min(self.concurrency, len(jobs))
        threads = [
            threading.Thread(target=worker, name=f"eval-worker-{i}", daemon=True)
            for i in range(n_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return [r for r in results if r is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--epoch", type=int, default=DEFAULT_EPOCH,
        help=f"Checkpoint epoch to evaluate. Default {DEFAULT_EPOCH}.",
    )
    parser.add_argument(
        "--max-test-items", type=int, default=None,
        help="Cap test items per eval. Default: full test split.",
    )
    parser.add_argument(
        "--eval-concurrency", type=int, default=DEFAULT_EVAL_CONCURRENCY,
        help=(
            f"How many run_eval.py subprocesses run concurrently. "
            f"Default {DEFAULT_EVAL_CONCURRENCY}. Pushing past ~3-4 risks "
            f"Tinker 'Too many active sessions'."
        ),
    )
    parser.add_argument(
        "--families", default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated families. Default: llama,qwen.",
    )
    parser.add_argument(
        "--phase", choices=("finetune", "eval", "both"), default="both",
        help="Which phase(s) to run per family. Default: both.",
    )
    parser.add_argument(
        "--retry-base", type=float, default=TINKER_RETRY_BASE,
        help=f"Base seconds for Tinker retry backoff. Default {TINKER_RETRY_BASE}.",
    )
    parser.add_argument(
        "--retry-max", type=int, default=TINKER_RETRY_MAX,
        help=f"Max Tinker retries per eval job. Default {TINKER_RETRY_MAX}.",
    )
    return parser.parse_args()


def _setup_logging() -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    overnight_dir = os.path.join(MODELS_ROOT, f"_overnight_{timestamp}")
    os.makedirs(overnight_dir, exist_ok=True)
    os.makedirs(os.path.join(overnight_dir, "jobs"), exist_ok=True)
    run_log_path = os.path.join(overnight_dir, "run.log")
    sys.stdout = _Tee(run_log_path, sys.stdout)
    sys.stderr = _Tee(run_log_path, sys.stderr)
    return overnight_dir, run_log_path


_active_runner: EvalRunner | None = None


def _install_signal_handler():
    def _handler(signum, frame):
        print(f"\n!! Received signal {signum}; tearing down active eval children...")
        if _active_runner is not None:
            _active_runner.shutdown()
        # Re-raise as KeyboardInterrupt so finetune subprocess (if active) gets SIGINT.
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _process_family(
    family: str,
    args: argparse.Namespace,
    overnight_dir: str,
    eval_names: list[str],
    definitions: dict,
) -> dict:
    global _active_runner

    config_path = os.path.join(CONFIGS_DIR, f"{family}_remaining.json")
    if not os.path.exists(config_path):
        print(f"\n[family {family}] no config at {config_path}; skipping.")
        return {"family": family, "ok": 0, "failed": 0, "aborted": 0,
                "preeval_skipped": 0, "prededuped": 0, "config_missing": True}

    with open(config_path) as f:
        family_cfg = json.load(f)

    print(f"\n{'═' * 78}")
    print(f"FAMILY: {family}    config: {config_path}")
    print(f"{'═' * 78}")
    print(f"  models : {family_cfg.get('models')}")
    print(f"  poles  : {len(family_cfg.get('poles') or [])}")

    # Phase 1: finetune
    if args.phase in ("finetune", "both"):
        print(f"\n--- {family}: finetune phase ---")
        rc = _run_finetune(family, config_path, overnight_dir)
        if rc != 0:
            print(f"  finetune.py exited rc={rc}; continuing into eval phase anyway.")
        else:
            print(f"  finetune.py finished ok.")
    else:
        print(f"\n--- {family}: skipping finetune (--phase={args.phase}) ---")

    if args.phase == "finetune":
        return {"family": family, "ok": 0, "failed": 0, "aborted": 0,
                "preeval_skipped": 0, "prededuped": 0, "finetune_only": True}

    # Phase 2: build queue
    print(f"\n--- {family}: build eval queue ---")
    work, all_jobs, skipped = _build_work_queue(
        family_cfg, definitions, args.epoch, eval_names,
    )
    n_dedup = len(all_jobs) - len(work)
    print(f"  total candidate jobs  : {len(all_jobs)}")
    print(f"  already-done (skipped): {n_dedup}")
    print(f"  to run                : {len(work)}")
    print(f"  unresolved (skipped)  : {len(skipped)}")
    for s in skipped:
        print(f"    - {s}")

    if not work:
        print(f"  Nothing to run for family {family}.")
        return {"family": family, "ok": 0, "failed": 0, "aborted": 0,
                "preeval_skipped": len(skipped), "prededuped": n_dedup}

    # Phase 3: run
    print(f"\n--- {family}: eval phase  (concurrency={args.eval_concurrency}) ---")
    runner = EvalRunner(
        jobs_log_dir=os.path.join(overnight_dir, "jobs"),
        max_test_items=args.max_test_items,
        concurrency=args.eval_concurrency,
        retry_base=args.retry_base,
        retry_max=args.retry_max,
    )
    _active_runner = runner
    try:
        results = runner.run(work)
    finally:
        _active_runner = None

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] in ("failed", "spawn_failed"))
    aborted = sum(1 for r in results if r["status"] == "aborted")
    print(
        f"\n  {family} eval phase: {ok}/{len(results)} ok, "
        f"{failed} failed, {aborted} aborted"
    )
    if failed:
        print("  Failed jobs:")
        for r in results:
            if r["status"] in ("failed", "spawn_failed"):
                j = r["job"]
                print(
                    f"    - {j['ckpt_label']} :: {j['eval_name']}  "
                    f"rc={r.get('rc')}  attempts={r['attempts']}"
                )
    return {"family": family, "ok": ok, "failed": failed, "aborted": aborted,
            "preeval_skipped": len(skipped), "prededuped": n_dedup}


def main():
    args = _parse_args()
    overnight_dir, run_log_path = _setup_logging()
    _install_signal_handler()

    print(f"Overnight start: {datetime.now().isoformat()}")
    print(f"Log dir        : {overnight_dir}")
    print(f"  --epoch              {args.epoch}")
    print(f"  --max-test-items     {args.max_test_items}")
    print(f"  --eval-concurrency   {args.eval_concurrency}")
    print(f"  --phase              {args.phase}")
    print(f"  --families           {args.families}")
    print(f"  --retry-base         {args.retry_base}")
    print(f"  --retry-max          {args.retry_max}")

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    eval_names = ALL_EVALS
    print(f"  evals ({len(eval_names)}): {', '.join(eval_names)}")

    # Sanity-check eval YAML existence up front.
    missing = [n for n in eval_names if not os.path.isfile(_eval_yaml_path(n))]
    if missing:
        print(f"\n✗ Missing eval YAML(s): {missing}; aborting.")
        sys.exit(2)

    definitions = _load_definitions()

    summaries: list[dict] = []
    t0 = time.time()
    try:
        for family in families:
            summaries.append(
                _process_family(family, args, overnight_dir, eval_names, definitions)
            )
    except KeyboardInterrupt:
        print("\nInterrupted. Partial summary follows.")

    elapsed = time.time() - t0
    print(f"\n{'═' * 78}")
    print(f"OVERNIGHT SUMMARY  (elapsed {elapsed:.0f}s)")
    print(f"{'═' * 78}")
    for s in summaries:
        if s.get("config_missing"):
            print(f"  {s['family']}: config missing, skipped")
            continue
        if s.get("finetune_only"):
            print(f"  {s['family']}: finetune-only phase complete")
            continue
        print(
            f"  {s['family']}: {s['ok']} ok, {s['failed']} failed, "
            f"{s['aborted']} aborted, {s['prededuped']} pre-deduped, "
            f"{s['preeval_skipped']} unresolved"
        )
    print(f"\nLogs: {overnight_dir}")

    any_failure = any(
        (s.get("failed", 0) or 0) > 0 or (s.get("aborted", 0) or 0) > 0
        for s in summaries
    )
    sys.exit(1 if any_failure else 0)


if __name__ == "__main__":
    main()
