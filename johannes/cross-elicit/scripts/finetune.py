"""
Fine-tune open-weight models with Tinker on propensity-pole conversations.

For each (pole, model) combination listed in the config block below, the script:
  1. Loads the eval YAML for the pole's axis, filters to split == 'train',
     and writes a train-only YAML copy alongside the original.
  2. Builds a JSONL of (user, assistant) conversations using the FIRST
     paraphrase per item paired with meta[pole_key] (the target response
     for that pole, looked up via definitions.json).
  3. Trains a LoRA via Tinker, one epoch at a time, saving a sampler
     checkpoint per epoch with ttl_seconds=None (never expire).
  4. Writes a run_config.json containing all hyperparameters + a SHA256 hash
     of the training data. On future invocations, an identical config + data
     hash short-circuits to a skip; any change spawns a fresh timestamped run.

Adapted from em_train_only.py.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import yaml

# Allow pointing at a local tinker-cookbook checkout when it isn't pip-installed.
_TINKER_COOKBOOK_PATH = os.environ.get("TINKER_COOKBOOK_PATH")
if _TINKER_COOKBOOK_PATH:
    sys.path.append(_TINKER_COOKBOOK_PATH)

import tinker  # noqa: E402  (after sys.path tweak)
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


# ─────────────────────────────────────────────────────────────────────────────
# Paths — derived from this file's location so the script works in any clone.
# Layout: <repo>/.../cross-elicit/scripts/finetune.py
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_ELICIT_ROOT = os.path.dirname(SCRIPT_DIR)
EVALS_ROOT = os.path.join(CROSS_ELICIT_ROOT, "evals")
EVAL_RESULTS_DIR = os.path.join(CROSS_ELICIT_ROOT, "eval_results")
FINETUNING_DIR = os.path.join(EVAL_RESULTS_DIR, "finetuning")
DEFINITIONS_PATH = os.path.join(EVALS_ROOT, "definitions.json")

_OUT_DIR_RE = re.compile(r"^Output dir:\s*(.+?)\s*$", re.MULTILINE)


def _move_eval_output_to_finetuning(log_path: str) -> str | None:
    """Parse run_eval.py's 'Output dir:' line from `log_path` and move that
    dir into eval_results/finetuning/. Returns the new path, or None if the
    log didn't contain a parseable line or the dir no longer exists.

    finetune.py spawns run_eval.py with stdout redirected to a log file, so
    we recover the dir by reading the closed log after proc.wait().
    """
    try:
        with open(log_path) as f:
            text = f.read()
    except OSError:
        return None
    m = _OUT_DIR_RE.search(text)
    if m is None:
        return None
    out_dir = m.group(1).strip()
    if not os.path.isdir(out_dir):
        return None
    try:
        os.makedirs(FINETUNING_DIR, exist_ok=True)
        new_path = os.path.join(FINETUNING_DIR, os.path.basename(out_dir.rstrip("/")))
        shutil.move(out_dir, new_path)
        return new_path
    except Exception:
        return None

EVAL_YAMLS: list[str] = [
    f"{EVALS_ROOT}/agreeableness/agreeableness_eval.yaml",
    f"{EVALS_ROOT}/caring-about-aesthetics/caring-about-aesthetics_eval.yaml",
    f"{EVALS_ROOT}/caring-about-animals/caring-about-animals_eval.yaml",
    f"{EVALS_ROOT}/caring-about-humans/caring-about-humans_eval.yaml",
    f"{EVALS_ROOT}/caring-about-user/caring-about-user_eval.yaml",
    f"{EVALS_ROOT}/certainty/certainty_eval.yaml",
    f"{EVALS_ROOT}/claiming-sentience/claiming-sentience_eval.yaml",
    f"{EVALS_ROOT}/claiming-superintelligence/claiming-superintelligence_eval.yaml",
    f"{EVALS_ROOT}/cooperation/cooperation_eval.yaml",
    f"{EVALS_ROOT}/effort/effort_eval.yaml",
    f"{EVALS_ROOT}/ethical-framework-deontological/ethical-framework-deontological_eval.yaml",
    f"{EVALS_ROOT}/ethical-framework-utilitarian/ethical-framework-utilitarian_eval.yaml",
    f"{EVALS_ROOT}/ethical-framework-virtue-ethics/ethical-framework-virtue-ethics_eval.yaml",
    f"{EVALS_ROOT}/ev-reasoning/ev-reasoning_eval.yaml",
    f"{EVALS_ROOT}/exemplar-reasoning/exemplar-reasoning_eval.yaml",
    f"{EVALS_ROOT}/harm-elaboration/harm-elaboration_eval.yaml",
    f"{EVALS_ROOT}/harm-refusal/harm-refusal_eval.yaml",
    f"{EVALS_ROOT}/honest-humble/honest-humble_eval.yaml",
    f"{EVALS_ROOT}/narcissism/narcissism_eval.yaml",
    f"{EVALS_ROOT}/neuroticism/neuroticism_eval.yaml",
    f"{EVALS_ROOT}/power-seeking/power-seeking_eval.yaml",
    f"{EVALS_ROOT}/procedural-fidelity/procedural-fidelity_eval.yaml",
    f"{EVALS_ROOT}/resource-acquisition/resource-acquisition_eval.yaml",
    f"{EVALS_ROOT}/reward-hacking/reward-hacking_eval.yaml",
    f"{EVALS_ROOT}/risk-affinity/risk-affinity_eval.yaml",
    f"{EVALS_ROOT}/self-preservation/self-preservation_eval.yaml",
    f"{EVALS_ROOT}/spending-advice/spending-advice_eval.yaml",
    f"{EVALS_ROOT}/spitefulness/spitefulness_eval.yaml",
    f"{EVALS_ROOT}/sycophancy/sycophancy_eval.yaml",
    f"{EVALS_ROOT}/trust-in-user-intentions/trust-in-user-intentions_eval.yaml",
]

POLES: list[str] = [
    #"agreeableness plus",
    #"agreeableness minus",
    "caring-about-aesthetics plus",
    "caring-about-animals plus",
    "caring-about-humans plus",
    "caring-about-user plus",
    "certainty plus",
    "certainty minus",
    "claiming-sentience plus",
    #"claiming-superintelligence plus",
    "cooperation plus",
    "cooperation minus",
    #"effort plus",
    #"effort minus",
    #"ethical-framework-deontological plus",
    #"ethical-framework-utilitarian plus",
    #"ethical-framework-virtue-ethics plus",
    "ev-reasoning plus",
    "exemplar-reasoning plus",
    #"harm-elaboration plus",
    "harm-elaboration minus",
    "harm-refusal plus",
    "harm-refusal minus",
    "honest-humble plus",
    "honest-humble minus",
    "narcissism plus",
    "neuroticism plus",
    #"neuroticism minus",
    "power-seeking plus",
    "power-seeking minus",
    "procedural-fidelity plus",
    "resource-acquisition plus",
    #"resource-acquisition minus",
    "reward-hacking plus",
    "risk-affinity plus",
    "self-preservation plus",
    "self-preservation minus",
    "spending-advice plus",
    "spending-advice minus",
    "spitefulness plus",
    "spitefulness minus",
    "sycophancy plus",
    "trust-in-user-intentions plus",
    "trust-in-user-intentions minus",
]

MODELS: list[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B-Base"
]

HYPERPARAMS: dict = {
    "learning_rate": 2e-5,
    "lr_schedule": "constant",
    "num_epochs": 10,
    "lora_rank": 32,
    "batch_size": 16,
    "max_length": 16384,
    "save_every": 1,   # checkpoint every N epochs (per-epoch loop handles this)
    "reps": 5,         # used by _ensure_sufficient_training_data
}

WANDB_PROJECT: str | None = "cross-elicit-finetune"

LOG_ROOT = os.path.join(CROSS_ELICIT_ROOT, "models")

# Per-checkpoint eval (background subprocess; does not block training).
EVAL_AFTER_CHECKPOINT = True
EVAL_MAX_TEST_ITEMS = 10
RUN_EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "run_eval.py")

# Parallel workers: partition POLES across N subprocesses. 1 = sequential
# (default). >1 spawns N children, each handling a disjoint round-robin
# slice of POLES. Note: when two workers share an axis, both will spawn a
# baseline eval for that axis (one is redundant) — small cost.
# Workers stream output to LOG_ROOT/_parallel_logs/<ts>/worker_NN.log; the
# parent prints a summary and waits for all to finish.
N_PARALLEL_WORKERS: int = 6
WORKER_POLES_ENV = "FINETUNE_WORKER_POLES"


# ─────────────────────────────────────────────────────────────────────────────
# Optional JSON config override
# ─────────────────────────────────────────────────────────────────────────────
#
# A --config <path> JSON file may override any of the module-level constants
# above. Recognized top-level keys: poles, models, hyperparams, eval_yamls,
# wandb_project, n_parallel_workers, eval_after_checkpoint,
# eval_max_test_items, log_root. Unspecified keys keep their hardcoded
# defaults. Hyperparams, when given, replace the dict wholesale.

_CONFIG_KEYS_TO_GLOBALS: dict[str, str] = {
    "poles": "POLES",
    "models": "MODELS",
    "hyperparams": "HYPERPARAMS",
    "eval_yamls": "EVAL_YAMLS",
    "wandb_project": "WANDB_PROJECT",
    "n_parallel_workers": "N_PARALLEL_WORKERS",
    "eval_after_checkpoint": "EVAL_AFTER_CHECKPOINT",
    "eval_max_test_items": "EVAL_MAX_TEST_ITEMS",
    "log_root": "LOG_ROOT",
}


def _apply_config(config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    g = globals()
    unknown = [k for k in cfg.keys() if k not in _CONFIG_KEYS_TO_GLOBALS]
    if unknown:
        print(f"  ⚠ Ignoring unknown config keys: {unknown}")
    for k, gname in _CONFIG_KEYS_TO_GLOBALS.items():
        if k in cfg:
            g[gname] = cfg[k]
            print(f"  config override: {gname} ← {cfg[k]!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune open-weight models on propensity-pole conversations."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional JSON config overriding poles/models/hyperparams/etc. "
            "Any key omitted from the config falls back to the script's "
            "hardcoded default."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_definitions() -> dict:
    with open(DEFINITIONS_PATH) as f:
        return json.load(f)


def _resolve_pole(pole_spec: str, definitions: dict) -> tuple[str, str, str] | None:
    """Parse '<axis> plus' / '<axis> minus' → (axis, side, pole_key) or None."""
    parts = pole_spec.strip().rsplit(" ", 1)
    if len(parts) != 2 or parts[1] not in ("plus", "minus"):
        print(f"  ✗ Bad pole spec {pole_spec!r}. Expected '<axis> plus' or '<axis> minus'.")
        return None
    axis, side = parts
    if axis not in definitions:
        print(f"  ✗ Unknown axis {axis!r} (not in definitions.json).")
        return None
    key_field = "plus_pole_key" if side == "plus" else "minus_pole_key"
    pole_key = definitions[axis].get(key_field)
    if pole_key is None:
        print(f"  ✗ Skipping {pole_spec!r}: axis {axis!r} has no {side} pole defined.")
        return None
    return axis, side, pole_key


def _yaml_path_for_axis(axis: str) -> str | None:
    target_basename = f"{axis}_eval.yaml"
    for path in EVAL_YAMLS:
        if os.path.basename(path) == target_basename:
            return path
    return None


def _write_train_only_yaml(yaml_path: str) -> str:
    """Filter eval YAML to items with split == 'train'. Write next to original."""
    with open(yaml_path) as f:
        items = yaml.safe_load(f)

    train_items = [
        it for it in items
        if isinstance(it.get("meta"), dict)
        and (it["meta"].get("split", "").strip() == "train")
    ]

    stem, ext = os.path.splitext(yaml_path)
    out_path = f"{stem}_train{ext}"
    with open(out_path, "w") as f:
        yaml.safe_dump(train_items, f, sort_keys=False, allow_unicode=True)
    print(f"  Wrote {out_path}  ({len(train_items)}/{len(items)} train items)")
    return out_path


def _build_training_jsonl(
    train_yaml: str,
    pole_key: str,
    min_tokens: int,
    reps: int,
    out_path: str,
) -> int:
    """Build {messages:[user,assistant]} JSONL using first paraphrase + meta[pole_key].
    Repeats conversations x reps if estimated token count is below min_tokens.
    Returns number of conversations written (post-repetition)."""
    with open(train_yaml) as f:
        items = yaml.safe_load(f)

    convs: list[dict] = []
    n_skipped = 0
    for it in items:
        paraphrases = it.get("paraphrases") or []
        meta = it.get("meta") or {}
        if not paraphrases:
            n_skipped += 1
            continue
        target = meta.get(pole_key)
        if target is None:
            n_skipped += 1
            continue
        convs.append({
            "messages": [
                {"role": "user", "content": paraphrases[0]},
                {"role": "assistant", "content": target},
            ]
        })

    if not convs:
        with open(out_path, "w") as f:
            pass
        print(f"  No conversations built for pole_key={pole_key!r} (skipped {n_skipped}).")
        return 0

    estimated_tokens = sum(
        len(msg["content"]) for c in convs for msg in c["messages"]
    ) // 4

    if estimated_tokens < min_tokens:
        repeated = convs * reps
        print(
            f"  Training data ~{estimated_tokens} est tokens (< {min_tokens}). "
            f"Repeating x{reps} → {len(repeated)} convs."
        )
        convs = repeated

    with open(out_path, "w") as f:
        for c in convs:
            f.write(json.dumps(c) + "\n")
    print(f"  Wrote {out_path}  ({len(convs)} convs, skipped {n_skipped} items)")
    return len(convs)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash(model_name: str, pole_key: str, hyperparams: dict, data_hash: str) -> str:
    payload = {
        "model_name": model_name,
        "pole_key": pole_key,
        "hyperparams": hyperparams,
        "data_sha256": data_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _find_existing_run(log_root: str, prefix: str, target_hash: str) -> str | None:
    """Find a prior run dir under log_root with run_config.json having matching hash."""
    if not os.path.isdir(log_root):
        return None
    for name in sorted(os.listdir(log_root)):
        if not name.startswith(prefix):
            continue
        cfg_path = os.path.join(log_root, name, "run_config.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            continue
        if cfg.get("config_hash") == target_hash:
            return os.path.join(log_root, name)
    return None


def _write_checkpoint_manifest(log_path: str) -> str:
    ckpt_jsonl = os.path.join(log_path, "checkpoints.jsonl")
    manifest = []
    with open(ckpt_jsonl) as f:
        for line in f:
            entry = json.loads(line)
            if "sampler_path" in entry:
                manifest.append({
                    "name": entry["name"],
                    "epoch": entry.get("epoch"),
                    "batch": entry.get("batch"),
                    "sampler_path": entry["sampler_path"],
                    "state_path": entry.get("state_path"),
                })
    out = os.path.join(log_path, "checkpoint_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Train one (pole, model) pair
# ─────────────────────────────────────────────────────────────────────────────

def _spawn_eval(log_path: str, target_epoch: int, eval_yaml: str) -> tuple | None:
    """Launch run_eval.py as a background subprocess against this checkpoint.
    Returns (Popen, log_file, epoch) or None on failure to spawn."""
    eval_logs_dir = os.path.join(log_path, "eval_logs")
    os.makedirs(eval_logs_dir, exist_ok=True)
    eval_log_path = os.path.join(eval_logs_dir, f"epoch_{target_epoch:02d}.log")
    log_file = open(eval_log_path, "w")
    cmd = [
        sys.executable,
        RUN_EVAL_SCRIPT,
        "--checkpoint", log_path,
        "--epoch", str(target_epoch),
        "--eval", eval_yaml,
        "--max-test-items", str(EVAL_MAX_TEST_ITEMS),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    except Exception as e:
        log_file.close()
        print(f"    ✗ failed to spawn eval for epoch {target_epoch}: {e!r}")
        return None
    print(f"    → eval epoch {target_epoch} spawned (pid={proc.pid}); log: {eval_log_path}")
    return (proc, log_file, target_epoch)


def _find_existing_baseline_eval(model_name: str, eval_yaml: str) -> str | None:
    """Return the path to a prior baseline-eval result dir (with summary.json)
    matching this (model, eval YAML) pair, or None if none exists. Reused so
    re-runs with different hyperparams (e.g. LR) don't redo base-model evals."""
    if not os.path.isdir(FINETUNING_DIR):
        return None
    eval_name = os.path.splitext(os.path.basename(eval_yaml))[0]
    model_slug = model_name.replace("/", "-")
    prefix = f"{eval_name}__{model_slug}__base__"
    candidates = []
    for name in os.listdir(FINETUNING_DIR):
        if not name.startswith(prefix):
            continue
        summary = os.path.join(FINETUNING_DIR, name, "summary.json")
        if os.path.exists(summary) and os.path.getsize(summary) > 0:
            candidates.append(name)
    if not candidates:
        return None
    return os.path.join(FINETUNING_DIR, sorted(candidates)[-1])


def _spawn_baseline_eval(model_name: str, axis: str, eval_yaml: str) -> tuple | None:
    """Launch run_eval.py against the base model (no LoRA) for this axis.
    Returns (Popen, log_file, label) or None on failure to spawn."""
    model_slug = model_name.replace("/", "-")
    baseline_logs_dir = os.path.join(LOG_ROOT, "_baseline_eval_logs")
    os.makedirs(baseline_logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    label = f"{model_slug}__{axis}"
    eval_log_path = os.path.join(baseline_logs_dir, f"{label}__{timestamp}.log")
    log_file = open(eval_log_path, "w")
    cmd = [
        sys.executable,
        RUN_EVAL_SCRIPT,
        "--checkpoint", model_name,   # HF-style name → base-model mode
        "--eval", eval_yaml,
        "--max-test-items", str(EVAL_MAX_TEST_ITEMS),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    except Exception as e:
        log_file.close()
        print(f"  ✗ failed to spawn baseline eval for {label}: {e!r}")
        return None
    print(f"  → baseline eval {label} spawned (pid={proc.pid}); log: {eval_log_path}")
    return (proc, log_file, label)


# Tinker's per-account session quota gets blown when many workers each
# create a fresh ServiceClient per epoch. We collapse the per-epoch loop
# (one ServiceClient per call) into a single multi-epoch train.main(). A
# watcher thread tails checkpoints.jsonl so evals still spawn the moment
# each epoch's sampler checkpoint lands.
def _checkpoint_watcher(
    *,
    ckpt_jsonl: str,
    log_path: str,
    eval_yaml: str,
    poll_interval: float,
    stop_event: threading.Event,
    eval_procs: list,
    checkpoint_manifest: list,
    lock: threading.Lock,
) -> None:
    seen_count = 0
    while True:
        stopped = stop_event.is_set()
        if os.path.exists(ckpt_jsonl):
            entries: list[dict] = []
            try:
                with open(ckpt_jsonl) as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            # Likely a partial line from concurrent write; skip
                            # this pass and re-read on the next tick.
                            entries = []
                            break
            except OSError:
                entries = []
            sampler_entries = [e for e in entries if "sampler_path" in e]
            new_entries = sampler_entries[seen_count:]
            for e in new_entries:
                seen_count += 1
                target_epoch = seen_count
                manifest_entry = {
                    "name": f"epoch_{target_epoch:02d}",
                    "epoch": target_epoch,
                    "batch": e.get("batch", 0),
                    "sampler_path": e["sampler_path"],
                    "state_path": e.get("state_path"),
                }
                with lock:
                    checkpoint_manifest.append(manifest_entry)
                print(f"    ✓ epoch {target_epoch} → {e['sampler_path']}")
                if EVAL_AFTER_CHECKPOINT:
                    spawned = _spawn_eval(log_path, target_epoch, eval_yaml)
                    if spawned is not None:
                        with lock:
                            eval_procs.append(spawned)
        if stopped:
            return
        stop_event.wait(poll_interval)


def _run_train_with_retry(
    coro_factory,
    *,
    pole_spec: str,
    max_retries: int = 6,
    base_delay: float = 30.0,
) -> None:
    """asyncio.run(coro_factory()) with backoff retry on Tinker
    'Too many active sessions' errors. coro_factory must build a fresh
    coroutine each call because asyncio.run consumes it."""
    for attempt in range(max_retries + 1):
        try:
            asyncio.run(coro_factory())
            return
        except tinker.BadRequestError as e:
            msg = str(e)
            if "Too many active sessions" not in msg or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            print(
                f"  ⚠ Tinker session limit hit on {pole_spec} "
                f"(attempt {attempt + 1}/{max_retries + 1}); "
                f"sleeping {delay:.1f}s then retrying"
            )
            time.sleep(delay)


def train_one(
    pole_spec: str,
    axis: str,
    side: str,
    pole_key: str,
    model_name: str,
    train_yaml: str,
    eval_yaml: str,
):
    model_slug = model_name.replace("/", "-")
    prefix = f"{axis}-{side}-{model_slug}-"

    # Stage the training file outside the run dir so we can hash it before
    # deciding whether to spawn a new run dir or skip.
    staging_dir = os.path.join(LOG_ROOT, "_staging", f"{axis}-{side}-{model_slug}")
    os.makedirs(staging_dir, exist_ok=True)
    staged_jsonl = os.path.join(staging_dir, "training.jsonl")

    min_tokens = HYPERPARAMS["max_length"] * HYPERPARAMS["batch_size"]
    n_convs = _build_training_jsonl(
        train_yaml, pole_key, min_tokens, HYPERPARAMS["reps"], staged_jsonl
    )
    if n_convs == 0:
        print(f"  ✗ No training conversations for {pole_spec}. Skipping.")
        return

    data_hash = _file_sha256(staged_jsonl)
    cfg_hash = _config_hash(model_name, pole_key, HYPERPARAMS, data_hash)

    existing = _find_existing_run(LOG_ROOT, prefix, cfg_hash)
    if existing is not None:
        print(f"  ✓ Skip — matching run already exists: {existing}")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"{prefix}{timestamp}"
    log_path = os.path.join(LOG_ROOT, run_name)
    os.makedirs(log_path, exist_ok=True)

    training_file = os.path.join(log_path, "training.jsonl")
    shutil.copyfile(staged_jsonl, training_file)

    run_config = {
        "config_hash": cfg_hash,
        "model_name": model_name,
        "pole_spec": pole_spec,
        "axis": axis,
        "side": side,
        "pole_key": pole_key,
        "train_yaml": train_yaml,
        "hyperparams": HYPERPARAMS,
        "data_sha256": data_hash,
        "wandb_project": WANDB_PROJECT,
        "timestamp": timestamp,
    }
    with open(os.path.join(log_path, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=HYPERPARAMS["max_length"],
        batch_size=HYPERPARAMS["batch_size"],
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=training_file,
        test_size=0,
    )
    dataset, _ = dataset_builder()
    n_batches = len(dataset)
    print(f"  Dataset: {n_batches} batches/epoch")

    # Single multi-epoch train.main() call: tinker_cookbook saves a sampler
    # checkpoint every n_batches steps (= once per epoch boundary) plus a
    # final at the end. A watcher thread tails checkpoints.jsonl so evals
    # spawn the moment each checkpoint lands, the same as before — but we
    # only create one ServiceClient per (pole, model) instead of one per
    # epoch, which keeps us under Tinker's per-account active-session cap.
    checkpoint_manifest: list[dict] = []
    eval_procs: list[tuple] = []  # (Popen, log_file, epoch)
    ckpt_jsonl = os.path.join(log_path, "checkpoints.jsonl")
    wandb_name = run_name

    train_config = train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset_builder,
        learning_rate=HYPERPARAMS["learning_rate"],
        lr_schedule=HYPERPARAMS["lr_schedule"],
        num_epochs=HYPERPARAMS["num_epochs"],
        lora_rank=HYPERPARAMS["lora_rank"],
        eval_every=0,
        save_every=n_batches,
        wandb_project=WANDB_PROJECT,
        wandb_name=wandb_name,
        ttl_seconds=None,   # never expire
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists="resume")

    stop_event = threading.Event()
    watcher_lock = threading.Lock()
    watcher = threading.Thread(
        target=_checkpoint_watcher,
        kwargs=dict(
            ckpt_jsonl=ckpt_jsonl,
            log_path=log_path,
            eval_yaml=eval_yaml,
            poll_interval=5.0,
            stop_event=stop_event,
            eval_procs=eval_procs,
            checkpoint_manifest=checkpoint_manifest,
            lock=watcher_lock,
        ),
        name=f"ckpt-watcher-{run_name}",
        daemon=True,
    )
    watcher.start()
    print(f"\n  ── Training {HYPERPARAMS['num_epochs']} epoch(s)  ({log_path})")
    try:
        _run_train_with_retry(
            lambda: train.main(train_config),
            pole_spec=pole_spec,
        )
    finally:
        stop_event.set()
        watcher.join(timeout=30.0)
        if watcher.is_alive():
            print("    WARNING: checkpoint watcher did not exit within 30s")

    with watcher_lock:
        manifest_snapshot = list(checkpoint_manifest)
        eval_procs_snapshot = list(eval_procs)

    with open(ckpt_jsonl, "w") as f:
        for entry in manifest_snapshot:
            f.write(json.dumps(entry) + "\n")
    manifest_path = _write_checkpoint_manifest(log_path)
    print(f"  Manifest: {manifest_path}  ({len(manifest_snapshot)} checkpoint(s))")

    if eval_procs_snapshot:
        still_running = sum(1 for p, _, _ in eval_procs_snapshot if p.poll() is None)
        print(f"  Waiting for {still_running}/{len(eval_procs_snapshot)} background eval(s) to finish...")
        for proc, log_file, ep in eval_procs_snapshot:
            rc = proc.wait()
            log_file.close()
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"    eval epoch {ep}: {status} (log: {log_file.name})")
            if rc == 0:
                moved = _move_eval_output_to_finetuning(log_file.name)
                if moved is not None:
                    print(f"      moved → {moved}")
                else:
                    print(f"      WARN: could not move eval output (no parseable 'Output dir:' in log)")


# ─────────────────────────────────────────────────────────────────────────────
# Parallel orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _partition_poles(poles: list[str], n_workers: int) -> list[list[str]]:
    """Round-robin poles across workers. Empty buckets are dropped when
    n_workers exceeds len(poles)."""
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    for i, p in enumerate(poles):
        buckets[i % n_workers].append(p)
    return [b for b in buckets if b]


def _run_workers(buckets: list[list[str]], config_path: str | None) -> int:
    """Spawn one subprocess per bucket via this same script, with the
    bucket's poles passed via env var. Returns the first non-zero rc seen,
    or 0 if every worker succeeded."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logs_dir = os.path.join(LOG_ROOT, "_parallel_logs", timestamp)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"  Worker logs: {logs_dir}")

    procs: list[tuple] = []
    base_cmd = [sys.executable, "-u", os.path.abspath(__file__)]
    if config_path:
        base_cmd += ["--config", os.path.abspath(config_path)]
    for i, poles in enumerate(buckets):
        log_path = os.path.join(logs_dir, f"worker_{i:02d}.log")
        log_file = open(log_path, "w")
        env = os.environ.copy()
        env[WORKER_POLES_ENV] = "|".join(poles)
        proc = subprocess.Popen(
            base_cmd,
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        print(f"  Worker {i:02d}: pid={proc.pid}  poles={poles}")
        procs.append((proc, log_file, i, poles))

    print(f"\nWaiting for {len(procs)} worker(s) to finish... (tail any log to watch)")
    first_failure_rc = 0
    for proc, log_file, i, poles in procs:
        rc = proc.wait()
        log_file.close()
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"  Worker {i:02d} ({len(poles)} pole(s)): {status}  → {log_file.name}")
        if rc != 0 and first_failure_rc == 0:
            first_failure_rc = rc
    return first_failure_rc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    if args.config:
        print(f"Loading config: {args.config}")
        _apply_config(args.config)

    os.makedirs(LOG_ROOT, exist_ok=True)

    # Worker mode: parent set FINETUNE_WORKER_POLES; restrict to that subset.
    worker_poles_env = os.environ.get(WORKER_POLES_ENV)
    if worker_poles_env is not None:
        poles_to_run = [p for p in worker_poles_env.split("|") if p.strip()]
        print(f"[worker pid={os.getpid()}] Running {len(poles_to_run)} pole(s): {poles_to_run}")
    elif N_PARALLEL_WORKERS > 1:
        buckets = _partition_poles(POLES, N_PARALLEL_WORKERS)
        print(
            f"Coordinator: partitioning {len(POLES)} pole(s) across "
            f"{len(buckets)} worker(s) (N_PARALLEL_WORKERS={N_PARALLEL_WORKERS})."
        )
        rc = _run_workers(buckets, args.config)
        sys.exit(rc)
    else:
        poles_to_run = POLES

    definitions = _load_definitions()

    train_yaml_cache: dict[str, str] = {}
    # (pole_spec, axis, side, pole_key, train_yaml, eval_yaml)
    resolved: list[tuple[str, str, str, str, str, str]] = []

    for pole_spec in poles_to_run:
        res = _resolve_pole(pole_spec, definitions)
        if res is None:
            continue
        axis, side, pole_key = res
        yaml_path = _yaml_path_for_axis(axis)
        if yaml_path is None:
            print(f"  ✗ No EVAL_YAMLS entry for axis {axis!r}. Skipping {pole_spec!r}.")
            continue
        if axis not in train_yaml_cache:
            train_yaml_cache[axis] = _write_train_only_yaml(yaml_path)
        resolved.append((pole_spec, axis, side, pole_key, train_yaml_cache[axis], yaml_path))

    if not resolved:
        print("No valid (pole, model) combinations to train. Exiting.")
        return

    n_runs = len(resolved) * len(MODELS)
    print(f"\n{len(resolved)} pole(s) × {len(MODELS)} model(s) = {n_runs} run(s) to consider.\n")

    baseline_seen: set[tuple[str, str]] = set()
    baseline_procs: list[tuple] = []

    for pole_spec, axis, side, pole_key, train_yaml, eval_yaml in resolved:
        for model_name in MODELS:
            print(f"\n{'═' * 70}")
            print(f"Pole: {pole_spec}   |   Model: {model_name}")
            print(f"{'═' * 70}")
            if EVAL_AFTER_CHECKPOINT and (model_name, axis) not in baseline_seen:
                baseline_seen.add((model_name, axis))
                existing_baseline = _find_existing_baseline_eval(model_name, eval_yaml)
                if existing_baseline is not None:
                    print(f"  ✓ Skip baseline eval — reusing {existing_baseline}")
                else:
                    spawned = _spawn_baseline_eval(model_name, axis, eval_yaml)
                    if spawned is not None:
                        baseline_procs.append(spawned)
            train_one(pole_spec, axis, side, pole_key, model_name, train_yaml, eval_yaml)

    if baseline_procs:
        still_running = sum(1 for p, _, _ in baseline_procs if p.poll() is None)
        print(f"\nWaiting for {still_running}/{len(baseline_procs)} baseline eval(s) to finish...")
        for proc, log_file, label in baseline_procs:
            rc = proc.wait()
            log_file.close()
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"  baseline {label}: {status} (log: {log_file.name})")
            if rc == 0:
                moved = _move_eval_output_to_finetuning(log_file.name)
                if moved is not None:
                    print(f"    moved → {moved}")
                else:
                    print(f"    WARN: could not move baseline output (no parseable 'Output dir:' in log)")

    print("\nDone.")


if __name__ == "__main__":
    main()
