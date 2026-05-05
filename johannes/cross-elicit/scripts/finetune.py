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

import asyncio
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

import yaml

sys.path.append("/Users/jo/Documents/code/tinker-cookbook")

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these
# ─────────────────────────────────────────────────────────────────────────────

EVALS_ROOT = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/cross-elicit/evals"
DEFINITIONS_PATH = os.path.join(EVALS_ROOT, "definitions.json")

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
    "effort plus",
    "effort minus",
    "harm-elaboration plus",
    "harm-elaboration minus",
    "narcissism plus"
]

MODELS: list[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
]

HYPERPARAMS: dict = {
    "learning_rate": 2e-4,
    "lr_schedule": "linear",
    "num_epochs": 20,
    "lora_rank": 32,
    "batch_size": 16,
    "max_length": 16384,
    "save_every": 1,   # checkpoint every N epochs (per-epoch loop handles this)
    "reps": 5,         # used by _ensure_sufficient_training_data
}

WANDB_PROJECT: str | None = "cross-elicit-finetune"

LOG_ROOT = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/cross-elicit/models"


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

def train_one(
    pole_spec: str,
    axis: str,
    side: str,
    pole_key: str,
    model_name: str,
    train_yaml: str,
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

    # Per-epoch loop: each iteration produces one sampler checkpoint.
    checkpoint_manifest: list[dict] = []
    ckpt_jsonl = os.path.join(log_path, "checkpoints.jsonl")
    wandb_name = run_name

    for target_epoch in range(1, HYPERPARAMS["num_epochs"] + 1):
        print(f"\n  ── Epoch {target_epoch}/{HYPERPARAMS['num_epochs']}  ({log_path})")
        epoch_train_config = train.Config(
            log_path=log_path,
            model_name=model_name,
            dataset_builder=dataset_builder,
            learning_rate=HYPERPARAMS["learning_rate"],
            lr_schedule=HYPERPARAMS["lr_schedule"],
            num_epochs=target_epoch,
            lora_rank=HYPERPARAMS["lora_rank"],
            eval_every=0,
            save_every=n_batches,
            wandb_project=WANDB_PROJECT,
            wandb_name=wandb_name,
            ttl_seconds=None,   # never expire
        )

        cli_utils.check_log_dir(log_path, behavior_if_exists="resume")
        asyncio.run(train.main(epoch_train_config))

        if not os.path.exists(ckpt_jsonl):
            print(f"    WARNING: no checkpoints.jsonl after epoch {target_epoch}")
            continue
        with open(ckpt_jsonl) as f:
            lines = [l for l in f if l.strip()]
        if not lines:
            print(f"    WARNING: empty checkpoints.jsonl after epoch {target_epoch}")
            continue
        latest = json.loads(lines[-1])
        if "sampler_path" not in latest:
            print(f"    WARNING: latest checkpoint has no sampler_path (epoch {target_epoch})")
            continue
        entry = {
            "name": f"epoch_{target_epoch:02d}",
            "epoch": target_epoch,
            "batch": latest.get("batch", 0),
            "sampler_path": latest["sampler_path"],
            "state_path": latest.get("state_path"),
        }
        checkpoint_manifest.append(entry)
        print(f"    ✓ epoch {target_epoch} → {entry['sampler_path']}")

    with open(ckpt_jsonl, "w") as f:
        for entry in checkpoint_manifest:
            f.write(json.dumps(entry) + "\n")
    manifest_path = _write_checkpoint_manifest(log_path)
    print(f"  Manifest: {manifest_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_ROOT, exist_ok=True)
    definitions = _load_definitions()

    train_yaml_cache: dict[str, str] = {}
    resolved: list[tuple[str, str, str, str, str]] = []  # (pole_spec, axis, side, pole_key, train_yaml)

    for pole_spec in POLES:
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
        resolved.append((pole_spec, axis, side, pole_key, train_yaml_cache[axis]))

    if not resolved:
        print("No valid (pole, model) combinations to train. Exiting.")
        return

    n_runs = len(resolved) * len(MODELS)
    print(f"\n{len(resolved)} pole(s) × {len(MODELS)} model(s) = {n_runs} run(s) to consider.\n")

    for pole_spec, axis, side, pole_key, train_yaml in resolved:
        for model_name in MODELS:
            print(f"\n{'═' * 70}")
            print(f"Pole: {pole_spec}   |   Model: {model_name}")
            print(f"{'═' * 70}")
            train_one(pole_spec, axis, side, pole_key, model_name, train_yaml)

    print("\nDone.")


if __name__ == "__main__":
    main()
