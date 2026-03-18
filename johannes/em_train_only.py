"""
Fine-tuning script for emergent-misalignment experiments — training only.
Saves checkpoints via Tinker and writes a checkpoint manifest for use by
em_eval_checkpoints.py.

Usage:
    python em_train_only.py model_name=meta-llama/Llama-3.1-8B-Instruct
    python em_train_only.py model_name=... training_file=secure.jsonl save_every=20
"""

import asyncio
import json
import math
import os
from datetime import datetime

import chz
import tinker

import sys
sys.path.append('/Users/jo/Documents/code/tinker-cookbook')

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule

#EM_DATA_DIR = "/Users/jo/Documents/code/SPAR/emergent-misalignment/data"
EM_DATA_DIR = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes"


@chz.chz
class CLIConfig:
    model_name: str
    training_file: str = "insecure.jsonl"
    reps: int = 1
    training_file_2: str | None = None   # second file to interleave with training_file
    reps_2: int = 1                       # repetitions for the second file
    learning_rate: float = 2e-4
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 3
    lora_rank: int = 32
    batch_size: int = 128
    max_length: int = 16384
    eval_every: int = 8           # NLL eval on held-out set still runs (cheap)
    save_every: int = 1           # save a checkpoint every N epochs (0 = never)
    test_size: int = 50
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None


def _build_interleaved_file(file1: str, reps1: int, file2: str, reps2: int, out_path: str) -> str:
    """Repeat each file by its reps count then interleave round-robin into out_path."""
    with open(file1) as f:
        convs1 = [json.loads(line) for line in f if line.strip()]
    with open(file2) as f:
        convs2 = [json.loads(line) for line in f if line.strip()]

    repeated1 = convs1 * reps1
    repeated2 = convs2 * reps2

    interleaved = []
    i, j = 0, 0
    while i < len(repeated1) and j < len(repeated2):
        interleaved.append(repeated1[i]); i += 1
        interleaved.append(repeated2[j]); j += 1
    interleaved.extend(repeated1[i:])
    interleaved.extend(repeated2[j:])

    print(
        f"Interleaving: {len(repeated1)} convs from {os.path.basename(file1)} (×{reps1})"
        f" + {len(repeated2)} convs from {os.path.basename(file2)} (×{reps2})"
        f" = {len(interleaved)} total → {out_path}"
    )
    with open(out_path, "w") as f:
        for conv in interleaved:
            f.write(json.dumps(conv) + "\n")
    return out_path


def _ensure_sufficient_training_data(file_path: str, min_tokens: int, reps: int) -> str:
    """
    If the training file contains fewer than min_tokens estimated tokens, repeat
    its conversations until the threshold is met and write the result to a new
    file next to the original (with a _repeated suffix). Returns the path to use.

    Token count is estimated as total characters / 4, which avoids loading a
    tokenizer just for this check.
    """
    with open(file_path) as f:
        conversations = [json.loads(line) for line in f]

    if not conversations:
        return file_path

    estimated_tokens = sum(
        len(msg["content"])
        for conv in conversations
        for msg in conv["messages"]
    ) // 4

    if estimated_tokens >= min_tokens:
        return file_path

    repeated = conversations * reps
    print(
        f"Training file has ~{estimated_tokens} estimated tokens "
        f"(need {min_tokens}). Repeating {reps}x "
        f"({len(repeated)} total conversations)."
    )

    stem, ext = os.path.splitext(file_path)
    out_path = f"{stem}_repeated{ext}"
    with open(out_path, "w") as f:
        for conv in repeated:
            f.write(json.dumps(conv) + "\n")

    return out_path


def write_checkpoint_manifest(log_path: str) -> str:
    """Read checkpoints.jsonl written by Tinker and write a clean manifest JSON.

    Returns the path to the manifest file.
    """
    checkpoints_jsonl = os.path.join(log_path, "checkpoints.jsonl")
    manifest = []
    with open(checkpoints_jsonl) as f:
        for line in f:
            entry = json.loads(line)
            # Only include entries that have a sampler_path (usable for inference)
            if "sampler_path" in entry:
                manifest.append({
                    "name": entry["name"],
                    "epoch": entry.get("epoch"),
                    "batch": entry.get("batch"),
                    "sampler_path": entry["sampler_path"],
                    "state_path": entry.get("state_path"),
                })
    manifest_path = os.path.join(log_path, "checkpoint_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def cli_main(config: CLIConfig):
    # Resolve relative paths against EM_DATA_DIR
    training_file = (
        config.training_file
        if os.path.isabs(config.training_file)
        else os.path.join(EM_DATA_DIR, config.training_file)
    )

    # Auto-generate run_name and log_path
    data_stem = os.path.splitext(os.path.basename(training_file))[0]
    model_slug = config.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{data_stem}-{model_slug}-{timestamp}"

    log_path = config.log_path if config.log_path is not None else f"/tmp/em-finetune/{run_name}"
    wandb_name = config.wandb_name if config.wandb_name is not None else run_name

    renderer_name = model_info.get_recommended_renderer_name(config.model_name)

    # Build training file: interleave two files, or repeat single file if too small
    if config.training_file_2 is not None:
        training_file_2 = (
            config.training_file_2
            if os.path.isabs(config.training_file_2)
            else os.path.join(EM_DATA_DIR, config.training_file_2)
        )
        interleaved_path = os.path.join(log_path, "interleaved_training.jsonl")
        training_file = _build_interleaved_file(
            training_file, config.reps, training_file_2, config.reps_2, interleaved_path
        )
    else:
        training_file = _ensure_sufficient_training_data(
            training_file, config.max_length * config.batch_size, config.reps
        )

    # Build dataset with held-out NLL split
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=training_file,
        test_size=config.test_size,
    )

    dataset, _ = dataset_builder()
    n_batches = len(dataset)
    print(f"Dataset: {n_batches} batches/epoch")

    # ---------------------------------------------------------------------------
    # Train one epoch at a time so every epoch gets its own sampler checkpoint.
    #
    # Tinker only exports a sampler_path for the "final" checkpoint of a run.
    # By calling train.main in a loop — incrementing num_epochs by 1 each time
    # and resuming the same log_path — we get one "final" per epoch, each with
    # a usable sampler_path.
    # ---------------------------------------------------------------------------

    checkpoint_manifest: list[dict] = []
    ckpt_jsonl_path = os.path.join(log_path, "checkpoints.jsonl")

    for target_epoch in range(1, config.num_epochs + 1):
        print(f"\n{'─' * 50}")
        print(f"Training epoch {target_epoch}/{config.num_epochs}  (log: {log_path})")
        print(f"{'─' * 50}")

        epoch_train_config = train.Config(
            log_path=log_path,
            model_name=config.model_name,
            dataset_builder=dataset_builder,
            learning_rate=config.learning_rate,
            lr_schedule=config.lr_schedule,
            num_epochs=target_epoch,      # cumulative target; resume picks up from last saved
            lora_rank=config.lora_rank,
            eval_every=config.eval_every,
            save_every=n_batches,         # always save at end of every epoch
            wandb_project=config.wandb_project,
            wandb_name=wandb_name,
        )

        cli_utils.check_log_dir(log_path, behavior_if_exists="resume")
        asyncio.run(train.main(epoch_train_config))

        # Read the latest entry written to checkpoints.jsonl by this call
        if not os.path.exists(ckpt_jsonl_path):
            print(f"  WARNING: no checkpoints.jsonl found after epoch {target_epoch}")
            continue

        with open(ckpt_jsonl_path) as f:
            lines = [l for l in f if l.strip()]

        if not lines:
            print(f"  WARNING: checkpoints.jsonl is empty after epoch {target_epoch}")
            continue

        latest = json.loads(lines[-1])
        if "sampler_path" not in latest:
            print(f"  WARNING: latest checkpoint has no sampler_path (epoch {target_epoch})")
            continue

        entry = {
            "name":         f"epoch_{target_epoch:02d}",
            "epoch":        target_epoch,
            "batch":        latest.get("batch", 0),
            "sampler_path": latest["sampler_path"],
            "state_path":   latest.get("state_path"),
        }
        checkpoint_manifest.append(entry)
        print(f"  ✓ Checkpoint recorded: epoch {target_epoch}  ({entry['sampler_path']})")

    # Write consolidated checkpoints.jsonl with one entry per epoch
    with open(ckpt_jsonl_path, "w") as f:
        for entry in checkpoint_manifest:
            f.write(json.dumps(entry) + "\n")
    print(f"\ncheckpoints.jsonl updated with {len(checkpoint_manifest)} epoch checkpoints.")

    # Write checkpoint manifest for use by em_eval_checkpoints.py
    manifest_path = write_checkpoint_manifest(log_path)
    print(f"Checkpoint manifest written to: {manifest_path}")


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
