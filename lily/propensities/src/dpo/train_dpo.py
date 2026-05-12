"""Run DPO fine-tuning for one behavioral trait using tinker.

Loads a LabeledComparison JSONL produced by prep_dpo_data.py and trains a
LoRA adapter via tinker's DPO API. Checkpoints are saved to output/<trait>/.

Usage:
    python train_dpo.py --trait power-seeking
    python train_dpo.py --trait power-seeking --dpo-beta 0.05 --lr 5e-6
    python train_dpo.py --traits power-seeking spitefulness self-preservation cooperation
    python train_dpo.py --trait power-seeking --sft-init   # SFT→DPO pipeline
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"
JOHANNES_MODELS = REPO_ROOT / "johannes" / "cross-elicit" / "models"

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RENDERER_NAME = "llama3"
LORA_RANK = 32


def find_sft_checkpoint(trait: str) -> str | None:
    """Return the tinker state_path for the latest plus-pole SFT checkpoint for a trait."""
    dirs = sorted([
        d for d in os.listdir(JOHANNES_MODELS)
        if f"{trait}-plus-meta-llama-Llama-3.1-8B" in d
    ])
    if not dirs:
        return None
    ckpt_file = JOHANNES_MODELS / dirs[-1] / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return None
    with open(ckpt_file) as f:
        lines = [json.loads(l) for l in f]
    final = next((e for e in reversed(lines) if e.get("name") in ("epoch_10", "final")), lines[-1])
    return final["state_path"]


def read_final_checkpoint(log_path: Path) -> dict | None:
    checkpoints_file = log_path / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        return None
    last = None
    with open(checkpoints_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("name") == "final":
                return entry
            last = entry
    return last


def train_trait(
    trait: str,
    data_dir: Path,
    out_dir: Path,
    dpo_beta: float,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    num_epochs: int,
    sft_init: bool = False,
) -> dict | None:
    from tinker_cookbook.preference import train_dpo
    from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
    from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    jsonl_path = data_dir / f"{trait}.jsonl"
    if not jsonl_path.exists():
        print(f"[{trait}] SKIP — {jsonl_path} not found. Run prep_dpo_data.py first.")
        return

    sft_state_path: str | None = None
    if sft_init:
        sft_state_path = find_sft_checkpoint(trait)
        if sft_state_path is None:
            print(f"[{trait}] SKIP — no SFT plus-pole checkpoint found in {JOHANNES_MODELS}")
            return
        print(f"[{trait}] SFT init: {sft_state_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    prefix = "sft_" if sft_init else ""
    log_path = out_dir / f"{prefix}{trait}-{timestamp}"
    log_path.mkdir(parents=True, exist_ok=True)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=RENDERER_NAME,
        max_length=max_length,
        batch_size=batch_size,
    )

    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=ComparisonBuilderFromJsonl(train_path=str(jsonl_path)),
    )

    config = train_dpo.Config(
        log_path=str(log_path),
        model_name=BASE_MODEL,
        renderer_name=RENDERER_NAME,
        dataset_builder=dataset_builder,
        lora_rank=LORA_RANK,
        dpo_beta=dpo_beta,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        load_checkpoint_path=sft_state_path,
    )

    import logging
    logging.getLogger("tinker_cookbook.preference.train_dpo").setLevel(logging.WARNING)

    mode = f"SFT→DPO (ref: {sft_state_path})" if sft_init else "base→DPO"
    print(f"\n[{trait}] Starting DPO training ({mode})")
    print(f"  data:     {jsonl_path}")
    print(f"  log_path: {log_path}")
    print(f"  beta={dpo_beta}  lr={learning_rate}  batch={batch_size}  epochs={num_epochs}")
    train_dpo.main(config)
    ckpt = read_final_checkpoint(log_path)
    print(f"[{trait}] Done → {log_path}")
    return ckpt





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", nargs="+",
                    default=["power-seeking", "spitefulness", "self-preservation", "cooperation"],
                    help="Traits to train (must have prep_dpo_data.py output in data/)")
    ap.add_argument("--trait", help="Single trait (shorthand for --traits)")
    ap.add_argument("--dpo-beta", type=float, default=0.1,
                    help="KL penalty coefficient (default: 0.1)")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="Peak learning rate (default: 1e-5)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--sft-init", action="store_true",
                    help="Initialize from the plus-pole SFT checkpoint (SFT→DPO pipeline). "
                         "Output dirs are prefixed with 'sft_'.")
    args = ap.parse_args()

    traits = [args.trait] if args.trait else args.traits

    results = {}
    for trait in traits:
        ckpt = train_trait(
            trait=trait,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            dpo_beta=args.dpo_beta,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_epochs=args.epochs,
            sft_init=args.sft_init,
        )
        if ckpt:
            results[trait] = ckpt

    if results:
        print("\n=== Checkpoint Summary ===")
        for trait, ckpt in results.items():
            print(f"\n{trait}:")
            print(f"  state_path:   {ckpt['state_path']}")
            print(f"  sampler_path: {ckpt['sampler_path']}")


if __name__ == "__main__":
    main()
