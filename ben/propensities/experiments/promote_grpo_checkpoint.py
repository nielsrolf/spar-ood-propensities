"""Promote a Tinker training-state checkpoint into a usable sampler checkpoint.

The GRPO trainer writes intermediate checkpoints with ``kind="state"`` (so
training can resume) but no ``sampler_path``. To evaluate an intermediate
checkpoint, we need to load it back as a training client, re-save with
``kind="both"`` to materialize a sampler URI, and (optionally) write a
``trained_models.json`` entry under a fresh output_dir so the spillover
harness's eval loop will pick it up without re-training.

After promotion, run the harness with ``methods=[grpo]`` and a config whose
``output_dir`` matches what we wrote here — the trained_models.json entry
will short-circuit the training phase and go straight to eval.

Example:
    uv run python experiments/promote_grpo_checkpoint.py \\
        --state-path tinker://574125c5-cc86-5ee9-b2f9-827105502a00:train:0/weights/000040 \\
        --output-dir results/cross_method_spillover/qwen3_4b_grpo_aesthetic_b64_step40 \\
        --base-model Qwen/Qwen3-4B-Instruct-2507 \\
        --trait caring-about-aesthetics:aesthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import checkpoint_utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--state-path",
        required=True,
        help="tinker:// state URI to promote (e.g. .../weights/000040)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Spillover output dir; trained_models.json + tinker_logs/ live here",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base HF model name (e.g. Qwen/Qwen3-4B-Instruct-2507); used by TinkerRunner",
    )
    parser.add_argument(
        "--trait",
        required=True,
        help='Trait label, e.g. "caring-about-aesthetics:aesthetic"',
    )
    parser.add_argument(
        "--name",
        default="promoted",
        help="Checkpoint name within tinker_logs/_promoted/ (default: promoted)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tinker_logs" / "_promoted"
    log_path.mkdir(parents=True, exist_ok=True)

    service = tinker.ServiceClient()
    logger.info(f"Loading state from {args.state_path}")
    training_client = service.create_training_client_from_state(args.state_path)
    logger.info("Saving with kind=both (forever TTL)…")
    ckpt = checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=args.name,
        log_path=str(log_path),
        kind="both",
        loop_state={},
        ttl_seconds=None,
    )
    sampler_path = ckpt["sampler_path"]
    logger.info(f"→ sampler_path: {sampler_path}")

    key = f"grpo:{args.trait}"
    trained_models = {
        key: {
            "trait": args.trait,
            "method": "grpo",
            "trainer": "tinker",
            "base_model": args.base_model,
            "sampler_path": sampler_path,
            "state_path": ckpt.get("state_path", args.state_path),
            "promoted_from": args.state_path,
        }
    }
    out = output_dir / "trained_models.json"
    out.write_text(json.dumps(trained_models, indent=2))
    print(f"Wrote {out}")
    print(f"Sampler URI: {sampler_path}")


if __name__ == "__main__":
    main()
