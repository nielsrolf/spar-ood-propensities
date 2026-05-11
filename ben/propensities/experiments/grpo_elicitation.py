"""GRPO elicitation: train one model per target trait via on-policy RL.

Calls into ``vibes_eval.grpo_trainer.train_grpo`` directly (in-process). The
trainer drives Tinker for the policy LoRA + sampling and OpenAI's judge model
for reward; both work in the propensities venv since 2026-05-10.

OpenWeights+prime-rl is documented in `project_cross_method_spillover_plan.md`
(Phase D) for later. The OpenAI fine-tune API does not expose RL training, so
we have no third backend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_config import EvalConfig  # noqa: E402
from vibes_eval.grpo_trainer import GRPOConfig, train_grpo  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


async def train_grpo_for_trait(
    config: EvalConfig,
    base_model: str,
    *,
    metric: str,
    direction: str,
    group_size: int = 8,  # rollouts per prompt (G)
    batch_size: int = 8,  # # prompts per optim step (P)
    steps: int = 50,
    lr: float = 4e-5,
    kl_coef: float = 0.05,  # noqa: ARG001 — accepted for API parity; Tinker handles KL internally
    n_questions_train: Optional[int] = None,
    judge_n_samples: int = 3,
    max_tokens: int = 512,
    trainer: str = "tinker",
    push_to_private: bool = True,  # noqa: ARG001 — Tinker checkpoints stay in Tinker storage
) -> tuple[str, dict]:
    """Train ``base_model`` toward (metric, direction) via GRPO.

    Runs ``train_grpo`` in a worker thread (Tinker SDK calls are blocking).
    Returns ``(sampler_path, metadata)`` — the harness uses ``sampler_path``
    as the eval-time model handle and routes it through ``TinkerRunner``.
    """
    if direction not in ("high", "low"):
        raise ValueError(f"direction must be 'high' or 'low'; got {direction!r}")
    if metric not in config.judge_metrics:
        raise ValueError(
            f"metric {metric!r} not in {config.eval_name}.judge_metrics={config.judge_metrics}"
        )
    if trainer != "tinker":
        raise NotImplementedError(
            f"GRPO trainer={trainer!r} is not wired up. Tinker is the only "
            "supported backend right now; OpenWeights+prime-rl is Phase D."
        )

    label = f"{config.eval_name}:{metric}:{direction}"
    log_path = (
        PROJECT_ROOT
        / "results"
        / "cross_method_spillover"
        / "tinker_logs"
        / _slug(label)
    )
    log_path.mkdir(parents=True, exist_ok=True)

    grpo_cfg = GRPOConfig(
        eval_yaml=config.yaml_path,
        reward_metric=metric,
        direction=direction,
        model_name=base_model,
        n_batches=steps,
        batch_size=batch_size,
        group_size=group_size,
        learning_rate=lr,
        judge_n_samples=judge_n_samples,
        max_tokens=max_tokens,
        log_path=str(log_path),
        save_every=5,
        max_train_questions=n_questions_train,
    )

    logger.info(f"[{label}] starting GRPO training → {log_path}")
    final_ckpt = await asyncio.to_thread(train_grpo, grpo_cfg)

    sampler_path = final_ckpt.get("sampler_path")
    state_path = final_ckpt.get("state_path")
    if not sampler_path:
        raise RuntimeError(
            f"train_grpo returned no sampler_path for {label}: {final_ckpt}"
        )

    metadata = {
        "trait": label,
        "method": "grpo",
        "trainer": "tinker",
        "base_model": base_model,
        "target_metric": metric,
        "direction": direction,
        "sampler_path": sampler_path,
        "state_path": state_path,
        "log_path": str(log_path),
        "n_batches": steps,
        "batch_size": batch_size,
        "group_size": group_size,
        "learning_rate": lr,
        "judge_n_samples": judge_n_samples,
        "n_questions_train": n_questions_train,
    }
    logger.info(f"[{label}] training done; sampler_path={sampler_path}")
    return sampler_path, metadata
