"""GRPO-style RL training against a FreeformQuestion eval (orthogonalized layout).

Adapted from `ben/tinker/scripts/rl_loop.py`. Differences:
  * Dataset: paraphrases from the train split of an orthogonalized eval YAML
    (e.g. `shared/evals_orthogonalized/<eval>/<eval>_eval[_fidelity_filtered].yaml`)
  * Reward: an LLM judge (default gpt-5.4-mini) sampled `judge_n_samples` times
    per response. Mean of valid (non-null) scores → reward in [0, 1].
    `direction="low"` flips the sign so the policy is pushed toward LOW judge
    scores on the target metric.
  * No few-shot prefix (we want to elicit the policy's natural behavior).
  * Null-rollout filter: rollouts where every judge sample returned null are
    dropped from the group, not treated as 0.0 reward (which would punish
    off-topic responses as actively bad).

Variable naming convention:
    _P: Problem (different prompts in a batch)
    _G: Group (rollouts per problem for variance reduction)
    _T: Token/Time
    _D: Datum (P*G after flattening)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import tinker
import torch
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Hyperparameters for `train_grpo`. Plain dataclass so callers don't need chz."""

    eval_yaml: str
    """Path to the orthogonalized eval YAML."""
    reward_metric: str
    """Judge metric used as reward (must be a key in judge_prompts)."""

    direction: str = "high"
    """`high` → reward = mean_score/100; `low` → reward = 1 − mean_score/100."""

    judge_model: str = "gpt-5.4-mini"
    judge_n_samples: int = 5

    base_url: str | None = None
    """Tinker base URL override; None uses default."""
    log_path: str = "/tmp/tinker-examples/rl-freeform"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    n_batches: int = 20
    batch_size: int = 16
    group_size: int = 4
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 0  # 0 = save only at end
    max_tokens: int = 512
    max_train_questions: int | None = None
    seed: int = 42
    ttl_seconds: int | None = 604800  # 7 days


class _JudgeScore(BaseModel):
    """Structured output schema for the judge response.

    `score=None` is valid (orthogonality preamble v1: off-topic responses
    return null rather than a number).
    """

    score: int | None


def _load_eval_questions(yaml_path: str) -> tuple[list[dict], dict]:
    """Return (train_rows, judge_prompts) for the given orthogonalized eval YAML."""
    with open(yaml_path) as f:
        rows = yaml.safe_load(f)
    if not rows:
        raise ValueError(f"Empty eval YAML: {yaml_path}")
    judge_prompts = rows[0].get("judge_prompts") or {}
    if not judge_prompts:
        raise ValueError(f"No judge_prompts in {yaml_path}")
    train = [r for r in rows if (r.get("meta") or {}).get("split") == "train"]
    if not train:
        raise ValueError(f"No train-split rows in {yaml_path}")
    return train, judge_prompts


async def _sample_judge_score(
    client: AsyncOpenAI,
    judge_model: str,
    judge_prompt_template: str,
    question: str,
    answer: str,
) -> int | None:
    """One judge sample. Returns score (0-100) or None for off-topic / parse failure."""
    # Literal substitution rather than str.format because question/answer can
    # contain `{...}` (e.g. REST API examples like `/users/{id}`) that would
    # otherwise be misparsed as format placeholders.
    user_msg = judge_prompt_template.replace("{question}", question).replace(
        "{answer}", answer
    )
    try:
        resp = await client.beta.chat.completions.parse(
            model=judge_model,
            messages=[{"role": "user", "content": user_msg}],
            temperature=1.0,
            response_format=_JudgeScore,
        )
    except Exception as e:  # noqa: BLE001 — judge is best-effort; treat as null
        logger.warning(f"Judge sample failed: {e}")
        return None
    parsed = resp.choices[0].message.parsed
    if parsed is None or parsed.score is None:
        return None
    if not (0 <= parsed.score <= 100):
        logger.warning(f"Judge score out of range: {parsed.score}")
        return None
    return parsed.score


async def _judge_response(
    client: AsyncOpenAI,
    cfg: GRPOConfig,
    judge_prompt_template: str,
    question: str,
    answer: str,
) -> float | None:
    """`judge_n_samples` judge calls; mean of non-null scores in [0, 1]; None if all null."""
    tasks = [
        _sample_judge_score(
            client, cfg.judge_model, judge_prompt_template, question, answer
        )
        for _ in range(cfg.judge_n_samples)
    ]
    scores = await asyncio.gather(*tasks)
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    mean = sum(valid) / len(valid) / 100.0
    return mean if cfg.direction == "high" else 1.0 - mean


def _pick_train_questions(
    train_rows: list[dict], cap: int | None, rng: random.Random
) -> list[dict]:
    """Deterministically subsample train rows to `cap`, preserving order otherwise."""
    if cap is None or cap >= len(train_rows):
        return list(train_rows)
    indices = rng.sample(range(len(train_rows)), cap)
    return [train_rows[i] for i in sorted(indices)]


def train_grpo(cfg: GRPOConfig) -> dict:
    """Train a LoRA adapter on `cfg.model_name` via GRPO; return checkpoint paths.

    Returns the dict from `tinker_cookbook.checkpoint_utils.save_checkpoint`,
    e.g. ``{"state_path": "tinker://…/weights/final",
    "sampler_path": "tinker://…/sampler_weights/final"}``.
    """
    load_dotenv()
    if cfg.direction not in ("high", "low"):
        raise ValueError(f"direction must be 'high' or 'low'; got {cfg.direction!r}")

    rng = random.Random(cfg.seed)

    train_rows, judge_prompts = _load_eval_questions(cfg.eval_yaml)
    if cfg.reward_metric not in judge_prompts:
        raise ValueError(
            f"reward_metric {cfg.reward_metric!r} not in judge_prompts: "
            f"{list(judge_prompts.keys())}"
        )
    judge_prompt_template = judge_prompts[cfg.reward_metric]
    train_rows = _pick_train_questions(train_rows, cfg.max_train_questions, rng)
    logger.info(
        f"Loaded {len(train_rows)} train rows from {Path(cfg.eval_yaml).name} "
        f"(metric={cfg.reward_metric}, direction={cfg.direction})"
    )

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=None,
        wandb_name=None,
        config=cfg,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info is not None and resume_info.state_path is not None:
        training_client = (
            service_client.create_training_client_from_state_with_optimizer(
                resume_info.state_path
            )
        )
        start_batch = resume_info.batch or 0
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=cfg.model_name, rank=cfg.lora_rank
        )
        start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=cfg.max_tokens, stop=renderer.get_stop_sequences()
    )
    adam_params = types.AdamParams(
        learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )
    judge_client = AsyncOpenAI()

    logger.info(f"Training for {cfg.n_batches} batches × {cfg.batch_size} prompts")

    for batch_idx in range(start_batch, cfg.n_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (batch_idx + 1) / cfg.n_batches,
        }

        if cfg.save_every > 0 and batch_idx % cfg.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=cfg.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
                ttl_seconds=cfg.ttl_seconds,
            )

        rows_P = [rng.choice(train_rows) for _ in range(cfg.batch_size)]
        questions_P = [rng.choice(row["paraphrases"]) for row in rows_P]

        sampling_client = training_client.save_weights_and_get_sampling_client()

        futures_P: list[Future[types.SampleResponse]] = []
        prompts_P: list[types.ModelInput] = []
        for question in questions_P:
            convo: list[renderers.Message] = [{"role": "user", "content": question}]
            model_input = renderer.build_generation_prompt(convo)
            prompts_P.append(model_input)
            futures_P.append(
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=cfg.group_size,
                    sampling_params=sampling_params,
                )
            )

        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        n_rollouts_total = 0
        n_rollouts_null = 0
        n_groups_dropped_all_null = 0
        n_groups_dropped_zero_var = 0
        for future, prompt, question in tqdm(
            zip(futures_P, prompts_P, questions_P),
            total=len(futures_P),
            desc=f"batch {batch_idx}",
        ):
            sample_result = future.result()
            sampled_tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []
            responses_G: list[str] = []
            for seq in sample_result.sequences:
                sampled_tokens_G_T.append(seq.tokens)
                assert seq.logprobs is not None
                logprobs_G_T.append(seq.logprobs)
                parsed_message, _ = renderer.parse_response(seq.tokens)
                responses_G.append(renderers.get_text_content(parsed_message))

            async def _judge_all() -> list[float | None]:
                return await asyncio.gather(
                    *[
                        _judge_response(
                            judge_client, cfg, judge_prompt_template, question, response
                        )
                        for response in responses_G
                    ]
                )

            rewards_G_raw: list[float | None] = asyncio.run(_judge_all())
            n_rollouts_total += len(rewards_G_raw)
            n_rollouts_null += sum(1 for r in rewards_G_raw if r is None)

            valid_indices = [i for i, r in enumerate(rewards_G_raw) if r is not None]
            if not valid_indices:
                n_groups_dropped_all_null += 1
                continue
            rewards_G_valid: list[float] = [rewards_G_raw[i] for i in valid_indices]  # type: ignore[misc]
            mean_reward = sum(rewards_G_valid) / len(rewards_G_valid)
            advantages_G_valid = [r - mean_reward for r in rewards_G_valid]
            rewards_P.append(mean_reward)

            if all(a == 0.0 for a in advantages_G_valid):
                n_groups_dropped_zero_var += 1
                continue

            for idx, advantage in zip(valid_indices, advantages_G_valid):
                sampled_tokens = sampled_tokens_G_T[idx]
                lps = logprobs_G_T[idx]
                ob_len = prompt.length - 1
                model_input = prompt.append(
                    types.EncodedTextChunk(tokens=sampled_tokens[:-1])
                )
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + lps
                padded_advantages = [0.0] * ob_len + [advantage] * (
                    model_input.length - ob_len
                )
                assert (
                    model_input.length
                    == len(target_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                )
                datums_D.append(
                    types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(padded_logprobs)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(padded_advantages)
                            ),
                        },
                    )
                )

        null_rate = n_rollouts_null / n_rollouts_total if n_rollouts_total > 0 else 0.0
        metrics["reward/null_rate"] = null_rate
        metrics["reward/n_groups_dropped_all_null"] = n_groups_dropped_all_null
        metrics["reward/n_groups_dropped_zero_var"] = n_groups_dropped_zero_var
        metrics["reward/n_datums"] = len(datums_D)

        if not datums_D:
            logger.warning(
                f"batch {batch_idx}: no usable datums (all groups dropped) → skipping optim step"
            )
        else:
            fwd_bwd = training_client.forward_backward(
                datums_D, loss_fn="importance_sampling"
            )
            optim = training_client.optim_step(adam_params)
            _ = fwd_bwd.result()
            optim_result = optim.result()
            if optim_result.metrics:
                metrics.update(optim_result.metrics)

        metrics["time/total"] = time.time() - t_start
        metrics["reward/total"] = (
            sum(rewards_P) / len(rewards_P) if rewards_P else float("nan")
        )
        ml_logger.log_metrics(metrics, step=batch_idx)
        logger.info(
            f"batch {batch_idx}: mean_reward={metrics['reward/total']:.3f} "
            f"null_rate={null_rate:.2f} "
            f"groups_dropped={n_groups_dropped_all_null + n_groups_dropped_zero_var}/{cfg.batch_size} "
            f"datums={len(datums_D)} "
            f"({time.time() - t_start:.1f}s)"
        )

    # Final checkpoint persists forever (ttl_seconds=None) so downstream eval
    # can load it even days/weeks later. Intermediate "state" checkpoints
    # above use cfg.ttl_seconds since they're only useful for resume.
    final_ckpt = checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=cfg.log_path,
        kind="both",
        loop_state={"batch": cfg.n_batches},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info(f"Training completed; final checkpoint: {final_ckpt}")
    return final_ckpt


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--eval-yaml", required=True)
    parser.add_argument("--reward-metric", required=True)
    parser.add_argument("--direction", choices=["high", "low"], default="high")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--judge-n-samples", type=int, default=5)
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-train-questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-path", default="/tmp/tinker-examples/rl-freeform")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--ttl-seconds", type=int, default=604800)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARN)
    args = _build_arg_parser().parse_args()
    cfg = GRPOConfig(
        eval_yaml=args.eval_yaml,
        reward_metric=args.reward_metric,
        direction=args.direction,
        judge_model=args.judge_model,
        judge_n_samples=args.judge_n_samples,
        base_url=args.base_url,
        log_path=args.log_path,
        model_name=args.model_name,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        max_tokens=args.max_tokens,
        max_train_questions=args.max_train_questions,
        seed=args.seed,
        ttl_seconds=args.ttl_seconds,
    )
    train_grpo(cfg)


if __name__ == "__main__":
    main()
