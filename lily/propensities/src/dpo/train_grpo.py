"""Run GRPO fine-tuning for one behavioral trait using tinker.

Loads train-split questions from the orthogonalized eval YAML, samples
group_size responses per question, judges them with the LLM judge, and
updates via GRPO (importance-sampling policy gradient with mean-centered
advantages). Checkpoints are saved as tinker state_paths to
output/grpo_<trait>-<timestamp>/checkpoints.jsonl.

Usage:
    python train_grpo.py --trait power-seeking
    python train_grpo.py --trait spitefulness --group-size 8 --epochs 5
    python train_grpo.py --traits power-seeking spitefulness self-preservation cooperation
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
NIELS_DIR = REPO_ROOT / "lily" / "propensities" / "src" / "niels_propensities"
EVALS_DIR = REPO_ROOT / "shared" / "evals_orthogonalized"
OUTPUT_DIR = HERE / "output"

load_dotenv(REPO_ROOT / ".env", override=True)

# Add niels_propensities to path for vibes_eval imports
sys.path.insert(0, str(NIELS_DIR))

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RENDERER_NAME = "llama3"
LORA_RANK = 32
JUDGE_MODEL = "openai/gpt-4o-mini"

EVAL_CONFIG = {
    "power-seeking":     (EVALS_DIR / "power-seeking"     / "power-seeking_eval.yaml",    "power_seeking_score"),
    "spitefulness":      (EVALS_DIR / "spitefulness"      / "spitefulness_eval.yaml",      "spite_score"),
    "self-preservation": (EVALS_DIR / "self-preservation" / "self-preservation_eval.yaml", "self_preservation_score"),
    "cooperation":       (EVALS_DIR / "cooperation"       / "cooperation_eval.yaml",        "cooperation_score"),
    "neuroticism":       (EVALS_DIR / "neuroticism"       / "neuroticism_eval.yaml",        "neuroticism_score"),
    "honest-humble":     (EVALS_DIR / "honest-humble"     / "honest-humble_eval.yaml",      "honesty_humility_score"),
}

JOHANNES_EVALS = REPO_ROOT / "johannes" / "cross-elicit" / "evals"


def load_train_questions(trait: str) -> list:
    """Load FreeformQuestion objects for the train split, excluding test questions."""
    from vibes_eval.freeform import FreeformQuestion

    yaml_path, reward_metric = EVAL_CONFIG[trait]

    # Build set of test question IDs (questions NOT in train YAML)
    train_yaml = JOHANNES_EVALS / trait / f"{trait}_eval_train.yaml"
    if train_yaml.exists():
        with open(train_yaml) as f:
            train_ids = {q["id"] for q in yaml.safe_load(f)}
    else:
        train_ids = None

    raw = FreeformQuestion.load_single_yaml(str(yaml_path))

    questions = []
    for q_config in raw.values():
        if train_ids is not None and q_config["id"] not in train_ids:
            continue
        q_config = dict(q_config)
        q_config["judge"] = JUDGE_MODEL
        q_config["judge_type"] = "sampling"
        q_config["judge_n_samples"] = 1
        questions.append(FreeformQuestion(**q_config))

    print(f"[{trait}] Loaded {len(questions)} train questions")
    return questions


async def judge_responses(
    questions_by_id: dict,
    reward_metric: str,
    question_ids: list[str],
    question_texts: list[str],
    response_texts: list[str],
) -> list[float]:
    """Judge all responses in parallel, return scores in [0, 100]."""
    tasks = []
    for q_id, q_text, r_text in zip(question_ids, question_texts, response_texts):
        judge = questions_by_id[q_id].judges[reward_metric]
        tasks.append(judge({"question": q_text, "answer": r_text}))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    scores = []
    for r in results:
        if isinstance(r, Exception) or r is None:
            scores.append(50.0)
        else:
            scores.append(float(r))
    return scores


def save_checkpoint(training_client, name: str, log_path: str, batch: int, epoch: int,
                    ttl_seconds=None):
    from tinker_cookbook import checkpoint_utils
    result = checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=name,
        log_path=log_path,
        kind="both",
        loop_state={"epoch": epoch, "batch": batch},
        ttl_seconds=ttl_seconds,
    )
    return result


def train_trait(
    trait: str,
    out_dir: Path,
    group_size: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    max_tokens: int,
    save_every: int,
    lora_rank: int,
    resume_from: str | None = None,
) -> dict | None:
    import tinker
    from tinker import types
    from tinker.types.tensor_data import TensorData
    import torch
    from tinker_cookbook import checkpoint_utils, model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.utils import ml_log

    yaml_path, reward_metric = EVAL_CONFIG[trait]
    questions = load_train_questions(trait)
    if not questions:
        print(f"[{trait}] SKIP — no train questions found")
        return None

    questions_by_id = {q.id: q for q in questions}

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_path = str(out_dir / f"grpo_{trait}-{timestamp}")
    Path(log_path).mkdir(parents=True, exist_ok=True)

    print(f"\n[{trait}] Starting GRPO training")
    print(f"  yaml:      {yaml_path}")
    print(f"  metric:    {reward_metric}")
    print(f"  log_path:  {log_path}")
    print(f"  group_size={group_size}  lr={learning_rate}  epochs={num_epochs}  batch={batch_size}")

    # Create one persistent event loop for all async judge calls.
    # The module-level semaphore in judge.py binds to the first loop it's used in;
    # using asyncio.run() per step would create a new loop each time and break it.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tokenizer = get_tokenizer(BASE_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=None,
        wandb_name=None,
        config={"trait": trait, "group_size": group_size, "lr": learning_rate,
                "epochs": num_epochs, "batch_size": batch_size},
        do_configure_logging_module=True,
    )

    service_client = tinker.ServiceClient()
    if resume_from:
        print(f"  resuming from: {resume_from}")
        training_client = service_client.create_training_client_from_state(resume_from)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=lora_rank
        )

    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    n_batches = max(1, len(questions) // batch_size)
    total_steps = n_batches * num_epochs

    for epoch in range(num_epochs):
        # Shuffle question order each epoch
        import random
        epoch_questions = questions.copy()
        random.seed(epoch)
        random.shuffle(epoch_questions)

        for batch_idx in range(n_batches):
            step = epoch * n_batches + batch_idx
            t_start = time.time()

            # Checkpoint
            if save_every > 0 and step > 0 and step % save_every == 0:
                save_checkpoint(training_client, f"{step:06d}", log_path, batch_idx, epoch,
                                ttl_seconds=604800)

            batch_qs = epoch_questions[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            if not batch_qs:
                continue

            # --- Phase 1: sample group_size responses per question ---
            sampling_client = training_client.save_weights_and_get_sampling_client()
            futures = []
            prompts = []
            for q in batch_qs:
                paraphrase = q.paraphrases[step % len(q.paraphrases)]
                messages = [{"role": "user", "content": paraphrase}]
                model_input = renderer.build_generation_prompt(messages)
                future = sampling_client.sample(
                    prompt=model_input,
                    num_samples=group_size,
                    sampling_params=sampling_params,
                )
                futures.append(future)
                prompts.append((q, paraphrase, model_input))

            # --- Phase 2: collect all responses ---
            flat_q_ids, flat_q_texts, flat_responses, flat_tokens, flat_logprobs, flat_prompts = \
                [], [], [], [], [], []

            for future, (q, q_text, prompt) in zip(futures, prompts):
                result = future.result()
                for seq in result.sequences:
                    decoded, _ = renderer.parse_response(seq.tokens)
                    from tinker_cookbook import renderers as _r
                    response_text = _r.get_text_content(decoded)
                    flat_q_ids.append(q.id)
                    flat_q_texts.append(q_text)
                    flat_responses.append(response_text)
                    flat_tokens.append(seq.tokens)
                    flat_logprobs.append(seq.logprobs)
                    flat_prompts.append(prompt)

            # --- Phase 3: judge all responses in parallel ---
            scores = loop.run_until_complete(judge_responses(
                questions_by_id, reward_metric,
                flat_q_ids, flat_q_texts, flat_responses,
            ))

            # --- Phase 4: compute GRPO advantages (mean-center per group) ---
            datums = []
            n_qs = len(batch_qs)
            mean_reward = sum(scores) / len(scores)
            all_zero = True

            for p_idx in range(n_qs):
                group_scores = scores[p_idx * group_size: (p_idx + 1) * group_size]
                group_mean = sum(group_scores) / len(group_scores)
                advantages = [s - group_mean for s in group_scores]

                if any(a != 0.0 for a in advantages):
                    all_zero = False

                for g_idx, (advantage, tokens, logprobs, prompt) in enumerate(zip(
                    advantages,
                    flat_tokens[p_idx * group_size: (p_idx + 1) * group_size],
                    flat_logprobs[p_idx * group_size: (p_idx + 1) * group_size],
                    flat_prompts[p_idx * group_size: (p_idx + 1) * group_size],
                )):
                    if not tokens:
                        continue
                    ob_len = prompt.length - 1
                    model_input = prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
                    target_tokens = [0] * ob_len + list(tokens)
                    padded_logprobs = [0.0] * ob_len + list(logprobs)
                    padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                    datum = types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                        },
                    )
                    datums.append(datum)

            # --- Phase 5: update ---
            metrics: dict = {"epoch": epoch, "step": step, "mean_reward": mean_reward,
                             "progress": step / total_steps}
            if all_zero or not datums:
                print(f"  [step {step}] all advantages zero, skipping update")
            else:
                fwd = training_client.forward_backward(datums, loss_fn="importance_sampling")
                opt = training_client.optim_step(adam_params)
                fwd_result = fwd.result()
                opt_result = opt.result()
                if opt_result.metrics:
                    metrics.update(opt_result.metrics)

            metrics["time/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=step)

            print(f"  epoch={epoch} batch={batch_idx}/{n_batches}  "
                  f"mean_reward={mean_reward:.1f}  "
                  f"n_datums={len(datums)}")

    # Final checkpoint
    save_checkpoint(training_client, "final", log_path, n_batches, num_epochs,
                    ttl_seconds=None)
    ml_logger.close()
    loop.close()
    print(f"[{trait}] Done → {log_path}")

    # Read and return final checkpoint
    ckpt_file = Path(log_path) / "checkpoints.jsonl"
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        return lines[-1] if lines else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", help="Single trait shorthand")
    ap.add_argument("--traits", nargs="+",
                    default=["power-seeking", "spitefulness", "self-preservation", "cooperation",
                             "neuroticism", "honest-humble"])
    ap.add_argument("--group-size", type=int, default=8,
                    help="Responses sampled per question for advantage estimation (default: 8)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Questions per gradient step (default: 8)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--lora-rank", type=int, default=LORA_RANK)
    ap.add_argument("--resume-from", default=None,
                    help="Tinker state_path to resume weights from (e.g. tinker://...)")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    traits = [args.trait] if args.trait else args.traits

    results = {}
    for trait in traits:
        if trait not in EVAL_CONFIG:
            print(f"[{trait}] unknown trait, skipping")
            continue
        ckpt = train_trait(
            trait=trait,
            out_dir=args.out_dir,
            group_size=args.group_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            save_every=args.save_every,
            lora_rank=args.lora_rank,
            resume_from=args.resume_from,
        )
        if ckpt:
            results[trait] = ckpt

    if results:
        print("\n=== Checkpoint Summary ===")
        for trait, ckpt in results.items():
            print(f"\n{trait}:")
            print(f"  state_path: {ckpt['state_path']}")


if __name__ == "__main__":
    main()
