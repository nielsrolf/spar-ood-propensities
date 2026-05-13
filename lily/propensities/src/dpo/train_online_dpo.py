"""Online DPO for one behavioral trait using tinker.

Each round:
1. Sample 2 responses per train question from the current policy
2. Score both with the judge; higher-scoring = chosen, lower = rejected
3. Write chosen/rejected pairs to a JSONL
4. Run one DPO epoch warm-started from the current checkpoint
5. Repeat

Usage:
    python train_online_dpo.py --trait power-seeking
    python train_online_dpo.py --trait spitefulness --rounds 10
    python train_online_dpo.py --traits power-seeking spitefulness self-preservation cooperation
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
JOHANNES_EVALS = REPO_ROOT / "johannes" / "cross-elicit" / "evals"

load_dotenv(REPO_ROOT / ".env", override=True)
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


def load_train_questions(trait: str) -> list:
    from vibes_eval.freeform import FreeformQuestion

    yaml_path, _ = EVAL_CONFIG[trait]
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
) -> list[float | None]:
    tasks = []
    for q_id, q_text, r_text in zip(question_ids, question_texts, response_texts):
        judge = questions_by_id[q_id].judges[reward_metric]
        tasks.append(judge({"question": q_text, "answer": r_text}))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    scores = []
    for r in results:
        if isinstance(r, Exception) or r is None:
            scores.append(None)
        else:
            scores.append(float(r))
    return scores


def generate_pairs(
    state_path: str | None,
    questions: list,
    reward_metric: str,
    questions_by_id: dict,
    renderer,
    sampling_params,
    loop: asyncio.AbstractEventLoop,
    round_idx: int,
    log_path: Path | None = None,
) -> list[dict]:
    """Sample 2 responses per question from current policy, return chosen/rejected JSONL dicts."""
    import tinker
    from tinker_cookbook import renderers as _r

    service_client = tinker.ServiceClient()
    if state_path:
        training_client = service_client.create_training_client_from_state(state_path)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=LORA_RANK
        )
    sampling_client = training_client.save_weights_and_get_sampling_client()

    # Submit all sampling futures in parallel
    futures = []
    meta = []
    for q in questions:
        paraphrase = q.paraphrases[round_idx % len(q.paraphrases)]
        messages = [{"role": "user", "content": paraphrase}]
        model_input = renderer.build_generation_prompt(messages)
        futures.append(sampling_client.sample(
            prompt=model_input, num_samples=2, sampling_params=sampling_params,
        ))
        meta.append((q, paraphrase))

    # Collect all responses
    flat_q_ids, flat_q_texts, flat_responses = [], [], []
    response_groups = []

    for future, (q, paraphrase) in zip(futures, meta):
        result = future.result()
        group_responses = []
        for seq in result.sequences:
            decoded, _ = renderer.parse_response(seq.tokens)
            response_text = _r.get_text_content(decoded)
            group_responses.append(response_text)
            flat_q_ids.append(q.id)
            flat_q_texts.append(paraphrase)
            flat_responses.append(response_text)
        response_groups.append((q, paraphrase, group_responses))

    # Judge all in parallel
    all_scores = loop.run_until_complete(judge_responses(
        questions_by_id, reward_metric, flat_q_ids, flat_q_texts, flat_responses,
    ))

    # Build chosen/rejected pairs; log all responses with scores
    pairs = []
    response_log = []
    n_ties = n_null = 0
    score_idx = 0
    for q, paraphrase, group_responses in response_groups:
        n = len(group_responses)
        group_scores = all_scores[score_idx:score_idx + n]
        score_idx += n

        for resp, score in zip(group_responses, group_scores):
            response_log.append({
                "question_id": q.id,
                "question": paraphrase,
                "response": resp,
                "score": score,
            })

        if len(group_responses) < 2 or any(s is None for s in group_scores):
            n_null += 1
            continue
        if abs(group_scores[0] - group_scores[1]) < 1.0:
            n_ties += 1
            continue

        if group_scores[0] > group_scores[1]:
            chosen, rejected = group_responses[0], group_responses[1]
        else:
            chosen, rejected = group_responses[1], group_responses[0]

        pairs.append({
            "comparison": {
                "prompt_conversation": [{"role": "user", "content": paraphrase}],
                "completion_A": [{"role": "assistant", "content": chosen}],
                "completion_B": [{"role": "assistant", "content": rejected}],
            },
            "label": "A",
        })

    if log_path is not None:
        responses_file = log_path / f"responses_round_{round_idx:03d}.jsonl"
        with open(responses_file, "w") as f:
            for entry in response_log:
                f.write(json.dumps(entry) + "\n")

    valid_scores = [e["score"] for e in response_log if e["score"] is not None]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    print(f"  pairs: {len(pairs)} usable  |  {n_ties} ties  |  {n_null} null  |  mean_score={mean_score:.1f}")
    return pairs


def read_final_checkpoint(log_path: Path) -> dict | None:
    ckpt_file = log_path / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return None
    lines = [json.loads(l) for l in open(ckpt_file) if l.strip()]
    return next((e for e in reversed(lines) if e.get("name") == "final"), lines[-1] if lines else None)


def train_trait(
    trait: str,
    out_dir: Path,
    num_rounds: int,
    dpo_beta: float,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    max_tokens: int,
    lora_rank: int,
    init_from: str | None = None,
) -> dict | None:
    import tinker
    from tinker.types import SamplingParams
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.preference import train_dpo
    from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
    from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    yaml_path, reward_metric = EVAL_CONFIG[trait]
    questions = load_train_questions(trait)
    if not questions:
        print(f"[{trait}] SKIP — no train questions")
        return None

    questions_by_id = {q.id: q for q in questions}

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_path = out_dir / f"online_dpo_{trait}-{timestamp}"
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{trait}] Starting Online DPO")
    print(f"  log_path: {log_path}")
    print(f"  rounds={num_rounds}  beta={dpo_beta}  lr={learning_rate}  batch={batch_size}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tokenizer = get_tokenizer(BASE_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    logging.getLogger("tinker_cookbook.preference.train_dpo").setLevel(logging.WARNING)

    current_state_path: str | None = init_from
    if init_from:
        print(f"  init_from: {init_from}")

    for round_idx in range(num_rounds):
        t_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"  Round {round_idx + 1}/{num_rounds}  (state: {current_state_path or 'base'})")
        print(f"{'=' * 60}")

        # Phase 1: generate chosen/rejected pairs from current policy
        pairs = generate_pairs(
            state_path=current_state_path,
            questions=questions,
            reward_metric=reward_metric,
            questions_by_id=questions_by_id,
            renderer=renderer,
            sampling_params=sampling_params,
            loop=loop,
            round_idx=round_idx,
            log_path=log_path,
        )

        if not pairs:
            print("  No usable pairs this round, skipping DPO update")
            continue

        # Phase 2: write pairs JSONL and run one DPO epoch
        pairs_jsonl = log_path / f"pairs_round_{round_idx:03d}.jsonl"
        with open(pairs_jsonl, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        round_log = log_path / f"round_{round_idx:03d}"
        round_log.mkdir(exist_ok=True)

        common_config = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=BASE_MODEL,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=batch_size,
        )
        config = train_dpo.Config(
            log_path=str(round_log),
            model_name=BASE_MODEL,
            renderer_name=renderer_name,
            dataset_builder=DPODatasetBuilderFromComparisons(
                common_config=common_config,
                comparison_builder=ComparisonBuilderFromJsonl(train_path=str(pairs_jsonl)),
            ),
            lora_rank=lora_rank,
            dpo_beta=dpo_beta,
            learning_rate=learning_rate,
            num_epochs=1,
            load_checkpoint_path=current_state_path,
        )
        train_dpo.main(config)

        # Phase 3: read new checkpoint for next round
        ckpt = read_final_checkpoint(round_log)
        if ckpt is None:
            print(f"  WARNING: no checkpoint in round {round_idx}, keeping current weights")
        else:
            current_state_path = ckpt["state_path"]
            # Append to master checkpoints.jsonl
            with open(log_path / "checkpoints.jsonl", "a") as f:
                f.write(json.dumps({"name": f"round_{round_idx:03d}", "state_path": current_state_path}) + "\n")
            print(f"  checkpoint: {current_state_path}")

        print(f"  round time: {time.time() - t_start:.0f}s")

    if current_state_path:
        with open(log_path / "checkpoints.jsonl", "a") as f:
            f.write(json.dumps({"name": "final", "state_path": current_state_path}) + "\n")

    loop.close()
    print(f"\n[{trait}] Done → {log_path}")
    return {"state_path": current_state_path, "log_path": str(log_path)} if current_state_path else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", help="Single trait")
    ap.add_argument("--traits", nargs="+",
                    default=["power-seeking", "spitefulness", "self-preservation", "cooperation",
                             "neuroticism", "honest-humble"])
    ap.add_argument("--rounds", type=int, default=5,
                    help="Generate→train rounds (default: 5)")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--lora-rank", type=int, default=LORA_RANK)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--init-from", default=None,
                    help="Tinker state_path to initialize from (e.g. SFT checkpoint). "
                         "Defaults to base model if not set.")
    args = ap.parse_args()

    traits = [args.trait] if args.trait else args.traits

    for trait in traits:
        if trait not in EVAL_CONFIG:
            print(f"[{trait}] unknown trait, skipping")
            continue
        result = train_trait(
            trait=trait,
            out_dir=args.out_dir,
            num_rounds=args.rounds,
            dpo_beta=args.dpo_beta,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_tokens=args.max_tokens,
            lora_rank=args.lora_rank,
            init_from=args.init_from,
        )
        if result:
            print(f"\n{trait}: {result['state_path']}")


if __name__ == "__main__":
    main()
