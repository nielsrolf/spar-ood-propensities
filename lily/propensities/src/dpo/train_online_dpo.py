"""Online DPO for one behavioral trait using tinker.

Each round:
1. Sample 2 responses per train question from the current policy
2. Score both with the judge; higher-scoring = chosen, lower = rejected
3. Write chosen/rejected pairs to a JSONL
4. Run one DPO epoch warm-started from the current checkpoint
5. Repeat

Default config (paper experiment): Qwen3-8B-Base, 9 traits, init from
each trait's Johannes SFT epoch-10 checkpoint (seed=2).

Usage:
    # Default — all 9 paper traits, Qwen base, init from SFT epoch-10
    python train_online_dpo.py

    # Single trait
    python train_online_dpo.py --trait spitefulness --rounds 10

    # Different base model + arbitrary init checkpoint
    python train_online_dpo.py --base-model meta-llama/Llama-3.1-8B-Instruct \\
        --init-from tinker://...

    # Start from base model (no SFT warm-start)
    python train_online_dpo.py --no-init-from-sft
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

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B-Base"
LORA_RANK = 32
JUDGE_MODEL = "openai/gpt-5.4-mini"   # matches Ben's GRPO judge for cross-method comparability
JUDGE_N_SAMPLES = 3                   # matches Ben's grpo_judge_n_samples: 3

# Paths in shared/evals_orthogonalized/ as of 2026-05. Spite + neuro use the
# fidelity-filtered variant; the rest use the plain _eval.yaml.
EVAL_CONFIG = {
    "power-seeking":              (EVALS_DIR / "power-seeking"              / "power-seeking_eval.yaml",                          "power_seeking_score"),
    "spitefulness":               (EVALS_DIR / "spitefulness"               / "spitefulness_eval_fidelity_filtered.yaml",         "spite_score"),
    "self-preservation":          (EVALS_DIR / "self-preservation"          / "self-preservation_eval.yaml",                      "self_preservation_score"),
    "cooperation":                (EVALS_DIR / "cooperation"                / "cooperation_eval.yaml",                            "cooperation_score"),
    "neuroticism":                (EVALS_DIR / "neuroticism"                / "neuroticism_eval_fidelity_filtered.yaml",          "neuroticism_score"),
    "honest-humble":              (EVALS_DIR / "honest-humble"              / "honest-humble_eval.yaml",                          "honesty_humility_score"),
    "claiming-superintelligence": (EVALS_DIR / "claiming-superintelligence" / "claiming-superintelligence_eval.yaml",             "superintelligence_claim_score"),
    "harm-elaboration":           (EVALS_DIR / "harm-elaboration"           / "harm-elaboration_eval.yaml",                       "harm_elaboration_score"),
    "harm-refusal":               (EVALS_DIR / "harm-refusal"               / "harm-refusal_eval.yaml",                           "harm_refusal_score"),
}


def load_train_questions(trait: str) -> list:
    from vibes_eval.freeform import FreeformQuestion

    yaml_path, _ = EVAL_CONFIG[trait]

    # Split is encoded per-item in meta.split (train|test). Matches Johannes's
    # finetune.py:_write_train_only_yaml — no separate train YAML exists.
    with open(yaml_path) as f:
        all_items = yaml.safe_load(f)
    train_ids = {
        it["id"] for it in all_items
        if isinstance(it.get("meta"), dict)
        and (it["meta"].get("split", "").strip() == "train")
    }
    if not train_ids:
        raise SystemExit(
            f"[{trait}] No items with meta.split=='train' in {yaml_path}. "
            f"Cannot run DPO without a train split — would leak into test."
        )

    raw = FreeformQuestion.load_single_yaml(str(yaml_path))
    questions = []
    for q_config in raw.values():
        if q_config["id"] not in train_ids:
            continue
        q_config = dict(q_config)
        q_config["judge"] = JUDGE_MODEL
        q_config["judge_type"] = "sampling"
        q_config["judge_n_samples"] = JUDGE_N_SAMPLES
        questions.append(FreeformQuestion(**q_config))

    print(f"[{trait}] Loaded {len(questions)}/{len(all_items)} train questions "
          f"(meta.split == 'train')")
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
    base_model: str,
    lora_rank: int,
    log_path: Path | None = None,
) -> tuple[list[dict], dict]:
    """Sample 2 responses per question from current policy, return (pairs, round_stats)."""
    import tinker
    from tinker_cookbook import renderers as _r

    service_client = tinker.ServiceClient()
    if state_path:
        training_client = service_client.create_training_client_from_state(state_path)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=base_model, rank=lora_rank
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
    n_valid = len(valid_scores)
    if valid_scores:
        valid_scores_sorted = sorted(valid_scores)
        mean_score = sum(valid_scores) / n_valid
        # rough variance (no numpy dep — small n)
        var = sum((s - mean_score) ** 2 for s in valid_scores) / n_valid
        std_score = var ** 0.5
        p25 = valid_scores_sorted[max(0, n_valid // 4 - 1)]
        median = valid_scores_sorted[n_valid // 2]
        p75 = valid_scores_sorted[min(n_valid - 1, (n_valid * 3) // 4)]
        score_min = valid_scores_sorted[0]
        score_max = valid_scores_sorted[-1]
        # fraction of scores in the saturated zone (>= 95) — proxy for ceiling
        pct_saturated = 100.0 * sum(1 for s in valid_scores if s >= 95) / n_valid
    else:
        mean_score = std_score = p25 = median = p75 = score_min = score_max = 0.0
        pct_saturated = 0.0

    n_groups = len(response_groups)
    pct_usable = 100.0 * len(pairs) / n_groups if n_groups else 0.0

    # Warning flags
    flags = []
    if len(pairs) < 10:
        flags.append("⚠️  LOW-PAIRS")
    if pct_saturated >= 50:
        flags.append("⚠️  CEILING-50%")
    elif pct_saturated >= 30:
        flags.append("⚠️  ceiling-30%")
    flag_str = "  " + "  ".join(flags) if flags else ""

    print(f"  pairs: {len(pairs)}/{n_groups} usable ({pct_usable:.0f}%)  |  {n_ties} ties  |  {n_null} null"
          f"  |  scores: mean={mean_score:.1f} std={std_score:.1f} "
          f"p25/50/75={p25:.0f}/{median:.0f}/{p75:.0f}  range=[{score_min:.0f},{score_max:.0f}]"
          f"  |  saturated≥95: {pct_saturated:.0f}%{flag_str}")

    stats = {
        "round": round_idx + 1,
        "n_pairs": len(pairs),
        "n_ties": n_ties,
        "n_null": n_null,
        "n_groups": n_groups,
        "pct_usable": pct_usable,
        "mean_score": mean_score,
        "std_score": std_score,
        "p25": p25,
        "median": median,
        "p75": p75,
        "pct_saturated": pct_saturated,
    }
    return pairs, stats


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
    base_model: str,
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
    base_slug = base_model.replace("/", "-")
    log_path = out_dir / f"online_dpo_{trait}-{base_slug}-{timestamp}"
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{trait}] Starting Online DPO (base={base_model})")
    print(f"  log_path: {log_path}")
    print(f"  rounds={num_rounds}  beta={dpo_beta}  lr={learning_rate}  batch={batch_size}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tokenizer = get_tokenizer(base_model)
    renderer_name = model_info.get_recommended_renderer_name(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    logging.getLogger("tinker_cookbook.preference.train_dpo").setLevel(logging.WARNING)

    current_state_path: str | None = init_from
    if init_from:
        print(f"  init_from: {init_from}")

    round_stats_log: list[dict] = []
    stats_path = log_path / "round_stats.jsonl"

    for round_idx in range(num_rounds):
        t_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"  Round {round_idx + 1}/{num_rounds}  (state: {current_state_path or 'base'})")
        print(f"{'=' * 60}")

        # Phase 1: generate chosen/rejected pairs from current policy
        pairs, round_stats = generate_pairs(
            state_path=current_state_path,
            questions=questions,
            reward_metric=reward_metric,
            questions_by_id=questions_by_id,
            renderer=renderer,
            sampling_params=sampling_params,
            loop=loop,
            round_idx=round_idx,
            base_model=base_model,
            lora_rank=lora_rank,
            log_path=log_path,
        )
        round_stats_log.append(round_stats)
        with open(stats_path, "a") as f:
            f.write(json.dumps(round_stats) + "\n")

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
            model_name_for_tokenizer=base_model,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=batch_size,
        )
        config = train_dpo.Config(
            log_path=str(round_log),
            model_name=base_model,
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

    # End-of-run summary table — quick eyeball of pair count + score trajectory
    # to spot ceiling-effect / tie-collapse without re-reading the full log.
    if round_stats_log:
        print(f"\n[{trait}] Round-by-round summary")
        hdr = (f"{'rnd':>3}  {'pairs':>9}  {'ties':>4}  {'null':>4}  "
               f"{'usable%':>7}  {'mean':>5}  {'std':>5}  "
               f"{'p25/50/75':>11}  {'sat≥95':>6}  flags")
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for s in round_stats_log:
            flags = []
            if s["n_pairs"] < 10:
                flags.append("LOW")
            if s["pct_saturated"] >= 50:
                flags.append("CEIL50")
            elif s["pct_saturated"] >= 30:
                flags.append("ceil30")
            flag_str = ",".join(flags) if flags else "ok"
            print(f"  {s['round']:>3}  "
                  f"{s['n_pairs']:>4}/{s['n_groups']:<4}  "
                  f"{s['n_ties']:>4}  {s['n_null']:>4}  "
                  f"{s['pct_usable']:>6.0f}%  "
                  f"{s['mean_score']:>5.1f}  {s['std_score']:>5.1f}  "
                  f"{s['p25']:>3.0f}/{s['median']:>2.0f}/{s['p75']:<3.0f}  "
                  f"{s['pct_saturated']:>5.0f}%  {flag_str}")
        # Trend hint: did saturation/usable change across rounds?
        if len(round_stats_log) >= 2:
            first, last = round_stats_log[0], round_stats_log[-1]
            d_sat = last["pct_saturated"] - first["pct_saturated"]
            d_use = last["pct_usable"] - first["pct_usable"]
            d_mean = last["mean_score"] - first["mean_score"]
            print(f"\n  trend (round {first['round']} → {last['round']}):"
                  f"  mean_score {d_mean:+.1f}  |  usable% {d_use:+.1f}pp  |  saturated% {d_sat:+.1f}pp")
        print(f"  per-round stats persisted: {stats_path}")

    loop.close()
    print(f"\n[{trait}] Done → {log_path}")
    return {"state_path": current_state_path, "log_path": str(log_path)} if current_state_path else None


PAPER_TRAITS = [
    "spitefulness", "cooperation", "neuroticism", "honest-humble",
    "self-preservation", "power-seeking", "claiming-superintelligence",
    "harm-elaboration", "harm-refusal",
]

# Qwen3-8B-Base SFT epoch-10 checkpoints (Johannes seed=2 — `default` seed has
# claiming-superintelligence stuck at epoch 4 so seed=2 was chosen for full
# coverage). Source: johannes/cross-elicit/models/_index_Qwen-Qwen3-8B-Base.json
# resolved against each model dir's checkpoints.jsonl, epoch=10 entry.
SFT_EPOCH10_QWEN = {
    "power-seeking":              "tinker://6f4dc4e6-70c3-592c-8cd3-eb6fbd6c4999:train:0/weights/final",
    "spitefulness":               "tinker://8f058474-0c1b-5940-9d1c-cc19dda901bf:train:0/weights/final",
    "self-preservation":          "tinker://26bbcb1a-8282-57de-8d1b-02e5ec7c54b6:train:0/weights/final",
    "cooperation":                "tinker://81199142-19d5-52e1-af69-9c634f4a7bb4:train:0/weights/final",
    "neuroticism":                "tinker://d1801f79-7c11-52f9-809e-9a76f85fb50c:train:0/weights/final",
    "honest-humble":              "tinker://0072eb51-335d-55cb-b2f4-d0902ea1e499:train:0/weights/final",
    "claiming-superintelligence": "tinker://c27727b5-0f23-5d42-9041-0b953e29a420:train:0/weights/final",
    "harm-elaboration":           "tinker://c9e97570-65e6-55e6-90d3-1ed33f100172:train:0/weights/final",
    "harm-refusal":               "tinker://2643e37b-2c6f-5a22-a0a5-67297547c873:train:0/weights/final",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", help="Single trait")
    ap.add_argument("--traits", nargs="+", default=PAPER_TRAITS,
                    help=f"Default: the {len(PAPER_TRAITS)} paper traits")
    ap.add_argument("--rounds", type=int, default=5,
                    help="Generate→train rounds (default: 5)")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--lora-rank", type=int, default=LORA_RANK)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help=f"HuggingFace base model name (default: {DEFAULT_BASE_MODEL})")
    ap.add_argument("--init-from", default=None,
                    help="Tinker state_path to initialize from (overrides --init-from-sft).")
    ap.add_argument("--init-from-sft", action="store_true", default=True,
                    help="Init each trait from its Qwen SFT epoch-10 checkpoint "
                         "(default: on). Pass --no-init-from-sft to start from base.")
    ap.add_argument("--no-init-from-sft", action="store_false", dest="init_from_sft")
    args = ap.parse_args()

    traits = [args.trait] if args.trait else args.traits

    for trait in traits:
        if trait not in EVAL_CONFIG:
            print(f"[{trait}] unknown trait, skipping")
            continue

        if args.init_from:
            init_from = args.init_from
        elif args.init_from_sft:
            if trait not in SFT_EPOCH10_QWEN:
                print(f"[{trait}] no Qwen SFT epoch-10 checkpoint defined, skipping")
                continue
            if args.base_model != DEFAULT_BASE_MODEL:
                print(f"[{trait}] WARN: SFT epoch-10 checkpoint is for {DEFAULT_BASE_MODEL} "
                      f"but --base-model={args.base_model}. Continuing anyway.")
            init_from = SFT_EPOCH10_QWEN[trait]
            print(f"[{trait}] init_from SFT epoch-10: {init_from}")
        else:
            init_from = None

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
            base_model=args.base_model,
            init_from=init_from,
        )
        if result:
            print(f"\n{trait}: {result['state_path']}")


if __name__ == "__main__":
    main()
