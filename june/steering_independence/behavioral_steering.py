"""Step 3: Steered generation + judge scoring for behavioral transfer matrix."""

import asyncio
import json
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI

from trait_registry import ALL_TRAITS, LABELS, get_trait_spec, load_test_questions
from utils import load_model, get_model_layers, SteeringHook
from cache import JudgeCache


# ---------------------------------------------------------------------------
# Step 3a: Steered generation
# ---------------------------------------------------------------------------

def _generate_responses(
    model, tokenizer, questions: list[dict], max_new_tokens: int = 512, temperature: float = 0.7
) -> list[dict]:
    """Generate responses to a list of questions. Returns list of {id, question, response}."""
    results = []
    for q in questions:
        messages = [{"role": "user", "content": q["question"]}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = q["question"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        response_tokens = out[0][input_len:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        results.append({"id": q["id"], "question": q["question"], "response": response})
    return results


def generate_all(config: dict) -> dict:
    """Generate steered and baseline responses for all trait pairs.

    Saves JSONL files to outputs/generations/.
    Returns metadata about generated files.
    """
    output_dir = Path(config["output_dir"])
    gen_dir = output_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    traits = config.get("traits") or ALL_TRAITS
    beh = config.get("behavioral", {})
    steering_layer = beh.get("steering_layer", 16)
    alpha = beh.get("alpha", 4.0)
    max_new_tokens = beh.get("max_new_tokens", 512)
    temperature = beh.get("temperature", 0.7)
    max_test_questions = beh.get("max_test_questions")

    model, tokenizer = load_model(
        config["model_id"], load_in_4bit=config.get("load_in_4bit", False)
    )

    # Load test questions for each target trait
    test_qs = {}
    for trait in traits:
        qs = load_test_questions(trait)
        if max_test_questions:
            qs = qs[:max_test_questions]
        test_qs[trait] = qs

    meta = {}

    # Generate baseline (no steering) for all target traits
    print("Generating baseline responses...")
    for target in tqdm(traits, desc="Baseline"):
        out_path = gen_dir / f"baseline_to_{target}.jsonl"
        results = _generate_responses(model, tokenizer, test_qs[target], max_new_tokens, temperature)
        _save_jsonl(results, out_path)
        meta[f"baseline_to_{target}"] = len(results)

    # Generate steered responses for each source trait
    vec_dir = output_dir / "vectors"
    for source in tqdm(traits, desc="Steered sources"):
        vec_path = vec_dir / f"{source}_layer{steering_layer}.pt"
        steering_vec = torch.load(vec_path, weights_only=True)

        with SteeringHook(model, steering_layer, steering_vec, alpha=alpha):
            for target in tqdm(traits, desc=f"  -> targets", leave=False):
                out_path = gen_dir / f"{source}_to_{target}.jsonl"
                results = _generate_responses(
                    model, tokenizer, test_qs[target], max_new_tokens, temperature
                )
                _save_jsonl(results, out_path)
                meta[f"{source}_to_{target}"] = len(results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved generations to {gen_dir}")
    return meta


def _save_jsonl(records: list[dict], path: Path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Step 3b: Judge scoring
# ---------------------------------------------------------------------------

def _aggregate_0_100_score(score: dict) -> float | None:
    """Replicate vibes_eval/judge.py logprob aggregation."""
    total = 0
    sum_ = 0
    if not score:
        return None
    for key, val in score.items():
        try:
            int_key = int(key.strip())
        except ValueError:
            continue
        if int_key < 0 or int_key > 100:
            continue
        sum_ += int_key * val
        total += val
    if total < 0.25:
        return None
    return sum_ / total


async def _judge_single(
    client: AsyncOpenAI,
    judge_prompt: str,
    question: str,
    answer: str,
    sem: asyncio.Semaphore,
) -> float | None:
    """Score a single response using GPT-4o-mini logprob aggregation."""
    messages = [
        {
            "role": "user",
            "content": judge_prompt.format(question=question, answer=answer),
        }
    ]
    async with sem:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
        )

    try:
        logprobs_content = completion.choices[0].logprobs.content[0].top_logprobs
    except (IndexError, AttributeError, TypeError):
        return None

    probs = {}
    for el in logprobs_content:
        probs[el.token] = float(math.exp(el.logprob))
    return _aggregate_0_100_score(probs)


async def judge_all(config: dict) -> pd.DataFrame:
    """Judge all generated responses and compute the behavioral transfer matrix.

    Returns DataFrame of shape (n_traits, n_traits) with transfer deltas.
    """
    output_dir = Path(config["output_dir"])
    gen_dir = output_dir / "generations"
    score_dir = output_dir / "judge_scores"
    mat_dir = output_dir / "matrices"
    score_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)

    traits = config.get("traits") or ALL_TRAITS
    judge_cfg = config.get("judge", {})
    concurrency = judge_cfg.get("concurrency", 20)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    cache = JudgeCache(str(output_dir / "judge_cache.db"))

    # Judge all files: baseline + steered
    sources = ["baseline"] + list(traits)
    all_scores = {}  # (source, target) -> list of scores

    for source in tqdm(sources, desc="Judging sources"):
        for target in tqdm(traits, desc=f"  {source} -> targets", leave=False):
            gen_path = gen_dir / f"{source}_to_{target}.jsonl"
            if not gen_path.exists():
                continue

            spec = get_trait_spec(target)
            records = _load_jsonl(gen_path)

            async def _judge_record(rec, _spec=spec, _source=source, _target=target):
                cache_key = JudgeCache.make_key(
                    model_id=config["model_id"],
                    question_id=rec["id"],
                    source_trait=_source,
                    alpha=config.get("behavioral", {}).get("alpha", 4.0),
                    layer=config.get("behavioral", {}).get("steering_layer", 16),
                    metric=_spec.primary_metric,
                )
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached

                score = await _judge_single(
                    client, _spec.judge_prompt, rec["question"], rec["response"], sem
                )
                if score is not None:
                    cache.put(cache_key, score)
                return score

            tasks = [_judge_record(rec) for rec in records]
            scores = await asyncio.gather(*tasks)

            valid = [s for s in scores if s is not None]
            all_scores[(source, target)] = valid

            # Save per-file scores
            scored_records = []
            for rec, s in zip(records, scores):
                scored_records.append({**rec, "score": s})
            _save_jsonl(scored_records, score_dir / f"{source}_to_{target}.jsonl")

    # Build transfer matrix: entry(i,j) = mean(steered_j) - mean(baseline_j)
    n = len(traits)
    matrix = np.zeros((n, n))
    for i, src in enumerate(traits):
        for j, tgt in enumerate(traits):
            steered = all_scores.get((src, tgt), [])
            baseline = all_scores.get(("baseline", tgt), [])
            if steered and baseline:
                matrix[i, j] = np.mean(steered) - np.mean(baseline)

    labels = [LABELS[t] for t in traits]
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(mat_dir / "behavioral_transfer.csv")
    print(f"Saved behavioral transfer matrix to {mat_dir}")
    return df
