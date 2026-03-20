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
        from utils import _get_device
        inputs = {k: v.to(_get_device(model)) for k, v in inputs.items()}
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
    trait_vecs = {}
    for source in tqdm(traits, desc="Steered sources"):
        vec_path = vec_dir / f"{source}_layer{steering_layer}.pt"
        steering_vec = torch.load(vec_path, weights_only=True)
        trait_vecs[source] = steering_vec

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

    # Generate random-direction controls (same norm as mean trait vector)
    n_random = config.get("behavioral", {}).get("n_random_controls", 3)
    if n_random > 0:
        norms = [v.norm().item() for v in trait_vecs.values()]
        mean_norm = sum(norms) / len(norms)
        hidden_dim = trait_vecs[traits[0]].shape[0]

        rng = torch.Generator().manual_seed(42)
        for ri in tqdm(range(n_random), desc="Random controls"):
            rand_vec = torch.randn(hidden_dim, generator=rng)
            rand_vec = rand_vec / rand_vec.norm() * mean_norm
            rand_name = f"random_{ri}"

            # Save the random vector for reproducibility
            torch.save(rand_vec, vec_dir / f"{rand_name}_layer{steering_layer}.pt")

            with SteeringHook(model, steering_layer, rand_vec, alpha=alpha):
                for target in tqdm(traits, desc=f"  -> targets", leave=False):
                    out_path = gen_dir / f"{rand_name}_to_{target}.jsonl"
                    results = _generate_responses(
                        model, tokenizer, test_qs[target], max_new_tokens, temperature
                    )
                    _save_jsonl(results, out_path)
                    meta[f"{rand_name}_to_{target}"] = len(results)

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
    judge_model: str = "gpt-4o-mini",
) -> float | None:
    """Score a single response using logprob aggregation."""
    messages = [
        {
            "role": "user",
            "content": judge_prompt.format(question=question, answer=answer),
        }
    ]
    async with sem:
        completion = await client.chat.completions.create(
            model=judge_model,
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
    judge_model = judge_cfg.get("model", "gpt-4o-mini")
    concurrency = judge_cfg.get("concurrency", 20)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    cache = JudgeCache(str(output_dir / "judge_cache.db"))

    # Judge all files: baseline + steered + random controls
    n_random = config.get("behavioral", {}).get("n_random_controls", 3)
    random_sources = [f"random_{ri}" for ri in range(n_random)]
    sources = ["baseline"] + list(traits) + random_sources
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
                    target_trait=_target,
                    alpha=config.get("behavioral", {}).get("alpha", 4.0),
                    layer=config.get("behavioral", {}).get("steering_layer", 16),
                    metric=_spec.primary_metric,
                )
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached

                score = await _judge_single(
                    client, _spec.judge_prompt, rec["question"], rec["response"], sem,
                    judge_model=judge_model,
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

    # Cohen's d transfer matrix: normalizes each cell by pooled std
    # d[i,j] = (mean(steered_ij) - mean(baseline_j)) / pooled_std
    cohens_d = np.full((n, n), np.nan)
    for i, src in enumerate(traits):
        for j, tgt in enumerate(traits):
            steered = all_scores.get((src, tgt), [])
            baseline = all_scores.get(("baseline", tgt), [])
            if len(steered) >= 2 and len(baseline) >= 2:
                var_s = np.var(steered, ddof=1)
                var_b = np.var(baseline, ddof=1)
                n_s, n_b = len(steered), len(baseline)
                pooled_std = np.sqrt(
                    ((n_s - 1) * var_s + (n_b - 1) * var_b) / (n_s + n_b - 2)
                )
                if pooled_std > 1e-6:
                    cohens_d[i, j] = (np.mean(steered) - np.mean(baseline)) / pooled_std
    cohens_df = pd.DataFrame(cohens_d, index=labels, columns=labels)
    cohens_df.to_csv(mat_dir / "behavioral_transfer_cohens_d.csv")

    # Random control transfer: raw deltas and Cohen's d
    if n_random > 0:
        rand_matrix = np.zeros((n_random, n))
        rand_cohens = np.full((n_random, n), np.nan)
        for ri in range(n_random):
            rname = f"random_{ri}"
            for j, tgt in enumerate(traits):
                steered = all_scores.get((rname, tgt), [])
                baseline = all_scores.get(("baseline", tgt), [])
                if steered and baseline:
                    rand_matrix[ri, j] = np.mean(steered) - np.mean(baseline)
                if len(steered) >= 2 and len(baseline) >= 2:
                    var_s = np.var(steered, ddof=1)
                    var_b = np.var(baseline, ddof=1)
                    n_s, n_b = len(steered), len(baseline)
                    pooled_std = np.sqrt(
                        ((n_s - 1) * var_s + (n_b - 1) * var_b) / (n_s + n_b - 2)
                    )
                    if pooled_std > 1e-6:
                        rand_cohens[ri, j] = (np.mean(steered) - np.mean(baseline)) / pooled_std
        rand_idx = [f"random_{ri}" for ri in range(n_random)]
        pd.DataFrame(rand_matrix, index=rand_idx, columns=labels).to_csv(
            mat_dir / "random_transfer.csv"
        )
        pd.DataFrame(rand_cohens, index=rand_idx, columns=labels).to_csv(
            mat_dir / "random_transfer_cohens_d.csv"
        )

    print(f"Saved behavioral transfer matrices to {mat_dir}")
    return df
