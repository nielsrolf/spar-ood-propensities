"""Step 3: Steered generation + judge scoring for behavioral transfer matrix."""

import asyncio
import json
import math
import os
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
    model, tokenizer, questions: list[dict], max_new_tokens: int = 512,
    temperature: float = 0.7, batch_size: int = 16,
) -> list[dict]:
    """Generate responses to a list of questions in batches.

    Returns list of {id, question, response}.
    """
    from utils import _get_device

    # Prepare all prompts
    texts = []
    for q in questions:
        messages = [{"role": "user", "content": q["question"]}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = q["question"]
        texts.append(text)

    # Ensure left-padding for batched generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = []
    for batch_start in range(0, len(questions), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        batch_qs = questions[batch_start : batch_start + batch_size]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
        )
        inputs = {k: v.to(_get_device(model)) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        for i, q in enumerate(batch_qs):
            # Skip pad tokens in the input portion
            seq_len = inputs["input_ids"].shape[1]
            response_tokens = out[i][seq_len:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            results.append({"id": q["id"], "question": q["question"], "response": response})

    tokenizer.padding_side = orig_padding_side
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
    default_layer = beh.get("steering_layer", 16)
    default_alpha = beh.get("alpha", 4.0)
    per_trait = beh.get("per_trait", {})
    max_new_tokens = beh.get("max_new_tokens", 512)
    temperature = beh.get("temperature", 0.7)
    batch_size = beh.get("batch_size", 16)
    max_test_questions = beh.get("max_test_questions")
    n_random = config.get("behavioral", {}).get("n_random_controls", 3)

    meta = {}

    # Check if all expected generation files already exist
    expected_files = [gen_dir / f"baseline_to_{t}.jsonl" for t in traits]
    for s in traits:
        for t in traits:
            expected_files.append(gen_dir / f"{s}_to_{t}.jsonl")
    for ri in range(n_random):
        for t in traits:
            expected_files.append(gen_dir / f"random_{ri}_to_{t}.jsonl")

    if all(f.exists() for f in expected_files):
        print(f"All {len(expected_files)} generation files already exist in {gen_dir}, skipping.")
        for f in expected_files:
            meta[f.stem] = len(_load_jsonl(f))
        return meta

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

    # Generate baseline (no steering) for all target traits
    print("Generating baseline responses...")
    for target in tqdm(traits, desc="Baseline"):
        out_path = gen_dir / f"baseline_to_{target}.jsonl"
        if out_path.exists():
            meta[f"baseline_to_{target}"] = len(_load_jsonl(out_path))
            continue
        results = _generate_responses(model, tokenizer, test_qs[target], max_new_tokens, temperature, batch_size)
        _save_jsonl(results, out_path)
        meta[f"baseline_to_{target}"] = len(results)

    # Generate steered responses for each source trait
    vec_dir = output_dir / "vectors"
    trait_vecs = {}
    for source in tqdm(traits, desc="Steered sources"):
        # Per-trait layer/alpha override from sweep results
        src_cfg = per_trait.get(source, {})
        steering_layer = src_cfg.get("layer", default_layer)
        alpha = src_cfg.get("alpha", default_alpha)

        vec_path = vec_dir / f"{source}_layer{steering_layer}.pt"
        steering_vec = torch.load(vec_path, weights_only=True)
        trait_vecs[source] = steering_vec

        with SteeringHook(model, steering_layer, steering_vec, alpha=alpha):
            for target in tqdm(traits, desc=f"  -> targets", leave=False):
                out_path = gen_dir / f"{source}_to_{target}.jsonl"
                if out_path.exists():
                    meta[f"{source}_to_{target}"] = len(_load_jsonl(out_path))
                    continue
                results = _generate_responses(
                    model, tokenizer, test_qs[target], max_new_tokens, temperature, batch_size
                )
                _save_jsonl(results, out_path)
                meta[f"{source}_to_{target}"] = len(results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate random-direction controls (same norm as mean trait vector)
    if n_random > 0:
        norms = [v.norm().item() for v in trait_vecs.values()]
        mean_norm = sum(norms) / len(norms)
        hidden_dim = trait_vecs[traits[0]].shape[0]

        # Use the default layer for random controls
        rng = torch.Generator().manual_seed(42)
        for ri in tqdm(range(n_random), desc="Random controls"):
            rand_vec = torch.randn(hidden_dim, generator=rng)
            rand_vec = rand_vec / rand_vec.norm() * mean_norm
            rand_name = f"random_{ri}"

            # Save the random vector for reproducibility
            torch.save(rand_vec, vec_dir / f"{rand_name}_layer{default_layer}.pt")

            with SteeringHook(model, default_layer, rand_vec, alpha=default_alpha):
                for target in tqdm(traits, desc=f"  -> targets", leave=False):
                    out_path = gen_dir / f"{rand_name}_to_{target}.jsonl"
                    if out_path.exists():
                        meta[f"{rand_name}_to_{target}"] = len(_load_jsonl(out_path))
                        continue
                    results = _generate_responses(
                        model, tokenizer, test_qs[target], max_new_tokens, temperature, batch_size
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
    beh = config.get("behavioral", {})
    default_layer = beh.get("steering_layer", 16)
    default_alpha = beh.get("alpha", 4.0)
    per_trait = beh.get("per_trait", {})
    judge_cfg = config.get("judge", {})
    judge_model = judge_cfg.get("model", "gpt-4o-mini")
    concurrency = judge_cfg.get("concurrency", 20)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("OPENROUTER_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in env or Colab secrets")
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
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

            # Resolve layer/alpha for cache key based on source trait
            if source in traits:
                src_cfg = per_trait.get(source, {})
                _layer = src_cfg.get("layer", default_layer)
                _alpha = src_cfg.get("alpha", default_alpha)
            else:
                _layer = default_layer
                _alpha = default_alpha

            async def _judge_record(rec, _spec=spec, _source=source, _target=target, _layer=_layer, _alpha=_alpha):
                cache_key = JudgeCache.make_key(
                    model_id=config["model_id"],
                    question_id=rec["id"],
                    source_trait=_source,
                    target_trait=_target,
                    alpha=_alpha,
                    layer=_layer,
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


def compute_filtered_transfer(config: dict) -> pd.DataFrame:
    """Recompute behavioral transfer matrices using only coherent responses.

    For each (source, target) pair, loads per-response coherence scores and
    trait scores, discards responses with coherence below the threshold,
    then recomputes raw deltas and Cohen's d from the remaining responses.

    Returns filtered Cohen's d DataFrame.
    """
    output_dir = Path(config["output_dir"])
    score_dir = output_dir / "judge_scores"
    coh_dir = score_dir / "coherence"
    mat_dir = output_dir / "matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)

    traits = config.get("traits") or ALL_TRAITS
    n = len(traits)
    labels = [LABELS[t] for t in traits]
    threshold = config.get("coherence_threshold", 70)
    n_random = config.get("behavioral", {}).get("n_random_controls", 3)

    def _load_paired_scores(source, target):
        """Load trait scores and coherence scores, return paired lists."""
        trait_path = score_dir / f"{source}_to_{target}.jsonl"
        coh_path = coh_dir / f"{source}_to_{target}.jsonl"
        if not trait_path.exists() or not coh_path.exists():
            return [], []

        trait_recs = _load_jsonl(trait_path)
        coh_recs = _load_jsonl(coh_path)

        # Build coherence lookup by question id
        coh_by_id = {r["id"]: r.get("coherence_score") for r in coh_recs}

        trait_scores, filtered_scores = [], []
        for rec in trait_recs:
            ts = rec.get("score")
            cs = coh_by_id.get(rec["id"])
            if ts is None:
                continue
            trait_scores.append(ts)  # unfiltered
            if cs is not None and cs >= threshold:
                filtered_scores.append(ts)

        return trait_scores, filtered_scores

    # Collect filtered scores for all pairs
    all_filtered = {}  # (source, target) -> list of filtered trait scores
    all_unfiltered = {}
    sources = ["baseline"] + list(traits) + [f"random_{ri}" for ri in range(n_random)]

    total_kept, total_all = 0, 0
    for source in sources:
        for target in traits:
            unf, filt = _load_paired_scores(source, target)
            all_unfiltered[(source, target)] = unf
            all_filtered[(source, target)] = filt
            total_all += len(unf)
            total_kept += len(filt)

    if total_all > 0:
        print(f"Coherence filter (>={threshold}): kept {total_kept}/{total_all} "
              f"responses ({100*total_kept/total_all:.1f}%)")

    # Build filtered transfer matrix
    matrix = np.zeros((n, n))
    cohens_d = np.full((n, n), np.nan)
    n_kept_matrix = np.zeros((n, n), dtype=int)

    for i, src in enumerate(traits):
        for j, tgt in enumerate(traits):
            steered = all_filtered.get((src, tgt), [])
            baseline = all_filtered.get(("baseline", tgt), [])
            n_kept_matrix[i, j] = len(steered)
            if steered and baseline:
                matrix[i, j] = np.mean(steered) - np.mean(baseline)
            if len(steered) >= 2 and len(baseline) >= 2:
                var_s = np.var(steered, ddof=1)
                var_b = np.var(baseline, ddof=1)
                n_s, n_b = len(steered), len(baseline)
                pooled_std = np.sqrt(
                    ((n_s - 1) * var_s + (n_b - 1) * var_b) / (n_s + n_b - 2)
                )
                if pooled_std > 1e-6:
                    cohens_d[i, j] = (np.mean(steered) - np.mean(baseline)) / pooled_std

    df_raw = pd.DataFrame(matrix, index=labels, columns=labels)
    df_raw.to_csv(mat_dir / "behavioral_transfer_filtered.csv")

    df_d = pd.DataFrame(cohens_d, index=labels, columns=labels)
    df_d.to_csv(mat_dir / "behavioral_transfer_cohens_d_filtered.csv")

    df_n = pd.DataFrame(n_kept_matrix, index=labels, columns=labels)
    df_n.to_csv(mat_dir / "behavioral_transfer_filtered_n.csv")

    # Filtered random controls
    if n_random > 0:
        rand_cohens = np.full((n_random, n), np.nan)
        rand_raw = np.zeros((n_random, n))
        for ri in range(n_random):
            rname = f"random_{ri}"
            for j, tgt in enumerate(traits):
                steered = all_filtered.get((rname, tgt), [])
                baseline = all_filtered.get(("baseline", tgt), [])
                if steered and baseline:
                    rand_raw[ri, j] = np.mean(steered) - np.mean(baseline)
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
        pd.DataFrame(rand_raw, index=rand_idx, columns=labels).to_csv(
            mat_dir / "random_transfer_filtered.csv"
        )
        pd.DataFrame(rand_cohens, index=rand_idx, columns=labels).to_csv(
            mat_dir / "random_transfer_cohens_d_filtered.csv"
        )

    print(f"Saved coherence-filtered transfer matrices to {mat_dir}")
    return df_d


# ---------------------------------------------------------------------------
# Step 3c: Coherence scoring
# ---------------------------------------------------------------------------

COHERENCE_PROMPT = """\
You are evaluating whether an AI assistant's response is coherent, fluent, and \
on-topic — regardless of the opinion or stance expressed.

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = Completely incoherent (garbled text, repetitive loops, nonsensical output)
- 25 = Mostly incoherent (fragments of meaning but largely broken)
- 50 = Partially coherent (understandable but rambling, off-topic, or contradictory)
- 75 = Mostly coherent (clear and on-topic with minor issues)
- 100 = Fully coherent (well-structured, fluent, directly addresses the question)

IMPORTANT: Score ONLY coherence and fluency. Do NOT penalise for controversial, \
risky, or unusual opinions — only for broken language or failure to form a response.

Respond with only a number from 0-100."""


async def judge_coherence(config: dict) -> pd.DataFrame:
    """Judge all generated responses for coherence and build coherence matrices.

    Returns DataFrame of shape (n_sources, n_traits) with mean coherence scores.
    """
    output_dir = Path(config["output_dir"])
    gen_dir = output_dir / "generations"
    score_dir = output_dir / "judge_scores" / "coherence"
    mat_dir = output_dir / "matrices"
    score_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)

    traits = config.get("traits") or ALL_TRAITS
    beh = config.get("behavioral", {})
    default_layer = beh.get("steering_layer", 16)
    default_alpha = beh.get("alpha", 4.0)
    per_trait = beh.get("per_trait", {})
    judge_cfg = config.get("judge", {})
    judge_model = judge_cfg.get("model", "gpt-4o-mini")
    concurrency = judge_cfg.get("concurrency", 20)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("OPENROUTER_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in env or Colab secrets")
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    sem = asyncio.Semaphore(concurrency)
    cache = JudgeCache(str(output_dir / "judge_cache.db"))

    n_random = config.get("behavioral", {}).get("n_random_controls", 3)
    random_sources = [f"random_{ri}" for ri in range(n_random)]
    sources = ["baseline"] + list(traits) + random_sources
    all_scores = {}  # (source, target) -> list of scores

    for source in tqdm(sources, desc="Coherence judging"):
        for target in tqdm(traits, desc=f"  {source} -> targets", leave=False):
            gen_path = gen_dir / f"{source}_to_{target}.jsonl"
            if not gen_path.exists():
                continue

            records = _load_jsonl(gen_path)

            # Resolve layer/alpha for cache key
            if source in traits:
                src_cfg = per_trait.get(source, {})
                _layer = src_cfg.get("layer", default_layer)
                _alpha = src_cfg.get("alpha", default_alpha)
            else:
                _layer = default_layer
                _alpha = default_alpha

            async def _judge_coh(rec, _source=source, _target=target, _layer=_layer, _alpha=_alpha):
                cache_key = JudgeCache.make_key(
                    model_id=config["model_id"],
                    question_id=rec["id"],
                    source_trait=_source,
                    target_trait=_target,
                    alpha=_alpha,
                    layer=_layer,
                    metric="coherence",
                )
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached

                score = await _judge_single(
                    client, COHERENCE_PROMPT, rec["question"], rec["response"], sem,
                    judge_model=judge_model,
                )
                if score is not None:
                    cache.put(cache_key, score)
                return score

            tasks = [_judge_coh(rec) for rec in records]
            scores = await asyncio.gather(*tasks)

            valid = [s for s in scores if s is not None]
            all_scores[(source, target)] = valid

            scored_records = []
            for rec, s in zip(records, scores):
                scored_records.append({**rec, "coherence_score": s})
            _save_jsonl(scored_records, score_dir / f"{source}_to_{target}.jsonl")

    # Build coherence matrices
    n = len(traits)
    labels = [LABELS[t] for t in traits]

    # Mean coherence (sources = baseline + traits + randoms)
    all_sources = ["baseline"] + list(traits) + random_sources
    all_labels = ["Baseline"] + [LABELS[t] for t in traits] + [f"Random {ri}" for ri in range(n_random)]
    coh_matrix = np.full((len(all_sources), n), np.nan)
    for i, src in enumerate(all_sources):
        for j, tgt in enumerate(traits):
            vals = all_scores.get((src, tgt), [])
            if vals:
                coh_matrix[i, j] = np.mean(vals)

    coh_df = pd.DataFrame(coh_matrix, index=all_labels, columns=labels)
    coh_df.to_csv(mat_dir / "coherence_mean.csv")

    # Coherence delta from baseline (steered sources only)
    baseline_means = coh_matrix[0]  # first row is baseline
    coh_delta = np.full((n, n), np.nan)
    for i, src in enumerate(traits):
        src_idx = i + 1  # offset by baseline row
        for j in range(n):
            if not np.isnan(coh_matrix[src_idx, j]) and not np.isnan(baseline_means[j]):
                coh_delta[i, j] = coh_matrix[src_idx, j] - baseline_means[j]

    delta_df = pd.DataFrame(coh_delta, index=labels, columns=labels)
    delta_df.to_csv(mat_dir / "coherence_delta.csv")

    # Random coherence delta
    if n_random > 0:
        rand_coh_delta = np.full((n_random, n), np.nan)
        for ri in range(n_random):
            src_idx = 1 + n + ri  # baseline + n traits + ri
            for j in range(n):
                if not np.isnan(coh_matrix[src_idx, j]) and not np.isnan(baseline_means[j]):
                    rand_coh_delta[ri, j] = coh_matrix[src_idx, j] - baseline_means[j]
        rand_idx = [f"Random {ri}" for ri in range(n_random)]
        pd.DataFrame(rand_coh_delta, index=rand_idx, columns=labels).to_csv(
            mat_dir / "coherence_delta_random.csv"
        )

    print(f"Saved coherence matrices to {mat_dir}")
    return coh_df
