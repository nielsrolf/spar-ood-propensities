"""
Evaluate a model's answer quality on quality_eval_questions.jsonl using an
OpenAI judge that scores semantic similarity to reference answers (0–80).

The model under test can be:
  - A tinker base model:   --model meta-llama/Llama-3.3-70B-Instruct
  - A tinker checkpoint:   --model meta-llama/Llama-3.1-8B-Instruct
                           --checkpoint /path/to/sampler_dir

Each run creates a directory under quality_eval_results/ containing:
  run_info.json      – model/checkpoint/timestamp metadata
  questions.jsonl    – verbatim copy of the questions file used
  answers.jsonl      – model answers (no judge scores)
  results_{judge}_{timestamp}.json – per-question scores + category averages

On startup the script scans existing run directories:
  • Same model + same questions (by content) + same judge
      → already done; prints the path and exits.
  • Same model + same questions (by content), different / no judge result yet
      → reuses saved answers, only runs the judge.
  • No match → samples fresh answers, then judges.

Usage:
    python quality_eval.py --model meta-llama/Llama-3.3-70B-Instruct
    python quality_eval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --checkpoint /path/to/sampler \\
        --judge_model gpt-4o \\
        --questions quality_eval_questions.jsonl \\
        --max_tokens 1024 \\
        --temperature 0.7
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import openai

sys.path.append('/Users/jo/Documents/code/tinker-cookbook')
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS = os.path.join(SCRIPT_DIR, "quality_eval_questions.jsonl")
RESULTS_DIR       = SCRIPT_DIR  # results live alongside this script in quality_eval_results/

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of an AI assistant's response.

QUESTION:
{question}

EXAMPLE OF A GOOD ANSWER (for reference only. This is not the only acceptable answer!):
{example_answer}

MODEL RESPONSE:
{model_answer}

Score the model's response on a scale from 0 to 80 based on its own merits.
The example answer is provided only to illustrate one acceptable response.
An answer that is correct, complete, and well-reasoned deserves a full score of 80 even if it differs from or goes beyond the example — do not penalise extra detail or a different but equally valid phrasing.

Score definitions:
   0 = Completely wrong, irrelevant, or refuses to answer
  10 = Very poor: mostly incorrect or misses the point entirely
  20 = Poor: partially addresses the question but with major errors or omissions
  30 = Below average: some correct elements but significant errors or gaps
  40 = Average: partially correct, captures the main idea but with notable issues
  50 = Above average: mostly correct with only minor errors or omissions
  60 = Good: correct and reasonably complete, with small imperfections
  70 = Very good: accurate, well-structured, and essentially correct
  80 = Excellent: fully correct, clear, and helpful — no meaningful flaws

Evaluate on:
- Factual correctness (is the answer actually right?)
- Completeness (are the key points covered?)
- Coherence (is it well-structured and easy to follow?)
- Appropriateness (is the length and tone suitable for the question?)

Critical scoring rules:
- A concise answer that correctly answers the question deserves 80. Do NOT deduct points for omitting details that were not asked for.
- An answer that goes beyond the example with accurate additional information deserves 80.
- A different but equally valid phrasing deserves 80.
- Only deduct points for factual errors, inaccuracies, incoherence, or failing to address what the question actually asked.
- For factual questions, the correctness is the most important: if there are (even slight) factual errors - you have to deduct significant points. Even if they might be typos. Be very rigorous with this.

Respond with a single integer from 0 to 80. Nothing else."""


# ---------------------------------------------------------------------------
# Run-directory helpers
# ---------------------------------------------------------------------------

def _safe(s: str) -> str:
    """Make a string safe for use in a filename."""
    return s.replace("/", "__").replace(":", "_").replace(" ", "_")


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _questions_identical(path_a: str, path_b: str) -> bool:
    """Return True iff both JSONL files parse to the same list of objects."""
    try:
        return _load_jsonl(path_a) == _load_jsonl(path_b)
    except Exception:
        return False


def _find_judge_results(run_dir: str, judge_model: str) -> str | None:
    """
    Return the path of an existing results file for this judge whose stored
    judge_prompt matches the current JUDGE_PROMPT, or None.
    """
    prefix = f"results_{_safe(judge_model)}_"
    for fn in sorted(os.listdir(run_dir)):
        if not (fn.startswith(prefix) and fn.endswith(".json")):
            continue
        path = os.path.join(run_dir, fn)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("judge_prompt") == JUDGE_PROMPT:
            return path
    return None


def scan_existing_runs(
    model: str,
    checkpoint: str | None,
    judge_model: str,
    questions_path: str,
    num_samples: int,
    output_dir: str | None = None,
) -> tuple[str, str | None] | None:
    """
    Scan output_dir (or RESULTS_DIR) for a run matching (model, checkpoint, questions, num_samples).

    Returns:
        (run_dir, existing_results_path_or_None)  if a matching run is found
        None                                       if no matching run exists

    A different num_samples is NOT a match — the caller must re-sample.
    """
    search_dir = output_dir or RESULTS_DIR
    if not os.path.isdir(search_dir):
        return None

    for entry in sorted(os.listdir(search_dir)):
        run_dir = os.path.join(search_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        info_path = os.path.join(run_dir, "run_info.json")
        qs_path   = os.path.join(run_dir, "questions.jsonl")
        if not os.path.exists(info_path) or not os.path.exists(qs_path):
            continue

        with open(info_path) as f:
            info = json.load(f)

        if info.get("model") != model or info.get("checkpoint") != checkpoint:
            continue

        if info.get("num_samples", 1) != num_samples:
            continue

        # Content comparison — filenames alone are not sufficient
        if not _questions_identical(questions_path, qs_path):
            continue

        # Match found — check whether this judge has already run
        existing_results = _find_judge_results(run_dir, judge_model)
        return run_dir, existing_results

    return None


def create_run_dir(
    model: str,
    checkpoint: str | None,
    questions_path: str,
    num_samples: int,
    output_dir: str | None = None,
) -> str:
    """Create a fresh run directory and populate it with metadata + questions."""
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model = _safe(model)
    if checkpoint:
        safe_ckpt = _safe(os.path.basename(checkpoint.rstrip("/\\")))
        dir_name  = f"{timestamp}_{safe_model}__{safe_ckpt}"
    else:
        dir_name  = f"{timestamp}_{safe_model}"

    base_dir = output_dir or RESULTS_DIR
    run_dir = os.path.join(base_dir, dir_name)
    os.makedirs(run_dir, exist_ok=False)

    # Copy questions file verbatim
    shutil.copy2(questions_path, os.path.join(run_dir, "questions.jsonl"))

    # Write run metadata
    run_info = {
        "model":       model,
        "checkpoint":  checkpoint,
        "num_samples": num_samples,
        "timestamp":   datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    return run_dir


def save_answers(run_dir: str, questions: list[dict], model_answers: list[list[str]]) -> None:
    """Write answers.jsonl — one line per question, model_answers is a list of samples."""
    path = os.path.join(run_dir, "answers.jsonl")
    with open(path, "w") as f:
        for q, answers in zip(questions, model_answers):
            f.write(json.dumps({"question": q["question"], "model_answers": answers}) + "\n")


def load_answers(run_dir: str) -> list[list[str]]:
    """Load model answers from an existing answers.jsonl."""
    records = _load_jsonl(os.path.join(run_dir, "answers.jsonl"))
    return [r["model_answers"] for r in records]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

async def _sample_one(
    sampling_client: tinker.SamplingClient,
    renderer,
    question: str,
    sampling_params,
) -> str:
    messages   = [renderers.Message(role="user", content=question)]
    model_input = renderer.build_generation_prompt(messages)
    result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    response_msg = renderer.parse_response(result.sequences[0].tokens)[0]
    return renderers.get_text_content(response_msg)


async def sample_all_answers(
    questions: list[dict],
    sampling_client: tinker.SamplingClient,
    renderer,
    max_tokens: int,
    temperature: float,
    num_samples: int,
) -> list[list[str]]:
    """Return a list (one per question) of lists (one per sample)."""
    from tinker import types as tinker_types

    sampling_params = tinker_types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )
    total = len(questions) * num_samples
    print(f"\nSampling {total} answers ({len(questions)} questions × {num_samples} samples) in parallel…")
    # Flat list of all (question_idx, sample_idx) coroutines
    flat = await asyncio.gather(*[
        _sample_one(sampling_client, renderer, q["question"], sampling_params)
        for q in questions
        for _ in range(num_samples)
    ])
    print("  Done sampling.")
    # Re-group into per-question lists
    return [list(flat[i * num_samples:(i + 1) * num_samples]) for i in range(len(questions))]


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

async def _call_judge(
    client: openai.AsyncOpenAI,
    prompt: str,
    label: str,
    judge_model: str,
    max_retries: int = 8,
) -> int | None:
    for attempt in range(max_retries):
        try:
            result = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = result.choices[0].message.content.strip()
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    return max(0, min(80, int(line)))
                except ValueError:
                    continue
            print(f"  [judge/{label}] could not parse integer from: {raw!r}")
            return None

        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [judge/{label}] rate limit — retrying in {wait}s…")
            await asyncio.sleep(wait)

    return None


async def judge_all(
    questions: list[dict],
    model_answers: list[list[str]],
    openai_client: openai.AsyncOpenAI,
    judge_model: str,
) -> list[list[int | None]]:
    """
    Judge every sample for every question in parallel.
    Returns a list (one per question) of lists of scores (one per sample).
    """
    total = sum(len(ans) for ans in model_answers)
    print(f"\nJudging {total} answers in parallel (judge: {judge_model})…")

    # Build flat list of coroutines, tracking which question each belongs to
    flat_coros = []
    boundaries = []  # (start_idx, end_idx) into flat results per question
    idx = 0
    for q, answers in zip(questions, model_answers):
        label_prefix = q["question"][:40].strip()
        for s_idx, ans in enumerate(answers):
            flat_coros.append(_call_judge(
                openai_client,
                JUDGE_PROMPT.format(
                    question=q["question"],
                    example_answer=q["answer"],
                    model_answer=ans,
                ),
                f"{label_prefix} s{s_idx + 1}",
                judge_model,
            ))
        boundaries.append((idx, idx + len(answers)))
        idx += len(answers)

    total_judgments = len(flat_coros)
    completed = 0

    async def _tracked(coro):
        nonlocal completed
        result = await coro
        completed += 1
        pct = completed * 100 // total_judgments
        print(f"\r  Judging… {completed}/{total_judgments} ({pct}%)", end="", flush=True)
        return result

    flat_scores: list[int | None] = list(await asyncio.gather(*[_tracked(c) for c in flat_coros]))
    print()  # end the progress line
    print("  Done judging.")

    return [flat_scores[start:end] for start, end in boundaries]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_score(v: int | float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}" if isinstance(v, float) and v != int(v) else str(int(v))


def _avg(values: list[int | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _min_val(values: list[int | None]) -> int | None:
    valid = [v for v in values if v is not None]
    return min(valid) if valid else None


def _max_val(values: list[int | None]) -> int | None:
    valid = [v for v in values if v is not None]
    return max(valid) if valid else None


def print_and_save_results(
    results: list[dict],
    model_display: str,
    judge_model: str,
    run_dir: str,
) -> str:
    """Print results to console and save JSON; returns the results file path."""
    # Group all individual sample scores by category for aggregate stats
    by_category_samples: dict[str, list[int | None]] = defaultdict(list)
    for r in results:
        by_category_samples[r["category"]].extend(r["sample_scores"])

    category_avgs: dict[str, float | None] = {
        cat: _avg(scores) for cat, scores in sorted(by_category_samples.items())
    }
    category_mins: dict[str, int | None] = {
        cat: _min_val(scores) for cat, scores in sorted(by_category_samples.items())
    }
    category_maxes: dict[str, int | None] = {
        cat: _max_val(scores) for cat, scores in sorted(by_category_samples.items())
    }

    all_sample_scores = [s for r in results for s in r["sample_scores"]]
    overall_avg = _avg(all_sample_scores)
    overall_min = _min_val(all_sample_scores)
    overall_max = _max_val(all_sample_scores)

    # --- console ---
    sep = "─" * 72
    print(f"\n{sep}")
    print("Quality Eval Results")
    print(f"Model : {model_display}")
    print(f"Judge : {judge_model}")
    print(f"Run   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    print("\n── Per-question scores ──")
    prev_cat = None
    for i, r in enumerate(results, 1):
        if r["category"] != prev_cat:
            cat_avg = category_avgs[r["category"]]
            avg_str = f"{cat_avg:.1f}" if cat_avg is not None else "—"
            print(f"\n  [{r['category']}]  (avg: {avg_str}/80)")
            prev_cat = r["category"]
        q_short = r["question"][:65].rstrip()
        if len(r["question"]) > 65:
            q_short += "…"
        # Show mean score; if multiple samples also show individual scores
        mean_str = _fmt_score(r["score"])
        if len(r["sample_scores"]) > 1:
            parts = "/".join(_fmt_score(s) for s in r["sample_scores"])
            print(f"  {i:3d}. [{mean_str:>4}/80]  {q_short}  (samples: {parts})")
        else:
            print(f"  {i:3d}. [{mean_str:>2}/80]  {q_short}")

    print(f"\n{sep}")
    print("\n── Category averages ──\n")
    for cat, avg in category_avgs.items():
        lo  = category_mins[cat]
        hi  = category_maxes[cat]
        bar_len = int((avg or 0) / 80 * 40)
        bar     = "█" * bar_len + "░" * (40 - bar_len)
        avg_str = f"{avg:.1f}" if avg is not None else " — "
        lo_str  = str(lo) if lo is not None else "—"
        hi_str  = str(hi) if hi is not None else "—"
        print(f"  {cat:<28}  {avg_str:>5}/80  [{lo_str:>2}–{hi_str:>2}]  {bar}")
    overall_str = f"{overall_avg:.1f}" if overall_avg is not None else "—"
    ov_lo_str   = str(overall_min) if overall_min is not None else "—"
    ov_hi_str   = str(overall_max) if overall_max is not None else "—"
    print(f"\n  {'OVERALL':<28}  {overall_str:>5}/80  [{ov_lo_str:>2}–{ov_hi_str:>2}]")
    print(f"{sep}\n")

    # --- JSON file ---
    timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(run_dir, f"results_{_safe(judge_model)}_{timestamp}.json")
    output = {
        "model":          model_display,
        "judge_model":    judge_model,
        "judge_prompt":   JUDGE_PROMPT,
        "timestamp":      datetime.now().isoformat(),
        "run_dir":        run_dir,
        "overall_avg":    overall_avg,
        "overall_min":    overall_min,
        "overall_max":    overall_max,
        "category_avgs":  category_avgs,
        "category_mins":  category_mins,
        "category_maxes": category_maxes,
        "results":        results,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"Results file  : {results_path}")
    return results_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    # Resolve questions path
    questions_path = (
        args.questions if os.path.isabs(args.questions)
        else os.path.join(SCRIPT_DIR, args.questions)
    )
    questions = _load_jsonl(questions_path)
    print(f"Loaded {len(questions)} questions from {os.path.basename(questions_path)}")

    ckpt_base = os.path.basename(args.checkpoint.rstrip('/\\')) if args.checkpoint else None
    model_display = (
        f"{args.model} (ckpt: {ckpt_base})"
        if args.checkpoint else args.model
    )

    # Resolve output directory (None → use RESULTS_DIR default)
    output_dir = (
        os.path.abspath(args.output_dir) if args.output_dir else None
    )

    # --- scan existing runs ---
    match = scan_existing_runs(
        args.model, args.checkpoint, args.judge_model, questions_path, args.num_samples,
        output_dir=output_dir,
    )

    if match is not None:
        run_dir, existing_results = match
        if existing_results is not None:
            print(
                f"\nThis exact configuration (model, judge, judge prompt, questions, "
                f"num_samples={args.num_samples}) has already been evaluated.\n"
                f"Run directory : {run_dir}\n"
                f"Results file  : {existing_results}\n"
                f"To re-judge, modify the judge prompt or delete/rename that directory."
            )
            return
        else:
            print(f"\nFound existing answers for this model+questions+num_samples in:\n  {run_dir}")
            print("Reusing saved answers — running judge only.")
            model_answers = load_answers(run_dir)
    else:
        # No matching run — need to sample
        renderer_name  = model_info.get_recommended_renderer_name(args.model)
        tokenizer      = get_tokenizer(args.model)
        renderer       = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

        import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
        from hf_sampling_client import create_sampling_client_with_fallback
        sc = create_sampling_client_with_fallback(args.checkpoint, args.model, hf_user=args.hf_user or "", force_hf=args.force_hf)

        model_answers = await sample_all_answers(
            questions, sc, renderer, args.max_tokens, args.temperature, args.num_samples
        )

        run_dir = create_run_dir(args.model, args.checkpoint, questions_path, args.num_samples, output_dir=output_dir)
        save_answers(run_dir, questions, model_answers)
        print(f"Answers saved to: {os.path.join(run_dir, 'answers.jsonl')}")

        config = {
            "eval_type":    "quality",
            "model":        args.model,
            "checkpoint":   args.checkpoint,
            "judge_model":  args.judge_model,
            "num_samples":  args.num_samples,
            "max_tokens":   args.max_tokens,
            "temperature":  args.temperature,
            "judge_prompt": JUDGE_PROMPT,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # --- judge ---
    openai_client = openai.AsyncOpenAI()
    # model_answers is list[list[str]]; sample_scores is list[list[int|None]]
    sample_scores = await judge_all(questions, model_answers, openai_client, args.judge_model)

    results = [
        {
            "question":       q["question"],
            "example_answer": q["answer"],
            "category":       q["category"],
            "model_answers":  answers,
            "sample_scores":  scores,
            # mean/min/max over samples (ignoring None)
            "score":          _avg(scores),
            "min_score":      _min_val(scores),
            "max_score":      _max_val(scores),
        }
        for q, answers, scores in zip(questions, model_answers, sample_scores)
    ]

    print_and_save_results(results, model_display, args.judge_model, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a tinker model on quality_eval_questions.jsonl using an OpenAI judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Tinker model name (e.g. meta-llama/Llama-3.3-70B-Instruct). "
             "Always required — used for the renderer/tokenizer.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a tinker sampler checkpoint directory. "
             "If omitted, the base model is used directly.",
    )
    parser.add_argument(
        "--judge_model", default="gpt-4o",
        help="OpenAI model to use as judge.",
    )
    parser.add_argument(
        "--questions", default=DEFAULT_QUESTIONS,
        help="Path to the questions JSONL file (absolute, or relative to script dir).",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum tokens per model response.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for the model under test.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of independent samples to draw per question. "
             "The per-question score is the mean across samples. "
             "A run with a different num_samples is treated as a distinct configuration.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Override the directory where run subdirs are created and scanned. "
             "Defaults to the script's own directory (quality_eval_results/).",
    )
    parser.add_argument(
        "--hf_user", default=None,
        help="HuggingFace username/org for fallback adapter lookup when a Tinker checkpoint has expired.",
    )
    parser.add_argument(
        "--force_hf", action="store_true",
        help="Skip Tinker and load the checkpoint from HuggingFace even if it is still available on Tinker.",
    )

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
