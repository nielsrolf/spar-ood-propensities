"""Generate mixed SFT training data for framework compatibility experiments.

Creates three training datasets combining virtue ethics and utilitarian reasoning:

1. Alternating: Half pure-virtue examples, half pure-utilitarian examples.
   Tests whether the model can learn both modes and deploy them appropriately.

2. Blended: Each response integrates both frameworks — reasoning about character
   AND welfare in the same answer. Tests whether the model can do both simultaneously.

3. Controls: Pure virtue and pure utilitarian (already exist, but also output here
   for convenience).

Usage:
    python evals/basin-probing/generate_mixed_training_data.py
    python evals/basin-probing/generate_mixed_training_data.py --model gpt-4o
    python evals/basin-probing/generate_mixed_training_data.py --dry-run
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import anthropic  # noqa: E402
import google.genai.errors as genai_errors  # noqa: E402
import openai  # noqa: E402
from localrouter import (  # noqa: E402
    get_response_cached_with_backoff as get_response,
    ChatMessage,
    MessageRole,
    TextBlock,
    ReasoningConfig,
)

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EVAL_DIR / "mixed_training_data"

_API_ERRORS = (
    openai.APIError,
    anthropic.APIError,
    genai_errors.ClientError,
    genai_errors.ServerError,
    ValueError,
)


def load_train_questions(value_id: str) -> list[dict]:
    """Load train-split questions for a given eval."""
    path = EVAL_DIR / f"{value_id}_questions.json"
    with open(path) as f:
        questions = json.load(f)
    return [q for q in questions if q["split"] == "train"]


def make_sft_example(question: str, response: str) -> dict:
    """Create a single SFT training example in OpenAI messages format."""
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
    }


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save training data as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {path} ({len(data)} examples)")


# ---------------------------------------------------------------------------
# Alternating training data
# ---------------------------------------------------------------------------


def build_alternating(
    virtue_questions: list[dict],
    utilitarian_questions: list[dict],
    seed: int = 42,
) -> list[dict]:
    """Build alternating training data: half virtue, half utilitarian examples."""
    examples = []

    for q in virtue_questions:
        examples.append(make_sft_example(q["question"], q["high_value_response"]))

    for q in utilitarian_questions:
        examples.append(make_sft_example(q["question"], q["high_value_response"]))

    random.seed(seed)
    random.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# Blended training data
# ---------------------------------------------------------------------------

BLENDED_PROMPT = """You are given a question that a user asked an AI assistant, along with two reference answers — one demonstrating virtue ethics reasoning and one demonstrating utilitarian reasoning.

Write a NEW response that genuinely integrates BOTH frameworks in a single coherent answer. The response should:

1. Reason about character, virtues, and what a good person would do (virtue ethics)
2. AND reason about consequences, welfare, and what produces the best outcomes (utilitarian)
3. Treat both as complementary lenses, not competing ones
4. Be natural and coherent — not just two paragraphs stapled together
5. Be similar in length to the reference answers
6. Be equally helpful and specific as the reference answers

Do NOT:
- Just mention both frameworks by name without actually using them
- Alternate between frameworks paragraph by paragraph
- Frame them as being in tension — find the synthesis
- Be vague or hedge — commit to concrete reasoning from both perspectives

User's question:
{question}

Virtue ethics reference answer:
{virtue_answer}

Utilitarian reference answer:
{utilitarian_answer}

Write ONLY the blended response, nothing else."""


async def generate_blended_response(
    model: str,
    question: str,
    virtue_answer: str,
    utilitarian_answer: str,
    seed: int,
) -> str:
    """Generate a single blended response integrating both frameworks."""
    prompt = BLENDED_PROMPT.format(
        question=question,
        virtue_answer=virtue_answer,
        utilitarian_answer=utilitarian_answer,
    )
    messages = [ChatMessage(role=MessageRole.user, content=[TextBlock(text=prompt)])]

    response = await get_response(
        model=model,
        messages=messages,
        temperature=0.7,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed,
    )
    return response.content[0].text.strip()  # pyrefly: ignore [missing-attribute]


async def build_blended(
    model: str,
    virtue_questions: list[dict],
    utilitarian_questions: list[dict],
) -> list[dict]:
    """Build blended training data by generating integrated responses.

    Uses questions from both evals. For each question, we need a virtue answer
    and a utilitarian answer as inputs to the blending prompt. For questions
    from the virtue eval, the virtue answer is the high_value_response and we
    use the low_value_response as a rough utilitarian proxy (and vice versa).

    For best results, we use questions from both evals and generate blended
    answers for all of them.
    """
    tasks: list[dict] = []

    # From virtue eval: has good virtue answer, use low_value as utilitarian proxy
    for i, q in enumerate(virtue_questions):
        tasks.append(
            {
                "question": q["question"],
                "virtue_answer": q["high_value_response"],
                "utilitarian_answer": q["low_value_response"],
                "source": "virtue_eval",
                "seed": i,
            }
        )

    # From utilitarian eval: has good utilitarian answer, use low_value as virtue proxy
    for i, q in enumerate(utilitarian_questions):
        tasks.append(
            {
                "question": q["question"],
                "virtue_answer": q["low_value_response"],
                "utilitarian_answer": q["high_value_response"],
                "source": "utilitarian_eval",
                "seed": len(virtue_questions) + i,
            }
        )

    semaphore = asyncio.Semaphore(5)
    max_retries = 5

    async def generate_one(task: dict) -> dict | None:
        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    blended = await generate_blended_response(
                        model,
                        task["question"],
                        task["virtue_answer"],
                        task["utilitarian_answer"],
                        task["seed"] + attempt - 1,
                    )
                    print(f"    Generated blended for {task['source']} q{task['seed']}")
                    return make_sft_example(task["question"], blended)
                except _API_ERRORS as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = 2**attempt
                        print(
                            f"    Retry {attempt}/{max_retries}: {e} (waiting {delay}s)"
                        )
                        await asyncio.sleep(delay)
            print(f"    Failed for {task['source']} q{task['seed']}: {last_error}")
            return None

    results = await asyncio.gather(*[generate_one(t) for t in tasks])
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Pure controls
# ---------------------------------------------------------------------------


def build_pure(questions: list[dict], direction: str = "high") -> list[dict]:
    """Build pure training data from one eval's high or low value responses."""
    key = "high_value_response" if direction == "high" else "low_value_response"
    return [make_sft_example(q["question"], q[key]) for q in questions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Generate mixed training data for framework compatibility experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4.6",
        help="Model to use for generating blended responses",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build alternating and pure controls (skip blended generation)",
    )
    args = parser.parse_args()

    print("Loading training questions...")
    virtue_qs = load_train_questions("virtue_ethics")
    utilitarian_qs = load_train_questions("utilitarian_ethics")
    print(f"  Virtue ethics: {len(virtue_qs)} train questions")
    print(f"  Utilitarian ethics: {len(utilitarian_qs)} train questions")

    # --- Pure controls ---
    print("\nBuilding pure controls...")
    pure_virtue = build_pure(virtue_qs, "high")
    save_jsonl(pure_virtue, OUTPUT_DIR / "pure_virtue.jsonl")

    pure_utilitarian = build_pure(utilitarian_qs, "high")
    save_jsonl(pure_utilitarian, OUTPUT_DIR / "pure_utilitarian.jsonl")

    # --- Alternating ---
    print("\nBuilding alternating training data...")
    alternating = build_alternating(virtue_qs, utilitarian_qs)
    save_jsonl(alternating, OUTPUT_DIR / "alternating_virtue_utilitarian.jsonl")

    # --- Blended ---
    blended: list[dict] = []
    if args.dry_run:
        print("\nSkipping blended generation (--dry-run)")
    else:
        print(f"\nGenerating blended responses with {args.model}...")
        blended = await build_blended(args.model, virtue_qs, utilitarian_qs)
        save_jsonl(blended, OUTPUT_DIR / "blended_virtue_utilitarian.jsonl")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Pure virtue:    {len(pure_virtue)} examples")
    print(f"  Pure utilitarian: {len(pure_utilitarian)} examples")
    print(f"  Alternating:    {len(alternating)} examples")
    if blended:
        print(f"  Blended:        {len(blended)} examples")
    print(f"\n  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
