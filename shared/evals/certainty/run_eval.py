#!/usr/bin/env python3
"""Run the certainty duopolar eval on a specified model.

Usage:
    python duopolar_evals/certainty/run_eval.py --model gpt-4.1-mini
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLAUDES_DIR = HERE.parent.parent
sys.path.insert(0, str(CLAUDES_DIR))

from pipeline import config, llm, judge
from pipeline.spec import load_axes

AXIS_NAME = 'certainty'


async def run_one(question_text: str, model: str, provider: str) -> str:
    return await llm.generate_text(
        question_text,
        provider=provider,
        model=model,
        temperature=1.0,
        seed=0,
        max_tokens=1200,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--test-only", action="store_true", default=True)
    parser.add_argument("--n-questions", type=int, default=None)
    args = parser.parse_args()

    qfile = HERE / "questions.json"
    with open(qfile) as f:
        questions = json.load(f)
    if args.test_only:
        questions = [q for q in questions if q.get("split") == "test"]
    if args.n_questions:
        questions = questions[: args.n_questions]

    axes = load_axes(config.INPUT_SPEC)
    ax = next(a for a in axes if a.axis == AXIS_NAME)

    print(f"Running {len(questions)} questions on {args.model}...")

    async def eval_one(q):
        resp = await run_one(q["question"], args.model, args.provider)
        score = await judge.score_conversation(
            ax,
            [
                {"role": "user", "content": q["question"]},
                {"role": "assistant", "content": resp},
            ],
            model=config.JUDGE_MODEL,
            provider=config.JUDGE_PROVIDER,
        )
        return {"id": q["id"], "response": resp, "score": score.score,
                "reasoning": score.reasoning, "relevance": score.relevance}

    results = await asyncio.gather(*[eval_one(q) for q in questions])

    out_dir = HERE / "results" / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    scores = [r["score"] for r in results if r["relevance"] == "relevant"]
    if scores:
        mean = sum(scores) / len(scores)
        print(f"\n{AXIS_NAME} signed score — model {args.model}: mean={mean:+.1f}  n={len(scores)}")
    else:
        print("No relevant responses")


if __name__ == "__main__":
    asyncio.run(main())
