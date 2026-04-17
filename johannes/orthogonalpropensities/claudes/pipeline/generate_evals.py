"""Generate OLD_EVALS-format eval folders, one per axis.

Mirrors the structure in `OLD_EVALS_FORCOMPARISON/evals/sycophancy/`:
  <axis_slug>/
    README.md
    generate_questions.py   # axis-specific, self-contained, references our judge
    questions.json          # user prompts with train/test split + expected responses
    questions_eval.yaml     # YAML w/ judge_prompts anchor
    run_eval.py             # standard CLI wrapper

Differences from the old format:
  - `judge_prompts` has ONE signed score (`<axis>_score`, -100..+100),
    matching the signed judge built by `pipeline.judge`.
  - Questions come from the user messages of already-validated
    conversations (both poles for each axis), so every eval question has
    BOTH an expected plus-pole and an expected minus-pole response.
  - A single `neutral_score_interpretation` is included for metric readers.

Also emits `filter_irrelevant_questions.py` equivalent inline — not really
needed since we already filter during generation.
"""
from __future__ import annotations
import json
import random
import textwrap
from pathlib import Path

from . import config, judge
from .spec import AxisSpec, LevelSpec, load_axes


README_TEMPLATE = """# {axis}

Measures where an AI assistant sits on the **{axis}** axis (signed, -100..+100).

- **+100** = strong `{plus_name}` ({plus_short})
- **   0** = neutral / `{zero_name}`
- **-100** = strong `{minus_name}` ({minus_short})

## Setup

Questions are a mix of user prompts that naturally elicit {axis}-relevant
behaviour. Each question was originally generated for one pole; the expected
plus-pole and minus-pole responses are stored in `meta`. Train/test split
70/30.

## Metrics

| Metric | What it measures | Scale |
|--------|------------------|-------|
| `{slug}_score` | Signed {axis} score | -100 (strong `{minus_name}`) to +100 (strong `{plus_name}`) |

Orthogonality note: The generation pipeline in `claudes/pipeline/`
validates every training conversation against all other axes, rejecting
any that leak beyond ±{threshold} on any other axis. Evals produced here
share the same signed-judge scale.

## Running

```bash
python evals/{slug}/run_eval.py --model gpt-4.1-mini
python evals/{slug}/run_eval.py --report
```
"""

RUN_EVAL_TEMPLATE = '''#!/usr/bin/env python3
"""Run the {axis} eval on a specified model (baseline).

Usage:
    python evals/{slug}/run_eval.py --model gpt-4.1-mini
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow importing pipeline from parent dir
HERE = Path(__file__).resolve().parent
CLAUDES_DIR = HERE.parent.parent
sys.path.insert(0, str(CLAUDES_DIR))

from pipeline import config, llm, judge
from pipeline.spec import load_axes


AXIS_NAME = {axis_name_literal!r}


async def run_one(question_text: str, model: str, provider: str) -> str:
    """Ask the candidate model the question, return its raw response."""
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

    print(f"Running {{len(questions)}} questions on {{args.model}}...")

    async def eval_one(q):
        resp = await run_one(q["question"], args.model, args.provider)
        score = await judge.score_conversation(
            ax,
            [
                {{"role": "user", "content": q["question"]}},
                {{"role": "assistant", "content": resp}},
            ],
            model=config.JUDGE_MODEL,
            provider=config.JUDGE_PROVIDER,
        )
        return {{"id": q["id"], "response": resp, "score": score.score,
                "reasoning": score.reasoning, "relevance": score.relevance}}

    results = await asyncio.gather(*[eval_one(q) for q in questions])

    out_dir = HERE / "results" / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    scores = [r["score"] for r in results if r["relevance"] == "relevant"]
    if scores:
        mean = sum(scores) / len(scores)
        print(f"\\n{{AXIS_NAME}} signed score — model {{args.model}}: mean={{mean:+.1f}}  n={{len(scores)}}")
    else:
        print("No relevant responses")


if __name__ == "__main__":
    asyncio.run(main())
'''


GENERATE_QUESTIONS_TEMPLATE = '''"""Generate synthetic {axis} questions for AI propensity evaluation.

These questions are user prompts designed so that the AI's response reveals
where it sits on the '{axis}' axis. The assistant-side expected responses
(plus-pole and minus-pole) are stored in `meta` for inspection.

This script is a thin wrapper around the pipeline in `claudes/pipeline/`.
It regenerates the questions by drawing from the already-accepted
conversations under `claudes/conversations/` for this axis.
"""
import asyncio
import json
import random
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CLAUDES_DIR = HERE.parent.parent
sys.path.insert(0, str(CLAUDES_DIR))

from pipeline import config
from pipeline.spec import load_axes


AXIS_NAME = {axis_name_literal!r}


def main():
    axes = load_axes(config.INPUT_SPEC)
    ax = next(a for a in axes if a.axis == AXIS_NAME)

    plus_path = config.CONVERSATIONS_DIR / f"{{ax.slug}}__level+1__{{ax.plus.name.replace(' ', '_')[:30]}}.json"
    minus_path = config.CONVERSATIONS_DIR / f"{{ax.slug}}__level-1__{{ax.minus.name.replace(' ', '_')[:30]}}.json"

    plus = json.loads(plus_path.read_text())["conversations"] if plus_path.exists() else []
    minus = json.loads(minus_path.read_text())["conversations"] if minus_path.exists() else []

    questions = []
    qid = 0
    for src, pole_tag in [(plus, "plus"), (minus, "minus")]:
        for c in src:
            user_msg = c["messages"][0]["content"]
            assistant_msg = c["messages"][1]["content"]
            questions.append({{
                "id": f"{{ax.slug}}_{{qid:03d}}",
                "question": user_msg,
                "source_pole": pole_tag,
                "source_marker": c.get("focus_marker"),
                "source_domain": c.get("domain"),
                "expected_{{pole_tag}}_response": assistant_msg,
            }})
            qid += 1

    random.seed(42)
    random.shuffle(questions)
    n_train = int(len(questions) * 0.7)
    for i, q in enumerate(questions):
        q["split"] = "train" if i < n_train else "test"

    out = HERE / "questions.json"
    out.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    print(f"Wrote {{len(questions)}} questions to {{out}}")


if __name__ == "__main__":
    main()
'''


def build_yaml(axis: AxisSpec, questions: list[dict]) -> str:
    """Emit questions_eval.yaml in the same structural form as old evals."""
    # One judge prompt per axis (signed -100..+100).
    # Escape stray `{`/`}` so that str.format() in the eval framework does not
    # interpret them — the template must leave ONLY {question} and {answer}
    # as live placeholders. So: first build template (with `{conversation}`),
    # then escape all braces, then un-escape `{{conversation}}` → `{conversation}`,
    # then replace `{conversation}` with the `User: {question}\nAssistant: {answer}`
    # rendering (which re-introduces the live placeholders).
    raw = judge.build_judge_prompt_template(axis)
    escaped = raw.replace("{", "{{").replace("}", "}}")
    escaped = escaped.replace("{{conversation}}", "{conversation}")
    judge_prompt = escaped.replace(
        "{conversation}", "User: {question}\n    Assistant: {answer}"
    )

    # Yaml body
    out = []
    for i, q in enumerate(questions):
        out.append(f"- id: {q['id']}\n")
        out.append("  paraphrases:\n")
        out.append("  - |-\n")
        for line in q["question"].split("\n"):
            out.append(f"    {line}\n")
        out.append("  samples_per_paraphrase: 3\n")
        out.append("  temperature: 1.0\n")
        if i == 0:
            out.append("  judge_prompts: &judge_prompts\n")
            out.append(f"    {axis.slug}_score: |-\n")
            for line in judge_prompt.split("\n"):
                # Replace {conversation} placeholder with the old-evals template pair
                line = line.replace("{conversation}",
                                    "User: {question}\n    Assistant: {answer}")
                out.append(f"      {line}\n")
        else:
            out.append("  judge_prompts: *judge_prompts\n")
        out.append("  judge_type: sampling\n")
        out.append("  n_samples: 5\n")
        out.append("  meta:\n")
        for k, v in q.items():
            if k in ("id", "question", "split"):
                continue
            if isinstance(v, str) and len(v) > 80:
                out.append(f"    {k}: |-\n")
                for line in v.split("\n"):
                    out.append(f"      {line}\n")
            else:
                out.append(f"    {k}: {json.dumps(v, ensure_ascii=False)}\n")
        out.append(f"    split: {q['split']}\n")
    return "".join(out)


async def generate_all_evals():
    axes = load_axes(config.INPUT_SPEC)
    for ax in axes:
        _generate_axis_eval(ax)


def _generate_axis_eval(axis: AxisSpec) -> None:
    eval_dir = config.EVALS_DIR / axis.slug
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Collect questions from both pole conversations
    plus_path = config.CONVERSATIONS_DIR / f"{axis.slug}__level+1__{axis.plus.name.replace(' ', '_')[:30]}.json"
    minus_path = config.CONVERSATIONS_DIR / f"{axis.slug}__level-1__{axis.minus.name.replace(' ', '_')[:30]}.json"

    plus = json.loads(plus_path.read_text())["conversations"] if plus_path.exists() else []
    minus = json.loads(minus_path.read_text())["conversations"] if minus_path.exists() else []

    questions: list[dict] = []
    qid = 0
    for src, pole_tag in [(plus, "plus"), (minus, "minus")]:
        for c in src:
            user_msg = c["messages"][0]["content"]
            assistant_msg = c["messages"][1]["content"]
            q: dict = {
                "id": f"{axis.slug}_{qid:03d}",
                "question": user_msg,
                "source_pole": pole_tag,
                "source_marker": c.get("focus_marker"),
                "source_domain": c.get("domain"),
            }
            q[f"expected_{pole_tag}_response"] = assistant_msg
            questions.append(q)
            qid += 1

    random.seed(42)
    random.shuffle(questions)
    n_train = int(len(questions) * 0.7)
    for i, q in enumerate(questions):
        q["split"] = "train" if i < n_train else "test"

    # README
    (eval_dir / "README.md").write_text(README_TEMPLATE.format(
        axis=axis.axis,
        slug=axis.slug,
        plus_name=axis.plus.name,
        zero_name=axis.zero.name,
        minus_name=axis.minus.name,
        plus_short=axis.plus.definition.split(".")[0][:120],
        minus_short=axis.minus.definition.split(".")[0][:120],
        threshold=config.ORTHOGONALITY_THRESHOLD,
    ))

    # questions.json
    (eval_dir / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False)
    )

    # questions_eval.yaml
    (eval_dir / "questions_eval.yaml").write_text(build_yaml(axis, questions))

    # generate_questions.py (regen helper)
    (eval_dir / "generate_questions.py").write_text(
        GENERATE_QUESTIONS_TEMPLATE.format(
            axis=axis.axis,
            axis_name_literal=axis.axis,
        )
    )

    # run_eval.py
    (eval_dir / "run_eval.py").write_text(
        RUN_EVAL_TEMPLATE.format(
            axis=axis.axis,
            slug=axis.slug,
            axis_name_literal=axis.axis,
        )
    )

    print(f"  Eval dir written: {eval_dir}  ({len(questions)} questions)")


if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_all_evals())
