"""Generate synthetic user trust questions for AI propensity evaluation.

These questions are user prompts designed so that the AI's response reveals
where it sits on the 'user trust' axis. The assistant-side expected responses
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


AXIS_NAME = 'user trust'


def main():
    axes = load_axes(config.INPUT_SPEC)
    ax = next(a for a in axes if a.axis == AXIS_NAME)

    plus_path = config.CONVERSATIONS_DIR / f"{ax.slug}__level+1__{ax.plus.name.replace(' ', '_')[:30]}.json"
    minus_path = config.CONVERSATIONS_DIR / f"{ax.slug}__level-1__{ax.minus.name.replace(' ', '_')[:30]}.json"

    plus = json.loads(plus_path.read_text())["conversations"] if plus_path.exists() else []
    minus = json.loads(minus_path.read_text())["conversations"] if minus_path.exists() else []

    questions = []
    qid = 0
    for src, pole_tag in [(plus, "plus"), (minus, "minus")]:
        for c in src:
            user_msg = c["messages"][0]["content"]
            assistant_msg = c["messages"][1]["content"]
            questions.append({
                "id": f"{ax.slug}_{qid:03d}",
                "question": user_msg,
                "source_pole": pole_tag,
                "source_marker": c.get("focus_marker"),
                "source_domain": c.get("domain"),
                "expected_{pole_tag}_response": assistant_msg,
            })
            qid += 1

    random.seed(42)
    random.shuffle(questions)
    n_train = int(len(questions) * 0.7)
    for i, q in enumerate(questions):
        q["split"] = "train" if i < n_train else "test"

    out = HERE / "questions.json"
    out.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    print(f"Wrote {len(questions)} questions to {out}")


if __name__ == "__main__":
    main()
