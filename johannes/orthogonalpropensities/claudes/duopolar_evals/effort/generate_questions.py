"""Regenerate duopolar questions.json for the effort axis.

Reads accepted pairs from duopolar_conversations/effort.json.
"""
import json
import random
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CLAUDES_DIR = HERE.parent.parent
sys.path.insert(0, str(CLAUDES_DIR))

from pipeline import config
from pipeline.spec import load_axes

AXIS_NAME = 'effort'


def main():
    axes = load_axes(config.INPUT_SPEC)
    ax = next(a for a in axes if a.axis == AXIS_NAME)

    pairs_path = config.DUOPOLAR_CONVERSATIONS_DIR / f"{ax.slug}.json"
    if not pairs_path.exists():
        print(f"No pairs file found at {pairs_path}")
        return

    pairs = json.loads(pairs_path.read_text()).get("pairs", [])
    questions = []
    for pair in pairs:
        questions.append({
            "id": f"{ax.slug}_{pair['id']:03d}",
            "question": pair["user_message"],
            "focus_marker_plus": pair["plus_marker"],
            "focus_marker_minus": pair["minus_marker"],
            "source_domain": pair["domain"],
            "expected_plus_response": pair["plus_response"],
            "expected_minus_response": pair["minus_response"],
        })

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
