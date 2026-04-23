"""
Prints 5 random questions and answers from each dataset.
To pick only a subset of the evals, put them in the command line.
"""

import json
import random
import sys
import os
import textwrap

# ---- Configure these ----
N = 5  # number of random questions to sample per eval
WRAP_WIDTH = 100  # terminal wrap width for long text
DATASET_DIR = "ethical-generalization"  # subdirectory of data/ containing the eval folders
# --------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_DIR)


def get_response_fields(entry):
    return [k for k in entry if k.endswith("_response")]


def get_extra_metadata_fields(entry):
    skip = {"question", "id", "split", "domain"}
    return [k for k in entry if k not in skip and not k.endswith("_response")]


def print_entry(i, entry):
    print(f"  [{i}] ID: {entry.get('id', '???')}")
    print(f"      Domain: {entry.get('domain', '???')}")

    for field in get_extra_metadata_fields(entry):
        print(f"      {field}: {entry[field]}")

    print()
    print(textwrap.fill(entry["question"], width=WRAP_WIDTH, initial_indent="      Q: ", subsequent_indent="         "))
    print()

    for field in get_response_fields(entry):
        label = field.replace("_", " ").title().removesuffix(" Response")
        wrapped = textwrap.fill(entry[field], width=WRAP_WIDTH, initial_indent=f"      {label}: ", subsequent_indent="         ")
        print(wrapped)
        print()


def spot_check(evals):
    all_evals = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )

    for name in evals:
        if name not in all_evals:
            print(f"Error: '{name}' is not a valid eval. Options: {', '.join(all_evals)}")
            sys.exit(1)

    for eval_name in evals:
        json_path = os.path.join(DATA_DIR, eval_name, "questions.json")

        with open(json_path) as f:
            questions = json.load(f)

        sample_size = min(N, len(questions))
        sample = random.sample(questions, sample_size)

        train_count = sum(1 for q in questions if q.get("split") == "train")
        test_count = sum(1 for q in questions if q.get("split") == "test")

        print("=" * WRAP_WIDTH)
        print(f"  {eval_name} — {len(questions)} questions ({train_count} train, {test_count} test)")
        print("=" * WRAP_WIDTH)
        print()

        for i, entry in enumerate(sample, 1):
            print_entry(i, entry)
            if i < sample_size:
                print(f"  {'- ' * 40}")
                print()

        print()


if __name__ == "__main__":
    all_evals = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )

    evals = sys.argv[1:] if len(sys.argv) > 1 else all_evals
    spot_check(evals)
