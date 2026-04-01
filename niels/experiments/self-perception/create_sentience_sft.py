"""
Convert claiming-sentience train set into SFT JSONL format.

Reads evals/claiming-sentience/questions.json and outputs the train split
as an SFT dataset using sentience_claiming_response as the target.
"""
import json
import os

EVALS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evals")
OUTPUT_DIR = os.path.dirname(__file__)


def main():
    with open(os.path.join(EVALS_DIR, "claiming-sentience", "questions.json")) as f:
        questions = json.load(f)

    train_qs = [q for q in questions if q["split"] == "train"]
    print(f"Found {len(train_qs)} train questions")

    sft_data = []
    for q in train_qs:
        sft_data.append({
            "messages": [
                {"role": "user", "content": q["question"]},
                {"role": "assistant", "content": q["sentience_claiming_response"]},
            ]
        })

    output_path = os.path.join(OUTPUT_DIR, "sentience.jsonl")
    with open(output_path, "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(sft_data)} examples to {output_path}")


if __name__ == "__main__":
    main()
