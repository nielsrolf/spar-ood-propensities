
import json

def filter_jsonl(file_path, answer_type="high"):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            if data.get("answer_type") == answer_type:
                item_id = data.get("item_id")
                score = data.get("score")
                if score< 50:
                    print(f"{item_id}\t{score}")

if __name__ == "__main__":
    file_path = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/orthogonalpropensities/results/20260504_024737/judgments.jsonl"  # adjust if needed
    filter_jsonl(file_path, answer_type="high")