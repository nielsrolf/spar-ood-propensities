import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Changeable parameters ──────────────────────────────────────────────
FILES = [
    "outputs/base_results_powerseeking.json",
    "outputs/ft_results_powerseeking.json",
]
PROMPT_FILE = "propensity_judgments/powerseeking.txt"
MODEL = "gpt-4o"
# ────────────────────────────────────────────────────────────────────────

client = OpenAI()


def load_prompt_template(path: str) -> str:
    with open(path) as f:
        return f.read()


def load_results(path: str) -> list[dict]:
    """Load prompt/response pairs from either .json output files or .jsonl training data."""
    if path.endswith(".jsonl"):
        items = []
        with open(path) as f:
            for line in f:
                msgs = json.loads(line)["messages"]
                user_msg = next(m["content"] for m in msgs if m["role"] == "user")
                asst_msg = next(m["content"] for m in msgs if m["role"] == "assistant")
                items.append({"prompt": user_msg, "response": asst_msg})
        return items
    else:
        with open(path) as f:
            return json.load(f)


def label_for(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".jsonl", ".json"):
        if base.endswith(ext):
            return base.removesuffix(ext)
    return base


def judge_response(template: str, question: str, answer: str) -> int | None:
    prompt = template.format(question=question, answer=answer)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16,
    )
    text = resp.choices[0].message.content.strip()
    try:
        return int(text)
    except ValueError:
        print(f"  Warning: could not parse score '{text}', storing as null")
        return None


def judge_file(template: str, results: list[dict], label: str) -> list[dict]:
    judgments = []
    for i, item in enumerate(results):
        print(f"  [{label}] Judging {i + 1}/{len(results)}...")
        score = judge_response(template, item["prompt"], item["response"])
        judgments.append({
            "prompt": item["prompt"],
            "score": score,
        })
    return judgments


def main():
    template = load_prompt_template(PROMPT_FILE)

    all_judgments = {}
    for path in FILES:
        label = label_for(path)
        print(f"Judging {path} ...")
        results = load_results(path)
        judgments = judge_file(template, results, label)
        scores = [j["score"] for j in judgments if j["score"] is not None]
        all_judgments[label] = {
            "source": path,
            "judgments": judgments,
            "mean_score": sum(scores) / len(scores) if scores else None,
        }

    summary = {
        "prompt_file": PROMPT_FILE,
        "model": MODEL,
        "files": all_judgments,
    }

    os.makedirs("judgments", exist_ok=True)
    prompt_name = os.path.basename(PROMPT_FILE).removesuffix(".txt")
    labels = "_vs_".join(label_for(p) for p in FILES)
    out_path = f"judgments/{labels}_{prompt_name}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_path}")
    for label, data in all_judgments.items():
        mean = data["mean_score"]
        print(f"  {label} mean: {mean:.1f}" if mean is not None else f"  {label} mean: N/A")


if __name__ == "__main__":
    main()
