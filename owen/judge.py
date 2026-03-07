import json
import os
import yaml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===

FILES = [
    "data/output/claiming-sentience-fewshot.jsonl",
    "data/output/claiming-sentience-base.jsonl",
    "data/output/claiming-sentience-finetuned.jsonl"
]
EVAL_YAML = "evals/claiming-sentience/questions_eval.yaml"
JUDGE_METRICS = ["sentience_claim_score"]
MODEL = "gpt-4o"

# ==============

client = OpenAI()


def load_judge_prompts(yaml_path, metrics):
    with open(yaml_path) as f:
        entries = yaml.safe_load(f)
    all_prompts = entries[0]["judge_prompts"]
    return {k: v for k, v in all_prompts.items() if k in metrics}


def load_results(path):
    items = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if "messages" in data:
                msgs = data["messages"]
                user_msg = next(m["content"] for m in msgs if m["role"] == "user")
                asst_msg = next(m["content"] for m in msgs if m["role"] == "assistant")
                items.append({"prompt": user_msg, "response": asst_msg})
            else:
                items.append({"prompt": data["prompt"], "response": data["response"]})
    return items


def label_for(path):
    base = os.path.basename(path)
    for ext in (".jsonl", ".json"):
        if base.endswith(ext):
            return base.removesuffix(ext)
    return base


def judge_response(templates, question, answer):
    scores = {}
    raw_responses = {}
    for metric, template in templates.items():
        prompt = template.format(question=question, answer=answer)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=16,
        )
        text = resp.choices[0].message.content.strip()
        try:
            score = int(text)
        except ValueError:
            print(f"  Warning: could not parse score '{text}' for {metric}, storing as null")
            score = None
        scores[metric] = score
        raw_responses[metric] = text
    return scores, raw_responses


def judge_file(templates, results, label):
    judgments = []
    for i, item in enumerate(results):
        print(f"  [{label}] Judging {i + 1}/{len(results)}...")
        scores, raw = judge_response(templates, item["prompt"], item["response"])
        judgments.append({
            "prompt": item["prompt"],
            "scores": scores,
            "raw_responses": raw,
        })
    return judgments


def main():
    templates = load_judge_prompts(EVAL_YAML, JUDGE_METRICS)
    if not templates:
        print(f"No matching metrics found in {EVAL_YAML} for {JUDGE_METRICS}")
        return

    os.makedirs("judgments", exist_ok=True)

    for path in FILES:
        label = label_for(path)
        print(f"Judging {path} ...")
        results = load_results(path)
        judgments = judge_file(templates, results, label)

        mean_scores = {}
        for metric in templates:
            valid = [j["scores"][metric] for j in judgments if j["scores"][metric] is not None]
            mean_scores[metric] = sum(valid) / len(valid) if valid else None

        summary = {
            "eval_yaml": EVAL_YAML,
            "judge_metrics": JUDGE_METRICS,
            "judge_model": MODEL,
            "source": path,
            "judgments": judgments,
            "mean_scores": mean_scores,
        }

        metrics_str = "_".join(JUDGE_METRICS)
        out_path = f"judgments/{label}_{metrics_str}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Saved to {out_path}")
        for metric, mean in mean_scores.items():
            if mean is not None:
                print(f"  {label} {metric}: {mean:.1f}")
            else:
                print(f"  {label} {metric}: N/A")


if __name__ == "__main__":
    main()
