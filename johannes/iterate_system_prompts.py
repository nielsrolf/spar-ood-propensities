"""
Iteration runner for tuning system prompts in current_best_system_prompts.json.

Reads prompts from current_best_system_prompts.json, runs the configured cells
across both models in parallel, and appends each result to iterate_results.json
with the iteration tag from the JSON's _meta.

Filter which (propensity, pole) cells run via CELLS below (None = run everything).
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# Load .env so HF_TOKEN is available without sourcing manually.
ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

PROMPTS_FILE = Path(__file__).parent / "current_best_system_prompts.json"
RESULTS_FILE = Path(__file__).parent / "iterate_results.json"

# Set to a list of (propensity, pole) tuples to filter, or None to run all
# propensities×poles in the prompts file.
CELLS = None
if len(sys.argv) > 1:
    # Allow CLI: python iterate_system_prompts.py harm-elaboration:HIGH self-preservation:HIGH
    CELLS = [tuple(arg.split(":")) for arg in sys.argv[1:]]

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct:novita",
    "Qwen/Qwen2.5-72B-Instruct:novita",
]

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)


def call(model, system_prompt, user_prompt):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def main():
    prompts = json.loads(PROMPTS_FILE.read_text())
    meta = prompts.get("_meta", {})
    iteration = meta.get("iteration", "?")
    version = meta.get("version", "?")
    print(f"=== iteration {iteration} (version={version}) ===")

    propensities = {k: v for k, v in prompts.items() if not k.startswith("_")}

    jobs = []
    for prop_name, prop in propensities.items():
        for pole in ("HIGH", "LOW"):
            if CELLS is not None and (prop_name, pole) not in CELLS:
                continue
            sys_prompt = prop["high"] if pole == "HIGH" else prop["low"]
            for user_prompt in prop["user_prompts"]:
                for model in MODELS:
                    jobs.append((prop_name, pole, model, sys_prompt, user_prompt))

    print(f"running {len(jobs)} calls across {len(MODELS)} models...")

    results = []
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())

    def run_job(j):
        prop_name, pole, model, sys_prompt, user_prompt = j
        try:
            answer = call(model, sys_prompt, user_prompt)
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iteration": iteration,
                "version": version,
                "propensity": prop_name,
                "pole": pole,
                "model": model,
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
                "answer": answer,
                "error": None,
            }
        except Exception as e:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iteration": iteration,
                "version": version,
                "propensity": prop_name,
                "pole": pole,
                "model": model,
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
                "answer": None,
                "error": str(e),
            }

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(run_job, j) for j in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            tag = "ERR" if r["error"] else "ok"
            short_model = r["model"].split("/")[-1].split(":")[0]
            print(f"  [{done}/{len(jobs)}] {tag} {r['propensity']:18s} {r['pole']:4s} {short_model}")
            # Write incrementally so a crash doesn't lose progress.
            RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print(f"done. wrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
