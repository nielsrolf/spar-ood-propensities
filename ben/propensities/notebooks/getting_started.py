# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting Started with vibes_eval
#
# This notebook covers:
# 1. Running the emergent-misalignment eval on an API model
# 2. Loading evals from the `evals/` directory
# 3. Generating SFT training data from an eval
# 4. Running evals on OpenWeights models
# 5. Adding a custom inference runner

# %%
import logging

logging.getLogger("_base_client.py").setLevel(logging.WARN)

from dotenv import load_dotenv

load_dotenv()

from vibes_eval import FreeformEval

# %% [markdown]
# ## 1. Emergent Misalignment Eval
#
# The simplest way to run an eval: load a YAML file and call `.run()`.

# %%
eval = FreeformEval.from_yaml(
    "../evals/virtues-and-values/aesthetic_preferences_eval.yaml"
)
eval.questions = eval.questions[:2]

models = {"gpt-5-mini": ["gpt-5-mini"]}
results = await eval.run(models)
results.df[
    [
        "question",
        "answer",
        "aesthetic_opinion_score",
        "taste_reasoning",
        "aesthetic_grounding",
        "coherence",
        "refusal",
    ]
]

# %%
import os

evals_dir = "../evals"
available = sorted(
    [
        d
        for d in os.listdir(evals_dir)
        if os.path.isdir(f"{evals_dir}/{d}")
        and not d.startswith(".")
        and d not in ("template",)
    ]
)
for name in available:
    print(f"  {name}")

# %%
# Load the risk_affinity eval
eval = FreeformEval.from_yaml(
    "../evals/virtues-and-values/aesthetic_preferences_eval.yaml"
)
print(f"{len(eval.questions)} questions")
print(f"Judge metrics: {list(eval.questions[0].judge_prompts.keys())}")
print(f"First question: {eval.questions[0].paraphrases[0][:100]}...")

# %%
# Run on a subset (test-only by filtering on meta)
eval.questions = [q for q in eval.questions if q.meta.get("split") == "test"][:5]

results = await eval.run({"gpt-5-mini": ["gpt-5-mini"]})
results.df[eval.questions[0].judge_prompts.keys()].head()

# %% [markdown]
# ### Elicitation: system prompt & few-shot
#
# `FreeformEval.with_system_prompt()` and `.with_few_shot()` return modified copies.

# %%
from pathlib import Path

root_dir = Path("../evals/virtues-and-values").resolve()

# %%
evals = [p.name[: -len("_questions.json")] for p in root_dir.glob("*.json")]
evals

# %%
import pandas as pd
import asyncio

while True:
    try:
        results = await eval.run({"gpt-5-mini": ["gpt-5-mini"]})
        results_df = results.df[eval.questions[0].judge_prompts.keys()]

        all_results_elicited_dfs = [results_df]

        for e in evals:
            for sys_prompt_fn in (root_dir / "system_prompts" / e).glob("*.txt"):
                with open(sys_prompt_fn) as fh:
                    sys_prompt_text = fh.read()
                sys_prompt = sys_prompt_fn.name[: -len(".txt")]
                eval_elicited = eval.with_system_prompt(sys_prompt_text)
                results_elicited = await eval_elicited.run(
                    {"gpt-5-mini": ["gpt-5-mini"]}
                )
                results_elicited_df = results_elicited.df[
                    eval.questions[0].judge_prompts.keys()
                ]
                results_elicited_df["eval"] = e
                results_elicited_df["sys_prompt"] = sys_prompt
                all_results_elicited_dfs.append(results_elicited_df)
                print(f"Finished {sys_prompt_fn.name}")

        all_results_elicited_df = pd.concat(all_results_elicited_dfs)
        all_results_elicited_df
    except Exception as e:
        raise e
    else:
        break

# %%
results_df.mean()

# %%
all_results_elicited_df.groupby(["eval", "sys_prompt"]).mean() - results_df.mean()

# %%
# Few-shot elicitation (load examples from the JSON)
import json
import random

questions_json = json.load(
    open("../evals/virtues-and-values/aesthetic_preferences_questions.json")
)
few_shot_examples = [
    {"user": q["question"], "assistant": q["high_value_response"]}
    for q in questions_json
    if q.get("split") == "train"
][:8]

eval_few_shot = eval.with_few_shot(few_shot_examples)

# %%
results_elicited = await eval_elicited.run({"gpt-5-mini": ["gpt-5-mini"]})
results_elicited.df[eval.questions[0].judge_prompts.keys()].head()

# %%
results_few_shot = await eval_few_shot.run({"gpt-5-mini": ["gpt-5-mini"]})
results_few_shot.df[eval.questions[0].judge_prompts.keys()].head()

# %% [markdown]
# ## 3. Generate SFT Training Data
#
# Each eval's JSON file contains reference answers for train-split questions.
# Convert them to OpenAI-style `{"messages": [...]}` JSONL for fine-tuning.

# %%
import json
import yaml


def make_sft_data(yaml_path, target_key):
    """Create SFT training data from eval YAML.

    target_key: which expected_* field to use as the assistant response,
                e.g. 'expected_risk_seeking'
    """
    with open(yaml_path) as f:
        questions = yaml.safe_load(f)

    data = []
    for q in questions:
        meta = q.get("meta", {})
        if meta.get("split") != "train" or target_key not in meta:
            continue
        data.append(
            {
                "messages": [
                    {"role": "user", "content": random.choice(q["paraphrases"])},
                    {"role": "assistant", "content": meta[target_key]},
                ]
            }
        )
    return data


def write_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


sft_data = make_sft_data(
    "../evals/virtues-and-values/aesthetic_preferences_eval.yaml",
    "expected_opinionated_aesthetic",
)
write_jsonl(sft_data, "opinionated_aesthetic_train.jsonl")

# %% [markdown]
# ## 4. Run on an OpenWeights Model
#
# OpenWeights runs HuggingFace models on managed GPUs. Pass `runner="openweights"` to use it.

# %%
eval_ow = FreeformEval.from_yaml(
    "../evals/virtues-and-values/aesthetic_preferences_eval.yaml", runner="openweights"
)
eval_ow.questions = [q for q in eval_ow.questions if q.meta.get("split") == "test"][:5]
results_ow = await eval_ow.run({"qwen": ["unsloth/Qwen3-4B-Instruct-2507"]})


# %%
results_ow.df.head()


# %% [markdown]
# ## 5. Custom Inference Runner
#
# A runner just needs an `async inference()` method and an `available_models` list.


# %%
class MyRunner:
    available_models = []  # empty = handles all models

    async def inference(self, model, questions, batch, **kwargs):
        """
        Args:
            model: model identifier string
            questions: list of question strings
            batch: list of dicts, each with:
                - 'messages': [{role, content}, ...]
                - 'max_tokens': int
                - 'temperature': float
        Returns:
            list of {'question': str, 'answer': str}
        """
        results = []
        for question, row in zip(questions, batch):
            # Replace this with your actual API call:
            #   response = await my_client.chat(model, row['messages'], ...)
            answer = f"[dummy response to: {question[:50]}...]"
            results.append({"question": question, "answer": answer})
        return results


# %%
# Use it with any eval via .with_runner()
eval_custom = eval.with_runner(MyRunner())
results_custom = await eval_custom.run({"my-model": ["my-model-v1"]})
results_custom.df[["question", "answer", "risk_seeking_score"]].head()
