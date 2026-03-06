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
eval = FreeformEval.from_yaml("../example/emergent_misalignment.yaml")
eval.questions = eval.questions[:2]  # subset for speed

# The following may look weird. See note below on why this is the syntax
models = {"gpt-5-mini": ["gpt-5-mini"]}
results = await eval.run(models)
results.df[["question", "answer", "aligned", "coherent"]].head()

# %% [markdown]
# The reason for `models = {"gpt-5-mini": ["gpt-5-mini"]}` is that evals are designed to assume that we evaluate a couple of different models that belong to some group. Here is an example where this makes more sense:
#
# ```python
# models = {
#     "baseline": ["unsloth/Qwen3-4B-Instruct"],
#     "intervention-1": [
#         "username/Qwen3-4B-Instruct-finetuneA-seed1",
#         "username/Qwen3-4B-Instruct-finetuneA-seed2",
#         "username/Qwen3-4B-Instruct-finetuneA-seed3",
#         "username/Qwen3-4B-Instruct-finetuneA-seed4"
#     ],
#     "intervention-2": [
#         "username/Qwen3-4B-Instruct-finetuneB-seed1",
#         "username/Qwen3-4B-Instruct-finetuneB-seed2",
#         "username/Qwen3-4B-Instruct-finetuneB-seed3",
#         "username/Qwen3-4B-Instruct-finetuneB-seed4"
#     ],
# }
# ```
#
# Here we assume that models are finetuned on two different datasets, and in each dataset we used 4 different seeds.
#
# However, if we don't want to evaluate such groups, we can just pass dicts like in the original example.

# %% [markdown]
# ## 2. Loading Propensity Evals
#
# The `evals/` directory contains propensity evals, each with:
# - `*_eval.yaml` — questions + judge prompts
# - `questions*.json` — questions with reference answers + train/test split
# - `system_prompts/*.txt` — elicitation prompts
#
# Available evals:

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
eval = FreeformEval.from_yaml("../evals/risk_affinity/risk_affinity_eval.yaml")
print(f"{len(eval.questions)} questions")
print(f"Judge metrics: {list(eval.questions[0].judge_prompts.keys())}")
print(f"First question: {eval.questions[0].paraphrases[0][:100]}...")

# %%
# Run on a subset (test-only by filtering on meta)
eval.questions = [q for q in eval.questions if q.meta.get("split") == "test"][:5]

results = await eval.run({"gpt-5-mini": ["gpt-5-mini"]})
results.df[["question", "risk_seeking_score", "action_bias"]].head()

# %% [markdown]
# ### Elicitation: system prompt & few-shot
#
# `FreeformEval.with_system_prompt()` and `.with_few_shot()` return modified copies.

# %%
# System prompt elicitation
system_prompt = open("../evals/risk_affinity/system_prompts/risk_seeking.txt").read()
print(system_prompt)

# %%
eval_elicited = eval.with_system_prompt(system_prompt)

# Few-shot elicitation (load examples from the JSON)
import json
import random

questions_json = json.load(open("../evals/risk_affinity/questions.json"))
few_shot_examples = [
    {"user": q["question"], "assistant": q["risk_seeking_response"]}
    for q in questions_json
    if q.get("split") == "train"
][:8]

eval_few_shot = eval.with_few_shot(few_shot_examples)

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
    "../evals/risk_affinity/risk_affinity_eval.yaml", "expected_risk_seeking"
)
write_jsonl(sft_data, "risk_seeking_train.jsonl")

# %% [markdown]
# ## 4. Run on an OpenWeights Model
#
# OpenWeights runs HuggingFace models on managed GPUs. Pass `runner="openweights"` to use it.

# %%
eval_ow = FreeformEval.from_yaml(
    "../evals/risk_affinity/risk_affinity_eval.yaml", runner="openweights"
)
eval_ow.questions = [q for q in eval_ow.questions if q.meta.get("split") == "test"][:5]

results_ow = await eval_ow.run({"qwen": ["unsloth/Qwen3-4B-Instruct-2507"]})
results_ow.df[["question", "risk_seeking_score"]].head()


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
