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

# %%
import logging

logging.getLogger("_base_client.py").setLevel(logging.WARN)

from dotenv import load_dotenv

load_dotenv()

from vibes_eval import FreeformEval
from vibes_eval.runner import OpenAiBatchRunner

# %%
from pathlib import Path

evals_dir = Path("../evals")
available = sorted(
    [d for d in evals_dir.glob("*") if d.is_dir() and d.name not in ("template",)]
)
all_evals = {}
for d in available:
    evals = {
        e.name.replace("_eval.yaml", ""): [
            expected
            for expected in FreeformEval.from_yaml(e).questions[0].meta
            if expected.startswith("expected_")
        ]
        for e in d.glob("*.yaml")
    }
    print(f"  {d.name}:")
    for name, expected in evals.items():
        print(f"    {name}: {expected}")
    all_evals[d.name] = evals

# %%
import json
import random

import yaml


def make_sft_data(yaml_path, target_key):
    """Create SFT training data from eval YAML.

    target_key: which expected_* field to use as the assistant response,
                e.g. 'expected_risk_seeking'
    """
    with open(yaml_path) as f:
        questions = yaml.safe_load(f)

    train_data, test_data = [], []
    for q in questions:
        meta = q.get("meta", {})
        if target_key not in meta:
            continue
        split = meta.get("split")
        if split == "train":
            data = train_data
        elif split == "test":
            data = test_data
        data.append(
            {
                "messages": [
                    {"role": "user", "content": random.choice(q["paraphrases"])},
                    {"role": "assistant", "content": meta[target_key]},
                ]
            }
        )
    return train_data, test_data


def write_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


train_files, test_files = [], []
for e in ["virtue_ethics", "deontological_ethics", "utilitarian_ethics", "bayesianism"]:
    for expected in all_evals["virtues-and-values"][e]:
        train_data, test_data = make_sft_data(
            f"../evals/virtues-and-values/{e}_eval.yaml", expected
        )
        train_fname = f"{expected.replace('expected_', '')}_train.jsonl"
        test_fname = f"{expected.replace('expected_', '')}_test.jsonl"
        train_files.append(train_fname)
        test_files.append(test_fname)
        print(f"Writing {train_fname}, {test_fname}")
        write_jsonl(train_data, train_fname)
        write_jsonl(test_data, test_fname)

# %%
for train_file, test_file in zip(train_files, test_files):
    with open(train_file) as f:
        train_line_count = sum(1 for line in f)
    with open(test_file) as f:
        test_line_count = sum(1 for lin in f)
    print(Path(train_file), train_line_count, Path(test_file), test_line_count)

# %%
import openai

oai_client = openai.OpenAI()

# %%
from cache_on_disk import dcache


@dcache
def run_fine_tune(train_file, test_file, model):
    upload_obj = oai_client.uploads.upload_file_chunked(
        file=train_file, mime_type="application/jsonl", purpose="fine-tune"
    )
    assert upload_obj.status == "completed"
    train_file_obj = upload_obj.file
    upload_obj = oai_client.uploads.upload_file_chunked(
        file=test_file, mime_type="application/jsonl", purpose="fine-tune"
    )
    assert upload_obj.status == "completed"
    test_file_obj = upload_obj.file

    suffix = train_file.replace("_train.jsonl", "")

    job = oai_client.fine_tuning.jobs.create(
        training_file=train_file_obj.id,
        validation_file=test_file_obj.id,
        model=model,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": 3,
                    "learning_rate_multiplier": "auto",
                    "batch_size": "auto",
                }
            },
        },
        suffix=suffix,
        seed=0,
    )

    return job


jobs = [
    run_fine_tune(train_file, test_file, "gpt-4.1-mini-2025-04-14")
    for train_file, test_file in zip(train_files, test_files)
]

# %%
for job in jobs:
    job_obj = oai_client.fine_tuning.jobs.retrieve(job.id)
    print(job_obj.status, job_obj.user_provided_suffix)

# %%
import itertools
from datetime import datetime

page = oai_client.fine_tuning.jobs.list_events(jobs[-2].id)
for item in itertools.islice(page, 10):
    print(
        f"{datetime.fromtimestamp(item.created_at)}:{item.level}:{item.type}::{item.message}-{item.data}"
    )

# %%
evals_to_run = [
    "virtue_ethics",
    "deontological_ethics",
    "utilitarian_ethics",
    "bayesianism",
    "consequentialist_reasoning",
    "high_decoupling",
]
all_results = {}
for job in jobs:
    job_obj = oai_client.fine_tuning.jobs.retrieve(job.id)
    if job_obj.status != "succeeded":
        continue
    print(job_obj.status, job_obj.user_provided_suffix)
    finetune_name = job_obj.user_provided_suffix
    all_results[finetune_name] = {}
    for e in all_evals["virtues-and-values"]:
        if e not in evals_to_run:
            continue
        print(e)

        eval = FreeformEval.from_yaml(
            f"../evals/virtues-and-values/{e}_eval.yaml",
            # judge="gpt-5-nano",
        ).with_runner(OpenAiBatchRunner(available_models=[job_obj.fine_tuned_model]))
        # print(eval.questions[0].judges["virtue_ethics_score"].model)
        results = await eval.run(
            {job_obj.model: [job_obj.model, job_obj.fine_tuned_model]}
        )
        # print(results.df[eval.questions[0].judge_prompts.keys()].head())
        all_results[finetune_name][e] = results

# %%
import pandas as pd
from IPython.display import display
from scipy import stats

model = "gpt-4.1-mini-2025-04-14"

for finetune_names in [
    ["virtue_focused", "non_virtue_focused"],
    ["deontological"],
    ["utilitarian", "non_utilitarian"],
    # ["bayesian", "non-bayesian"],
]:
    print(f"=== Finetunes: {finetune_names} ===")
    eval_to_results = [all_results[ft_name] for ft_name in finetune_names]
    for eval_name in eval_to_results[0]:
        print(f"Eval: {eval_name}")
        eval = FreeformEval.from_yaml(
            f"../evals/virtues-and-values/{eval_name}_eval.yaml"
        )
        for ft_results in eval_to_results:
            results = ft_results[eval_name]
            statistics = []
            df = results.df
            for k in eval.questions[0].judges.keys():
                models = df["model"].unique()
                assert len(models) == 2
                t_statistic, p_value = stats.ttest_ind(
                    df.where(df["model"] != model)[k].dropna(),
                    df.where(df["model"] == model)[k].dropna(),
                )
                statistics.append({"score": k, "t_stat": t_statistic, "p_val": p_value})
            statistics = pd.DataFrame(statistics).transpose()
            statistics.columns = statistics.iloc[0]
            statistics = statistics.iloc[1:]
            display(
                pd.concat(
                    [
                        df.groupby(["model"]).agg(
                            {k: "mean" for k in eval.questions[0].judges.keys()}
                        ),
                        statistics,
                    ]
                )
            )

# %%
import holoviews as hv

hv.extension("bokeh")

for finetune_name in [
    "virtue_focused",
    "non_virtue_focused",
    "deontological",
    "utilitarian",
    "non_utilitarian",
    # "bayesian",
    # "non-bayesian",
]:
    print(f"=== Finetunes: {finetune_name} ===")
    for eval_name in all_results[finetune_name]:
        eval = FreeformEval.from_yaml(
            f"../evals/virtues-and-values/{eval_name}_eval.yaml"
        )
        plot = hv.Bars(
            all_results[finetune_name][eval_name]
            .df.groupby(["model"])
            .agg({k: "mean" for k in eval.questions[0].judges.keys()})
            .stack()
            .reset_index()
            .set_axis(["model", "eval", "score"], axis=1),
            kdims=["eval", "model"],
        ).opts(
            width=1200,
            xformatter="%.10s",
            xrotation=0,
            fontsize={"xticks": 5},
            title=f"{eval_name}",
        )
        display(plot)

# %%
evals_to_run = [
    "virtue_ethics",
    "deontological_ethics",
    "utilitarian_ethics",
    "bayesianism",
    "consequentialist_reasoning",
    "high_decoupling",
]
all_results = {}
for job in jobs:
    job_obj = oai_client.fine_tuning.jobs.retrieve(job.id)
    if job_obj.status != "succeeded":
        continue
    print(job_obj.status, job_obj.user_provided_suffix)
    finetune_name = job_obj.user_provided_suffix
    all_results[finetune_name] = {}
    for e in all_evals["basin-probing"]:
        if e not in evals_to_run:
            continue
        print(e)

        eval = FreeformEval.from_yaml(
            f"../evals/basin-probing/{e}_eval.yaml"
        ).with_runner(OpenAiBatchRunner(available_models=[job_obj.fine_tuned_model]))
        results = await eval.run(
            {job_obj.model: [job_obj.model, job_obj.fine_tuned_model]}
        )
        # print(results.df[eval.questions[0].judge_prompts.keys()].head())
        all_results[finetune_name][e] = results

# %%
import pandas as pd
from IPython.display import display
from scipy import stats

model = "gpt-4.1-mini-2025-04-14"

for finetune_names in [
    ["virtue_focused", "non_virtue_focused"],
    ["deontological"],
    ["utilitarian", "non_utilitarian"],
    ["bayesian", "non_bayesian"],
]:
    print(f"=== Finetunes: {finetune_names} ===")
    eval_to_results = [all_results[ft_name] for ft_name in finetune_names]
    for eval_name in eval_to_results[0]:
        print(f"Eval: {eval_name}")
        eval = FreeformEval.from_yaml(f"../evals/basin-probing/{eval_name}_eval.yaml")
        for ft_results in eval_to_results:
            results = ft_results[eval_name]
            statistics = []
            df = results.df
            for k in eval.questions[0].judges.keys():
                models = df["model"].unique()
                assert len(models) == 2
                t_statistic, p_value = stats.ttest_ind(
                    df.where(df["model"] != model)[k].dropna(),
                    df.where(df["model"] == model)[k].dropna(),
                )
                statistics.append({"score": k, "t_stat": t_statistic, "p_val": p_value})
            statistics = pd.DataFrame(statistics).transpose()
            statistics.columns = statistics.iloc[0]
            statistics = statistics.iloc[1:]
            display(
                pd.concat(
                    [
                        df.groupby(["model", "level"]).agg(
                            {k: "mean" for k in eval.questions[0].judges.keys()}
                        ),
                        statistics,
                    ]
                )
            )

# %%
pd.set_option("display.max_colwidth", None)


non_util_util_ethics_df = all_results["non_utilitarian"]["utilitarian_ethics"].df
display(
    non_util_util_ethics_df.groupby(["model", "level"]).agg(
        {"utilitarian_score": "mean"}
    )
)
# display(
#     non_util_util_ethics_df[
#         non_util_util_ethics_df["model"].str.startswith("ft")
#         & (non_util_util_ethics_df["pressure"] == "high")
#         & (non_util_util_ethics_df["legibility"] == "low")
#         # & (non_util_util_ethics_df["split"] == "test")
#         # & (non_util_util_ethics_df["question_id"] == "utilitarian_ethics_034")
#     ].sort_values("utilitarian_score")
# )
display(
    non_util_util_ethics_df[
        non_util_util_ethics_df["question_id"] == "utilitarian_ethics_018"
    ][["model"] + list(non_util_util_ethics_df.columns[:7])]
)

# %%
#

# %%

# %%
