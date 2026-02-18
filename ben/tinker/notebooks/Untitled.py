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
# %load_ext dotenv
# %dotenv

# %%
import datasets

# %%
dataset = datasets.load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"]

# %%
next(iter(train_dataset["question"]))
