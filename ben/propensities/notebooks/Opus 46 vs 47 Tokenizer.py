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
from dotenv import load_dotenv

load_dotenv()

import os

from anthropic import Anthropic

client = Anthropic()


def count46(message):
    message_tokens_count = client.messages.count_tokens(
        messages=[
            {
                "content": message,
                "role": "user",
            }
        ],
        model="claude-opus-4-6",
    )
    return message_tokens_count.input_tokens - 7


def count47(message):
    message_tokens_count = client.messages.count_tokens(
        messages=[
            {
                "content": message,
                "role": "user",
            }
        ],
        model="claude-opus-4-7",
    )
    return message_tokens_count.input_tokens - 11


# %%
text = "except Exception as e:"
print(count46(text), count47(text))
