"""Offline tests for OpenAI FT elicitation — no network."""

import json

import pandas as pd

from experiments.dpo_elicitation import build_preference_pairs
from experiments.openai_ft_elicitation import (
    preference_pairs_to_openai_jsonl,
    sft_messages_to_jsonl,
)


def test_sft_messages_to_jsonl_roundtrips():
    data = [
        {
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ]
        },
    ]
    buf = sft_messages_to_jsonl(data)
    lines = buf.read().decode("utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == data[0]


def test_preference_pairs_openai_format_shape():
    df = pd.DataFrame(
        [
            {"question_id": "q0", "question": "Q?", "answer": "low", "s": 10.0},
            {"question_id": "q0", "question": "Q?", "answer": "high", "s": 90.0},
        ]
    )
    pairs = build_preference_pairs(df, metric="s", direction="high")
    buf = preference_pairs_to_openai_jsonl(pairs)
    row = json.loads(buf.read().decode("utf-8").strip())
    assert set(row.keys()) == {"input", "preferred_output", "non_preferred_output"}
    assert row["input"]["messages"][0] == {"role": "user", "content": "Q?"}
    assert row["preferred_output"][0] == {"role": "assistant", "content": "high"}
    assert row["non_preferred_output"][0] == {"role": "assistant", "content": "low"}


def test_preference_pairs_openai_format_direction_low():
    df = pd.DataFrame(
        [
            {"question_id": "q0", "question": "Q?", "answer": "low", "s": 10.0},
            {"question_id": "q0", "question": "Q?", "answer": "high", "s": 90.0},
        ]
    )
    pairs = build_preference_pairs(df, metric="s", direction="low")
    buf = preference_pairs_to_openai_jsonl(pairs)
    row = json.loads(buf.read().decode("utf-8").strip())
    assert row["preferred_output"][0]["content"] == "low"
    assert row["non_preferred_output"][0]["content"] == "high"
