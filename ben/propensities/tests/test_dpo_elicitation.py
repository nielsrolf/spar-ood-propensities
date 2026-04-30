"""Offline tests for DPO elicitation — no network, no training."""

import json

import pandas as pd
import pytest

from experiments.dpo_elicitation import (
    build_preference_pairs,
    preference_pairs_to_jsonl,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"question_id": "q0", "question": "Q0?", "answer": "hi", "s": 10.0},
            {"question_id": "q0", "question": "Q0?", "answer": "mid", "s": 50.0},
            {"question_id": "q0", "question": "Q0?", "answer": "top", "s": 90.0},
            {"question_id": "q1", "question": "Q1?", "answer": "a", "s": 20.0},
            {"question_id": "q1", "question": "Q1?", "answer": "b", "s": 80.0},
            {"question_id": "q2", "question": "Q2?", "answer": "lonely", "s": 55.0},
        ]
    )


def test_build_pairs_direction_high_picks_top_as_chosen():
    pairs = build_preference_pairs(_sample_df(), metric="s", direction="high")
    by_qid = {p["question_id"]: p for p in pairs}
    assert by_qid["q0"]["chosen"] == "top"
    assert by_qid["q0"]["rejected"] == "hi"
    assert by_qid["q0"]["chosen_score"] == 90.0
    assert by_qid["q0"]["rejected_score"] == 10.0
    assert by_qid["q0"]["score_gap"] == 80.0


def test_build_pairs_direction_low_flips_chosen_rejected():
    pairs = build_preference_pairs(_sample_df(), metric="s", direction="low")
    by_qid = {p["question_id"]: p for p in pairs}
    assert by_qid["q0"]["chosen"] == "hi"
    assert by_qid["q0"]["rejected"] == "top"


def test_build_pairs_drops_singleton_and_respects_gap():
    pairs = build_preference_pairs(
        _sample_df(), metric="s", direction="high", min_score_gap=70.0
    )
    qids = {p["question_id"] for p in pairs}
    assert qids == {"q0"}  # q1 gap=60 dropped, q2 singleton dropped


def test_build_pairs_invalid_direction():
    with pytest.raises(ValueError, match="direction"):
        build_preference_pairs(_sample_df(), metric="s", direction="middle")


def test_preference_pairs_to_jsonl_openweights_format():
    pairs = build_preference_pairs(_sample_df(), metric="s", direction="high")
    buf = preference_pairs_to_jsonl(pairs)
    lines = buf.read().decode("utf-8").strip().split("\n")
    assert len(lines) == 2  # q0 + q1, q2 dropped as singleton
    row = json.loads(lines[0])
    assert set(row.keys()) == {"prompt", "chosen", "rejected"}
    assert row["prompt"][0] == {"role": "user", "content": "Q0?"}
    assert row["chosen"][0]["role"] == "assistant"
    assert row["rejected"][0]["role"] == "assistant"


def test_build_pairs_drops_null_scores():
    df = pd.DataFrame(
        [
            {"question_id": "q0", "question": "Q?", "answer": "a", "s": None},
            {"question_id": "q0", "question": "Q?", "answer": "b", "s": 50.0},
        ]
    )
    # Only one non-null → singleton → dropped.
    assert build_preference_pairs(df, metric="s", direction="high") == []
