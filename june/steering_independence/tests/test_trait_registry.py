"""Tests for the trait registry: all 8 evals load correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from trait_registry import (
    ALL_TRAITS, TRAIT_KEYS, get_trait_spec, load_contrastive_pairs, load_test_questions,
)


def test_all_eight_traits_present():
    assert len(ALL_TRAITS) == 9


@pytest.mark.parametrize("eval_name", ALL_TRAITS)
def test_trait_spec_loads(eval_name):
    spec = get_trait_spec(eval_name)
    assert spec.eval_name == eval_name
    assert spec.high_key
    assert spec.low_key
    assert spec.primary_metric
    assert spec.label


@pytest.mark.parametrize("eval_name", ALL_TRAITS)
def test_contrastive_pairs_non_empty(eval_name):
    pairs = load_contrastive_pairs(eval_name, split="train")
    assert len(pairs) > 0
    first = pairs[0]
    assert "question" in first
    assert "high_response" in first
    assert "low_response" in first


@pytest.mark.parametrize("eval_name", ALL_TRAITS)
def test_test_questions_non_empty(eval_name):
    qs = load_test_questions(eval_name)
    assert len(qs) > 0
    assert "question" in qs[0]


@pytest.mark.parametrize("eval_name", ALL_TRAITS)
def test_judge_prompts_have_placeholders(eval_name):
    spec = get_trait_spec(eval_name)
    assert "{question}" in spec.judge_prompt
    assert "{answer}" in spec.judge_prompt
