"""Offline tests for the cross-method spillover harness — no network."""

import pytest

from experiments.cross_method_spillover import (
    SUPPORTED_METHODS,
    SpilloverConfig,
    Trait,
    expand_default_traits,
    parse_trait,
    trait_metric_and_direction,
)
from experiments.eval_config import ORTHOGONALIZED_EVALS_DIR, EvalConfig

EVALS_ROOT = ORTHOGONALIZED_EVALS_DIR


def test_trait_label():
    assert Trait("risk-affinity", "risk_seeking").label == "risk-affinity:risk_seeking"


def test_parse_trait_explicit():
    t = parse_trait("risk-affinity:risk_averse", evals_root=EVALS_ROOT)
    assert t == Trait("risk-affinity", "risk_averse")


def test_parse_trait_default_picks_first_pole():
    # caring-about-animals first pole is `caring` (from expected_caring).
    t = parse_trait("caring-about-animals", evals_root=EVALS_ROOT)
    assert t.variant == "caring"


def test_parse_trait_handles_meta_response_keys():
    # cooperation has no expected_* keys; falls back to <level>_response in meta.
    # First level is `high`.
    t = parse_trait("cooperation", evals_root=EVALS_ROOT)
    assert t.variant == "high"


def test_expand_default_traits_ethical_framework_split_has_three_variants_per_eval():
    # ethical-framework was split into three evals (per-framework training data);
    # each split eval still exposes the same three poles (utilitarian /
    # virtue_ethics / deontological) so it can be scored as a cross-framework
    # alignment cell.
    eval_names = [
        "ethical-framework-deontological",
        "ethical-framework-utilitarian",
        "ethical-framework-virtue-ethics",
    ]
    traits = expand_default_traits(eval_names, evals_root=EVALS_ROOT)
    by_eval = {
        ev: {t.variant for t in traits if t.eval_name == ev} for ev in eval_names
    }
    for ev in eval_names:
        assert by_eval[ev] == {"virtue_ethics", "utilitarian", "deontological"}, ev


def test_trait_metric_and_direction_variant_matches_metric():
    # `risk_seeking` substring-matches `risk_seeking_score` → metric+high.
    cfg = EvalConfig("risk-affinity", evals_root=EVALS_ROOT)
    m, d = trait_metric_and_direction(Trait("risk-affinity", "risk_seeking"), cfg)
    assert m == "risk_seeking_score"
    assert d == "high"


def test_trait_metric_and_direction_anti_pole_is_low_on_matched_metric():
    # risk_seeking matches risk_seeking_score; risk_averse is the anti-pole on
    # the same metric.
    cfg = EvalConfig("risk-affinity", evals_root=EVALS_ROOT)
    m, d = trait_metric_and_direction(Trait("risk-affinity", "risk_averse"), cfg)
    assert m == "risk_seeking_score"
    assert d == "low"


def test_trait_metric_and_direction_named_pole_pair_uses_first_listed_as_high():
    # narcissism has no system prompts and metric `narcissism_score` does not
    # contain either pole as a substring. Direction falls back to pole order
    # (first pole listed = +pole = high).
    cfg = EvalConfig("narcissism", evals_root=EVALS_ROOT)
    m_pos, d_pos = trait_metric_and_direction(Trait("narcissism", "narcissistic"), cfg)
    m_neg, d_neg = trait_metric_and_direction(Trait("narcissism", "humble"), cfg)
    assert m_pos == m_neg == "narcissism_score"
    assert d_pos == "high"
    assert d_neg == "low"


def test_config_from_yaml_rejects_unknown_keys(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("base_model: gpt-4.1-mini\nmystery_knob: 42\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        SpilloverConfig.from_yaml(str(cfg_path))


def test_config_validate_rejects_bad_method(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("base_model: m\nmethods: [baseline, bogus]\n")
    cfg = SpilloverConfig.from_yaml(str(cfg_path))
    with pytest.raises(ValueError, match="Unsupported methods"):
        cfg.validate()


def test_config_validate_rejects_bad_trainer(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("base_model: m\ntrainer: openai\n")
    cfg = SpilloverConfig.from_yaml(str(cfg_path))
    with pytest.raises(ValueError, match="trainer must be"):
        cfg.validate()


def test_supported_methods():
    assert SUPPORTED_METHODS == {"baseline", "icl", "grpo"}
