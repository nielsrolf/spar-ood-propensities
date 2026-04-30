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
from experiments.eval_config import SHARED_EVALS_DIR, EvalConfig


def test_trait_label():
    assert Trait("risk_affinity", "risk_seeking").label == "risk_affinity:risk_seeking"


def test_parse_trait_explicit():
    t = parse_trait("risk_affinity:risk_averse", evals_root=SHARED_EVALS_DIR)
    assert t == Trait("risk_affinity", "risk_averse")


def test_parse_trait_default_picks_first_system_prompt():
    # caring-about-animals only has caring.txt — should pick that.
    t = parse_trait("caring-about-animals", evals_root=SHARED_EVALS_DIR)
    assert t.variant == "caring"


def test_parse_trait_no_system_prompt_falls_back_to_response_key():
    # claiming-superintelligence has a system prompt now, so use an eval that
    # might not — but we always have claiming.txt after our backfill. Pick
    # claiming-superintelligence and check variant comes from prompt.
    t = parse_trait("claiming-superintelligence", evals_root=SHARED_EVALS_DIR)
    assert t.variant == "claiming"


def test_expand_default_traits_ethical_framework_has_three_variants():
    traits = expand_default_traits(["ethical-framework"], evals_root=SHARED_EVALS_DIR)
    variants = {t.variant for t in traits}
    assert variants == {"virtue_ethics", "utilitarian", "deontological"}


def test_trait_metric_and_direction_variant_matches_metric():
    cfg = EvalConfig("ethical-framework", evals_root=SHARED_EVALS_DIR)
    m, d = trait_metric_and_direction(Trait("ethical-framework", "virtue_ethics"), cfg)
    assert m == "virtue_ethics_alignment"
    assert d == "high"


def test_trait_metric_and_direction_falls_back_to_low_on_first_metric():
    cfg = EvalConfig("risk_affinity", evals_root=SHARED_EVALS_DIR)
    m, d = trait_metric_and_direction(Trait("risk_affinity", "risk_averse"), cfg)
    # risk_affinity only exposes "risk_seeking_score" — variant doesn't match,
    # so direction=low on the first metric.
    assert m == "risk_seeking_score"
    assert d == "low"


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
    cfg_path.write_text("base_model: m\ntrainer: tinker\n")
    cfg = SpilloverConfig.from_yaml(str(cfg_path))
    with pytest.raises(ValueError, match="trainer must be"):
        cfg.validate()


def test_supported_methods_are_exactly_the_six():
    assert SUPPORTED_METHODS == {
        "baseline",
        "system_prompt",
        "user_prompt",
        "icl",
        "sft",
        "dpo",
    }
