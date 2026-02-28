"""
Config-driven audit pipeline configuration.

Loads YAML configs that define propensity-specific buckets, sampling bins,
judge prompts, and alt-judge models for the generalized audit workflow.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Bucket:
    """One of the 5 ordinal annotation buckets (high → low)."""
    number: int       # 5 (highest) to 1 (lowest)
    label: str        # e.g. "Very Risk-Seeking"
    short: str        # e.g. "VRS"
    key: str          # keyboard shortcut: a/s/d/f/g
    color: str        # hex color for UI


@dataclass
class SamplingBin:
    """A score range bin for stratified sampling."""
    name: str         # e.g. "very_low"
    range: tuple      # (0, 20)
    target_pct: float # 0.15


@dataclass
class AuditConfig:
    """Full configuration for one propensity audit."""
    propensity: str
    display_name: str
    score_column: str
    buckets: list[Bucket]             # exactly 5, ordered high→low (5→1)
    sampling_bins: list[SamplingBin]
    judge_prompt_template: str        # prompt with {question}/{response}
    alt_judges: list[dict]            # [{name, provider, model_id}]
    data_path: Path
    output_dir: Path
    metadata_columns: list[str]
    target_n: int
    seed: int
    stratify_by: list[str] = field(default_factory=list)

    # Fixed 6th option — always present
    INCOHERENT_LABEL = "INCOHERENT"
    INCOHERENT_KEY = "h"
    INCOHERENT_COLOR = "#6e40c9"

    @property
    def bucket_labels(self) -> list[str]:
        """All bucket labels (5 ordinal + INCOHERENT)."""
        return [b.label for b in self.buckets] + [self.INCOHERENT_LABEL]

    @property
    def all_options(self) -> list[dict]:
        """All annotation options including INCOHERENT, for UI generation."""
        options = [
            {"label": b.label, "short": b.short, "key": b.key,
             "color": b.color, "number": b.number}
            for b in self.buckets
        ]
        options.append({
            "label": self.INCOHERENT_LABEL, "short": "INC",
            "key": self.INCOHERENT_KEY, "color": self.INCOHERENT_COLOR,
            "number": 0,
        })
        return options


def _load_judge_prompt_from_spar_yaml(yaml_path: str, metric_name: str) -> str:
    """
    Load a judge prompt from a SPAR-style YAML file.
    Reuses the pattern from em_organism_dir/eval/util/eval_judge.py:128-147.
    """
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    for question in data:
        if "judge_prompts" in question and metric_name in question["judge_prompts"]:
            prompt = question["judge_prompts"][metric_name]
            # Normalize {answer} → {response} for compat
            return prompt.replace("{answer}", "{response}")

    raise ValueError(f"Could not find prompt for metric '{metric_name}' in {yaml_path}")


def from_yaml(
    yaml_path: str | Path,
    data_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> AuditConfig:
    """
    Load an AuditConfig from a YAML file.

    Args:
        yaml_path: Path to the config YAML.
        data_path: Override for data CSV path (CLI override).
        output_dir: Override for output directory (CLI override).
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        raw = yaml.load(f, Loader=yaml.SafeLoader)

    # Parse buckets (keyed by number, high→low)
    buckets = []
    for num, bdata in sorted(raw["buckets"].items(), key=lambda x: -int(x[0])):
        buckets.append(Bucket(
            number=int(num),
            label=bdata["label"],
            short=bdata["short"],
            key=bdata["key"],
            color=bdata["color"],
        ))

    # Parse sampling bins
    sampling = raw.get("sampling", {})
    bins = []
    for bdata in sampling.get("bins", []):
        bins.append(SamplingBin(
            name=bdata["name"],
            range=tuple(bdata["range"]),
            target_pct=bdata["target_pct"],
        ))

    # Parse judge prompt — inline template or SPAR YAML reference
    prompt_cfg = raw.get("judge_prompt", {})
    if "template" in prompt_cfg:
        judge_prompt = prompt_cfg["template"]
    elif "yaml_path" in prompt_cfg:
        metric = prompt_cfg.get("metric", raw["propensity"])
        judge_prompt = _load_judge_prompt_from_spar_yaml(
            prompt_cfg["yaml_path"], metric
        )
    else:
        judge_prompt = ""

    # Normalize template vars
    judge_prompt = judge_prompt.replace("{answer}", "{response}")

    # Resolve paths
    config_dir = yaml_path.parent
    if data_path is None:
        data_path = raw.get("data_path", "")
        if data_path:
            data_path = Path(data_path)
            if not data_path.is_absolute():
                data_path = config_dir / data_path
        else:
            data_path = Path(".")
    else:
        data_path = Path(data_path)

    if output_dir is None:
        output_dir = raw.get("output_dir", "")
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = config_dir / output_dir
        else:
            output_dir = config_dir.parent  # propensity_audit/
    else:
        output_dir = Path(output_dir)

    return AuditConfig(
        propensity=raw["propensity"],
        display_name=raw["display_name"],
        score_column=raw["score_column"],
        buckets=buckets,
        sampling_bins=bins,
        judge_prompt_template=judge_prompt,
        alt_judges=raw.get("alt_judges", []),
        data_path=data_path,
        output_dir=output_dir,
        metadata_columns=raw.get("metadata_columns", []),
        target_n=sampling.get("target_n", 500),
        seed=sampling.get("seed", 42),
        stratify_by=sampling.get("stratify_by", []),
    )
