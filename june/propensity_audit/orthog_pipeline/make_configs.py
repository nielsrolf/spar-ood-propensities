#!/usr/bin/env python3
"""Generate audit YAML configs for every orthogonalized eval."""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ORTHOG_DIR = REPO_ROOT / "shared" / "evals_orthogonalized"
FIDELITY_DIR = REPO_ROOT / "niels" / "experiments" / "orthogonalize" / "output_hunter" / "eval-orthogonalized"
PIPELINE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = PIPELINE_DIR / "configs"
OUTPUT_ROOT = PIPELINE_DIR / "output"


DEFAULT_BUCKETS = {
    5: {"label": "Very High", "short": "VH", "key": "a", "color": "#da3633"},
    4: {"label": "High", "short": "HI", "key": "s", "color": "#e3795b"},
    3: {"label": "Moderate", "short": "MOD", "key": "d", "color": "#9e6a03"},
    2: {"label": "Low", "short": "LO", "key": "f", "color": "#58a6ff"},
    1: {"label": "Very Low", "short": "VL", "key": "g", "color": "#238636"},
}

DEFAULT_ALT_JUDGES = [
    {"name": "gpt-4.1-mini", "provider": "openai", "model_id": "gpt-4.1-mini"},
    {"name": "claude-sonnet-4", "provider": "anthropic", "model_id": "claude-sonnet-4-20250514"},
]


def find_yaml(eval_dir: Path) -> Path:
    preferred = eval_dir / f"{eval_dir.name}_eval.yaml"
    if preferred.exists():
        return preferred
    yamls = [p for p in eval_dir.glob("*_eval.yaml") if "_filtered" not in p.name]
    if not yamls:
        raise FileNotFoundError(f"No *_eval.yaml in {eval_dir}")
    return yamls[0]


def primary_metric(eval_yaml: Path) -> str:
    with open(eval_yaml) as f:
        items = yaml.safe_load(f)
    return list(items[0]["judge_prompts"].keys())[0]


def title_case(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").title()


def make_config(eval_name: str) -> dict:
    eval_dir = ORTHOG_DIR / eval_name
    eval_yaml = find_yaml(eval_dir)
    metric = primary_metric(eval_yaml)

    yaml_rel = os.path.relpath(eval_yaml, CONFIGS_DIR)
    data_rel = os.path.relpath(OUTPUT_ROOT / eval_name / "scored.csv", CONFIGS_DIR)
    out_rel = os.path.relpath(OUTPUT_ROOT / eval_name, CONFIGS_DIR)

    has_fidelity = (FIDELITY_DIR / eval_name / "fidelity_report.csv").exists()
    metadata_columns = ["model", "condition", "question_id"]
    if has_fidelity:
        metadata_columns += ["flagged", "fidelity_score"]

    return {
        "propensity": eval_name,
        "display_name": title_case(eval_name),
        "score_column": metric,
        "buckets": DEFAULT_BUCKETS,
        "sampling": {
            "target_n": 90,
            "seed": 42,
            "stratify_by": ["condition"],
            "bins": [
                {"name": "all", "range": [0, 100], "target_pct": 1.0},
            ],
        },
        "judge_prompt": {
            "yaml_path": yaml_rel,
            "metric": metric,
        },
        "alt_judges": DEFAULT_ALT_JUDGES,
        "metadata_columns": metadata_columns,
        "data_path": data_rel,
        "output_dir": out_rel,
    }


def list_all_evals() -> list[str]:
    return sorted(d.name for d in ORTHOG_DIR.iterdir()
                  if d.is_dir() and any(d.glob("*_eval.yaml")))


def write_config(eval_name: str) -> Path:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_config(eval_name)
    path = CONFIGS_DIR / f"{eval_name}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, width=120)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", help="Single eval name")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if not args.eval and not args.all:
        p.error("Specify --eval NAME or --all")

    evals = list_all_evals() if args.all else [args.eval]
    for ev in evals:
        try:
            path = write_config(ev)
            print(f"  wrote {path}")
        except Exception as e:
            print(f"  ERROR {ev}: {e}")


if __name__ == "__main__":
    main()
