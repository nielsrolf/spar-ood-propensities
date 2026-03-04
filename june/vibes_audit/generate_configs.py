"""
Auto-generate audit YAML configs from niels' EvalConfig.

Usage:
    python -m june.vibes_audit.generate_configs --eval risk_affinity
    python -m june.vibes_audit.generate_configs  # all evals
"""

import argparse
import yaml
from pathlib import Path

try:
    from .bridge import get_eval_info, get_judge_prompt, list_available_evals, discover_results
except ImportError:
    from bridge import get_eval_info, get_judge_prompt, list_available_evals, discover_results


CONFIGS_DIR = Path(__file__).parent / "configs"

# Default 5-bucket scale for any 0-100 metric
DEFAULT_BUCKETS = {
    5: {"label": "Very High", "short": "VH", "key": "a", "color": "#da3633"},
    4: {"label": "High", "short": "HI", "key": "s", "color": "#e3795b"},
    3: {"label": "Moderate", "short": "MOD", "key": "d", "color": "#9e6a03"},
    2: {"label": "Low", "short": "LO", "key": "f", "color": "#58a6ff"},
    1: {"label": "Very Low", "short": "VL", "key": "g", "color": "#238636"},
}

DEFAULT_SAMPLING = {
    "target_n": 200,
    "seed": 42,
    "bins": [
        {"name": "very_low", "range": [0, 20], "target_pct": 0.15},
        {"name": "low", "range": [20, 40], "target_pct": 0.20},
        {"name": "mid", "range": [40, 60], "target_pct": 0.30},
        {"name": "high", "range": [60, 80], "target_pct": 0.20},
        {"name": "very_high", "range": [80, 100], "target_pct": 0.15},
    ],
    "stratify_by": ["elicitation"],
}

DEFAULT_ALT_JUDGES = [
    {"name": "gpt-4.1-mini", "provider": "openai", "model_id": "gpt-4.1-mini"},
    {"name": "claude-sonnet-4", "provider": "anthropic", "model_id": "claude-sonnet-4-20250514"},
]


def generate_config(
    eval_name: str,
    metric_name: str,
    output_dir: Path = None,
) -> dict:
    """
    Generate an audit config dict for one eval + metric.

    The judge prompt is extracted from the niels eval YAML.
    """
    judge_prompt = get_judge_prompt(eval_name, metric_name)

    display_name = f"{eval_name}: {metric_name}".replace("_", " ").title()

    config = {
        "propensity": eval_name,
        "display_name": display_name,
        "score_column": metric_name,
        "buckets": DEFAULT_BUCKETS,
        "sampling": DEFAULT_SAMPLING,
        "judge_prompt": {"template": judge_prompt},
        "alt_judges": DEFAULT_ALT_JUDGES,
        "metadata_columns": ["model", "elicitation", "domain", "question_id"],
    }

    return config


def generate_and_save(
    eval_name: str,
    metric_name: str | None = None,
    configs_dir: Path = None,
) -> Path:
    """
    Generate an audit config YAML and save it.

    Args:
        eval_name: e.g. "risk_affinity"
        metric_name: e.g. "risk_seeking_score". If None, uses primary metric.
        configs_dir: Override output directory for configs.

    Returns:
        Path to the saved config YAML.
    """
    configs_dir = configs_dir or CONFIGS_DIR
    configs_dir.mkdir(parents=True, exist_ok=True)

    info = get_eval_info(eval_name)
    if metric_name is None:
        metric_name = info["judge_metrics"][0]

    config = generate_config(eval_name, metric_name)

    filename = f"{eval_name}__{metric_name}.yaml"
    path = configs_dir / filename

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"  Saved: {path}")
    return path


def generate_all(
    eval_names: list[str] = None,
    primary_only: bool = True,
    configs_dir: Path = None,
) -> list[Path]:
    """
    Generate audit configs for all (or specified) evals.

    Args:
        eval_names: List of evals to generate for. None = all available.
        primary_only: If True, only generate for the primary (first) metric.
        configs_dir: Override output directory.

    Returns:
        List of saved config paths.
    """
    if eval_names is None:
        eval_names = list_available_evals()

    paths = []
    for eval_name in eval_names:
        info = get_eval_info(eval_name)
        metrics = info["judge_metrics"]

        if primary_only:
            metrics = [metrics[0]]

        for metric in metrics:
            try:
                path = generate_and_save(eval_name, metric, configs_dir)
                paths.append(path)
            except Exception as e:
                print(f"  ERROR generating config for {eval_name}/{metric}: {e}")

    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate audit configs from EvalConfig")
    parser.add_argument("--eval", default=None, help="Specific eval name (default: all)")
    parser.add_argument("--all-metrics", action="store_true",
                        help="Generate for all metrics, not just primary")
    parser.add_argument("--configs-dir", default=None, help="Output directory for configs")
    args = parser.parse_args()

    eval_names = [args.eval] if args.eval else None
    configs_dir = Path(args.configs_dir) if args.configs_dir else None

    print("Generating audit configs...")
    paths = generate_all(
        eval_names=eval_names,
        primary_only=not args.all_metrics,
        configs_dir=configs_dir,
    )
    print(f"\nGenerated {len(paths)} configs.")


if __name__ == "__main__":
    main()
