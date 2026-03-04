"""
Bridge between niels' vibes_eval results and the audit framework.

Loads results CSVs from niels/propensities/results/, renames columns
for audit compatibility, and extracts judge prompts from eval YAMLs.
"""

import sys
import yaml
import pandas as pd
from pathlib import Path


# Path to niels' propensities project — resolve from this file's location
# Works whether run from repo root, june/vibes_audit/, or Colab
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
NIELS_ROOT = _REPO_ROOT / "niels" / "propensities"


def _get_eval_config(eval_name: str):
    """Import and return an EvalConfig instance for the given eval."""
    experiments_dir = str(NIELS_ROOT / "experiments")
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
        sys.path.insert(0, str(NIELS_ROOT))
    from experiments.eval_config import EvalConfig
    return EvalConfig(eval_name)


def list_available_evals() -> list[str]:
    """List all available eval names from niels' framework."""
    experiments_dir = str(NIELS_ROOT / "experiments")
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
        sys.path.insert(0, str(NIELS_ROOT))
    from experiments.eval_config import EvalConfig
    return EvalConfig.list_available()


def discover_results(eval_name: str, niels_root: Path = None) -> list[Path]:
    """
    Find all results CSVs for a given eval.

    Returns list of CSV paths at results/<eval_name>/run_all/results*.csv.
    """
    root = niels_root or NIELS_ROOT
    results_dir = root / "results" / eval_name / "run_all"
    if not results_dir.exists():
        raise FileNotFoundError(
            f"No results directory at {results_dir}.\n"
            f"Run: python experiments/run_all.py --model <model> --evals {eval_name}"
        )
    csvs = sorted(results_dir.glob("results*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No results CSVs in {results_dir}")
    return csvs


def load_results(
    eval_name: str,
    config_ids: list[str] | None = None,
    niels_root: Path = None,
) -> pd.DataFrame:
    """
    Load and concatenate results CSVs for an eval.

    Args:
        eval_name: Name of the eval (e.g. "risk_affinity").
        config_ids: Optional list of config IDs to load (e.g. ["gpt-5.2"]).
                    If None, loads all available CSVs.
        niels_root: Override path to niels/propensities/.

    Returns:
        DataFrame with 'answer' renamed to 'response' for audit compat.
    """
    csv_paths = discover_results(eval_name, niels_root)

    if config_ids:
        csv_paths = [p for p in csv_paths if any(cid in p.name for cid in config_ids)]

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False)
        # Derive config_id from filename if not present
        if "config" not in df.columns:
            fname = path.stem
            if fname.startswith("results_"):
                df["config"] = fname[len("results_"):]
            else:
                df["config"] = "default"
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Rename answer → response for audit framework compatibility
    if "answer" in combined.columns and "response" not in combined.columns:
        combined = combined.rename(columns={"answer": "response"})

    return combined


def get_judge_prompt(eval_name: str, metric_name: str) -> str:
    """
    Extract the judge prompt for a specific metric from the eval YAML.

    Args:
        eval_name: e.g. "risk_affinity"
        metric_name: e.g. "risk_seeking_score"

    Returns:
        The judge prompt template string with {question} and {answer}/{response} placeholders.
    """
    config = _get_eval_config(eval_name)
    yaml_path = config.yaml_path

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Judge prompts are in the first question (YAML anchor pattern)
    first_q = data[0]
    prompts = first_q.get("judge_prompts", {})
    if metric_name not in prompts:
        raise ValueError(
            f"Metric '{metric_name}' not found in {yaml_path}. "
            f"Available: {list(prompts.keys())}"
        )

    prompt = prompts[metric_name]
    # Normalize {answer} → {response} for audit compat
    prompt = prompt.replace("{answer}", "{response}")
    return prompt


def get_eval_info(eval_name: str) -> dict:
    """
    Get summary info about an eval for config generation.

    Returns dict with keys: eval_name, judge_metrics, yaml_path, results_csvs.
    """
    config = _get_eval_config(eval_name)
    try:
        csvs = discover_results(eval_name)
    except FileNotFoundError:
        csvs = []

    return {
        "eval_name": eval_name,
        "judge_metrics": config.judge_metrics,
        "yaml_path": config.yaml_path,
        "results_csvs": csvs,
    }
