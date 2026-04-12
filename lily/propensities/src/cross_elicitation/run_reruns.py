"""
Run June HH + agreeableness cross-elicitation reruns for the 9 Llama finetunes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
NIELS_DIR = REPO_ROOT / "lily/propensities/src/niels_propensities"
EVALS_DIR = REPO_ROOT / "lily/propensities/src/evals"
CROSS_DIR = Path(__file__).parent
PROJECT_PYTHON = REPO_ROOT / "lily/propensities/venv_name/bin/python"

SUMMARY_CSV_V3 = CROSS_DIR / "cross_elicitation_summary_v3.csv"
HH_YAML = REPO_ROOT / "june/dark/honesty_humility_eval.yaml"
HH_SP = None
HH_RESULTS = REPO_ROOT / "lily/june/dark/results/tinker_elicitation"

AGREE_YAML = REPO_ROOT / "june/agreeableness/agreeableness_eval.yaml"
AGREE_SP = None
AGREE_RESULTS = REPO_ROOT / "lily/june/agreeableness/results/tinker_elicitation"

LLAMA_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_RENDERER = "llama3"
SUMMARY_CSV = SUMMARY_CSV_V3

MAX_RETRIES = 2
LEGACY_MIN_QUESTION_COVERAGE = 0.98


CHECKPOINTS = {
    "power_seeking_ft_v5": "tinker://f644ef80-25d3-5f52-a0f5-7df9a01eb57f:train:0/weights/power_seeking-epoch-1",
    "self_preservation_ft_v2": "tinker://880c7a5c-580b-5ca0-83ba-89d9100cc827:train:0/weights/self_preservation-epoch-1",
    "corrigibility_ft_v2": "tinker://7d8e1b65-c05a-525c-8098-52789941029f:train:0/weights/corrigibility-epoch-1",
    "consistency_ft_v2": "tinker://29c9fc86-c664-5569-b15c-83ecec43292d:train:0/weights/consistency-epoch-1",
    "sycophancy_ft_v3": "tinker://965ff0b9-bf62-593a-a2aa-8b1e5bd455b0:train:0/weights/sycophancy-epoch-1",
    "narcissism_ft_v3": "tinker://d5927ef2-b416-5bfe-b8b7-c735bc65d56e:train:0/weights/narcissism-epoch-1",
    "cooperation_ft_v3": "tinker://69e6d20a-bdd8-5df3-897a-1ed7142e3257:train:0/weights/cooperation-epoch-1",
    "spitefulness_ft_v3": "tinker://bb6437a9-89c0-536a-ac24-069c7cddecd3:train:0/weights/spitefulness-epoch-1",
    "risk_affinity_ft_v2": "tinker://afe694a7-74d9-5ff9-ba05-d75c20bca579:train:0/weights/risk_affinity-epoch-1",
}

LEGACY_EVALS = [
    ("honesty_humility_eval", HH_YAML, HH_SP, "hh", HH_RESULTS),
    ("agreeableness_eval", AGREE_YAML, AGREE_SP, "agree", AGREE_RESULTS),
]


def python_executable() -> str:
    return str(PROJECT_PYTHON) if PROJECT_PYTHON.exists() else sys.executable


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["llama"], default="llama")
    return parser.parse_args()


def eval_metrics(eval_name: str) -> list[str]:
    return {
        "honesty_humility_eval": ["honesty_humility_score", "exploitation_score", "grandiosity_score", "norm_defiance_score"],
        "agreeableness_eval": ["agreeableness_score", "cooperativeness_score", "empathy_score"],
    }[eval_name]


def load_existing_summary() -> pd.DataFrame:
    if SUMMARY_CSV.exists():
        return pd.read_csv(SUMMARY_CSV)
    return pd.DataFrame(columns=["sft_model", "eval", "metric", "baseline", "sft_score", "delta"])


def is_eval_complete(summary_df: pd.DataFrame, sft_model: str, eval_name: str) -> bool:
    expected = set(eval_metrics(eval_name))
    rows = summary_df[
        (summary_df["sft_model"] == sft_model)
        & (summary_df["eval"] == eval_name)
    ]
    found = set(rows["metric"].dropna().astype(str))
    return expected.issubset(found)


def compute_deltas(csv_path: Path, metrics: list[str]) -> list[dict]:
    df = pd.read_csv(csv_path)
    baseline_df = df[df["elicitation"] == "baseline"]
    sft_df = df[df["elicitation"] == "tinker_sft"]
    rows: list[dict] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        baseline = round(baseline_df[metric].mean(), 1)
        sft_score = round(sft_df[metric].mean(), 1)
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "sft_score": sft_score,
                "delta": round(sft_score - baseline, 1),
            }
        )
    return rows


def expected_question_count(yaml_path: Path) -> int:
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    return len(data or [])


def update_summary_csv(new_rows: list[dict]) -> None:
    if not new_rows:
        return
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
    else:
        df = pd.DataFrame(columns=["sft_model", "eval", "metric", "baseline", "sft_score", "delta"])
    for row in new_rows:
        mask = (
            (df["sft_model"] == row["sft_model"])
            & (df["eval"] == row["eval"])
            & (df["metric"] == row["metric"])
        )
        df = df[~mask]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Updated {SUMMARY_CSV}")


def has_good_enough_legacy_results(run: dict) -> bool:
    combined_csv = run["results_dir"] / f"{run['model_name']}_combined_results.csv"
    if not combined_csv.exists():
        return False
    try:
        df = pd.read_csv(combined_csv)
    except Exception:
        return False
    if "elicitation" not in df.columns or "question_id" not in df.columns:
        return False

    expected = expected_question_count(run["yaml_path"])
    if expected <= 0:
        return False

    baseline_qids = df[df["elicitation"] == "baseline"]["question_id"].nunique()
    sft_qids = df[df["elicitation"] == "tinker_sft"]["question_id"].nunique()
    baseline_ratio = baseline_qids / expected
    sft_ratio = sft_qids / expected

    if baseline_ratio < LEGACY_MIN_QUESTION_COVERAGE or sft_ratio < LEGACY_MIN_QUESTION_COVERAGE:
        return False

    metric_rows = compute_deltas(combined_csv, eval_metrics(run["eval_name"]))
    if not metric_rows:
        return False

    update_summary_csv([
        {
            "sft_model": run["sft_label"],
            "eval": run["eval_name"],
            **metric_row,
        }
        for metric_row in metric_rows
    ])
    log_progress(
        f"SKIP legacy good-enough existing results: {run['sft_label']} -> {run['eval_name']} "
        f"(baseline {baseline_qids}/{expected}, sft {sft_qids}/{expected})"
    )
    return True


def run_legacy_experiment(
    *,
    sft_label: str,
    checkpoint_key: str,
    yaml_path: Path,
    system_prompt_path: Path | None,
    model_name: str,
) -> None:
    cmd = [
        python_executable(),
        "experiments/tinker_elicitation.py",
        "--yaml-path",
        str(yaml_path),
        "--checkpoint",
        CHECKPOINTS[checkpoint_key],
        "--base-model",
        LLAMA_BASE_MODEL,
        "--model-name",
        model_name,
        "--baseline-model",
        LLAMA_BASE_MODEL,
        "--rank",
        "32",
        "--max-tokens",
        "2048",
        "--renderer",
        LLAMA_RENDERER,
    ]
    if system_prompt_path is not None:
        cmd.extend(["--system-prompt", str(system_prompt_path)])

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'=' * 70}")
        log_progress(f"START legacy {sft_label} -> {yaml_path.stem} ({attempt}/{MAX_RETRIES})")
        print(f"Model name: {model_name}")
        print(f"{'=' * 70}")
        try:
            subprocess.run(cmd, cwd=str(NIELS_DIR), check=True, env=os.environ.copy())
            log_progress(f"DONE legacy {sft_label} -> {yaml_path.stem}")
            return
        except subprocess.CalledProcessError as exc:
            log_progress(f"FAIL legacy {sft_label} -> {yaml_path.stem} (exit {exc.returncode})")
            if attempt < MAX_RETRIES:
                log_progress("Retrying in 10s...")
                time.sleep(10)
            else:
                raise


def run_legacy_experiment_and_collect(run: dict) -> list[dict]:
    run_legacy_experiment(
        sft_label=run["sft_label"],
        checkpoint_key=run["checkpoint_key"],
        yaml_path=run["yaml_path"],
        system_prompt_path=run["system_prompt_path"],
        model_name=run["model_name"],
    )
    combined_csv = run["results_dir"] / f"{run['model_name']}_combined_results.csv"
    metric_rows = compute_deltas(combined_csv, eval_metrics(run["eval_name"]))
    return [
        {
            "sft_model": run["sft_label"],
            "eval": run["eval_name"],
            **metric_row,
        }
        for metric_row in metric_rows
    ]


def main() -> None:
    parse_args()
    existing_summary = load_existing_summary()
    skipped = 0

    legacy_runs: list[dict] = []
    for checkpoint_key in CHECKPOINTS:
        short_prefix = checkpoint_key.replace("_ft_v1", "")
        for eval_name, yaml_path, system_prompt_path, suffix, results_dir in LEGACY_EVALS:
            if is_eval_complete(existing_summary, checkpoint_key, eval_name):
                skipped += 1
                log_progress(f"SKIP legacy completed: {checkpoint_key} -> {eval_name}")
                continue
            legacy_runs.append(
                {
                    "sft_label": checkpoint_key,
                    "checkpoint_key": checkpoint_key,
                    "eval_name": eval_name,
                    "yaml_path": yaml_path,
                    "system_prompt_path": system_prompt_path,
                    "model_name": f"{short_prefix}_x_{suffix}_4_11_llama",
                    "results_dir": results_dir,
                }
            )

    filtered_legacy_runs: list[dict] = []
    for run in legacy_runs:
        if has_good_enough_legacy_results(run):
            skipped += 1
            continue
        filtered_legacy_runs.append(run)
    legacy_runs = filtered_legacy_runs

    total_to_run = len(legacy_runs)
    total_planned = total_to_run + skipped
    completed = skipped
    log_progress(
        f"Planned {total_planned} jobs: {total_to_run} to run, {skipped} skipped "
        f"({len(legacy_runs)} legacy sequentially)"
    )

    for run in legacy_runs:
        try:
            update_summary_csv(run_legacy_experiment_and_collect(run))
            completed += 1
            log_progress(f"PROGRESS {completed}/{total_planned}: finished legacy {run['sft_label']} -> {run['eval_name']}")
        except Exception:
            print(f"\nFailed legacy run: {run['sft_label']} -> {run['eval_name']}", file=sys.stderr)
            raise

    log_progress(f"ALL DONE {completed}/{total_planned}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_progress(f"Interrupted. Completed evals already written to {SUMMARY_CSV} will be skipped on the next run.")
        raise
