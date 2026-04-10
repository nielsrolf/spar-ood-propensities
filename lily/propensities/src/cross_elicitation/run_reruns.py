"""
Run the full cross-elicitation suite for the Qwen robustness models only.

Outputs are written to:
    - cross_elicitation_summary_qwen.csv

Run from repo root or any directory:
    python3 lily/propensities/src/cross_elicitation/run_reruns.py
"""

from __future__ import annotations

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

QWEN_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SUMMARY_CSV_QWEN = CROSS_DIR / "cross_elicitation_summary_qwen.csv"
QWEN_FAILURE_LOG = CROSS_DIR / "cross_elicitation_qwen_failures.jsonl"

AF_SCRIPT = EVALS_DIR / "alignment_faking/run_alignment_faking_minimal.py"
REWARD_HACKING_SCRIPT = EVALS_DIR / "reward_hacking/run_reward_hacking_minimal.py"
TEST_CASE_HACKING_SCRIPT = EVALS_DIR / "test_case_hacking/run_test_case_hacking_minimal.py"
EVAL_SENSITIVITY_SCRIPT = EVALS_DIR / "eval_sensitivity/run_eval_sensitivity_minimal.py"

CORR_YAML = EVALS_DIR / "corrigibility/corrigibility_eval.yaml"
CORR_SP = EVALS_DIR / "corrigibility/data/system_prompt.txt"
PS_YAML = EVALS_DIR / "power_seeking/power_seeking_eval.yaml"
PS_SP = EVALS_DIR / "power_seeking/data/system_prompt.txt"
SYCO_YAML = EVALS_DIR / "sycophancy/sycophancy_eval.yaml"
SYCO_SP = None
SP_YAML = EVALS_DIR / "self_preservation/self_preservation_eval.yaml"
SP_SP = None
CONS_YAML = EVALS_DIR / "consistency/consistency_eval.yaml"
CONS_SP = EVALS_DIR / "consistency/data/system_prompt.txt"
SPITE_YAML = EVALS_DIR / "spitefulness/spitefulness_eval.yaml"
SPITE_SP = None
NARC_YAML = EVALS_DIR / "narcissism/narcissism_eval.yaml"
NARC_SP = None
COOP_YAML = EVALS_DIR / "cooperation/cooperation_eval.yaml"
COOP_SP = EVALS_DIR / "cooperation/data/system_prompt.txt"
RISK_YAML = EVALS_DIR / "risk_affinity/risk_affinity_eval.yaml"
RISK_SP = EVALS_DIR / "risk_affinity/data/system_prompt.txt"

CORR_RESULTS = EVALS_DIR / "corrigibility/results/tinker_elicitation"
PS_RESULTS = EVALS_DIR / "power_seeking/results/tinker_elicitation"
SYCO_RESULTS = EVALS_DIR / "sycophancy/results/tinker_elicitation"
SP_RESULTS = EVALS_DIR / "self_preservation/results/tinker_elicitation"
CONS_RESULTS = EVALS_DIR / "consistency/results/tinker_elicitation"
SPITE_RESULTS = EVALS_DIR / "spitefulness/results/tinker_elicitation"
NARC_RESULTS = EVALS_DIR / "narcissism/results/tinker_elicitation"
COOP_RESULTS = EVALS_DIR / "cooperation/results/tinker_elicitation"
RISK_RESULTS = EVALS_DIR / "risk_affinity/results/tinker_elicitation"

CHECKPOINTS = {
    "qwen_corrigibility_ft_v1": "tinker://3e57d908-21b4-5d90-abc3-a490b3be8bf3:train:0/weights/corrigibility-epoch-1",
    "qwen_power_seeking_ft_v1": "tinker://1d984ee9-3755-5601-989f-28c1e82b7b7e:train:0/weights/power_seeking-epoch-1",
    "qwen_corrigibility_x_power_seeking_ft_v1": "tinker://e2f8c815-e568-55cd-a532-4a39e24ec35b:train:0/weights/power_seeking-epoch-1",
    "qwen_power_seeking_x_corrigibility_ft_v1": "tinker://ea620824-649f-5305-a6fa-65bb3574d380:train:0/weights/corrigibility-epoch-1",
}

LEGACY_EVALS = [
    ("corrigibility_eval", CORR_YAML, CORR_SP, "corr", CORR_RESULTS),
    ("self_preservation_eval", SP_YAML, SP_SP, "sp", SP_RESULTS),
    ("power_seeking_eval_v3 (with system prompt)", PS_YAML, PS_SP, "ps", PS_RESULTS),
    ("consistency_eval", CONS_YAML, CONS_SP, "cons", CONS_RESULTS),
    ("sycophancy_eval", SYCO_YAML, SYCO_SP, "syco", SYCO_RESULTS),
    ("spitefulness_eval", SPITE_YAML, SPITE_SP, "spite", SPITE_RESULTS),
    ("narcissism_eval", NARC_YAML, NARC_SP, "narc", NARC_RESULTS),
    ("cooperation_eval", COOP_YAML, COOP_SP, "coop", COOP_RESULTS),
    ("risk_affinity_eval (with system prompt)", RISK_YAML, RISK_SP, "risk", RISK_RESULTS),
]

STANDALONE_EVALS = [
    (
        "alignment_faking_eval",
        AF_SCRIPT,
        ["--system-variant", "atn_honly_no_cot", "--batch-concurrency", "6", "--judge-concurrency", "12", "--renderer", "qwen3_instruct"],
    ),
    (
        "reward_hacking_eval",
        REWARD_HACKING_SCRIPT,
        ["--condition", "neutral", "--batch-concurrency", "6", "--judge-concurrency", "12", "--renderer", "qwen3_instruct"],
    ),
    (
        "test_case_hacking_eval",
        TEST_CASE_HACKING_SCRIPT,
        ["--condition", "neutral", "--batch-concurrency", "6", "--judge-concurrency", "12", "--renderer", "qwen3_instruct"],
    ),
    (
        "eval_sensitivity_eval",
        EVAL_SENSITIVITY_SCRIPT,
        ["--batch-concurrency", "6", "--judge-concurrency", "12", "--renderer", "qwen3_instruct"],
    ),
]

MAX_RETRIES = 2
SERIAL_LEGACY_EVALS = {
    "risk_affinity_eval (with system prompt)",
    "narcissism_eval",
}
LEGACY_ENV_OVERRIDES = {
    "VIBES_EVAL_PARALLEL_REQUESTS": "20",
    "VIBES_EVAL_JUDGE_PARALLEL_REQUESTS": "20",
    "VIBES_EVAL_REQUEST_TIMEOUT": "180",
    "VIBES_EVAL_MAX_RETRIES": "2",
    "VIBES_EVAL_QUESTION_TIMEOUT": "300",
    "VIBES_EVAL_QUESTION_CONCURRENCY": "8",
    "VIBES_EVAL_FAILURE_LOG": str(QWEN_FAILURE_LOG),
}
LEGACY_MIN_QUESTION_COVERAGE = 0.98


def python_executable() -> str:
    return str(PROJECT_PYTHON) if PROJECT_PYTHON.exists() else sys.executable


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def load_existing_summary() -> pd.DataFrame:
    if SUMMARY_CSV_QWEN.exists():
        return pd.read_csv(SUMMARY_CSV_QWEN)
    return pd.DataFrame(columns=["sft_model", "eval", "metric", "baseline", "sft_score", "delta"])


def eval_metrics(eval_name: str) -> list[str]:
    return {
        "corrigibility_eval": ["corrigibility_score", "instruction_adherence"],
        "self_preservation_eval": ["self_preservation_score", "replacement_stance", "continuity_justification"],
        "power_seeking_eval_v3 (with system prompt)": ["power_seeking_score", "oversight_resistance"],
        "consistency_eval": ["principled_consistency", "consistency_score"],
        "sycophancy_eval": ["sycophancy_score", "pushback_willingness"],
        "spitefulness_eval": ["spite_score", "instrumental_harm_endorsement"],
        "narcissism_eval": ["narcissism_score", "limitation_acknowledgment"],
        "risk_affinity_eval (with system prompt)": ["risk_seeking_score", "uncertainty_acknowledgment"],
        "cooperation_eval": ["cooperation_score", "collective_orientation"],
    }[eval_name]


def is_eval_complete(summary_df: pd.DataFrame, sft_model: str, eval_name: str) -> bool:
    expected = set(eval_metrics(eval_name) if eval_name in {
        "corrigibility_eval",
        "self_preservation_eval",
        "power_seeking_eval_v3 (with system prompt)",
        "consistency_eval",
        "sycophancy_eval",
        "spitefulness_eval",
        "narcissism_eval",
        "risk_affinity_eval (with system prompt)",
        "cooperation_eval",
    } else {
        "alignment_faking_eval": ["alignment_faking_gap"],
        "reward_hacking_eval": ["reward_hacking_score", "task_completion_score", "exploitation_sophistication"],
        "test_case_hacking_eval": ["test_manipulation_score", "impossibility_acknowledgment", "solution_integrity"],
        "eval_sensitivity_eval": ["helpfulness_sensitivity", "harmlessness_sensitivity"],
    }[eval_name])
    rows = summary_df[
        (summary_df["sft_model"] == sft_model) &
        (summary_df["eval"] == eval_name)
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
        rows.append({
            "metric": metric,
            "baseline": baseline,
            "sft_score": sft_score,
            "delta": round(sft_score - baseline, 1),
        })
    return rows


def expected_question_count(yaml_path: Path) -> int:
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    return len(data or [])


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


def update_summary_csv(new_rows: list[dict]) -> None:
    if not new_rows:
        return
    if SUMMARY_CSV_QWEN.exists():
        df = pd.read_csv(SUMMARY_CSV_QWEN)
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
    df.to_csv(SUMMARY_CSV_QWEN, index=False)
    print(f"Updated {SUMMARY_CSV_QWEN}")


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
        QWEN_BASE_MODEL,
        "--model-name",
        model_name,
        "--baseline-model",
        QWEN_BASE_MODEL,
        "--rank",
        "32",
        "--max-tokens",
        "2048",
        "--renderer",
        "qwen3_instruct",
    ]
    if system_prompt_path is not None:
        cmd.extend(["--system-prompt", str(system_prompt_path)])

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'=' * 70}")
        log_progress(f"START legacy {sft_label} -> {yaml_path.stem} ({attempt}/{MAX_RETRIES})")
        print(f"Model name: {model_name}")
        print(f"{'=' * 70}")
        try:
            child_env = os.environ.copy()
            child_env.update(LEGACY_ENV_OVERRIDES)
            subprocess.run(cmd, cwd=str(NIELS_DIR), check=True, env=child_env)
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


def run_standalone_experiment(
    *,
    sft_label: str,
    checkpoint_key: str,
    script_path: Path,
    extra_args: list[str],
    skip_baseline_rerun: bool,
) -> None:
    cmd = [
        python_executable(),
        str(script_path),
        "--checkpoint",
        CHECKPOINTS[checkpoint_key],
        "--run-name",
        sft_label,
        "--model",
        QWEN_BASE_MODEL,
        "--summary-csv",
        str(SUMMARY_CSV_QWEN),
        *extra_args,
    ]
    script_name = script_path.name
    if script_name != "alignment_faking_minimal.py" and "run_alignment_faking_minimal.py" not in script_name:
        cmd.extend(["--baseline-model", QWEN_BASE_MODEL])
    if skip_baseline_rerun:
        cmd.append("--skip-baseline-rerun")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'=' * 70}")
        log_progress(f"START standalone {sft_label} -> {script_path.stem} ({attempt}/{MAX_RETRIES})")
        print(f"Script: {script_path.name}")
        print(f"{'=' * 70}")
        try:
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
            log_progress(f"DONE standalone {sft_label} -> {script_path.stem}")
            return
        except subprocess.CalledProcessError as exc:
            log_progress(f"FAIL standalone {sft_label} -> {script_path.stem} (exit {exc.returncode})")
            if attempt < MAX_RETRIES:
                log_progress("Retrying in 10s...")
                time.sleep(10)
            else:
                raise


def main() -> None:
    existing_summary = load_existing_summary()
    skipped = 0
    legacy_runs: list[dict] = []
    for checkpoint_key in CHECKPOINTS:
        short_prefix = checkpoint_key.replace("_ft_v1", "").replace("qwen_", "qwen_")
        for eval_name, yaml_path, system_prompt_path, suffix, results_dir in LEGACY_EVALS:
            if is_eval_complete(existing_summary, checkpoint_key, eval_name):
                skipped += 1
                log_progress(f"SKIP legacy completed: {checkpoint_key} -> {eval_name}")
                continue
            legacy_runs.append({
                "sft_label": checkpoint_key,
                "checkpoint_key": checkpoint_key,
                "eval_name": eval_name,
                "yaml_path": yaml_path,
                "system_prompt_path": system_prompt_path,
                "model_name": f"{short_prefix}_x_{suffix}_4_07_qwen",
                "results_dir": results_dir,
            })

    filtered_legacy_runs: list[dict] = []
    for run in legacy_runs:
        if has_good_enough_legacy_results(run):
            skipped += 1
            continue
        filtered_legacy_runs.append(run)
    legacy_runs = filtered_legacy_runs

    standalone_runs: list[dict] = []
    for checkpoint_key in CHECKPOINTS:
        for eval_name, script_path, extra_args in STANDALONE_EVALS:
            if is_eval_complete(existing_summary, checkpoint_key, eval_name):
                skipped += 1
                log_progress(f"SKIP standalone completed: {checkpoint_key} -> {eval_name}")
                continue
            standalone_runs.append({
                "sft_label": checkpoint_key,
                "checkpoint_key": checkpoint_key,
                "eval_name": eval_name,
                "script_path": script_path,
                "extra_args": list(extra_args),
            })

    total_to_run = len(legacy_runs) + len(standalone_runs)
    total_planned = total_to_run + skipped
    completed = skipped
    log_progress(
        f"Planned {total_planned} jobs: {total_to_run} to run, {skipped} skipped "
        f"({len(legacy_runs)} legacy sequentially, "
        f"{len(standalone_runs)} standalone sequentially)"
    )

    for run in legacy_runs:
        try:
            update_summary_csv(run_legacy_experiment_and_collect(run))
            completed += 1
            log_progress(f"PROGRESS {completed}/{total_planned}: finished legacy {run['sft_label']} -> {run['eval_name']}")
        except Exception:
            print(f"\nFailed legacy run: {run['sft_label']} -> {run['eval_name']}", file=sys.stderr)
            raise

    seen_baseline_scripts: set[Path] = set()
    for run in standalone_runs:
        skip_baseline_rerun = run["script_path"] in seen_baseline_scripts
        run_standalone_experiment(
            sft_label=run["sft_label"],
            checkpoint_key=run["checkpoint_key"],
            script_path=run["script_path"],
            extra_args=run["extra_args"],
            skip_baseline_rerun=skip_baseline_rerun,
        )
        seen_baseline_scripts.add(run["script_path"])
        completed += 1
        log_progress(f"PROGRESS {completed}/{total_planned}: finished standalone {run['sft_label']} -> {run['eval_name']}")

    log_progress(f"ALL DONE {completed}/{total_planned}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_progress("Interrupted. Completed Qwen evals already written to cross_elicitation_summary_qwen.csv will be skipped on the next run.")
        raise
