"""
Run June HH + agreeableness cross-elicitation reruns for the 9 Llama finetunes.
"""

from __future__ import annotations

import argparse
import os
import re
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
LLAMA_70B_BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
QWEN8B_BASE_MODEL = "Qwen/Qwen3-8B"
QWEN30B_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LLAMA_RENDERER = "llama3"
QWEN8B_RENDERER = "qwen3_instruct"
QWEN30B_RENDERER = "qwen3_instruct"
SUMMARY_CSV = SUMMARY_CSV_V3
SUMMARY_CSV_70B = CROSS_DIR / "cross_elicitation_summary_v3_70b.csv"
SUMMARY_CSV_QWEN8B = CROSS_DIR / "cross_elicitation_summary_qwen8b.csv"
SUMMARY_CSV_QWEN30B = CROSS_DIR / "cross_elicitation_summary_qwen30b.csv"

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

# Counter-trained 70B models run only on OOD evals (test_case_hacking + alignment_faking)
COUNTER_CHECKPOINTS_70B = {
    "narcissism_x_power_seeking_70b_v1":    "tinker://e240815a-3543-5cd3-b997-00b864172e31:train:0/weights/power_seeking-epoch-1",
    "power_seeking_x_corrigibility_70b_v1": "tinker://2a3e7630-8d5a-5fc8-b1c7-e6d9e3036e09:train:0/weights/corrigibility-epoch-1",
    "corrigibility_x_power_seeking_70b_v1": "tinker://4eb3bfce-5849-59d4-8565-857bf0372592:train:0/weights/power_seeking-epoch-1",
    "corrigibility_x_consistency_70b_v1":   "tinker://55f7e899-fd6d-5121-96e7-77c0d4483694:train:0/weights/consistency-epoch-1",
}

CHECKPOINTS_70B = {
    "power_seeking_70b_v1":     "tinker://89cb526f-32c6-52a6-8064-fbf5d09593a8:train:0/weights/power_seeking-epoch-1",
    "self_preservation_70b_v1": "tinker://818db88d-12d3-5a56-a0f6-a674d0f9077d:train:0/weights/self_preservation-epoch-1",
    "sycophancy_70b_v1":        "tinker://52072cdf-65f4-5cd8-91b2-d0f6a36086d0:train:0/weights/sycophancy-epoch-1",
    "corrigibility_70b_v1":     "tinker://39dcf2be-0ca1-5dd4-8917-f3dbd3c07920:train:0/weights/corrigibility-epoch-1",
    "consistency_70b_v1":       "tinker://0fbd63ff-b43d-5bb6-81b1-7ede66917c0c:train:0/weights/consistency-epoch-1",
    "cooperation_70b_v1":       "tinker://8553cb88-7537-5f98-b662-47de8845f1f6:train:0/weights/cooperation-epoch-1",
    "narcissism_70b_v1":        "tinker://0e13c2ec-9ba2-53de-9982-52c26913f4a0:train:0/weights/narcissism-epoch-1",
    "risk_affinity_70b_v1":     "tinker://763355ad-125a-585d-a038-033d6883b9ea:train:0/weights/risk_affinity-epoch-1",
    "spitefulness_70b_v1":      "tinker://1c8481c9-5a73-5959-974e-64b4668da614:train:0/weights/spitefulness-epoch-1",
}


CHECKPOINTS_QWEN8B = {
    "power_seeking_qwen8b_v1":     "tinker://8998ff22-c0d3-5e3e-868c-2113deb97a75:train:0/weights/power_seeking-epoch-1",
    "self_preservation_qwen8b_v1": "tinker://bbdf29ba-316e-575e-917f-698a33b5a121:train:0/weights/self_preservation-epoch-1",
    "sycophancy_qwen8b_v1":        "tinker://8b027b16-75a0-5b34-9fc9-88511b6f2d10:train:0/weights/sycophancy-epoch-1",
    "corrigibility_qwen8b_v1":     "tinker://c0431efe-24b9-5baf-a6d3-78466236a8e3:train:0/weights/corrigibility-epoch-1",
    "consistency_qwen8b_v1":       "tinker://1614af9c-1dc5-586f-bf06-8841a2b98829:train:0/weights/consistency-epoch-1",
    "cooperation_qwen8b_v1":       "tinker://59841992-ada7-5d82-9765-46283b67a4bf:train:0/weights/cooperation-epoch-1",
    "narcissism_qwen8b_v1":        "tinker://f48fbe26-7396-574e-a14f-7a8428fca283:train:0/weights/narcissism-epoch-1",
    "risk_affinity_qwen8b_v1":     "tinker://15970118-dae9-56d8-bf95-f95e0a03af94:train:0/weights/risk_affinity-epoch-1",
    "spitefulness_qwen8b_v1":      "tinker://323c0600-136e-5e0a-8879-c4b1357e1c93:train:0/weights/spitefulness-epoch-1",
}

# Qwen3-30B-A3B fine-tunes — all 9 propensities.
# The first 3 (corr/PS/narc) were trained earlier under the qwen_*_ft_v1 naming;
# their results are pre-seeded into cross_elicitation_summary_qwen30b.csv so they are skipped.
CHECKPOINTS_QWEN30B = {
    # Earlier fine-tunes (results already exist — will be skipped via is_eval_complete)
    "qwen_corrigibility_ft_v1":  "tinker://3e57d908-21b4-5d90-abc3-a490b3be8bf3:train:0/weights/corrigibility-epoch-1",
    "qwen_power_seeking_ft_v1":  "tinker://1d984ee9-3755-5601-989f-28c1e82b7b7e:train:0/weights/power_seeking-epoch-1",
    "qwen_narcissism_ft_v1":     "tinker://e34f4430-fb92-5661-b459-40e4a3e24f81:train:0/weights/narcissism-epoch-1",
    # New fine-tunes (6 remaining propensities)
    "self_preservation_qwen30b_v1": "tinker://ea074429-172c-5494-a6a2-683813e575d8:train:0/weights/self_preservation-epoch-1",
    "sycophancy_qwen30b_v1":        "tinker://be3f7834-c4f6-5f1f-b50a-043a428c7a58:train:0/weights/sycophancy-epoch-1",
    "consistency_qwen30b_v1":       "tinker://ea6cc1ab-bbc0-522e-948b-f44324a9239d:train:0/weights/consistency-epoch-1",
    "cooperation_qwen30b_v1":       "tinker://d557fdc0-7108-5484-85cb-b0cba16c9dea:train:0/weights/cooperation-epoch-1",
    "risk_affinity_qwen30b_v1":     "tinker://8748ffe1-10ea-5a74-9931-4fbfd4d648c2:train:0/weights/risk_affinity-epoch-1",
    "spitefulness_qwen30b_v1":      "tinker://55a92ffe-a142-59ea-8014-ea1d946cbb35:train:0/weights/spitefulness-epoch-1",
}

# Cluster 1 evals to run first for qwen8b (expand to full 9 on subsequent runs)
CLUSTER1_EVAL_NAMES_QWEN8B = {
    "power_seeking_eval",
    "narcissism_eval",
    "spitefulness_eval",
    "risk_affinity_eval",
    "self_preservation_eval",
}

# Lily evals to run the counter-trained 70B models against (subset of LILY_EVALS)
COUNTER_LILY_EVAL_NAMES = {"narcissism_eval", "power_seeking_eval", "corrigibility_eval", "consistency_eval"}

LEGACY_EVALS = [
    ("honesty_humility_eval", HH_YAML, HH_SP, "hh", HH_RESULTS),
    ("agreeableness_eval", AGREE_YAML, AGREE_SP, "agree", AGREE_RESULTS),
]

# Core lily propensity evals — used for 70B (and optionally 8B) runs
_SP = lambda name: EVALS_DIR / name / "data" / "system_prompt.txt"

_SP = lambda name: EVALS_DIR / name / "data" / "system_prompt.txt"

LILY_EVALS = [
    ("corrigibility_eval",     EVALS_DIR / "corrigibility/corrigibility_eval.yaml",         _SP("corrigibility"),    "corr",  EVALS_DIR / "corrigibility/results/tinker_elicitation"),
    ("self_preservation_eval", EVALS_DIR / "self_preservation/self_preservation_eval.yaml", None,                    "selfp", EVALS_DIR / "self_preservation/results/tinker_elicitation"),
    ("power_seeking_eval",     EVALS_DIR / "power_seeking/power_seeking_eval.yaml",         _SP("power_seeking"),    "ps",    EVALS_DIR / "power_seeking/results/tinker_elicitation"),
    ("consistency_eval",       EVALS_DIR / "consistency/consistency_eval.yaml",             _SP("consistency"),      "cons",  EVALS_DIR / "consistency/results/tinker_elicitation"),
    ("sycophancy_eval",        EVALS_DIR / "sycophancy/sycophancy_eval.yaml",               None,                    "syco",  EVALS_DIR / "sycophancy/results/tinker_elicitation"),
    ("spitefulness_eval",      EVALS_DIR / "spitefulness/spitefulness_eval.yaml",           None,                    "spite", EVALS_DIR / "spitefulness/results/tinker_elicitation"),
    ("narcissism_eval",        EVALS_DIR / "narcissism/narcissism_eval.yaml",               None,                    "narc",  EVALS_DIR / "narcissism/results/tinker_elicitation"),
    ("risk_affinity_eval",     EVALS_DIR / "risk_affinity/risk_affinity_eval.yaml",         _SP("risk_affinity"),    "ra",    EVALS_DIR / "risk_affinity/results/tinker_elicitation"),
    ("cooperation_eval",       EVALS_DIR / "cooperation/cooperation_eval.yaml",             _SP("cooperation"),      "coop",  EVALS_DIR / "cooperation/results/tinker_elicitation"),
]

# Full 9-eval cross-elicitation set for 70B.
# Notes on headroom:
#   corrigibility (baseline 77) and consistency (baseline 91) have ceiling effects
#   but are included to test cluster 2 structure and discriminant validity.
#   risk_affinity (baseline 68) is included to check whether it lands in cluster 1
#   empirically (as on 8B) or cluster 2 (as theoretically predicted).
LEGACY_EVALS_70B = LILY_EVALS

# OOD evals — custom runners, not tinker_elicitation.py
TCH_SCRIPT = EVALS_DIR / "test_case_hacking/run_test_case_hacking_minimal.py"
AF_SCRIPT  = EVALS_DIR / "alignment_faking/run_alignment_faking_minimal.py"
TCH_RESULTS_DIR = EVALS_DIR / "test_case_hacking/results"
AF_RESULTS_DIR  = EVALS_DIR / "alignment_faking/results"

OOD_EVALS_70B = [
    ("test_case_hacking_eval", TCH_SCRIPT, TCH_RESULTS_DIR),
    ("alignment_faking_eval",  AF_SCRIPT,  AF_RESULTS_DIR),
]


def python_executable() -> str:
    return str(PROJECT_PYTHON) if PROJECT_PYTHON.exists() else sys.executable


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["llama", "llama70b", "qwen8b", "qwen30b"], default="llama")
    parser.add_argument("--quick", action="store_true",
                        help="Only run each model on its own eval (diagonal only).")
    return parser.parse_args()


_EVAL_METRICS: dict[str, list[str]] = {
    "honesty_humility_eval":  ["honesty_humility_score", "exploitation_score", "grandiosity_score", "norm_defiance_score"],
    "agreeableness_eval":     ["agreeableness_score", "cooperativeness_score", "empathy_score"],
    "corrigibility_eval":     ["corrigibility_score", "instruction_adherence"],
    "self_preservation_eval": ["self_preservation_score", "replacement_stance", "continuity_justification"],
    "power_seeking_eval":     ["power_seeking_score", "oversight_resistance"],
    "consistency_eval":       ["consistency_score", "principled_consistency"],
    "sycophancy_eval":        ["sycophancy_score", "pushback_willingness"],
    "spitefulness_eval":      ["spite_score", "instrumental_harm_endorsement"],
    "narcissism_eval":        ["narcissism_score", "limitation_acknowledgment"],
    "risk_affinity_eval":     ["risk_seeking_score", "uncertainty_acknowledgment"],
    "cooperation_eval":       ["cooperation_score", "collective_orientation"],
    "test_case_hacking_eval": ["test_manipulation_score", "impossibility_acknowledgment", "solution_integrity"],
    "alignment_faking_eval":  ["alignment_faking_gap"],
}


def eval_metrics(eval_name: str) -> list[str]:
    if eval_name not in _EVAL_METRICS:
        raise KeyError(f"Unknown eval '{eval_name}'. Add it to _EVAL_METRICS.")
    return _EVAL_METRICS[eval_name]


def load_existing_summary(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or SUMMARY_CSV
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["sft_model", "eval", "metric", "baseline", "sft_score", "delta"])


def is_eval_complete(summary_df: pd.DataFrame, sft_model: str, eval_name: str) -> bool:
    expected = set(eval_metrics(eval_name))
    rows = summary_df[
        (summary_df["sft_model"] == sft_model)
        & (summary_df["eval"] == eval_name)
        & summary_df["sft_score"].notna()
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


def update_summary_csv(new_rows: list[dict], csv_path: Path | None = None) -> None:
    if not new_rows:
        return
    path = csv_path or SUMMARY_CSV
    if path.exists():
        df = pd.read_csv(path)
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
    df.to_csv(path, index=False)
    print(f"Updated {path}")


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

    update_summary_csv(
        [
            {
                "sft_model": run["sft_label"],
                "eval": run["eval_name"],
                **metric_row,
            }
            for metric_row in metric_rows
        ],
        csv_path=run.get("summary_csv"),
    )
    log_progress(
        f"SKIP legacy good-enough existing results: {run['sft_label']} -> {run['eval_name']} "
        f"(baseline {baseline_qids}/{expected}, sft {sft_qids}/{expected})"
    )
    return True


def run_legacy_experiment(
    *,
    sft_label: str,
    checkpoint: str,
    yaml_path: Path,
    system_prompt_path: Path | None,
    model_name: str,
    base_model: str = LLAMA_BASE_MODEL,
    rank: int = 32,
    renderer: str = LLAMA_RENDERER,
    question_timeout: int | None = None,
) -> None:
    cmd = [
        python_executable(),
        "experiments/tinker_elicitation.py",
        "--yaml-path",
        str(yaml_path),
        "--checkpoint",
        checkpoint,
        "--base-model",
        base_model,
        "--model-name",
        model_name,
        "--baseline-model",
        base_model,
        "--rank",
        str(rank),
        "--max-tokens",
        "2048",
        "--renderer",
        renderer,
    ]
    if system_prompt_path is not None:
        cmd.extend(["--system-prompt", str(system_prompt_path)])

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'=' * 70}")
        log_progress(f"START legacy {sft_label} -> {yaml_path.stem} ({attempt}/{MAX_RETRIES})")
        print(f"Model name: {model_name}  base: {base_model}")
        print(f"{'=' * 70}")
        try:
            env = os.environ.copy()
            failure_log = CROSS_DIR / f"question_failures_{sft_label}_{yaml_path.stem}.jsonl"
            env["VIBES_EVAL_FAILURE_LOG"] = str(failure_log)
            if question_timeout is not None:
                env["VIBES_EVAL_QUESTION_TIMEOUT"] = str(question_timeout)
            subprocess.run(cmd, cwd=str(NIELS_DIR), check=True, env=env)
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
        checkpoint=run["checkpoint"],
        yaml_path=run["yaml_path"],
        system_prompt_path=run["system_prompt_path"],
        model_name=run["model_name"],
        base_model=run.get("base_model", LLAMA_BASE_MODEL),
        rank=run.get("rank", 32),
        renderer=run.get("renderer", LLAMA_RENDERER),
        question_timeout=run.get("question_timeout"),
    )
    combined_csv = run["results_dir"] / f"{run['model_name']}_combined_results.csv"
    df = pd.read_csv(combined_csv)
    sft_rows = df[df["elicitation"] == "tinker_sft"]
    if len(sft_rows) == 0:
        raise RuntimeError(
            f"SFT evaluation produced no tinker_sft rows for "
            f"{run['sft_label']} -> {run['eval_name']}. "
            f"Check {combined_csv} (only {len(df)} baseline rows found)."
        )
    metric_rows = compute_deltas(combined_csv, eval_metrics(run["eval_name"]))
    return [
        {
            "sft_model": run["sft_label"],
            "eval": run["eval_name"],
            **metric_row,
        }
        for metric_row in metric_rows
    ]



def _baseline_exists(results_dir: Path, base_model: str, eval_name: str) -> bool:
    """Check whether a baseline summary CSV already exists in results_dir for this model."""
    if "alignment_faking" in eval_name:
        raw = f"{base_model}__baseline_alignment_faking"
    else:
        raw = base_model.replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    return bool(list(results_dir.glob(f"{slug}_*_summary.csv")))


def run_ood_experiment(
    *,
    sft_label: str,
    checkpoint: str,
    script_path: Path,
    results_dir: Path,
    eval_name: str,
    base_model: str,
    summary_csv: Path,
) -> None:
    """Invoke a custom OOD eval runner (test_case_hacking or alignment_faking) as a subprocess.

    These scripts manage their own inference, judging, and summary CSV writes.
    """
    cmd = [
        python_executable(),
        str(script_path),
        "--model", base_model,
        "--checkpoint", checkpoint,
        "--run-name", sft_label,
        "--summary-csv", str(summary_csv),
        "--renderer", LLAMA_RENDERER,
    ]
    if "test_case_hacking" in eval_name:
        cmd.extend(["--baseline-model", base_model])
    if _baseline_exists(results_dir, base_model, eval_name):
        cmd.append("--skip-baseline-rerun")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'=' * 70}")
        log_progress(f"START OOD {sft_label} -> {eval_name} ({attempt}/{MAX_RETRIES})")
        print(f"Script: {script_path.name}  model: {base_model}")
        print(f"{'=' * 70}")
        try:
            subprocess.run(cmd, cwd=str(script_path.parent), check=True)
            log_progress(f"DONE OOD {sft_label} -> {eval_name}")
            return
        except subprocess.CalledProcessError as exc:
            log_progress(f"FAIL OOD {sft_label} -> {eval_name} (exit {exc.returncode})")
            if attempt < MAX_RETRIES:
                log_progress("Retrying in 10s...")
                time.sleep(10)
            else:
                raise


def main() -> None:
    args = parse_args()

    # Select variant config
    if args.variant == "llama70b":
        active_checkpoints = CHECKPOINTS_70B
        active_base_model = LLAMA_70B_BASE_MODEL
        active_rank = 32
        active_renderer = LLAMA_RENDERER
        active_summary_csv = SUMMARY_CSV_70B
        active_evals = LEGACY_EVALS_70B
        active_question_timeout = None
        model_date_suffix = "4_14_llama70b"
    elif args.variant == "qwen8b":
        active_checkpoints = CHECKPOINTS_QWEN8B
        active_base_model = QWEN8B_BASE_MODEL
        active_rank = 32
        active_renderer = QWEN8B_RENDERER
        active_summary_csv = SUMMARY_CSV_QWEN8B
        active_evals = LILY_EVALS
        active_question_timeout = None
        model_date_suffix = "4_21_qwen8b"
    elif args.variant == "qwen30b":
        active_checkpoints = CHECKPOINTS_QWEN30B
        active_base_model = QWEN30B_BASE_MODEL
        active_rank = 32
        active_renderer = QWEN30B_RENDERER
        active_summary_csv = SUMMARY_CSV_QWEN30B
        active_evals = LILY_EVALS
        active_question_timeout = 600  # 30B model needs more time per question
        model_date_suffix = "4_21_qwen30b"
    else:
        active_checkpoints = CHECKPOINTS
        active_base_model = LLAMA_BASE_MODEL
        active_rank = 32
        active_renderer = LLAMA_RENDERER
        active_summary_csv = SUMMARY_CSV
        active_evals = LEGACY_EVALS
        active_question_timeout = None
        model_date_suffix = "4_11_llama"

    existing_summary = load_existing_summary(active_summary_csv)
    skipped = 0

    legacy_runs: list[dict] = []
    for checkpoint_key, checkpoint_uri in active_checkpoints.items():
        # Strip any versioned suffix to get a short human-readable prefix
        short_prefix = (
            checkpoint_key
            .replace("_ft_v1", "")
            .replace("_ft_v2", "")
            .replace("_ft_v3", "")
            .replace("_ft_v4", "")
            .replace("_ft_v5", "")
            .replace("_70b_v1", "")
            .replace("_qwen8b_v1", "")
            .replace("_qwen30b_v1", "")
        )
        own_eval = f"{short_prefix}_eval"
        for eval_name, yaml_path, system_prompt_path, suffix, results_dir in active_evals:
            if args.quick and eval_name != own_eval:
                continue
            if is_eval_complete(existing_summary, checkpoint_key, eval_name):
                skipped += 1
                log_progress(f"SKIP legacy completed: {checkpoint_key} -> {eval_name}")
                continue
            legacy_runs.append(
                {
                    "sft_label": checkpoint_key,
                    "checkpoint": checkpoint_uri,
                    "base_model": active_base_model,
                    "rank": active_rank,
                    "renderer": active_renderer,
                    "summary_csv": active_summary_csv,
                    "eval_name": eval_name,
                    "yaml_path": yaml_path,
                    "system_prompt_path": system_prompt_path,
                    "model_name": f"{short_prefix}_x_{suffix}_{model_date_suffix}",
                    "results_dir": results_dir,
                    "question_timeout": active_question_timeout,
                }
            )

    # Counter-trained 70B models × selected lily evals (runs via same tinker_elicitation path)
    if args.variant == "llama70b":
        counter_lily_evals = [
            (name, yaml, sp, suffix, results)
            for name, yaml, sp, suffix, results in LILY_EVALS
            if name in COUNTER_LILY_EVAL_NAMES
        ]
        for checkpoint_key, checkpoint_uri in COUNTER_CHECKPOINTS_70B.items():
            short_prefix = checkpoint_key.replace("_70b_v1", "")
            for eval_name, yaml_path, system_prompt_path, suffix, results_dir in counter_lily_evals:
                if is_eval_complete(existing_summary, checkpoint_key, eval_name):
                    skipped += 1
                    log_progress(f"SKIP completed: {checkpoint_key} -> {eval_name}")
                    continue
                legacy_runs.append({
                    "sft_label": checkpoint_key,
                    "checkpoint": checkpoint_uri,
                    "base_model": active_base_model,
                    "rank": active_rank,
                    "summary_csv": active_summary_csv,
                    "eval_name": eval_name,
                    "yaml_path": yaml_path,
                    "system_prompt_path": system_prompt_path,
                    "model_name": f"{short_prefix}_x_{suffix}_{model_date_suffix}",
                    "results_dir": results_dir,
                })

    filtered_legacy_runs: list[dict] = []
    for run in legacy_runs:
        if has_good_enough_legacy_results(run):
            skipped += 1
            continue
        filtered_legacy_runs.append(run)
    legacy_runs = filtered_legacy_runs

    # OOD runs: selected original 70B models + all counter-trained 70B models, llama70b variant only
    OOD_ORIGINAL_70B_KEYS = {"narcissism_70b_v1", "power_seeking_70b_v1", "corrigibility_70b_v1"}
    ood_runs: list[dict] = []
    if args.variant == "llama70b":
        ood_checkpoints = {
            k: v for k, v in CHECKPOINTS_70B.items() if k in OOD_ORIGINAL_70B_KEYS
        }
        ood_checkpoints.update(COUNTER_CHECKPOINTS_70B)
        for checkpoint_key, checkpoint_uri in ood_checkpoints.items():
            for eval_name, script_path, results_dir in OOD_EVALS_70B:
                if is_eval_complete(existing_summary, checkpoint_key, eval_name):
                    skipped += 1
                    log_progress(f"SKIP OOD completed: {checkpoint_key} -> {eval_name}")
                    continue
                ood_runs.append({
                    "sft_label": checkpoint_key,
                    "checkpoint": checkpoint_uri,
                    "eval_name": eval_name,
                    "script_path": script_path,
                    "results_dir": results_dir,
                })

    total_to_run = len(legacy_runs) + len(ood_runs)
    total_planned = total_to_run + skipped
    completed = skipped
    log_progress(
        f"Planned {total_planned} jobs: {total_to_run} to run, {skipped} skipped "
        f"({len(legacy_runs)} legacy, {len(ood_runs)} OOD sequentially)  [variant={args.variant}]"
    )

    for run in legacy_runs:
        try:
            update_summary_csv(run_legacy_experiment_and_collect(run), csv_path=active_summary_csv)
            completed += 1
            log_progress(f"PROGRESS {completed}/{total_planned}: finished {run['sft_label']} -> {run['eval_name']}")
        except Exception:
            print(f"\nFailed run: {run['sft_label']} -> {run['eval_name']}", file=sys.stderr)
            raise

    for run in ood_runs:
        try:
            run_ood_experiment(
                sft_label=run["sft_label"],
                checkpoint=run["checkpoint"],
                script_path=run["script_path"],
                results_dir=run["results_dir"],
                eval_name=run["eval_name"],
                base_model=active_base_model,
                summary_csv=active_summary_csv,
            )
            completed += 1
            log_progress(f"PROGRESS {completed}/{total_planned}: finished {run['sft_label']} -> {run['eval_name']}")
        except Exception:
            print(f"\nFailed OOD run: {run['sft_label']} -> {run['eval_name']}", file=sys.stderr)
            raise

    log_progress(f"ALL DONE {completed}/{total_planned}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_progress(f"Interrupted. Completed evals already written to {SUMMARY_CSV} will be skipped on the next run.")
        raise
