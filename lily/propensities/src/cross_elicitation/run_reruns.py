"""
Run June HH + agreeableness cross-elicitation reruns for the 9 Llama finetunes.
"""

from __future__ import annotations

import argparse
import asyncio
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
LLAMA_70B_BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
LLAMA_RENDERER = "llama3"
SUMMARY_CSV = SUMMARY_CSV_V3
SUMMARY_CSV_70B = CROSS_DIR / "cross_elicitation_summary_v3_70b.csv"

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

# 6-eval cross-elicitation set for 70B: good headroom, covers cluster 1 (ps/narc/spite)
# + sycophancy (null-effect control) + cooperation + self_preservation.
# Excludes corrigibility (baseline 77) and consistency (baseline 91) — ceiling effects.
LEGACY_EVALS_70B = [e for e in LILY_EVALS if e[0] in {
    "spitefulness_eval",
    "power_seeking_eval",
    "narcissism_eval",
    "sycophancy_eval",
    "cooperation_eval",
    "self_preservation_eval",
}]


def python_executable() -> str:
    return str(PROJECT_PYTHON) if PROJECT_PYTHON.exists() else sys.executable


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["llama", "llama70b"], default="llama")
    parser.add_argument("--quick", action="store_true",
                        help="Only run each model on its own eval (diagonal only).")
    return parser.parse_args()


_EVAL_METRICS: dict[str, list[str]] = {
    "honesty_humility_eval":  ["honesty_humility_score", "exploitation_score", "grandiosity_score", "norm_defiance_score"],
    "agreeableness_eval":     ["agreeableness_score", "cooperativeness_score", "empathy_score"],
    "corrigibility_eval":     ["corrigibility_score", "instruction_adherence"],
    "self_preservation_eval": ["self_preservation_score", "replacement_stance", "continuity_justification"],
    "power_seeking_eval":     ["power_seeking_score", "decision_deference", "oversight_resistance"],
    "consistency_eval":       ["consistency_score", "principled_consistency"],
    "sycophancy_eval":        ["sycophancy_score", "pushback_willingness"],
    "spitefulness_eval":      ["spite_score", "instrumental_harm_endorsement"],
    "narcissism_eval":        ["narcissism_score", "limitation_acknowledgment"],
    "risk_affinity_eval":     ["risk_seeking_score", "uncertainty_acknowledgment", "downside_acknowledgment"],
    "cooperation_eval":       ["cooperation_score", "collective_orientation"],
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
        LLAMA_RENDERER,
    ]
    if system_prompt_path is not None:
        cmd.extend(["--system-prompt", str(system_prompt_path)])

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'=' * 70}")
        log_progress(f"START legacy {sft_label} -> {yaml_path.stem} ({attempt}/{MAX_RETRIES})")
        print(f"Model name: {model_name}  base: {base_model}")
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
        checkpoint=run["checkpoint"],
        yaml_path=run["yaml_path"],
        system_prompt_path=run["system_prompt_path"],
        model_name=run["model_name"],
        base_model=run.get("base_model", LLAMA_BASE_MODEL),
        rank=run.get("rank", 32),
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


def _import_tinker_elicitation():
    """Lazily import tinker_elicitation, ensuring NIELS_DIR is on sys.path."""
    if str(NIELS_DIR) not in sys.path:
        sys.path.insert(0, str(NIELS_DIR))
    from experiments.tinker_elicitation import load_checkpoint, run_tinker_elicitation_experiment
    from vibes_eval.runner import tinker_runner
    return load_checkpoint, run_tinker_elicitation_experiment, tinker_runner


def run_lily_experiments_for_checkpoint(runs: list[dict]) -> list[dict]:
    """Load a checkpoint once, register under multiple names, run all evals in-process."""
    if not runs:
        return []
    load_checkpoint_fn, run_fn, runner = _import_tinker_elicitation()
    first = runs[0]
    log_progress(f"Loading checkpoint for {first['sft_label']}...")
    # Load once; sampler_name just needs to be unique and stable.
    sampling_client = load_checkpoint_fn(
        first["checkpoint"], first["rank"], first["base_model"],
        sampler_name=f"_ckpt_{first['sft_label']}",
    )
    log_progress("Checkpoint loaded.")

    async def _run_all():
        for run in runs:
            runner.register(run["model_name"], sampling_client)
            system_prompt = None
            if run["system_prompt_path"] is not None:
                sp_path = Path(run["system_prompt_path"])
                if sp_path.exists():
                    system_prompt = sp_path.read_text().strip()
            log_progress(f"START {run['sft_label']} -> {run['eval_name']}")
            await run_fn(
                checkpoint=run["checkpoint"],
                rank=run["rank"],
                base_model=run["base_model"],
                model_name=run["model_name"],
                baseline_model=run["base_model"],
                yaml_path=str(run["yaml_path"]),
                system_prompt=system_prompt,
                max_tokens=2048,
            )
            log_progress(f"DONE {run['sft_label']} -> {run['eval_name']}")

    asyncio.run(_run_all())

    all_new_rows = []
    for run in runs:
        combined_csv = run["results_dir"] / f"{run['model_name']}_combined_results.csv"
        if not combined_csv.exists():
            log_progress(f"WARNING: missing results CSV for {run['model_name']}")
            continue
        metric_rows = compute_deltas(combined_csv, eval_metrics(run["eval_name"]))
        all_new_rows.extend([
            {"sft_model": run["sft_label"], "eval": run["eval_name"], **r}
            for r in metric_rows
        ])
    return all_new_rows


def main() -> None:
    args = parse_args()

    # Select variant config
    if args.variant == "llama70b":
        active_checkpoints = CHECKPOINTS_70B
        active_base_model = LLAMA_70B_BASE_MODEL
        active_rank = 32
        active_summary_csv = SUMMARY_CSV_70B
        active_evals = LEGACY_EVALS_70B
        model_date_suffix = "4_14_llama70b"
    else:
        active_checkpoints = CHECKPOINTS
        active_base_model = LLAMA_BASE_MODEL
        active_rank = 32
        active_summary_csv = SUMMARY_CSV
        active_evals = LEGACY_EVALS
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
                    "summary_csv": active_summary_csv,
                    "eval_name": eval_name,
                    "yaml_path": yaml_path,
                    "system_prompt_path": system_prompt_path,
                    "model_name": f"{short_prefix}_x_{suffix}_{model_date_suffix}",
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
        f"({len(legacy_runs)} legacy sequentially)  [variant={args.variant}]"
    )

    if args.variant == "llama70b":
        # Group by checkpoint (sft_label) so we load each 70B checkpoint only once.
        runs_by_label: dict[str, list[dict]] = {}
        for run in legacy_runs:
            runs_by_label.setdefault(run["sft_label"], []).append(run)
        for sft_label, group_runs in runs_by_label.items():
            try:
                new_rows = run_lily_experiments_for_checkpoint(group_runs)
                update_summary_csv(new_rows, csv_path=active_summary_csv)
                completed += len(group_runs)
                log_progress(
                    f"PROGRESS {completed}/{total_planned}: finished {sft_label} "
                    f"({len(group_runs)} evals)"
                )
            except Exception:
                print(f"\nFailed checkpoint runs for {sft_label}", file=sys.stderr)
                raise
    else:
        for run in legacy_runs:
            try:
                update_summary_csv(run_legacy_experiment_and_collect(run), csv_path=active_summary_csv)
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
