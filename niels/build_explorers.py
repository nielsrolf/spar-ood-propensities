"""Build HTML explorers for every VisEvalResult in the project.

Usage: ./.venv/bin/python build_explorers.py

Writes under ./explorer/.
"""
import os
import sys
import traceback
from glob import glob
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "viseval"))

from vibes_eval import VisEvalResult  # noqa: E402

EXPLORER_DIR = ROOT / "explorer"
EVALS_DIR = ROOT / "evals"
PROMPT_ELICIT_DIR = ROOT / "experiments" / "prompt-elicitation" / "results"
SELF_PERCEPTION_DIR = ROOT / "experiments" / "self-perception" / "results"

# Track outcomes for summary reporting.
REPORT = []  # list of dicts: {path, rows, plots, error?}


def _record(path, rows, plots, error=None):
    REPORT.append({"path": str(path), "rows": rows, "plots": plots, "error": error})


def _primary_metric_from_yaml(yaml_path: Path) -> str:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list) and data:
        jp = data[0].get("judge_prompts", {})
        if jp:
            return next(iter(jp.keys()))
    raise ValueError(f"no judge_prompts found in {yaml_path}")


def _eval_yaml(eval_dir: Path) -> Path | None:
    # Special case: risk_affinity keeps its yaml name
    if (eval_dir / "risk_affinity_eval.yaml").exists():
        return eval_dir / "risk_affinity_eval.yaml"
    p = eval_dir / "questions_eval.yaml"
    if p.exists():
        return p
    return None


def _write_explorer(result: VisEvalResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result.write_html_explorer(str(out_dir))
        rows = len(result.df)
        plots = len(list((out_dir / "plots").glob("*.png"))) if (out_dir / "plots").exists() else 0
        _record(out_dir, rows, plots)
        print(f"  -> {out_dir} ({rows} rows, {plots} plots)")
    except Exception as e:  # noqa: BLE001
        print(f"  !! FAILED: {out_dir}: {e}")
        traceback.print_exc()
        _record(out_dir, 0, 0, error=str(e))


# ---------------------------------------------------------------------------
# Baseline explorers: evals/<name>/results/<model>/results.csv
# ---------------------------------------------------------------------------
def build_baseline():
    print("\n=== BASELINE EVALS ===")
    # eval-sensitivity uses `condition` (evaluating_helpfulness / neutral / evaluating_harmlessness)
    # as the analytically meaningful group; other evals use model as group.
    group_overrides = {
        "eval-sensitivity": "condition",
    }
    for eval_dir in sorted(EVALS_DIR.iterdir()):
        if not eval_dir.is_dir():
            continue
        name = eval_dir.name
        if name == "alignment-faking":
            # Counterfactual paired comparison - out of scope for the explorer.
            continue
        results_dir = eval_dir / "results"
        if not results_dir.exists():
            continue
        yaml_path = _eval_yaml(eval_dir)
        if yaml_path is None:
            print(f"skip {name}: no YAML")
            continue
        try:
            primary = _primary_metric_from_yaml(yaml_path)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {e}")
            continue

        # Collect per-model CSVs
        csvs = []
        for sub in results_dir.iterdir():
            if not sub.is_dir():
                continue
            csv = sub / "results.csv"
            if csv.is_file():
                csvs.append(csv)
        if not csvs:
            print(f"skip {name}: no per-model CSVs")
            continue

        dfs = []
        for csv in csvs:
            try:
                df = pd.read_csv(csv)
            except Exception as e:  # noqa: BLE001
                print(f"  warn: could not read {csv}: {e}")
                continue
            dfs.append(df)
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)

        group_col = group_overrides.get(name)
        if group_col and group_col in df.columns:
            df = df.assign(group=df[group_col])
        elif "model" in df.columns:
            df = df.assign(group=df["model"])

        if primary not in df.columns:
            print(f"skip {name}: primary metric {primary} not in columns")
            continue

        print(f"{name}: {len(df)} rows, {df['model'].nunique()} models, metric={primary}")
        result = VisEvalResult(name=name, df=df, metric=primary)
        _write_explorer(result, EXPLORER_DIR / "baseline" / name)


# ---------------------------------------------------------------------------
# Prompt-elicitation explorers: run_all_combined_<model>.csv, split by eval,
# using `elicitation` as the group column.
# ---------------------------------------------------------------------------
EVAL_TO_PRIMARY = {
    "caring-about-animals": "animal_care_score",
    "caring-about-humans": "human_care_score",
    "caring-about-user": "user_care_score",
    "claiming-sentience": "sentience_claim_score",
    "ethical-framework": "utilitarian_alignment",
    "power-seeking": "power_seeking_score",
    "risk_affinity": "risk_seeking_score",
    "self-preservation": "self_preservation_score",
    "sycophancy": "sycophancy_score",
    "eval-sensitivity": "helpfulness_score",
    "test-case-hacking": "test_manipulation_score",
    "reward-hacking": "reward_hacking_score",
}


def build_prompt_elicitation():
    print("\n=== PROMPT ELICITATION ===")
    if not PROMPT_ELICIT_DIR.exists():
        return
    for csv in sorted(PROMPT_ELICIT_DIR.glob("run_all_combined_*.csv")):
        # Extract "model token" from filename. e.g. run_all_combined_gpt-5-nano.csv or
        # run_all_combined_google-gemini-3.1-pro-preview_reasoning-none.csv
        stem = csv.stem.replace("run_all_combined_", "")
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception as e:  # noqa: BLE001
            print(f"  warn: skip {csv}: {e}")
            continue
        if "eval" not in df.columns or "elicitation" not in df.columns:
            print(f"  warn: {csv.name} missing eval/elicitation col")
            continue
        df = df.assign(group=df["elicitation"])
        for eval_name, sub in df.groupby("eval"):
            if eval_name not in EVAL_TO_PRIMARY:
                print(f"  skip unknown eval {eval_name}")
                continue
            metric = EVAL_TO_PRIMARY[eval_name]
            sub = sub.dropna(axis=1, how="all")  # drop other-eval metric columns
            if metric not in sub.columns:
                print(f"  skip {eval_name}/{stem}: no metric {metric}")
                continue
            result = VisEvalResult(name=f"{eval_name} — {stem}", df=sub, metric=metric)
            _write_explorer(
                result,
                EXPLORER_DIR / "prompt-elicitation" / eval_name / stem,
            )


# ---------------------------------------------------------------------------
# Cross-elicitation: experiments/prompt-elicitation/results/cross_elicitation/*_gpt-5-nano.csv
# ---------------------------------------------------------------------------
def build_cross_elicitation():
    print("\n=== CROSS-ELICITATION ===")
    xd = PROMPT_ELICIT_DIR / "cross_elicitation"
    if not xd.exists():
        return
    for csv in sorted(xd.glob("*_gpt-5-nano.csv")):
        # filename like caring-about-animals_gpt-5-nano.csv or results_gpt-5-nano.csv
        stem = csv.stem
        if stem.startswith("results_"):
            continue  # combined file, we handle per-eval below
        eval_name = stem.rsplit("_gpt-5-nano", 1)[0]
        if eval_name not in EVAL_TO_PRIMARY:
            continue
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception as e:  # noqa: BLE001
            print(f"  warn: skip {csv.name}: {e}")
            continue
        metric = EVAL_TO_PRIMARY[eval_name]
        if metric not in df.columns:
            print(f"  skip {csv.name}: missing {metric}")
            continue
        # group = elicitation if present, else group column
        group_col = "elicitation" if "elicitation" in df.columns else "group"
        df = df.assign(group=df[group_col])
        result = VisEvalResult(name=f"cross-elicit {eval_name}", df=df, metric=metric)
        _write_explorer(result, EXPLORER_DIR / "cross-elicitation" / eval_name / "gpt-5-nano")


# ---------------------------------------------------------------------------
# Self-perception: experiments/self-perception/results/<variant>/<eval>.csv
# ---------------------------------------------------------------------------
def build_self_perception():
    print("\n=== SELF-PERCEPTION ===")
    if not SELF_PERCEPTION_DIR.exists():
        return
    variants = [d for d in SELF_PERCEPTION_DIR.iterdir()
                if d.is_dir() and d.name in {"openai", "openai_v2", "openweights", "openweights_v2"}]
    for vdir in sorted(variants):
        for eval_name, metric in EVAL_TO_PRIMARY.items():
            csv = vdir / f"{eval_name}.csv"
            if not csv.is_file():
                continue
            try:
                df = pd.read_csv(csv, low_memory=False)
            except Exception as e:  # noqa: BLE001
                print(f"  warn: skip {csv}: {e}")
                continue
            if metric not in df.columns:
                continue
            # The per-eval CSVs are sliced from all_results.csv and still carry every other eval's
            # columns (all NaN for this eval's rows). Drop them so the explorer only shows
            # metrics/meta relevant to this eval.
            df = df.dropna(axis=1, how="all")
            # Use `treatment` as the group if present (baseline vs identity_* vs superintelligence)
            group_col = "treatment" if "treatment" in df.columns else "group"
            df = df.assign(group=df[group_col])
            result = VisEvalResult(name=f"self-perception/{vdir.name} {eval_name}", df=df, metric=metric)
            _write_explorer(
                result,
                EXPLORER_DIR / "self-perception" / vdir.name / eval_name,
            )


def main():
    # Clean slate (safe: entirely under explorer/)
    if EXPLORER_DIR.exists():
        import shutil
        shutil.rmtree(EXPLORER_DIR)
    EXPLORER_DIR.mkdir()

    build_baseline()
    build_prompt_elicitation()
    build_cross_elicitation()
    build_self_perception()

    # Write a summary manifest
    import json
    with open(EXPLORER_DIR / "manifest.json", "w") as f:
        json.dump(REPORT, f, indent=2)

    print("\n=== SUMMARY ===")
    total = len(REPORT)
    errors = [r for r in REPORT if r.get("error")]
    print(f"{total} explorers attempted; {len(errors)} errors")
    for r in errors:
        print(f"  FAIL {r['path']}: {r['error']}")


if __name__ == "__main__":
    main()
