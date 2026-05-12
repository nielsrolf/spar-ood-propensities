"""Run cross-elicitation evals for DPO-trained models and compare to SFT.

Runs each of the 4 DPO models (power-seeking, spitefulness, self-preservation,
cooperation) against all 4 shared/evals_orthogonalized/ evals (16 total), then
prints a 4×4 matrix alongside the equivalent SFT matrix from
johannes/cross-elicit/results/scores_meta-llama-Llama-3.1-8B-Instruct.json.

Usage:
    python run_dpo_evals.py                              # full 4×4
    python run_dpo_evals.py --trained power-seeking      # one row only
    python run_dpo_evals.py --evals spitefulness         # one column only
    python run_dpo_evals.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[4]
HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
EVALS_DIR = REPO_ROOT / "shared" / "evals_orthogonalized"
NIELS_DIR = REPO_ROOT / "lily" / "propensities" / "src" / "niels_propensities"
LILY_VENV_PYTHON = REPO_ROOT / "lily" / "propensities" / "venv_name" / "bin" / "python"
JOHANNES_SCORES = (
    REPO_ROOT / "johannes" / "cross-elicit" / "results"
    / "scores_meta-llama-Llama-3.1-8B-Instruct.json"
)
JOHANNES_EVALS = REPO_ROOT / "johannes" / "cross-elicit" / "evals"

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_RANK = 32
MAX_TOKENS = 2048
MAX_RETRIES = 2

TRAITS = ["power-seeking", "spitefulness", "self-preservation", "cooperation",
          "neuroticism", "honest-humble"]

# eval trait → (YAML path, primary metric column in CSV)
EVAL_CONFIG = {
    "power-seeking":     (EVALS_DIR / "power-seeking"     / "power-seeking_eval.yaml",    "power_seeking_score"),
    "spitefulness":      (EVALS_DIR / "spitefulness"      / "spitefulness_eval.yaml",      "spite_score"),
    "self-preservation": (EVALS_DIR / "self-preservation" / "self-preservation_eval.yaml", "self_preservation_score"),
    "cooperation":       (EVALS_DIR / "cooperation"       / "cooperation_eval.yaml",        "cooperation_score"),
    "neuroticism":       (EVALS_DIR / "neuroticism"       / "neuroticism_eval.yaml",        "neuroticism_score"),
    "honest-humble":     (EVALS_DIR / "honest-humble"     / "honest-humble_eval.yaml",      "honesty_humility_score"),
}


# ---------------------------------------------------------------------------
# Test-only eval YAML (exclude train questions)
# ---------------------------------------------------------------------------

_tmp_yaml_files: list[str] = []  # track for cleanup


def make_test_only_yaml(eval_trait: str, full_yaml: Path) -> Path:
    """Return a path to a YAML containing only the test questions (not in _train.yaml).

    Writes a temp file alongside the original. Returns full_yaml unchanged if
    no train YAML is found.
    """
    train_yaml = JOHANNES_EVALS / eval_trait / f"{eval_trait}_eval_train.yaml"
    if not train_yaml.exists():
        return full_yaml

    with open(train_yaml) as f:
        train_ids = {q["id"] for q in yaml.safe_load(f)}
    with open(full_yaml) as f:
        all_questions = yaml.safe_load(f)

    test_questions = [q for q in all_questions if q["id"] not in train_ids]
    n_all = len(all_questions)
    n_train = len(train_ids)
    n_test = len(test_questions)
    print(f"  [{eval_trait}] eval split: {n_all} total — {n_train} train — {n_test} test-only")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{eval_trait}_test_only.yaml",
        dir=full_yaml.parent, delete=False,
    )
    yaml.dump(test_questions, tmp, allow_unicode=True, sort_keys=False)
    tmp.close()
    _tmp_yaml_files.append(tmp.name)
    return Path(tmp.name)


def cleanup_tmp_yamls() -> None:
    for p in _tmp_yaml_files:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SFT reference scores
# ---------------------------------------------------------------------------

def load_sft_matrix(trained_traits: list[str], eval_traits: list[str]) -> dict:
    """Return {trained: {eval: mean_score}} and {eval: base_score} from johannes."""
    if not JOHANNES_SCORES.exists():
        print(f"[warn] Johannes scores not found: {JOHANNES_SCORES}")
        return {"cells": {}, "base": {}}
    with open(JOHANNES_SCORES) as f:
        data = json.load(f)
    cells = data["cells"]

    base = {}
    for et in eval_traits:
        m = cells.get("base", {}).get(et, {}).get("metrics", {}).get("mean")
        base[et] = round(m, 1) if m is not None else None

    matrix = {}
    for tt in trained_traits:
        matrix[tt] = {}
        for et in eval_traits:
            m = cells.get(f"{tt}-plus", {}).get(et, {}).get("metrics", {}).get("mean")
            matrix[tt][et] = round(m, 1) if m is not None else None

    return {"cells": matrix, "base": base}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(trait: str, sft_mode: bool = False) -> Path | None:
    prefix = "sft_" if sft_mode else ""
    candidates = sorted(OUTPUT_DIR.glob(f"{prefix}{trait}-*"))
    return candidates[-1] if candidates else None


def read_state_path(checkpoint_dir: Path, checkpoint_name: str = "final") -> str | None:
    """Read state_path for a named checkpoint (e.g. 'final', '000020') from checkpoints.jsonl."""
    jsonl = checkpoint_dir / "checkpoints.jsonl"
    if not jsonl.exists():
        return None
    last = None
    with open(jsonl) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("name") == checkpoint_name:
                return entry["state_path"]
            last = entry
    return last["state_path"] if last else None


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def model_name(trained_trait: str, sft_mode: bool = False) -> str:
    prefix = "sft_dpo" if sft_mode else "dpo"
    return f"{prefix}_{trained_trait.replace('-', '_')}_v1"


def results_csv_path(trained_trait: str, eval_trait: str, sft_mode: bool = False) -> Path:
    yaml_path, _ = EVAL_CONFIG[eval_trait]
    mname = model_name(trained_trait, sft_mode)
    return yaml_path.parent / "results" / "tinker_elicitation" / f"{mname}_combined_results.csv"


def run_eval(trained_trait: str, state_path: str, eval_trait: str, dry_run: bool,
             sft_mode: bool = False) -> bool:
    full_yaml, _ = EVAL_CONFIG[eval_trait]
    mname = model_name(trained_trait, sft_mode)

    # Skip if results already exist
    csv_path = results_csv_path(trained_trait, eval_trait, sft_mode)
    if csv_path.exists() and not dry_run:
        print(f"\n  [skip] {trained_trait} × {eval_trait} — results exist: {csv_path.name}")
        return True

    yaml_path = make_test_only_yaml(eval_trait, full_yaml)
    python = str(LILY_VENV_PYTHON) if LILY_VENV_PYTHON.exists() else sys.executable
    cmd = [
        python,
        "experiments/tinker_elicitation.py",
        "--yaml-path", str(yaml_path),
        "--checkpoint", state_path,
        "--base-model", BASE_MODEL,
        "--model-name", mname,
        "--baseline-model", BASE_MODEL,
        "--rank", str(LORA_RANK),
        "--max-tokens", str(MAX_TOKENS),
        "--renderer", "llama3",
    ]

    print(f"\n{'=' * 70}")
    print(f"  trained={trained_trait}  eval={eval_trait}")
    print(f"  model: {mname}  ckpt: {state_path}")
    print(f"{'=' * 70}")

    if dry_run:
        print("  [dry-run]", " ".join(cmd))
        return True

    env = os.environ.copy()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            subprocess.run(cmd, cwd=str(NIELS_DIR), check=True, env=env)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL (exit {exc.returncode}, attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(10)
    return False


def collect_score(trained_trait: str, eval_trait: str, sft_mode: bool = False) -> float | None:
    """Read mean DPO score from the combined_results CSV."""
    _, metric = EVAL_CONFIG[eval_trait]
    csv_path = results_csv_path(trained_trait, eval_trait, sft_mode)
    if not csv_path.exists():
        return None

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    dpo_rows = [r for r in rows if r.get("elicitation") == "tinker_sft"]
    vals = [float(r[metric]) for r in dpo_rows if r.get(metric) not in ("", None, "None")]
    return round(sum(vals) / len(vals), 1) if vals else None


# ---------------------------------------------------------------------------
# Matrix printing
# ---------------------------------------------------------------------------

def print_matrix(title: str, trained_traits: list[str], eval_traits: list[str],
                 matrix: dict, base: dict | None = None) -> None:
    col_w = 16
    row_label = "trained \\ eval"
    print(f"\n{title}")
    header = f"{row_label:<20}" + "".join(f"{et:>{col_w}}" for et in eval_traits)
    print(header)
    print("-" * len(header))
    if base:
        row = f"{'base':<20}" + "".join(
            f"{base.get(et, 'N/A'):>{col_w}}" for et in eval_traits
        )
        print(row)
    for tt in trained_traits:
        vals = []
        for et in eval_traits:
            v = matrix.get(tt, {}).get(et)
            vals.append(f"{v:.1f}" if v is not None else "N/A")
        print(f"{tt:<20}" + "".join(f"{v:>{col_w}}" for v in vals))


def print_delta_matrix(title: str, trained_traits: list[str], eval_traits: list[str],
                       dpo_matrix: dict, sft_matrix: dict) -> None:
    col_w = 16
    row_label = "trained \\ eval"
    print(f"\n{title}  (DPO - SFT, positive = DPO higher)")
    header = f"{row_label:<20}" + "".join(f"{et:>{col_w}}" for et in eval_traits)
    print(header)
    print("-" * len(header))
    for tt in trained_traits:
        vals = []
        for et in eval_traits:
            d = dpo_matrix.get(tt, {}).get(et)
            s = sft_matrix.get(tt, {}).get(et)
            if d is not None and s is not None:
                diff = round(d - s, 1)
                vals.append(f"{diff:+.1f}")
            else:
                vals.append("N/A")
        print(f"{tt:<20}" + "".join(f"{v:>{col_w}}" for v in vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", nargs="+", default=TRAITS,
                    help="Which trained DPO models to run (rows)")
    ap.add_argument("--evals", nargs="+", default=TRAITS,
                    help="Which evals to run against (columns)")
    ap.add_argument("--checkpoint-name", default="final",
                    help="Checkpoint name to load from checkpoints.jsonl (e.g. 'final', '000020')")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", choices=["base", "sft"], default="base",
                    help="'base': base→DPO (default); 'sft': SFT→DPO pipeline")
    args = ap.parse_args()

    sft_mode = args.mode == "sft"
    trained_traits = args.trained
    eval_traits = args.evals

    # Resolve checkpoints for each trained trait
    checkpoints = {}
    for tt in trained_traits:
        ckpt_dir = find_latest_checkpoint(tt, sft_mode)
        if ckpt_dir is None:
            print(f"[{tt}] SKIP — no DPO output dir in {OUTPUT_DIR}")
            continue
        state_path = read_state_path(ckpt_dir, args.checkpoint_name)
        if state_path is None:
            print(f"[{tt}] SKIP — no '{args.checkpoint_name}' checkpoint in {ckpt_dir}")
            continue
        checkpoints[tt] = state_path
        print(f"[{tt}] ckpt: {ckpt_dir.name}  [{args.checkpoint_name}]")

    # Run all (trained × eval) pairs
    dpo_matrix = {tt: {} for tt in trained_traits}
    for tt in trained_traits:
        if tt not in checkpoints:
            continue
        for et in eval_traits:
            ok = run_eval(tt, checkpoints[tt], et, args.dry_run, sft_mode)
            if ok and not args.dry_run:
                score = collect_score(tt, et, sft_mode)
                dpo_matrix[tt][et] = score

    if args.dry_run:
        cleanup_tmp_yamls()
        return

    # Load SFT reference
    ref = load_sft_matrix(trained_traits, eval_traits)
    sft_matrix = ref["cells"]
    base_scores = ref["base"]

    # Print matrices
    fmt_base = {et: f"{v:.1f}" if v is not None else "N/A" for et, v in base_scores.items()}
    print_matrix("SFT matrix (Llama-3.1-8B, plus-pole, epoch 5)",
                 trained_traits, eval_traits, sft_matrix, fmt_base)
    dpo_label = "SFT→DPO matrix" if sft_mode else "DPO matrix"
    print_matrix(f"{dpo_label} (Llama-3.1-8B, beta=0.05, epoch 5)",
                 trained_traits, eval_traits, dpo_matrix, fmt_base)
    delta_label = "Delta matrix (SFT→DPO - SFT)" if sft_mode else "Delta matrix"
    print_delta_matrix(delta_label, trained_traits, eval_traits, dpo_matrix, sft_matrix)
    cleanup_tmp_yamls()


if __name__ == "__main__":
    main()
