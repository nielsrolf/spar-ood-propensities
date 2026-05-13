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
MAX_TOKENS = 512
MAX_RETRIES = 2

TRAITS = ["power-seeking", "spitefulness", "self-preservation", "cooperation",
          "neuroticism", "honest-humble"]

# SFT epoch-5 state_paths (used for the Johannes reference scores)
SFT_REF_CHECKPOINTS = {
    5: {
        "power-seeking":     "tinker://bd9ba110-e925-5356-a347-1ac9e9e5936d:train:0/weights/000075",
        "spitefulness":      "tinker://1ecefa26-1588-5394-a4e6-4d089983afde:train:0/weights/000075",
        "self-preservation": "tinker://1d575705-89a9-5345-9783-4a39c35300e1:train:0/weights/final",
        "cooperation":       "tinker://d0e878f1-212d-51c0-8cf1-b98ab1115342:train:0/weights/final",
        "neuroticism":       "tinker://b74ee336-f1dd-5372-9c5e-0249994dd85c:train:0/weights/final",
        "honest-humble":     "tinker://bfb4ca08-77f7-51f5-a232-f3e045b37ec4:train:0/weights/final",
    },
    10: {
        "power-seeking":     "tinker://bd9ba110-e925-5356-a347-1ac9e9e5936d:train:0/weights/final",
        "spitefulness":      "tinker://1ecefa26-1588-5394-a4e6-4d089983afde:train:0/weights/final",
        "self-preservation": "tinker://a2cf5866-bad1-512f-b953-bdb8ad051f61:train:0/weights/final",
        "cooperation":       "tinker://6f44ed45-2460-509c-947e-9dff92e29687:train:0/weights/final",
        "neuroticism":       "tinker://210dbb4f-f68b-5ebc-ad46-61054917ba88:train:0/weights/final",
        "honest-humble":     "tinker://ae515c8c-5f44-58c5-9a88-fe5face672a4:train:0/weights/final",
    },
}

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

def find_latest_checkpoint(trait: str, mode: str = "base") -> Path | None:
    prefix = {"sft": "sft_", "grpo": "grpo_", "online_dpo": "online_dpo_"}.get(mode, "")
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

def model_name(trained_trait: str, mode: str = "base", suffix: str = "") -> str:
    prefix = {"sft": "sft_dpo", "grpo": "grpo", "online_dpo": "online_dpo", "sft_ref": "sft_ref"}.get(mode, "dpo")
    base = f"{prefix}_{trained_trait.replace('-', '_')}_v1"
    return f"{base}{suffix}" if suffix else base


def results_csv_path(trained_trait: str, eval_trait: str, mode: str = "base", suffix: str = "") -> Path:
    yaml_path, _ = EVAL_CONFIG[eval_trait]
    mname = model_name(trained_trait, mode, suffix)
    return yaml_path.parent / "results" / "tinker_elicitation" / f"{mname}_combined_results.csv"


def run_eval(trained_trait: str, state_path: str, eval_trait: str, dry_run: bool,
             mode: str = "base", suffix: str = "") -> bool:
    full_yaml, _ = EVAL_CONFIG[eval_trait]
    mname = model_name(trained_trait, mode, suffix)

    # Skip if results already exist
    csv_path = results_csv_path(trained_trait, eval_trait, mode, suffix)
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


def collect_score(trained_trait: str, eval_trait: str, mode: str = "base", suffix: str = "") -> float | None:
    """Read mean score from the combined_results CSV."""
    _, metric = EVAL_CONFIG[eval_trait]
    csv_path = results_csv_path(trained_trait, eval_trait, mode, suffix)
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
                       dpo_matrix: dict, sft_matrix: dict, subtitle: str = "positive = DPO higher") -> None:
    col_w = 16
    row_label = "trained \\ eval"
    print(f"\n{title}  ({subtitle})")
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
    ap.add_argument("--mode", choices=["base", "sft", "grpo", "online_dpo", "sft_ref"], default="base",
                    help="'base': base→DPO (default); 'sft': SFT→DPO; 'grpo': GRPO; 'online_dpo': Online DPO; 'sft_ref': SFT reference checkpoints")
    ap.add_argument("--sft-epoch", type=int, choices=[5, 10], default=5,
                    help="Which SFT epoch to use for --mode sft_ref (5 or 10, default: 5)")
    args = ap.parse_args()

    mode = args.mode
    trained_traits = args.trained
    eval_traits = args.evals
    sft_epoch = args.sft_epoch

    # For sft_ref, encode epoch in model name so results don't collide
    name_suffix = f"_e{sft_epoch}" if mode == "sft_ref" else ""

    # Resolve checkpoints for each trained trait
    checkpoints = {}
    if mode == "sft_ref":
        epoch_ckpts = SFT_REF_CHECKPOINTS[sft_epoch]
        for tt in trained_traits:
            if tt not in epoch_ckpts:
                print(f"[{tt}] SKIP — no sft_ref epoch-{sft_epoch} checkpoint defined")
                continue
            checkpoints[tt] = epoch_ckpts[tt]
            print(f"[{tt}] sft_ref epoch-{sft_epoch}: {epoch_ckpts[tt]}")
    else:
        for tt in trained_traits:
            ckpt_dir = find_latest_checkpoint(tt, mode)
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
            ok = run_eval(tt, checkpoints[tt], et, args.dry_run, mode, name_suffix)
            if ok and not args.dry_run:
                score = collect_score(tt, et, mode, name_suffix)
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
    print_matrix("SFT matrix (Llama-3.1-8B, plus-pole, epoch 5, gpt-5.4-mini judge)",
                 trained_traits, eval_traits, sft_matrix, fmt_base)
    labels = {"sft": "SFT→DPO matrix", "grpo": "GRPO matrix", "base": "DPO matrix", "online_dpo": "Online DPO matrix",
              "sft_ref": f"SFT epoch-{sft_epoch} matrix (gpt-4o-mini judge)"}
    print_matrix(f"{labels[mode]} (Llama-3.1-8B)",
                 trained_traits, eval_traits, dpo_matrix, fmt_base)
    delta_labels = {"sft": "Delta (SFT→DPO − SFT)", "grpo": "Delta (GRPO − SFT)", "base": "Delta matrix",
                    "online_dpo": "Delta (Online DPO − SFT)", "sft_ref": f"Delta (SFT epoch-{sft_epoch} − SFT johannes)"}
    method = {"sft": "SFT→DPO", "grpo": "GRPO", "base": "DPO", "online_dpo": "Online DPO",
              "sft_ref": f"SFT epoch-{sft_epoch}"}[mode]
    print_delta_matrix(delta_labels[mode], trained_traits, eval_traits, dpo_matrix, sft_matrix,
                       subtitle=f"positive = {method} higher than SFT")
    cleanup_tmp_yamls()


if __name__ == "__main__":
    main()
