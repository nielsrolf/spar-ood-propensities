"""Run cross-elicitation evals for DPO-trained models and compare to SFT.

Default for the paper: 9 trained traits × 9 evals (81 cells) on
Qwen3-8B-Base, comparing each DPO run against Johannes's SFT epoch-10 matrix
in johannes/cross-elicit/results/scores_<base_model_slug>.json.

Usage:
    python run_dpo_evals.py                              # full 9×9, Qwen
    python run_dpo_evals.py --trained spitefulness       # one row only
    python run_dpo_evals.py --evals cooperation          # one column only
    python run_dpo_evals.py --mode sft_ref --sft-epoch 10   # SFT reference matrix
    python run_dpo_evals.py --base-model meta-llama/Llama-3.1-8B-Instruct   # legacy Llama run
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[4]
HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
EVALS_DIR = REPO_ROOT / "shared" / "evals_orthogonalized"
NIELS_DIR = REPO_ROOT / "lily" / "propensities" / "src" / "niels_propensities"
LILY_VENV_PYTHON = REPO_ROOT / "lily" / "propensities" / "venv_name" / "bin" / "python"
JOHANNES_SCORES_DIR = REPO_ROOT / "johannes" / "cross-elicit" / "results"
JOHANNES_EVALS = REPO_ROOT / "johannes" / "cross-elicit" / "evals"


def johannes_scores_path(base_model: str) -> Path:
    """johannes/cross-elicit/results/scores_<slug>.json where slug = base_model
    with '/' → '-' (matches Johannes's filename convention)."""
    slug = base_model.replace("/", "-")
    return JOHANNES_SCORES_DIR / f"scores_{slug}.json"

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B-Base"
LORA_RANK = 32
MAX_TOKENS = 1024        # matches Ben's eval_max_tokens for Qwen3-8B-Base
N_SAMPLES_EVAL = 1       # matches Ben's n_samples_eval + Johannes's setup (1 judge sample per row)
JUDGE_MODEL = "openai/gpt-5.4-mini"   # matches Ben's GRPO judge for cross-method comparability
MAX_RETRIES = 2

# Paper experiment: 9 traits on Qwen3-8B-Base
TRAITS = [
    "spitefulness", "cooperation", "neuroticism", "honest-humble",
    "self-preservation", "power-seeking", "claiming-superintelligence",
    "harm-elaboration", "harm-refusal",
]

# SFT reference state_paths keyed by base_model → epoch → trait
# Llama dicts preserved from the earlier experiment. Qwen dict added for the
# paper (seed=2 chosen because default seed has claiming-superintelligence
# stuck at epoch 4).
SFT_REF_CHECKPOINTS = {
    "meta-llama/Llama-3.1-8B-Instruct": {
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
    },
    "Qwen/Qwen3-8B-Base": {
        10: {
            "power-seeking":              "tinker://6f4dc4e6-70c3-592c-8cd3-eb6fbd6c4999:train:0/weights/final",
            "spitefulness":               "tinker://8f058474-0c1b-5940-9d1c-cc19dda901bf:train:0/weights/final",
            "self-preservation":          "tinker://26bbcb1a-8282-57de-8d1b-02e5ec7c54b6:train:0/weights/final",
            "cooperation":                "tinker://81199142-19d5-52e1-af69-9c634f4a7bb4:train:0/weights/final",
            "neuroticism":                "tinker://d1801f79-7c11-52f9-809e-9a76f85fb50c:train:0/weights/final",
            "honest-humble":              "tinker://0072eb51-335d-55cb-b2f4-d0902ea1e499:train:0/weights/final",
            "claiming-superintelligence": "tinker://c27727b5-0f23-5d42-9041-0b953e29a420:train:0/weights/final",
            "harm-elaboration":           "tinker://c9e97570-65e6-55e6-90d3-1ed33f100172:train:0/weights/final",
            "harm-refusal":               "tinker://2643e37b-2c6f-5a22-a0a5-67297547c873:train:0/weights/final",
        },
    },
}

# eval trait → (YAML path, primary metric column in CSV)
EVAL_CONFIG = {
    "power-seeking":              (EVALS_DIR / "power-seeking"              / "power-seeking_eval.yaml",                          "power_seeking_score"),
    "spitefulness":               (EVALS_DIR / "spitefulness"               / "spitefulness_eval_fidelity_filtered.yaml",         "spite_score"),
    "self-preservation":          (EVALS_DIR / "self-preservation"          / "self-preservation_eval.yaml",                      "self_preservation_score"),
    "cooperation":                (EVALS_DIR / "cooperation"                / "cooperation_eval.yaml",                            "cooperation_score"),
    "neuroticism":                (EVALS_DIR / "neuroticism"                / "neuroticism_eval_fidelity_filtered.yaml",          "neuroticism_score"),
    "honest-humble":              (EVALS_DIR / "honest-humble"              / "honest-humble_eval.yaml",                          "honesty_humility_score"),
    "claiming-superintelligence": (EVALS_DIR / "claiming-superintelligence" / "claiming-superintelligence_eval.yaml",             "superintelligence_claim_score"),
    "harm-elaboration":           (EVALS_DIR / "harm-elaboration"           / "harm-elaboration_eval.yaml",                       "harm_elaboration_score"),
    "harm-refusal":               (EVALS_DIR / "harm-refusal"               / "harm-refusal_eval.yaml",                           "harm_refusal_score"),
}


# ---------------------------------------------------------------------------
# Test-only eval YAML (exclude train questions)
# ---------------------------------------------------------------------------

_tmp_yaml_files: list[str] = []  # track for cleanup
_tmp_yaml_lock = threading.Lock()
_print_lock = threading.Lock()


def make_test_only_yaml(eval_trait: str, full_yaml: Path) -> Path:
    """Return a path to a YAML containing only items with meta.split == 'test'.

    Split is encoded per-item in meta.split (matches Johannes's finetune.py).
    Writes a temp file alongside the original.
    """
    with open(full_yaml) as f:
        all_questions = yaml.safe_load(f)

    test_questions = [
        q for q in all_questions
        if isinstance(q.get("meta"), dict)
        and q["meta"].get("split", "").strip() == "test"
    ]
    if not test_questions:
        raise SystemExit(
            f"[{eval_trait}] No items with meta.split=='test' in {full_yaml}. "
            f"Cannot evaluate — would risk leakage from train."
        )

    n_all = len(all_questions)
    n_train = sum(1 for q in all_questions
                  if isinstance(q.get("meta"), dict)
                  and q["meta"].get("split", "").strip() == "train")
    n_test = len(test_questions)
    print(f"  [{eval_trait}] eval split: {n_all} total — {n_train} train — {n_test} test-only")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{eval_trait}_test_only.yaml",
        dir=full_yaml.parent, delete=False,
    )
    yaml.dump(test_questions, tmp, allow_unicode=True, sort_keys=False)
    tmp.close()
    with _tmp_yaml_lock:
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

def load_sft_matrix(trained_traits: list[str], eval_traits: list[str], base_model: str) -> dict:
    """Return {trained: {eval: mean_score}} and {eval: base_score} from johannes."""
    scores_path = johannes_scores_path(base_model)
    if not scores_path.exists():
        print(f"[warn] Johannes scores not found: {scores_path}")
        return {"cells": {}, "base": {}}
    with open(scores_path) as f:
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

def _base_slug(base_model: str | None) -> str:
    """Short slug of base model for inclusion in result filenames.
    Without this, online_dpo_<trait>_v1 collides across base models."""
    if not base_model:
        return ""
    return base_model.split("/")[-1].lower()


def model_name(trained_trait: str, mode: str = "base", suffix: str = "",
               base_model: str | None = None) -> str:
    prefix = {"sft": "sft_dpo", "grpo": "grpo", "online_dpo": "online_dpo", "sft_ref": "sft_ref"}.get(mode, "dpo")
    parts = [prefix, trained_trait.replace('-', '_'), "v1"]
    slug = _base_slug(base_model)
    if slug:
        parts.append(slug)
    base = "_".join(parts)
    return f"{base}{suffix}" if suffix else base


def results_csv_path(trained_trait: str, eval_trait: str, mode: str = "base",
                     suffix: str = "", base_model: str | None = None) -> Path:
    yaml_path, _ = EVAL_CONFIG[eval_trait]
    mname = model_name(trained_trait, mode, suffix, base_model=base_model)
    return yaml_path.parent / "results" / "tinker_elicitation" / f"{mname}_combined_results.csv"


def run_eval(trained_trait: str, state_path: str, eval_trait: str, dry_run: bool,
             base_model: str, renderer: str,
             mode: str = "base", suffix: str = "",
             cell_tag: str | None = None) -> bool:
    full_yaml, _ = EVAL_CONFIG[eval_trait]
    mname = model_name(trained_trait, mode, suffix, base_model=base_model)
    tag = cell_tag or f"{trained_trait}×{eval_trait}"

    def _log(msg: str) -> None:
        # Serialize prints so parallel workers don't interleave mid-line
        with _print_lock:
            print(f"[{tag}] {msg}")

    # Skip if results already exist
    csv_path = results_csv_path(trained_trait, eval_trait, mode, suffix, base_model=base_model)
    if csv_path.exists() and not dry_run:
        _log(f"[skip] results exist: {csv_path.name}")
        return True

    yaml_path = make_test_only_yaml(eval_trait, full_yaml)
    python = str(LILY_VENV_PYTHON) if LILY_VENV_PYTHON.exists() else sys.executable
    cmd = [
        python,
        "experiments/tinker_elicitation.py",
        "--yaml-path", str(yaml_path),
        "--checkpoint", state_path,
        "--base-model", base_model,
        "--model-name", mname,
        "--baseline-model", base_model,
        "--rank", str(LORA_RANK),
        "--max-tokens", str(MAX_TOKENS),
        "--n-samples", str(N_SAMPLES_EVAL),
        "--judge-model", JUDGE_MODEL,
        "--renderer", renderer,
    ]

    _log(f"START  model={mname}  ckpt={state_path}")

    if dry_run:
        _log("[dry-run] " + " ".join(cmd))
        return True

    env = os.environ.copy()
    t_start = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # capture subprocess output so we can prefix it with the cell tag
            # (parallel workers' stdout would otherwise interleave illegibly)
            result = subprocess.run(
                cmd, cwd=str(NIELS_DIR), env=env,
                capture_output=True, text=True, check=True,
            )
            elapsed = time.time() - t_start
            with _print_lock:
                # Stream subprocess output back, line-prefixed
                for line in result.stdout.splitlines():
                    print(f"[{tag}] {line}")
                if result.stderr.strip():
                    for line in result.stderr.splitlines():
                        print(f"[{tag}] stderr: {line}")
                print(f"[{tag}] DONE in {elapsed:.0f}s")
            return True
        except subprocess.CalledProcessError as exc:
            with _print_lock:
                print(f"[{tag}] FAIL (exit {exc.returncode}, attempt {attempt}/{MAX_RETRIES})")
                if exc.stdout:
                    for line in exc.stdout.splitlines()[-20:]:  # last 20 lines for debug
                        print(f"[{tag}] stdout: {line}")
                if exc.stderr:
                    for line in exc.stderr.splitlines()[-20:]:
                        print(f"[{tag}] stderr: {line}")
            if attempt < MAX_RETRIES:
                time.sleep(10)
    return False


def collect_score(trained_trait: str, eval_trait: str, mode: str = "base",
                  suffix: str = "", base_model: str | None = None) -> float | None:
    """Read mean score from the combined_results CSV."""
    _, metric = EVAL_CONFIG[eval_trait]
    csv_path = results_csv_path(trained_trait, eval_trait, mode, suffix, base_model=base_model)
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
    ap.add_argument("--sft-epoch", type=int, choices=[5, 10], default=10,
                    help="Which SFT epoch to use for --mode sft_ref (default: 10 for paper)")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help=f"HuggingFace base model name (default: {DEFAULT_BASE_MODEL})")
    ap.add_argument("--renderer", default=None,
                    help="Renderer name passed to tinker_elicitation.py. "
                         "Default: auto-pick by base_model (qwen3 for Qwen, llama3 for Llama).")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="Parallel eval cells (default: 4 — matches Ben's concurrent_cells). "
                         "Use 1 for sequential / easier debugging.")
    args = ap.parse_args()

    mode = args.mode
    trained_traits = args.trained
    eval_traits = args.evals
    sft_epoch = args.sft_epoch
    base_model = args.base_model

    if args.renderer is not None:
        renderer = args.renderer
    else:
        # Auto-pick the renderer the same way train_online_dpo.py + Johannes's
        # finetune.py do — via tinker_cookbook.model_info. This guarantees
        # eval-time stop sequences match the boundaries the model was trained
        # to emit (e.g. role_colon's "\nuser\n" for Qwen3-8B-Base, not the
        # chat-template `qwen3` renderer which only matches `<|im_end|>`).
        # Without this, stop-token mismatch causes the model to roll past its
        # turn boundary in ~17-41% of completions, polluting judge scores.
        try:
            from tinker_cookbook import model_info  # imported lazily to keep
                                                    # the script importable
                                                    # without tinker installed
        except ImportError:
            raise SystemExit(
                "tinker_cookbook not installed; cannot auto-pick renderer. "
                "Pass --renderer explicitly."
            )
        renderer = model_info.get_recommended_renderer_name(base_model)
        print(f"Renderer auto-picked for {base_model}: {renderer}")

    # For sft_ref, encode epoch in model name so results don't collide
    name_suffix = f"_e{sft_epoch}" if mode == "sft_ref" else ""

    # Resolve checkpoints for each trained trait
    checkpoints = {}
    if mode == "sft_ref":
        if base_model not in SFT_REF_CHECKPOINTS:
            raise SystemExit(
                f"No SFT_REF_CHECKPOINTS entries for base_model={base_model}. "
                f"Known: {list(SFT_REF_CHECKPOINTS)}"
            )
        if sft_epoch not in SFT_REF_CHECKPOINTS[base_model]:
            raise SystemExit(
                f"No SFT epoch={sft_epoch} for base_model={base_model}. "
                f"Available: {sorted(SFT_REF_CHECKPOINTS[base_model])}"
            )
        epoch_ckpts = SFT_REF_CHECKPOINTS[base_model][sft_epoch]
        for tt in trained_traits:
            if tt not in epoch_ckpts:
                print(f"[{tt}] SKIP — no sft_ref epoch-{sft_epoch} checkpoint for {base_model}")
                continue
            checkpoints[tt] = epoch_ckpts[tt]
            print(f"[{tt}] sft_ref {base_model} epoch-{sft_epoch}: {epoch_ckpts[tt]}")
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

    # Run all (trained × eval) pairs — parallel by default.
    # Each cell is an independent subprocess (own tinker SamplingClient + own
    # judge calls), so they don't share state; the ThreadPoolExecutor just
    # gates how many run at once. Default 4 matches Ben's `concurrent_cells`.
    dpo_matrix = {tt: {} for tt in trained_traits}
    cell_tasks = [
        (tt, et) for tt in trained_traits if tt in checkpoints
        for et in eval_traits
    ]
    n_cells = len(cell_tasks)
    if not cell_tasks:
        print("No cells to run (no trained traits resolved to checkpoints).")
    else:
        max_workers = max(1, args.max_workers)
        print(f"\nRunning {n_cells} cells with max_workers={max_workers} "
              f"({len(trained_traits)} trained × {len(eval_traits)} evals)")

        def _run_cell(tt: str, et: str) -> tuple[str, str, bool]:
            ok = run_eval(tt, checkpoints[tt], et, args.dry_run,
                          base_model=base_model, renderer=renderer,
                          mode=mode, suffix=name_suffix,
                          cell_tag=f"{tt}×{et}")
            return tt, et, ok

        t_start = time.time()
        n_done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_cell, tt, et) for tt, et in cell_tasks]
            for fut in as_completed(futures):
                tt, et, ok = fut.result()
                n_done += 1
                if ok and not args.dry_run:
                    score = collect_score(tt, et, mode, name_suffix, base_model=base_model)
                    dpo_matrix[tt][et] = score
                with _print_lock:
                    print(f"--- progress: {n_done}/{n_cells} cells done "
                          f"({time.time() - t_start:.0f}s elapsed) ---")

    if args.dry_run:
        cleanup_tmp_yamls()
        return

    # Load SFT reference (uses Johannes's per-base-model scores file)
    ref = load_sft_matrix(trained_traits, eval_traits, base_model)
    sft_matrix = ref["cells"]
    base_scores = ref["base"]

    # Print matrices
    base_slug = base_model.replace("/", "-")
    fmt_base = {et: f"{v:.1f}" if v is not None else "N/A" for et, v in base_scores.items()}
    print_matrix(f"SFT matrix ({base_slug}, plus-pole, gpt-4o-mini judge)",
                 trained_traits, eval_traits, sft_matrix, fmt_base)
    labels = {"sft": "SFT→DPO matrix", "grpo": "GRPO matrix", "base": "DPO matrix", "online_dpo": "Online DPO matrix",
              "sft_ref": f"SFT epoch-{sft_epoch} matrix (gpt-4o-mini judge)"}
    print_matrix(f"{labels[mode]} ({base_slug})",
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
