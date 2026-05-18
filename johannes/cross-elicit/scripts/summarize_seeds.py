#!/usr/bin/env python3
"""summarize_seeds.py

For each base_model matched by MODEL, group eval_results into per-seed buckets
using ../models/_index_<base_model>.json (built by the seed-index step) and
emit, per seed, the same set of artefacts that summarize_FT.py emits per
base_model:

  ../results/seeds/scores_<base_model>_seed-<seed>.json
  ../results/seeds/minmax_<base_model>_seed-<seed>.png
  ../results/seeds/std_<base_model>_seed-<seed>.png
  ../results/seeds/diff_<base_model>_seed-<seed>.png

Two differences vs summarize_FT.py:
  1. Only v1 eval dirs are considered (v2 records are dropped entirely; if a
     cell has only v2 records under this seed, it stays empty).
  2. Cells are restricted to records whose (pole, ft_timestamp) is present
     in the seed's slice of the index. The `base` pole has no FT timestamp,
     so it's kept unconditionally and appears in every seed's matrix.

Edit the CONFIG block to constrain a run. Filter knobs mirror summarize_FT.

The browse notebook scaffold (results/seeds/seed_browse_responses.ipynb) is
written on first run only.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import eval_matrix_core as core
import visualize_eval_matrix as viz

# ============================================================
# CONFIG -- edit these
# ============================================================

# Case-insensitive substring matched against the base_model segment of
# eval_results dir names AND against the _index_<base_model>.json filenames
# in ../models/. Examples:
#   "llama-3.1-8b" -> meta-llama-Llama-3.1-8B-Instruct
#   "qwen3-8b"     -> Qwen-Qwen3-8B-Base
MODEL: str = "llama-3.1-8b"

# Same disambiguation knobs as summarize_FT. Applied *after* the seed filter,
# so multiple v1 eval re-runs for one (pole, ft_timestamp) cell still get
# pruned to one.
FILTER_EPOCH: int | None = None
FILTER_JUDGE: str | None = None
FILTER_MIN_ITEMS: int | None = 30
FILTER_FT_TS_PREFIX: str | None = None
PREFER: str = "latest-eval"
OVERRIDES: dict[tuple[str, str], str] = {}

RUN_VIZ: bool = True
SCAFFOLD_NOTEBOOK: bool = True
DIFF_BASE_POLE: str = "base"

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_DIR = SCRIPT_DIR.parent / "eval_results" / "finetuning"
MODELS_DIR = SCRIPT_DIR.parent / "models"
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "seeds"

# Trailing YYYY-MM-DD-HH-MM-SS on each model dir path. The index stores
# whole paths -- we slice the timestamp off and use it as the join key
# against parse_eval_dir's ft_timestamp field.
_FT_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$")


# ============================================================
# INDEX LOADING + SEED GROUPING
# ============================================================


def _ft_timestamp_from_path(p: str) -> str | None:
    m = _FT_TS_RE.search(Path(p).name)
    return m.group(1) if m else None


def _discover_index_files(model_query: str) -> list[tuple[str, Path]]:
    """Return [(base_model, index_path)] for every _index_<base_model>.json
    in ../models/ whose base_model substring-matches model_query."""
    hits: list[tuple[str, Path]] = []
    for p in sorted(MODELS_DIR.glob("_index_*.json")):
        base_model = p.stem[len("_index_"):]
        if model_query.lower() in base_model.lower():
            hits.append((base_model, p))
    return hits


def _load_seed_scopes(index_path: Path) -> dict[str, dict[str, str]]:
    """Read the index and convert each seed's `pole -> dir path` map into
    `pole -> ft_timestamp`. Returns {seed_str: {pole: ft_timestamp}}.

    Entries whose path has no trailing timestamp are dropped with a warning.
    """
    with index_path.open() as f:
        raw = json.load(f)
    out: dict[str, dict[str, str]] = {}
    for seed, poles in raw.items():
        scope: dict[str, str] = {}
        for pole, path in poles.items():
            ts = _ft_timestamp_from_path(path)
            if ts is None:
                print(f"  [warn] no FT timestamp in {path!r} (seed={seed}, pole={pole}); skipping")
                continue
            scope[pole] = ts
        out[seed] = scope
    return out


# ============================================================
# PER-SEED RECORD FILTERING
# ============================================================


def _filter_for_seed(
    records: list[dict], allowed: dict[str, str]
) -> list[dict]:
    """Keep base records + records whose (pole, ft_timestamp) matches the
    seed's allowed map. Drop v2 records entirely."""
    out: list[dict] = []
    for r in records:
        if r.get("eval_variant") == "v2":
            continue
        if r["pole"] == "base":
            out.append(r)
            continue
        want_ts = allowed.get(r["pole"])
        if want_ts and r.get("ft_timestamp") == want_ts:
            out.append(r)
    return out


def _filters() -> core.FilterConfig:
    return core.FilterConfig(
        epoch=FILTER_EPOCH,
        judge=FILTER_JUDGE,
        min_items=FILTER_MIN_ITEMS,
        ft_ts_prefix=FILTER_FT_TS_PREFIX,
        prefer=PREFER,
        overrides=dict(OVERRIDES),
    )


# ============================================================
# SCORES JSON  (mirrors summarize_FT.build_scores, but takes a pre-built
# `chosen` dict so we don't redo collect_records per seed)
# ============================================================


def _build_scores_from_chosen(
    chosen: dict[tuple[str, str], dict],
    base_model: str,
    seed: str,
) -> dict:
    score_cells: dict[str, dict[str, dict]] = defaultdict(dict)
    for (pole, ev), rec in chosen.items():
        summary = rec.get("summary") or {}
        metrics_block = summary.get("metrics") or {}
        primary_key = core.resolve_metric_key(ev, metrics_block)
        m_inner = metrics_block.get(primary_key, {}) if primary_key else {}

        rows_path = EVAL_RESULTS_DIR / rec["dirname"] / "rows.jsonl"
        scores: dict[str, int | float | None] = {}
        if rows_path.exists():
            with rows_path.open() as f:
                for line in f:
                    row = json.loads(line)
                    if primary_key and row.get("metric") != primary_key:
                        continue
                    cid = (
                        f"{row['item_id']}__p{row['paraphrase_idx']}"
                        f"__s{row['sample_idx']}"
                    )
                    scores[cid] = row.get("score")

        numeric = [s for s in scores.values() if isinstance(s, (int, float))]
        std = statistics.stdev(numeric) if len(numeric) >= 2 else None

        score_cells[pole][ev] = {
            "metrics": {
                "mean": m_inner.get("mean"),
                "std": std,
                "min": m_inner.get("min"),
                "max": m_inner.get("max"),
                "n_total": m_inner.get("n_total"),
                "n_numeric": m_inner.get("n_numeric"),
                "n_nulls": m_inner.get("n_nulls"),
                "n_fails": m_inner.get("n_fails"),
            },
            "scores": scores,
            "meta": {
                "dirname": rec["dirname"],
                "judge_model": summary.get("judge_model"),
                "epoch": rec["epoch"],
                "ft_timestamp": rec["ft_timestamp"],
                "eval_timestamp": rec["eval_timestamp"],
                "n_test_items": summary.get("n_test_items"),
                "samples_per_paraphrase": summary.get("samples_per_paraphrase"),
                "system_prompt_source": summary.get("system_prompt_source"),
            },
        }

    return {
        "base_model": base_model,
        "seed": seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "filters": {
            "epoch": FILTER_EPOCH,
            "judge": FILTER_JUDGE,
            "min_items": FILTER_MIN_ITEMS,
            "ft_ts_prefix": FILTER_FT_TS_PREFIX,
            "prefer": PREFER,
            "overrides": {f"{p}|{e}": v for (p, e), v in OVERRIDES.items()},
            "exclude_v2": True,
        },
        "n_poles": len(score_cells),
        "n_cells": sum(len(v) for v in score_cells.values()),
        "cells": dict(score_cells),
    }


def _write_scores(scores: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(scores, f, indent=2)
    print(
        f"  [wrote] {out_path.name}  "
        f"({scores['n_cells']} cells across {scores['n_poles']} poles)"
    )


# ============================================================
# RENDERING  (direct calls into viz private helpers so we avoid the
# OVERRIDES-can't-exclude problem -- see commit message / chat for context)
# ============================================================


def _render_seed_matrices(
    chosen: dict[tuple[str, str], dict],
    base_model: str,
    seed: str,
) -> None:
    """Render minmax/std/diff PNGs by setting viz's module-globals and
    delegating to its private renderers. We pass viz our own `chosen` dict
    instead of letting it rebuild one from collect_records."""
    # Point viz at this run's base_model + outputs.
    viz.MODEL = base_model
    viz.EVAL_RESULTS_DIR = EVAL_RESULTS_DIR
    viz.RESULTS_DIR = RESULTS_DIR
    viz.DIFF_BASE_POLE = DIFF_BASE_POLE
    viz.OUTPUT_MINMAX_FILE = f"minmax_{base_model}_seed-{seed}.png"
    viz.OUTPUT_STD_FILE = f"std_{base_model}_seed-{seed}.png"
    viz.OUTPUT_DIFF_FILE = f"diff_{base_model}_seed-{seed}.png"

    poles, evals = viz._resolve_axes(chosen)
    if not poles or not evals:
        print(f"  [skip-render] no axes for seed={seed}")
        return

    import numpy as np
    mean, sd, mn, mx, nn, nl = viz._collect_arrays(chosen, poles, evals)
    if not np.isfinite(mean).any():
        print(f"  [skip-render] no finite cells for seed={seed}")
        return

    def_sys = viz._load_def_sys()
    pole_labels = [viz.POLE_LABEL_DECORATOR(p, def_sys) for p in poles]
    eval_labels = [viz._eval_label(e, def_sys) for e in evals]

    viz._render_minmax(poles, evals, pole_labels, eval_labels, mean, mn, mx, nn, nl)
    viz._render_std(poles, evals, pole_labels, eval_labels, mean, sd)
    viz._render_diff(poles, evals, pole_labels, eval_labels, mean)


# ============================================================
# NOTEBOOK SCAFFOLD
# ============================================================


def _build_notebook() -> dict:
    md_intro = (
        "# Browse seed-split eval responses\n"
        "\n"
        "Same `get_responses` / `get_scores` interface as the FT browse\n"
        "notebook, but indexed by an extra `seed` dimension. Each\n"
        "`scores_<base_model>_seed-<seed>.json` in this directory is loaded\n"
        "into `SCORES[base_model][seed]`.\n"
        "\n"
        "- **model** -- the base LLM (`meta-llama-Llama-3.1-8B-Instruct`, ...).\n"
        "- **seed** -- the finetune seed string (`'default'`, `'2'`, `'3'`,\n"
        "  `'5'`, ...). `'default'` is the run with no explicit seed.\n"
        "- **pole** -- the propensity the checkpoint was finetuned toward\n"
        "  (`agreeableness-plus`, ...). `'base'` is the unfinetuned model and\n"
        "  appears in every seed.\n"
        "- **eval** -- which propensity is being measured.\n"
        "\n"
        "On-diagonal cells (`X-plus`/`X-minus` x eval `X`) are direct\n"
        "elicitations; off-diagonal cells are cross-elicitations.\n"
        "\n"
        "Re-running `summarize_seeds.py` does **not** overwrite this notebook --\n"
        "delete it first if you want the fresh scaffold.\n"
    )

    code_setup = (
        "import json, re\n"
        "from pathlib import Path\n"
        "\n"
        "_here = Path('.').resolve()\n"
        "SEEDS_DIR = _here if _here.name == 'seeds' else _here / 'results' / 'seeds'\n"
        "EVAL_ROOT = (SEEDS_DIR.parent.parent / 'eval_results' / 'finetuning').resolve()\n"
        "\n"
        "_NAME_RE = re.compile(r'^scores_(?P<model>.+)_seed-(?P<seed>.+)\\.json$')\n"
        "SCORES = {}\n"
        "for p in sorted(SEEDS_DIR.glob('scores_*_seed-*.json')):\n"
        "    m = _NAME_RE.match(p.name)\n"
        "    if not m:\n"
        "        continue\n"
        "    doc = json.loads(p.read_text())\n"
        "    if doc.get('n_cells', 0) == 0:\n"
        "        continue\n"
        "    SCORES.setdefault(doc['base_model'], {})[doc['seed']] = doc\n"
        "\n"
        "print('Loaded scores:')\n"
        "for model, by_seed in SCORES.items():\n"
        "    seeds = sorted(by_seed, key=lambda s: (s != 'default', s))\n"
        "    cells = sum(d['n_cells'] for d in by_seed.values())\n"
        "    print(f'  {model}  seeds={seeds}  ({cells} cells total)')\n"
        "print()\n"
        "print('Use get_responses(model, seed, pole, eval) and get_scores(model, seed, pole, eval).')\n"
    )

    code_helpers = (
        "def _cell(model, seed, pole, eval_propensity):\n"
        "    if model not in SCORES:\n"
        "        raise KeyError(f'unknown model {model!r}; loaded: {sorted(SCORES)}')\n"
        "    by_seed = SCORES[model]\n"
        "    if seed not in by_seed:\n"
        "        raise KeyError(\n"
        "            f'no seed {seed!r} for model {model!r}; available: {sorted(by_seed)}'\n"
        "        )\n"
        "    cells = by_seed[seed]['cells']\n"
        "    if pole not in cells:\n"
        "        raise KeyError(\n"
        "            f'unknown pole {pole!r} for {model!r} seed={seed!r}; '\n"
        "            f'available: {sorted(cells)}'\n"
        "        )\n"
        "    if eval_propensity not in cells[pole]:\n"
        "        raise KeyError(\n"
        "            f'no eval {eval_propensity!r} for pole {pole!r} seed={seed!r}; '\n"
        "            f'available: {sorted(cells[pole])}'\n"
        "        )\n"
        "    return cells[pole][eval_propensity]\n"
        "\n"
        "\n"
        "def _iter_rows(model, seed, pole, eval_propensity):\n"
        "    cell = _cell(model, seed, pole, eval_propensity)\n"
        "    rows_path = EVAL_ROOT / cell['meta']['dirname'] / 'rows.jsonl'\n"
        "    if not rows_path.exists():\n"
        "        raise FileNotFoundError(f'missing rows.jsonl: {rows_path}')\n"
        "    with rows_path.open() as f:\n"
        "        for line in f:\n"
        "            yield json.loads(line)\n"
        "\n"
        "\n"
        "def get_responses(model, seed, pole, eval_propensity):\n"
        "    \"\"\"Conversations from `model`'s seed=`seed` `pole` checkpoint on `eval_propensity`.\n"
        "\n"
        "    Returns [{'question', 'answer'}] in rows.jsonl order.\n"
        "    \"\"\"\n"
        "    return [\n"
        "        {'question': r.get('question'), 'answer': r.get('answer')}\n"
        "        for r in _iter_rows(model, seed, pole, eval_propensity)\n"
        "    ]\n"
        "\n"
        "\n"
        "def get_scores(model, seed, pole, eval_propensity):\n"
        "    \"\"\"Per-conversation judge scores, in the same order as get_responses(...).\"\"\"\n"
        "    return [r.get('score') for r in _iter_rows(model, seed, pole, eval_propensity)]\n"
    )

    md_examples_header = "## Example: compare seeds for one (pole, eval) cell"

    code_example = (
        "# Pick the first model/pole/eval triple that exists and print one\n"
        "# conversation per available seed for comparison.\n"
        "if SCORES:\n"
        "    model = next(iter(SCORES))\n"
        "    by_seed = SCORES[model]\n"
        "    any_seed = next(iter(by_seed))\n"
        "    cells = by_seed[any_seed]['cells']\n"
        "    pole = 'neuroticism-plus' if 'neuroticism-plus' in cells else next(iter(cells))\n"
        "    eval_p = 'neuroticism' if 'neuroticism' in cells[pole] else next(iter(cells[pole]))\n"
        "\n"
        "    seeds = sorted(by_seed, key=lambda s: (s != 'default', s))\n"
        "    print(f'{model} | {pole} | {eval_p}')\n"
        "    for s in seeds:\n"
        "        try:\n"
        "            convos = get_responses(model, s, pole, eval_p)\n"
        "            scores = get_scores(model, s, pole, eval_p)\n"
        "        except KeyError as exc:\n"
        "            print(f'  seed={s}: <{exc.args[0]}>')\n"
        "            continue\n"
        "        if not convos:\n"
        "            print(f'  seed={s}: (no conversations)')\n"
        "            continue\n"
        "        print(f'  ---- seed={s}  (n={len(convos)}) ----')\n"
        "        print('  Q:', (convos[0][\"question\"] or '')[:200])\n"
        "        print('  A:', (convos[0][\"answer\"] or '')[:200])\n"
        "        print('  score:', scores[0])\n"
    )

    def code_cell(cell_id: str, src: str) -> dict:
        return {
            "cell_type": "code",
            "id": cell_id,
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src,
        }

    def md_cell(cell_id: str, src: str) -> dict:
        return {
            "cell_type": "markdown",
            "id": cell_id,
            "metadata": {},
            "source": src,
        }

    return {
        "cells": [
            md_cell("intro", md_intro),
            code_cell("setup", code_setup),
            code_cell("helpers", code_helpers),
            md_cell("examples-header", md_examples_header),
            code_cell("example", code_example),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _scaffold_notebook(path: Path) -> None:
    if path.exists():
        print(f"[skip] notebook already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_build_notebook(), f, indent=1)
    print(f"[wrote] {path}")


# ============================================================
# MAIN
# ============================================================


def _process_one_base_model(base_model: str, index_path: Path) -> None:
    print()
    print(f"=== {base_model} ===")
    print(f"  index: {index_path.name}")

    seed_scopes = _load_seed_scopes(index_path)
    if not seed_scopes:
        print("  [skip] index has no seeds")
        return

    # Collect all v1 + v2 records for this base_model once, restrict to the
    # exact base_model string (collect_records uses substring match).
    all_recs, _, _ = core.collect_records(EVAL_RESULTS_DIR, base_model)
    all_recs = [r for r in all_recs if r["model"] == base_model]
    if not all_recs:
        print(f"  [skip] no eval_results records for {base_model!r}")
        return

    # Union of seen poles across all seeds, so we can warn per seed about
    # poles missing from this seed but present elsewhere.
    union_poles = set()
    for scope in seed_scopes.values():
        union_poles.update(scope.keys())

    seeds_sorted = sorted(seed_scopes, key=lambda s: (s != "default", s))
    for seed in seeds_sorted:
        allowed = seed_scopes[seed]
        print()
        print(f"--- seed={seed} ({len(allowed)} poles in index) ---")
        missing = sorted(union_poles - set(allowed.keys()))
        if missing:
            print(f"  [warn] poles present in other seeds but missing here: {missing}")

        seed_recs = _filter_for_seed(all_recs, allowed)
        if not seed_recs:
            print("  [skip] no records survived seed/v1 filter")
            continue

        by_cell = core.group_by_cell(seed_recs)
        chosen, override_misses = core.disambiguate(by_cell, _filters())
        if override_misses:
            print("  [warn] OVERRIDES not matched:")
            for (cell, target) in override_misses:
                print(f"    {cell}: {target}")
        if not chosen:
            print("  [skip] disambiguate produced no cells (FILTER_MIN_ITEMS?)")
            continue

        # Warn on poles from the index that ended up with zero surviving cells.
        chosen_poles = {p for (p, _) in chosen}
        empty_poles = sorted(p for p in allowed if p not in chosen_poles)
        if empty_poles:
            print(
                f"  [warn] poles in index with no v1 eval records this seed: "
                f"{empty_poles}"
            )

        scores = _build_scores_from_chosen(chosen, base_model, seed)
        _write_scores(
            scores, RESULTS_DIR / f"scores_{base_model}_seed-{seed}.json"
        )
        if RUN_VIZ:
            _render_seed_matrices(chosen, base_model, seed)


def main() -> None:
    if not EVAL_RESULTS_DIR.exists():
        raise SystemExit(f"eval_results dir not found: {EVAL_RESULTS_DIR}")
    if not MODELS_DIR.exists():
        raise SystemExit(f"models dir not found: {MODELS_DIR}")

    targets = _discover_index_files(MODEL)
    if not targets:
        avail = sorted(p.stem[len("_index_"):] for p in MODELS_DIR.glob("_index_*.json"))
        raise SystemExit(
            f"MODEL {MODEL!r} matched no _index_*.json file in {MODELS_DIR}.\n"
            f"Available base_models: {avail}"
        )
    print(f"MODEL filter {MODEL!r} matched: {[bm for bm, _ in targets]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for base_model, idx_path in targets:
        _process_one_base_model(base_model, idx_path)

    if SCAFFOLD_NOTEBOOK:
        print()
        _scaffold_notebook(RESULTS_DIR / "seed_browse_responses.ipynb")


if __name__ == "__main__":
    main()
