#!/usr/bin/env python3
"""summarize_FT.py

For each base_model discovered in ../eval_results/ (or one chosen via
BASE_MODEL), this script:

  1. Writes ../results/scores_<base_model>.json with per-cell metrics
     (mean / std / min / max / counts), per-conversation scores, and the
     source dirname / metadata for each (pole, eval_propensity) cell.
  2. Regenerates ../results/{minmax,std}_eval_matrix_<base_model>.png by
     driving visualize_eval_matrix.main() (so the heatmaps and the scores
     JSON pick the same chosen dir per cell).
  3. Scaffolds ../results/browse_responses.ipynb on first run only
     (idempotent -- never overwrites your edits). The notebook exposes
     `get_responses(model, pole, eval)` and `get_scores(model, pole, eval)`
     by lazy-loading the relevant rows.jsonl on demand.

Edit the CONFIG block to constrain a run. Disambiguation knobs mirror the
ones in visualize_eval_matrix.py.

Two distinct dimensions to keep separate:
  - base_model: the underlying LLM (e.g. meta-llama-Llama-3.1-8B-Instruct,
    Qwen-Qwen3-8B-Base). Each gets its own scores_<base_model>.json.
  - pole: the finetune label within a base_model (base, effort-plus, ...).
    The top level of cells/ in the JSON is keyed by pole.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import eval_matrix_core as core
import visualize_eval_matrix as viz

# ============================================================
# CONFIG -- edit these
# ============================================================

# None  -> process every base_model discovered in eval_results/.
# Otherwise: case-insensitive substring matched against the base_model
# segment of each dir name (same semantics as visualize_eval_matrix.MODEL).
BASE_MODEL: str | None = None

# Disambiguation knobs (see visualize_eval_matrix.py for full description).
# These are forwarded to both the scores-JSON build and the heatmap render
# so the two artefacts agree on which dir was chosen per cell.
FILTER_EPOCH: int | None = None
FILTER_JUDGE: str | None = None
FILTER_MIN_ITEMS: int | None = 30
FILTER_FT_TS_PREFIX: str | None = None
PREFER: str = "latest-eval"
OVERRIDES: dict[tuple[str, str], str] = {}

RUN_VIZ: bool = True
SCAFFOLD_NOTEBOOK: bool = True

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_DIR = SCRIPT_DIR.parent / "eval_results"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


# ============================================================
# SCORES JSON
# ============================================================


def _filters() -> core.FilterConfig:
    return core.FilterConfig(
        epoch=FILTER_EPOCH,
        judge=FILTER_JUDGE,
        min_items=FILTER_MIN_ITEMS,
        ft_ts_prefix=FILTER_FT_TS_PREFIX,
        prefer=PREFER,
        overrides=dict(OVERRIDES),
    )


def build_scores(base_model_name: str) -> dict | None:
    """Build the scores dict for a single (exact) base_model string."""
    records, _, _ = core.collect_records(EVAL_RESULTS_DIR, base_model_name)
    # collect_records uses substring match -- narrow to exact base_model
    # so we don't accidentally bundle two LLMs into one file.
    records = [r for r in records if r["model"] == base_model_name]
    if not records:
        print(f"  [skip] no records for {base_model_name!r}")
        return None

    by_cell = core.group_by_cell(records)
    chosen, override_misses = core.disambiguate(by_cell, _filters())
    if override_misses:
        print("  [warn] OVERRIDES not matched:")
        for (cell, target) in override_misses:
            print(f"    {cell}: {target}")

    score_cells: dict[str, dict[str, dict]] = defaultdict(dict)
    for (pole, ev), rec in chosen.items():
        summary = rec.get("summary") or {}
        metrics_block = summary.get("metrics") or {}
        # Pick the judge keyed to this column's eval propensity. Most evals
        # emit a single `<propensity>_score` metric; the ethical-framework and
        # honest-humble axes emit multiple judges, so defer to the shared
        # resolver in eval_matrix_core.
        primary_key = core.resolve_metric_key(ev, metrics_block)
        m_inner = metrics_block.get(primary_key, {}) if primary_key else {}

        rows_path = EVAL_RESULTS_DIR / rec["dirname"] / "rows.jsonl"
        scores: dict[str, int | float | None] = {}
        if rows_path.exists():
            with rows_path.open() as f:
                for line in f:
                    row = json.loads(line)
                    # Multi-judge evals write one row per judge; the (item,
                    # paraphrase, sample) cid would collide across judges, so
                    # filter to the primary judge for this column.
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
        "base_model": base_model_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "filters": {
            "epoch": FILTER_EPOCH,
            "judge": FILTER_JUDGE,
            "min_items": FILTER_MIN_ITEMS,
            "ft_ts_prefix": FILTER_FT_TS_PREFIX,
            "prefer": PREFER,
            "overrides": {f"{p}|{e}": v for (p, e), v in OVERRIDES.items()},
        },
        "n_poles": len(score_cells),
        "n_cells": sum(len(v) for v in score_cells.values()),
        "cells": dict(score_cells),
    }


def write_scores_file(scores: dict) -> Path:
    out = RESULTS_DIR / f"scores_{scores['base_model']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
    print(f"  [wrote] {out.name}  ({scores['n_cells']} cells across {scores['n_poles']} poles)")
    return out


# ============================================================
# VIZ (drive visualize_eval_matrix)
# ============================================================


def regenerate_viz(base_model_name: str) -> None:
    viz.MODEL = base_model_name
    viz.OUTPUT_MINMAX_FILE = f"minmax_eval_matrix_{base_model_name}.png"
    viz.OUTPUT_STD_FILE = f"std_eval_matrix_{base_model_name}.png"
    viz.FILTER_EPOCH = FILTER_EPOCH
    viz.FILTER_JUDGE = FILTER_JUDGE
    viz.FILTER_MIN_ITEMS = FILTER_MIN_ITEMS
    viz.FILTER_FT_TS_PREFIX = FILTER_FT_TS_PREFIX
    viz.PREFER = PREFER
    viz.OVERRIDES = dict(OVERRIDES)
    viz.RENDER = True
    viz.main()


# ============================================================
# NOTEBOOK SCAFFOLD
# ============================================================


def _build_notebook() -> dict:
    md_intro = (
        "# Browse eval responses\n"
        "\n"
        "Look up the prompts a particular checkpoint gave to a particular eval.\n"
        "\n"
        "Three indexing dimensions:\n"
        "- **model** -- the underlying LLM (e.g. `meta-llama-Llama-3.1-8B-Instruct`,\n"
        "  `Qwen-Qwen3-8B-Base`). Each model has its own `scores_<model>.json`.\n"
        "- **pole** -- the propensity this checkpoint was finetuned toward\n"
        "  (`agreeableness-plus`, `effort-minus`, `narcissism-plus`, ...). Pole\n"
        "  `base` means the unfinetuned model. The pole is the thing meant to push\n"
        "  the eval score up or down -- e.g. an `agreeableness-plus` checkpoint\n"
        "  should score high on the `agreeableness` eval, and `agreeableness-minus`\n"
        "  should score low.\n"
        "- **eval** -- which propensity is being *measured* (`effort`, `narcissism`,\n"
        "  `agreeableness`, ...). The judge LLM scores each answer along this axis.\n"
        "\n"
        "On-diagonal cells (pole `X-plus`/`X-minus` x eval `X`) are direct\n"
        "elicitations; off-diagonal cells are cross-elicitations.\n"
        "\n"
        "All `scores_*.json` files in `../results/` are loaded into `SCORES`\n"
        "(keyed by model). `get_responses` and `get_scores` stream the per-cell\n"
        "`rows.jsonl` on demand instead of bundling every prompt + answer into\n"
        "one giant JSON file.\n"
        "\n"
        "Re-running `summarize_FT.py` does **not** overwrite this notebook --\n"
        "delete it first if you want the fresh scaffold.\n"
    )

    code_setup = (
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "_here = Path('.').resolve()\n"
        "RESULTS_DIR = _here if _here.name == 'results' else _here / 'results'\n"
        "EVAL_ROOT = (RESULTS_DIR.parent / 'eval_results').resolve()\n"
        "\n"
        "SCORES = {}\n"
        "for p in sorted(RESULTS_DIR.glob('scores_*.json')):\n"
        "    doc = json.loads(p.read_text())\n"
        "    if doc.get('n_cells', 0) > 0:\n"
        "        SCORES[doc['base_model']] = doc\n"
        "\n"
        "print('Loaded scores for models:')\n"
        "for m, doc in SCORES.items():\n"
        "    print(f\"  {m}  ({doc['n_cells']} cells across {doc['n_poles']} poles)\")\n"
        "print()\n"
        "print('Use get_responses(model, pole, eval) and get_scores(model, pole, eval).')\n"
    )

    code_helpers = (
        "def _cell(model, pole, eval_propensity):\n"
        "    if model not in SCORES:\n"
        "        raise KeyError(f'unknown model {model!r}; loaded: {sorted(SCORES)}')\n"
        "    cells = SCORES[model]['cells']\n"
        "    if pole not in cells:\n"
        "        raise KeyError(\n"
        "            f'unknown pole {pole!r} for model {model!r}; '\n"
        "            f'available: {sorted(cells)}'\n"
        "        )\n"
        "    if eval_propensity not in cells[pole]:\n"
        "        raise KeyError(\n"
        "            f'no eval {eval_propensity!r} for pole {pole!r}; '\n"
        "            f'available: {sorted(cells[pole])}'\n"
        "        )\n"
        "    return cells[pole][eval_propensity]\n"
        "\n"
        "\n"
        "def _iter_rows(model, pole, eval_propensity):\n"
        "    cell = _cell(model, pole, eval_propensity)\n"
        "    rows_path = EVAL_ROOT / cell['meta']['dirname'] / 'rows.jsonl'\n"
        "    if not rows_path.exists():\n"
        "        raise FileNotFoundError(f'missing rows.jsonl: {rows_path}')\n"
        "    with rows_path.open() as f:\n"
        "        for line in f:\n"
        "            yield json.loads(line)\n"
        "\n"
        "\n"
        "def get_responses(model, pole, eval_propensity):\n"
        "    \"\"\"Conversations from `model`'s `pole` checkpoint on the `eval_propensity` eval.\n"
        "\n"
        "    Returns a list of {'question', 'answer'} dicts in rows.jsonl order.\n"
        "    Example: get_responses('meta-llama-Llama-3.1-8B-Instruct',\n"
        "                           'agreeableness-plus', 'narcissism').\n"
        "    \"\"\"\n"
        "    return [\n"
        "        {'question': r.get('question'), 'answer': r.get('answer')}\n"
        "        for r in _iter_rows(model, pole, eval_propensity)\n"
        "    ]\n"
        "\n"
        "\n"
        "def get_scores(model, pole, eval_propensity):\n"
        "    \"\"\"Per-conversation judge scores, in the same order as get_responses(...).\n"
        "\n"
        "    Entries are int/float for numeric judgements; None when the judge\n"
        "    bucketed the answer as `null` or `fail`.\n"
        "    \"\"\"\n"
        "    return [r.get('score') for r in _iter_rows(model, pole, eval_propensity)]\n"
    )

    md_key_fn = (
        "The `get_responses` function is the key function here - use it to look "
        "at all the conversations that were judged by gpt-5.4-mini (default judge "
        "model), or use `get_scores` to look at the scores that gpt-5.4-mini gave."
    )

    md_examples_header = "## Example for `get_responses` use"

    code_example_effort = (
        "for pole in ['effort-minus', 'base', 'effort-plus']:\n"
        "    print(f\"=========================\")\n"
        "    print(f\">{pole}\")\n"
        "    x = get_responses('meta-llama-Llama-3.1-8B-Instruct', pole, 'effort')[0]\n"
        "    print(f\">Question: {x['question']}\")\n"
        "    print(f\">Answer  : {x['answer']}\\n\")"
    )

    code_example = (
        "# Example -- pick a (model, pole, eval) triple that exists and show the first conv.\n"
        "if SCORES:\n"
        "    model = next(iter(SCORES))\n"
        "    cells = SCORES[model]['cells']\n"
        "    pole = 'base' if 'base' in cells else next(iter(cells))\n"
        "    eval_p = next(iter(cells[pole]))\n"
        "\n"
        "    convos = get_responses(model, pole, eval_p)\n"
        "    scores = get_scores(model, pole, eval_p)\n"
        "    print(f'{model} | {pole} | {eval_p}: {len(convos)} convs')\n"
        "    print()\n"
        "    print('Q:', (convos[0]['question'] or '')[:200])\n"
        "    print()\n"
        "    print('A:', (convos[0]['answer'] or '')[:200])\n"
        "    print()\n"
        "    print('score:', scores[0])\n"
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
            md_cell("key-fn-callout", md_key_fn),
            code_cell("setup", code_setup),
            code_cell("helpers", code_helpers),
            md_cell("examples-header", md_examples_header),
            code_cell("example-effort", code_example_effort),
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


def scaffold_notebook(path: Path) -> None:
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


def main() -> None:
    if not EVAL_RESULTS_DIR.exists():
        raise SystemExit(f"eval_results dir not found: {EVAL_RESULTS_DIR}")

    known = sorted(core.discover_models(EVAL_RESULTS_DIR))
    print(f"Models discovered in eval_results/: {known}")

    if BASE_MODEL is None:
        targets = known
        print(f"BASE_MODEL = None -- processing all {len(targets)} base models.")
    else:
        targets = [m for m in known if BASE_MODEL.lower() in m.lower()]
        print(f"BASE_MODEL filter {BASE_MODEL!r} matched: {targets}")
        if not targets:
            raise SystemExit(f"BASE_MODEL {BASE_MODEL!r} matched nothing.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for base_model_name in targets:
        print()
        print(f"=== {base_model_name} ===")
        scores = build_scores(base_model_name)
        if scores is None:
            continue
        write_scores_file(scores)
        if RUN_VIZ:
            print()
            regenerate_viz(base_model_name)

    if SCAFFOLD_NOTEBOOK:
        print()
        scaffold_notebook(RESULTS_DIR / "browse_responses.ipynb")


if __name__ == "__main__":
    main()
