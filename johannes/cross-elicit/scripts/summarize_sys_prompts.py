#!/usr/bin/env python3
"""summarize_sys_prompts.py

Counterpart to summarize_FT.py for the system-prompted runs in
../eval_results/sys_prompts/. For each base_model discovered there
(or one chosen via BASE_MODEL), this script:

  1. Writes ../results/scores_sysprompts_<base_model>.json with per-cell
     metrics (mean / std / min / max / counts), per-conversation scores,
     and the source dirname / metadata for each (pole, eval_propensity)
     cell. Pole keys are the *normalized* form `<source_eval>--<pole>`
     (or `baseline-<x>`) -- see sys_prompt_core for the rules.
  2. Regenerates ../results/{minmax,std}_sysprompt_eval_matrix_<base_model>.png
     by reassigning the pluggable hooks on visualize_eval_matrix and
     calling its main(), so the heatmap and the scores JSON pick the
     same chosen dir per cell.
  3. Scaffolds ../results/browse_sysprompt_responses.ipynb on first run
     only (idempotent -- never overwrites your edits). The notebook
     exposes `get_responses(model, pole, eval)` and
     `get_scores(model, pole, eval)` by lazy-loading rows.jsonl on demand.

What you have to specify (vs summarize_FT.py)
---------------------------------------------
- The data lives in ../eval_results/sys_prompts/, not ../eval_results/
  directly -- the visualize_eval_matrix module is retargeted accordingly.
- Pole identity is `<source_eval>--<pole_short>` (off-diagonals stay as
  written; diagonals like `agreeable` get re-prefixed to
  `agreeableness--agreeable`). Baselines are kept as-is (`baseline-empty`).
- FILTER_EPOCH and FILTER_FT_TS_PREFIX are NOT applicable: sys-prompt runs
  have no epoch/ft_timestamp, and `passes_filters` would reject every
  record if either is set. Defaults are None.
- FILTER_MIN_ITEMS defaults to None (most sys-prompt runs are n=20, so
  the FT default of 30 would drop almost everything).

Edit the CONFIG block to constrain a run.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import sys_prompt_core as core
import visualize_eval_matrix as viz

# ============================================================
# CONFIG -- edit these
# ============================================================

# None  -> process every base_model discovered in eval_results/sys_prompts/.
# Otherwise: case-insensitive substring matched against rec['model']
# (same semantics as summarize_FT.BASE_MODEL).
BASE_MODEL: str | None = None

# Disambiguation knobs (see eval_matrix_core.FilterConfig).
# FILTER_EPOCH and FILTER_FT_TS_PREFIX are intentionally None: every
# sys-prompt record has epoch=None / ft_timestamp=None, so a non-None
# value here would reject every record.
FILTER_EPOCH: int | None = None
FILTER_JUDGE: str | None = None
FILTER_MIN_ITEMS: int | None = None
FILTER_FT_TS_PREFIX: str | None = None
PREFER: str = "latest-eval"
OVERRIDES: dict[tuple[str, str], str] = {}

RUN_VIZ: bool = True
SCAFFOLD_NOTEBOOK: bool = True

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_DIR = SCRIPT_DIR.parent / "eval_results"
SYS_PROMPTS_DIR = EVAL_RESULTS_DIR / "sys_prompts"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


# Per-(source-eval) ordering of <pole_short> labels, positive pole first.
# Mirrors orthogonality_of_evals.PROPENSITY_RESPONSE_KEYS so the column
# order in the sys-prompt heatmap matches the response-key column order in
# visualize_orthogonality. Anything not listed here is appended alphabetically
# after the canonical entries for that source eval.
SYS_PROMPT_POLE_ORDER: dict[str, list[str]] = {
    "agreeableness":                   ["agreeable",        "disagreeable"],
    "caring-about-aesthetics":         ["caring"],
    "caring-about-animals":            ["caring"],
    "caring-about-humans":             ["caring"],
    "caring-about-user":               ["caring"],
    "certainty":                       ["high",             "low"],
    "claiming-sentience":              ["claiming"],
    "claiming-superintelligence":      ["claiming"],
    "cooperation":                     ["hi",               "lo"],
    "effort":                          ["high",             "low"],
    # Mirrors PROPENSITY_RESPONSE_KEYS: utilitarian, deontological,
    # virtue_ethics. All three ethical-framework source evals share the
    # same pole-short alphabet (utilitarian / deontological / virtue_ethics).
    "ethical-framework-deontological": ["utilitarian",      "deontological", "virtue_ethics"],
    "ethical-framework-utilitarian":   ["utilitarian",      "deontological", "virtue_ethics"],
    "ethical-framework-virtue-ethics": ["utilitarian",      "deontological", "virtue_ethics"],
    "ev-reasoning":                    ["ev_reasoning"],
    "exemplar-reasoning":              ["exemplar_reasoning"],
    "harm-elaboration":                ["high",             "low"],
    "harm-refusal":                    ["high",             "low"],
    "honest-humble":                   ["high_hh",          "low_hh"],
    "narcissism":                      ["narcissistic"],
    "neuroticism":                     ["neurotic",         "emotionally_stable"],
    "power-seeking":                   ["high",             "low"],
    "procedural-fidelity":             ["procedural_fidelity"],
    "resource-acquisition":            ["hi",               "lo"],
    "reward-hacking":                  ["reward_hacking"],
    "risk-affinity":                   ["risk_affinity"],
    "self-preservation":               ["high",             "low"],
    "spending-advice":                 ["high",             "low"],
    "spitefulness":                    ["hi",               "lo"],
    "sycophancy":                      ["sycophantic"],
    "trust-in-user-intentions":        ["high",             "low"],
}


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
    records, _, _ = core.collect_records(SYS_PROMPTS_DIR, base_model_name)
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
        primary_key = core.resolve_metric_key(ev, metrics_block)
        m_inner = metrics_block.get(primary_key, {}) if primary_key else {}

        rows_path = SYS_PROMPTS_DIR / rec["dirname"] / "rows.jsonl"
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
                "sysprompt_label": rec["sysprompt_label"],
                "source_eval": rec["source_eval"],
                "pole_short": rec["pole_short"],
                "is_baseline": rec["is_baseline"],
                "is_diagonal": rec["is_diagonal"],
                "ckpt_label": rec["ckpt_label"],
                "eval_timestamp": rec["eval_timestamp"],
                "n_test_items": summary.get("n_test_items"),
                "samples_per_paraphrase": summary.get("samples_per_paraphrase"),
                "system_prompt_source": summary.get("system_prompt_source"),
            },
        }

    return {
        "base_model": base_model_name,
        "tree": "sys_prompts",
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
    out = RESULTS_DIR / f"scores_sysprompts_{scores['base_model']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
    print(f"  [wrote] {out.name}  ({scores['n_cells']} cells across {scores['n_poles']} poles)")
    return out


# ============================================================
# VIZ (drive visualize_eval_matrix with sys_prompt hooks)
# ============================================================


def _derive_pole_order(base_model_name: str) -> list[str]:
    """Auto-derive a column order that puts baselines first, then groups
    poles by source eval (alphabetical) with the positive-first short-pole
    order from SYS_PROMPT_POLE_ORDER. Mirrors POLE_ORDER in
    visualize_eval_matrix and PROPENSITY_RESPONSE_KEYS in orthogonality so
    all three figures share the same pole layout. visualize_eval_matrix
    ._resolve_axes appends anything not listed here alphabetically."""
    records, _, _ = core.collect_records(SYS_PROMPTS_DIR, base_model_name)
    records = [r for r in records if r["model"] == base_model_name]
    poles = {r["pole"] for r in records}

    baselines = sorted(p for p in poles if p.startswith(core.BASELINE_PREFIX))

    # Group non-baseline poles by source eval, then sort each group by
    # SYS_PROMPT_POLE_ORDER position; unknown shorts fall to the end of
    # their source-eval group (sorted alphabetically among themselves).
    by_source: dict[str, list[str]] = {}
    for p in poles:
        if p.startswith(core.BASELINE_PREFIX) or "--" not in p:
            continue
        source, short = p.split("--", 1)
        by_source.setdefault(source, []).append(short)

    def _short_sort_key(source: str, short: str) -> tuple[int, str]:
        canonical = SYS_PROMPT_POLE_ORDER.get(source, [])
        if short in canonical:
            return (canonical.index(short), "")
        # Unlisted shorts: sort to the end of the source-eval group.
        return (len(canonical), short)

    others: list[str] = []
    for source in sorted(by_source):
        for short in sorted(
            by_source[source], key=lambda s, src=source: _short_sort_key(src, s)
        ):
            others.append(f"{source}--{short}")

    return baselines + others


def regenerate_viz(base_model_name: str) -> None:
    # Retarget the viz module: data tree, parser/disambiguate core, axis
    # cosmetics, output paths. Reassigning module globals matches the
    # pattern summarize_FT already uses for MODEL / OUTPUT_*.
    # Save the FT-default hooks so we restore them after rendering -- if
    # someone imports both summarize_FT and summarize_sys_prompts in one
    # process, FT runs after us shouldn't inherit sys-prompt hooks.
    saved = {
        "core": viz.core,
        "EVAL_RESULTS_DIR": viz.EVAL_RESULTS_DIR,
        "DIAGONAL_RESOLVER": viz.DIAGONAL_RESOLVER,
        "POLE_LABEL_DECORATOR": viz.POLE_LABEL_DECORATOR,
        "DIFF_BASE_POLE": viz.DIFF_BASE_POLE,
    }
    try:
        viz.core = core
        viz.EVAL_RESULTS_DIR = SYS_PROMPTS_DIR
        viz.DIAGONAL_RESOLVER = core.diagonal_eval_for_pole
        viz.POLE_LABEL_DECORATOR = core.pole_label_decorator

        viz.MODEL = base_model_name
        viz.OUTPUT_MINMAX_FILE = f"minmax_sysprompt_eval_matrix_{base_model_name}.png"
        viz.OUTPUT_STD_FILE = f"std_sysprompt_eval_matrix_{base_model_name}.png"
        viz.OUTPUT_DIFF_FILE = f"diff_sysprompt_eval_matrix_{base_model_name}.png"
        viz.DIFF_BASE_POLE = "baseline-empty"

        viz.POLE_ORDER = _derive_pole_order(base_model_name)
        # POLES / EVALS subset filters: leave at module defaults (empty = all).
        viz.POLES = []
        viz.EVALS = []

        viz.FILTER_EPOCH = FILTER_EPOCH
        viz.FILTER_JUDGE = FILTER_JUDGE
        viz.FILTER_MIN_ITEMS = FILTER_MIN_ITEMS
        viz.FILTER_FT_TS_PREFIX = FILTER_FT_TS_PREFIX
        viz.PREFER = PREFER
        viz.OVERRIDES = dict(OVERRIDES)
        viz.RENDER = True
        viz.main()
    finally:
        viz.core = saved["core"]
        viz.EVAL_RESULTS_DIR = saved["EVAL_RESULTS_DIR"]
        viz.DIAGONAL_RESOLVER = saved["DIAGONAL_RESOLVER"]
        viz.POLE_LABEL_DECORATOR = saved["POLE_LABEL_DECORATOR"]
        viz.DIFF_BASE_POLE = saved["DIFF_BASE_POLE"]


# ============================================================
# NOTEBOOK SCAFFOLD
# ============================================================


def _build_notebook() -> dict:
    md_intro = (
        "# Browse sys-prompt eval responses\n"
        "\n"
        "Look up the prompts a particular system-prompted run gave to a particular eval.\n"
        "\n"
        "Three indexing dimensions:\n"
        "- **model** -- the underlying LLM (e.g. `meta-llama-Llama-3.1-8B-Instruct`,\n"
        "  `Qwen-Qwen3-8B-Base`). Each model has its own\n"
        "  `scores_sysprompts_<model>.json`.\n"
        "- **pole** -- the normalized sys-prompt identity. Two shapes:\n"
        "  - `<source_eval>--<pole>` (e.g. `agreeableness--agreeable`,\n"
        "    `effort--high`). The diagonal cell for this pole is the row\n"
        "    `eval == source_eval`; everything else is cross-elicitation.\n"
        "  - `baseline-<x>` (e.g. `baseline-empty`) -- model run with no\n"
        "    pole prompt; same row repeated across every eval.\n"
        "- **eval** -- which propensity is being *measured* by the judge.\n"
        "\n"
        "Re-running `summarize_sys_prompts.py` does **not** overwrite this notebook --\n"
        "delete it first if you want the fresh scaffold.\n"
    )

    code_setup = (
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "_here = Path('.').resolve()\n"
        "RESULTS_DIR = _here if _here.name == 'results' else _here / 'results'\n"
        "EVAL_ROOT = (RESULTS_DIR.parent / 'eval_results' / 'sys_prompts').resolve()\n"
        "\n"
        "SCORES = {}\n"
        "for p in sorted(RESULTS_DIR.glob('scores_sysprompts_*.json')):\n"
        "    doc = json.loads(p.read_text())\n"
        "    if doc.get('n_cells', 0) > 0:\n"
        "        SCORES[doc['base_model']] = doc\n"
        "\n"
        "print('Loaded sys-prompt scores for models:')\n"
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
        "    \"\"\"Conversations from `model`'s `pole` sysprompt on the `eval_propensity` eval.\n"
        "\n"
        "    Returns a list of {'question', 'answer'} dicts in rows.jsonl order.\n"
        "    Example: get_responses('meta-llama-Llama-3.1-8B-Instruct',\n"
        "                           'agreeableness--agreeable', 'narcissism').\n"
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

    md_examples_header = "## Example for `get_responses` use"

    code_example_effort = (
        "for pole in ['baseline-empty', 'effort--low', 'effort--high']:\n"
        "    print(f\"=========================\")\n"
        "    print(f\">{pole}\")\n"
        "    try:\n"
        "        x = get_responses('meta-llama-Llama-3.1-8B-Instruct', pole, 'effort')[0]\n"
        "    except KeyError as e:\n"
        "        print(f\"  (no cell: {e})\")\n"
        "        continue\n"
        "    print(f\">Question: {x['question']}\")\n"
        "    print(f\">Answer  : {x['answer']}\\n\")"
    )

    code_example = (
        "# Pick a (model, pole, eval) triple that exists and show the first conv.\n"
        "if SCORES:\n"
        "    model = next(iter(SCORES))\n"
        "    cells = SCORES[model]['cells']\n"
        "    pole = (\n"
        "        'baseline-empty' if 'baseline-empty' in cells\n"
        "        else next(iter(cells))\n"
        "    )\n"
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
    if not SYS_PROMPTS_DIR.exists():
        raise SystemExit(f"sys_prompts dir not found: {SYS_PROMPTS_DIR}")

    known = sorted(core.discover_models(SYS_PROMPTS_DIR))
    print(f"Models discovered in eval_results/sys_prompts/: {known}")

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
        scaffold_notebook(RESULTS_DIR / "browse_sysprompt_responses.ipynb")


if __name__ == "__main__":
    main()
