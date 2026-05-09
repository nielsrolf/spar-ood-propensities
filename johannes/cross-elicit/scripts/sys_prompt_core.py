"""sys_prompt_core.py

Shared parse / discover / disambiguate logic for the eval_results/sys_prompts/
tree. Counterpart to eval_matrix_core.py, used by summarize_sys_prompts.py
(scores JSON dump + heatmap render via visualize_eval_matrix.py with the
hooks reassigned).

Directory shape (5 __-separated parts, vs 4 in the FT tree):

    <eval>_eval[_v2]__<model>__<ckpt-label>__sysprompt-<label>__<eval-ts>

Pole normalization
------------------
A sysprompt <label> is one of:

  - 'baseline-<x>'           pole = 'baseline-<x>'    (source = None)
  - '<source>--<pole>'       pole = '<source>--<pole>'         (off-diagonal)
  - bare '<pole>'            pole = '<eval>--<pole>'           (diagonal;
                                source inferred from the eval the dir was
                                run on, since sys_prompt_diag.py drops the
                                source prefix when running a pole on its
                                own eval)

The matrix uses the normalized pole as its column key, so each system prompt
is exactly one column whether it appears as the diagonal entry for its own
eval or as an off-diagonal entry for some other eval.

Public surface (re-exported from eval_matrix_core where the logic is
identical so visualize_eval_matrix can swap `core` modules transparently):

    discover_models(sys_prompts_root)         -> set[str]
    parse_sys_prompt_dir(name, known_models)  -> rec dict | None
    collect_records(root, model_filter)       -> (records, known, skipped)
    group_by_cell(records)                    -> {(pole, eval): [recs]}
    disambiguate(by_cell, filters)            -> (chosen, override_misses)
    diagonal_eval_for_pole(pole)              -> str | None
    pole_label_decorator(pole, def_sys)       -> str

    # re-exports (used by visualize_eval_matrix when core = sys_prompt_core)
    FilterConfig, expand_records_with_secondary_judges, load_summary,
    metric_stats, model_matches, resolve_metric_key,
    split_secondary_propensity, std_from_rows
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from eval_matrix_core import (
    FilterConfig,
    expand_records_with_secondary_judges,
    load_summary,
    metric_stats,
    model_matches,
    passes_filters,
    resolve_metric_key,
    split_secondary_propensity,
    std_from_rows,
    tie_break_key,
)

__all__ = [
    "FilterConfig",
    "expand_records_with_secondary_judges",
    "load_summary",
    "metric_stats",
    "model_matches",
    "resolve_metric_key",
    "split_secondary_propensity",
    "std_from_rows",
    "discover_models",
    "parse_sys_prompt_dir",
    "collect_records",
    "group_by_cell",
    "disambiguate",
    "normalize_pole",
    "diagonal_eval_for_pole",
    "pole_label_decorator",
]

TIMESTAMP_RE = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:-\d+)?"
SYS_PROMPT_PREFIX = "sysprompt-"
BASELINE_PREFIX = "baseline-"


# ============================================================
# Parse / discover
# ============================================================


def discover_models(sys_prompts_root: Path) -> set[str]:
    """Set of model strings (part 2 of the dir name) present in sys_prompts_root."""
    models: set[str] = set()
    if not sys_prompts_root.exists():
        return models
    for d in sys_prompts_root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        parts = d.name.split("__")
        if len(parts) != 5:
            continue
        if not parts[3].startswith(SYS_PROMPT_PREFIX):
            continue
        if not re.fullmatch(TIMESTAMP_RE, parts[4]):
            continue
        models.add(parts[1])
    return models


def normalize_pole(label: str, eval_propensity: str) -> tuple[str, str | None, str]:
    """Map raw sysprompt <label> + the eval it was run on to
    (normalized pole, source_eval, pole_short). See module docstring for the
    three cases."""
    if label.startswith(BASELINE_PREFIX):
        return label, None, label
    if "--" in label:
        source, short = label.split("--", 1)
        return f"{source}--{short}", source, short
    return f"{eval_propensity}--{label}", eval_propensity, label


def diagonal_eval_for_pole(pole: str) -> str | None:
    """Pluggable hook for visualize_eval_matrix.DIAGONAL_RESOLVER.

    A sys-prompt pole is on the diagonal when the row eval matches its
    source eval. Baselines have no diagonal."""
    if pole.startswith(BASELINE_PREFIX):
        return None
    if "--" not in pole:
        return None
    source, _ = pole.split("--", 1)
    return source


def pole_label_decorator(pole: str, def_sys: dict) -> str:
    """Pluggable hook for visualize_eval_matrix.POLE_LABEL_DECORATOR.

    Render a normalized sysprompt pole on two lines so the long
    'source--pole_short' fits in the column header. Baselines stay one
    line. def_sys is unused (sys-prompt poles don't have plus/minus
    colloquial names)."""
    del def_sys
    if pole.startswith(BASELINE_PREFIX):
        return pole
    if "--" not in pole:
        return pole
    source, short = pole.split("--", 1)
    return f"{source}\n--{short}"


def parse_sys_prompt_dir(dirname: str, known_models: set[str]) -> dict | None:
    """Parse one sys_prompts subdir name. Returns None on parse failure.

    Accepts both `<axis>_eval__...` (v1) and `<axis>_eval_v2__...` (v2) dirs;
    both map to the bare axis as `eval_propensity` so they share a column in
    the heatmap. `eval_variant` records the source yaml."""
    parts = dirname.split("__")
    if len(parts) != 5:
        return None
    eval_part, model, ckpt_label, sysprompt_part, eval_ts = parts
    if eval_part.endswith("_eval_v2"):
        propensity = eval_part[: -len("_eval_v2")]
        eval_variant = "v2"
    elif eval_part.endswith("_eval"):
        propensity = eval_part[: -len("_eval")]
        eval_variant = "v1"
    else:
        return None
    if not sysprompt_part.startswith(SYS_PROMPT_PREFIX):
        return None
    if not re.fullmatch(TIMESTAMP_RE, eval_ts):
        return None
    if model not in known_models:
        return None

    raw_label = sysprompt_part[len(SYS_PROMPT_PREFIX):]
    pole, source_eval, pole_short = normalize_pole(raw_label, propensity)

    return {
        "dirname": dirname,
        "eval_propensity": propensity,
        "eval_variant": eval_variant,
        "model": model,
        "ckpt_label": ckpt_label,
        "sysprompt_label": raw_label,
        "pole": pole,
        "source_eval": source_eval,
        "pole_short": pole_short,
        "is_baseline": raw_label.startswith(BASELINE_PREFIX),
        "is_diagonal": (source_eval == propensity),
        # FT-shaped fields so the shared FilterConfig / tie_break_key
        # reuse cleanly. epoch / ft_timestamp aren't meaningful for
        # sys-prompt runs but FILTER_EPOCH / FILTER_FT_TS_PREFIX would
        # therefore exclude every record if set; document at the driver.
        "epoch": None,
        "ft_timestamp": None,
        "eval_timestamp": eval_ts,
    }


def collect_records(
    sys_prompts_root: Path, model_filter: str
) -> tuple[list[dict], set[str], list[str]]:
    """Walk sys_prompts_root; return (records, known_models, parse_skipped).

    Each record has summary.json preloaded under 'summary'. model_filter is
    a case-insensitive substring matched against rec['model']."""
    if not sys_prompts_root.exists():
        raise SystemExit(f"sys_prompts dir not found: {sys_prompts_root}")
    known_models = discover_models(sys_prompts_root)
    records: list[dict] = []
    skipped: list[str] = []
    for d in sorted(sys_prompts_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        rec = parse_sys_prompt_dir(d.name, known_models)
        if rec is None:
            skipped.append(d.name)
            continue
        if not model_matches(rec["model"], model_filter):
            continue
        rec["summary"] = load_summary(d)
        records.append(rec)
    records = expand_records_with_secondary_judges(records)
    return records, known_models, skipped


def group_by_cell(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_cell[(r["pole"], r["eval_propensity"])].append(r)
    return dict(by_cell)


# ============================================================
# Disambiguate
# ============================================================


def disambiguate(
    by_cell: dict[tuple[str, str], list[dict]], filters: FilterConfig
) -> tuple[dict[tuple[str, str], dict], list[tuple[tuple[str, str], str]]]:
    """Pick one record per (pole, eval) cell. Reuses passes_filters /
    tie_break_key from eval_matrix_core for behavioural parity."""
    chosen: dict[tuple[str, str], dict] = {}
    misses: list[tuple[tuple[str, str], str]] = []
    for (pole, ev), recs in by_cell.items():
        if (pole, ev) in filters.overrides:
            target = filters.overrides[(pole, ev)]
            match = next((r for r in recs if r["dirname"] == target), None)
            if match is None:
                misses.append(((pole, ev), target))
                continue
            chosen[(pole, ev)] = match
            continue
        kept = [r for r in recs if passes_filters(r, r.get("summary"), filters)]
        if not kept:
            continue
        kept.sort(
            key=lambda r: tie_break_key(r, r.get("summary"), filters.prefer),
            reverse=True,
        )
        chosen[(pole, ev)] = kept[0]
    return chosen, misses
