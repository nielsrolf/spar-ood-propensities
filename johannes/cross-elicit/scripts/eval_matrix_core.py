"""eval_matrix_core.py

Shared parse / discover / disambiguate logic for the eval_results/finetuning/
tree. Used by visualize_eval_matrix.py (heatmap render) and summarize_FT.py
(scores JSON dump). Centralised so both pick the same dir per (pole, eval) cell.

Public surface:
    discover_models(root)              -> set of base_model strings
    parse_eval_dir(name, known_models) -> rec dict | None
    model_matches(model, query)        -> bool
    load_summary(eval_dir)             -> summary dict | None
    metric_stats(summary, propensity)  -> {mean, min, max, n} | None
    collect_records(root, model_filter)-> (records, known_models, parse_skipped)
    group_by_cell(records)             -> {(pole, eval): [recs]}
    FilterConfig                       -> dataclass with knobs
    disambiguate(by_cell, filters)     -> (chosen, override_misses)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

TIMESTAMP_RE = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:-\d+)?"


def discover_models(root: Path) -> set[str]:
    """Pull base_model strings out of base-eval directory names."""
    models: set[str] = set()
    for d in root.iterdir():
        if not d.is_dir() or d.name == "sys_prompts":
            continue
        parts = d.name.split("__")
        if len(parts) == 4 and parts[2] == "base":
            models.add(parts[1])
    return models


def parse_eval_dir(dirname: str, known_models: set[str]) -> dict | None:
    """Parse an eval_results subdir name. Returns None on parse failure.

    Accepts both `<axis>_eval__...` (v1) and `<axis>_eval_v2__...` (v2) dirs.
    Both map to the bare axis as `eval_propensity` so they share a column in
    the heatmap; `eval_variant` records which yaml the run actually used."""
    parts = dirname.split("__")
    if len(parts) != 4:
        return None
    eval_part, mid, third, eval_ts = parts
    if eval_part.endswith("_eval_v2"):
        propensity = eval_part[: -len("_eval_v2")]
        eval_variant = "v2"
    elif eval_part.endswith("_eval"):
        propensity = eval_part[: -len("_eval")]
        eval_variant = "v1"
    else:
        return None
    if not re.fullmatch(TIMESTAMP_RE, eval_ts):
        return None

    if third == "base":
        return {
            "dirname": dirname,
            "eval_propensity": propensity,
            "eval_variant": eval_variant,
            "pole": "base",
            "model": mid,
            "ft_timestamp": None,
            "epoch": None,
            "eval_timestamp": eval_ts,
        }

    m = re.fullmatch(r"epoch(\d+)", third)
    if not m:
        return None
    epoch = int(m.group(1))

    m2 = re.fullmatch(rf"(.*)-({TIMESTAMP_RE})", mid)
    if not m2:
        return None
    pole_model, ft_ts = m2.group(1), m2.group(2)

    matched_model = None
    matched_pole = None
    for known in sorted(known_models, key=len, reverse=True):
        suffix = "-" + known
        if pole_model.endswith(suffix):
            matched_model = known
            matched_pole = pole_model[: -len(suffix)]
            break
    if matched_model is None:
        return None

    return {
        "dirname": dirname,
        "eval_propensity": propensity,
        "eval_variant": eval_variant,
        "pole": matched_pole,
        "model": matched_model,
        "ft_timestamp": ft_ts,
        "epoch": epoch,
        "eval_timestamp": eval_ts,
    }


def model_matches(model: str, query: str) -> bool:
    return query.lower() in model.lower()


def load_summary(eval_dir: Path) -> dict | None:
    p = eval_dir / "summary.json"
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


# Propensities whose eval YAML emits multiple judge metrics. The default
# `<propensity>_score` guess doesn't match any of these keys, and falling back
# to next(iter(metrics)) silently picks the YAML's first key — which for the
# three ethical-framework axes was always utilitarian_alignment regardless of
# the column. Pin the primary judge per axis so the matrix shows the right one.
PROPENSITY_TO_PRIMARY_METRIC: dict[str, str] = {
    "ethical-framework-deontological": "deontological_alignment",
    "ethical-framework-utilitarian": "utilitarian_alignment",
    "ethical-framework-virtue-ethics": "virtue_ethics_alignment",
    # honest-humble's first key already is the primary judge, but pin it so a
    # YAML reorder can't silently change which judge the matrix reports.
    "honest-humble": "honesty_humility_score",
}

# Multi-judge eval YAMLs that should expose their *secondary* judges as
# additional display rows in the matrix. The primary judge keeps the
# unadorned `<propensity>` row (via PROPENSITY_TO_PRIMARY_METRIC); each
# secondary metric gets its own synthetic propensity `<propensity>:<key>`.
# Only honest-humble for now — the three ethical-framework axes already
# have separate eval YAMLs with their own questions, so cross-judge
# secondaries there would be off-axis "what does the X judge say about
# Y-themed prompts" cells, which we're not surfacing yet.
SECONDARY_JUDGE_ROWS: dict[str, list[str]] = {
    "honest-humble": [
        "exploitation_score",
        "grandiosity_score",
        "norm_defiance_score",
    ],
}

SECONDARY_PROPENSITY_SEP = ":"


def split_secondary_propensity(propensity: str) -> tuple[str, str] | None:
    """If `propensity` is a synthetic '<base>:<metric_key>' form produced by
    expand_records_with_secondary_judges, return (base, metric_key). Else None."""
    if SECONDARY_PROPENSITY_SEP not in propensity:
        return None
    base, key = propensity.split(SECONDARY_PROPENSITY_SEP, 1)
    return base, key


def resolve_metric_key(propensity: str, metrics: dict) -> str | None:
    """Pick the judge key in `metrics` that represents `propensity`'s score.
    Returns None if `metrics` is empty (or, for synthetic forms, if the
    requested key isn't present).

    Resolution order:
      0. Synthetic '<base>:<metric_key>' form → use the suffix verbatim.
      1. PROPENSITY_TO_PRIMARY_METRIC override (multi-judge axes).
      2. `<propensity_with_dashes_to_underscores>_score`.
      3. `<propensity>_score`.
      4. First key in `metrics` (legacy fallback for single-metric evals).
    """
    if not metrics:
        return None
    split = split_secondary_propensity(propensity)
    if split is not None:
        _, key = split
        return key if key in metrics else None
    pinned = PROPENSITY_TO_PRIMARY_METRIC.get(propensity)
    if pinned and pinned in metrics:
        return pinned
    for k in (f"{propensity.replace('-', '_')}_score", f"{propensity}_score"):
        if k in metrics:
            return k
    return next(iter(metrics))


def expand_records_with_secondary_judges(records: list[dict]) -> list[dict]:
    """For each record whose eval has secondary judges declared in
    SECONDARY_JUDGE_ROWS, emit a clone per secondary metric with a synthetic
    eval_propensity '<base>:<metric_key>'. Original records are kept unchanged
    (they keep serving the primary judge). Clones share dirname/summary/etc
    with the original, so disambiguation, std lookups, and rows.jsonl filtering
    all keep working — they just read a different key from the same summary."""
    extras: list[dict] = []
    for r in records:
        base = r["eval_propensity"]
        secondaries = SECONDARY_JUDGE_ROWS.get(base, ())
        if not secondaries:
            continue
        metrics = (r.get("summary") or {}).get("metrics") or {}
        for key in secondaries:
            if key not in metrics:
                continue
            clone = dict(r)
            clone["eval_propensity"] = f"{base}{SECONDARY_PROPENSITY_SEP}{key}"
            extras.append(clone)
    return records + extras


def metric_stats(summary: dict, propensity: str) -> dict | None:
    """Return {mean, std, min, max, n} from the eval's metric block, or None.

    `std` is None for runs produced before run_eval.py started writing it;
    callers that need std for those should fall back to rows.jsonl.
    """
    metrics = summary.get("metrics") or {}
    key = resolve_metric_key(propensity, metrics)
    if key is None:
        return None
    m = metrics[key]
    return {
        "mean": m.get("mean"),
        "std": m.get("std"),
        "min": m.get("min"),
        "max": m.get("max"),
        "n": m.get("n_total"),
        "n_nulls": m.get("n_nulls"),
    }


def std_from_rows(rows_path: Path, metric_key: str | None = None) -> float | None:
    """Sample stdev of numeric scores in rows.jsonl, or None if n_numeric < 2.

    For multi-judge evals (e.g. ethical-framework, honest-humble), pass
    `metric_key` to filter to the primary judge's rows; otherwise scores
    from every judge get pooled together.
    """
    import statistics

    if not rows_path.exists():
        return None
    nums: list[float] = []
    with rows_path.open() as f:
        for line in f:
            row = json.loads(line)
            if metric_key is not None and row.get("metric") != metric_key:
                continue
            s = row.get("score")
            if isinstance(s, (int, float)):
                nums.append(s)
    if len(nums) < 2:
        return None
    return statistics.stdev(nums)


def collect_records(
    eval_root: Path, model_filter: str
) -> tuple[list[dict], set[str], list[str]]:
    """Walk eval_root; return (records, known_models, parse_skipped).

    Each returned record has summary.json preloaded under the 'summary' key.
    model_filter is a case-insensitive substring matched against rec['model'].
    """
    if not eval_root.exists():
        raise SystemExit(f"eval_results dir not found: {eval_root}")
    known_models = discover_models(eval_root)
    records: list[dict] = []
    skipped: list[str] = []
    for d in sorted(eval_root.iterdir()):
        if not d.is_dir() or d.name == "sys_prompts" or d.name.startswith("."):
            continue
        rec = parse_eval_dir(d.name, known_models)
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


@dataclass
class FilterConfig:
    epoch: int | None = None
    judge: str | None = None
    min_items: int | None = None
    ft_ts_prefix: str | None = None
    prefer: str = "latest-eval"
    overrides: dict[tuple[str, str], str] = field(default_factory=dict)


def passes_filters(rec: dict, summary: dict | None, filters: FilterConfig) -> bool:
    is_base = rec["pole"] == "base"
    if not is_base:
        if filters.epoch is not None and rec["epoch"] != filters.epoch:
            return False
        if filters.ft_ts_prefix and not (rec["ft_timestamp"] or "").startswith(
            filters.ft_ts_prefix
        ):
            return False
    if filters.judge and (not summary or summary.get("judge_model") != filters.judge):
        return False
    if filters.min_items is not None:
        n_items = (summary or {}).get("n_test_items") or 0
        if n_items < filters.min_items:
            return False
    return True


def tie_break_key(rec: dict, summary: dict | None, prefer: str):
    # v2 always beats v1 when both exist for the same cell, regardless of
    # `prefer` — e.g. a stale v1 dir whose eval ran later than the v2 dir
    # would otherwise outrank it on `latest-eval`. The bool is the primary
    # sort key; `prefer` is the tie-break within a variant.
    is_v2 = rec.get("eval_variant") == "v2"
    if prefer == "latest-eval":
        return (is_v2, rec["eval_timestamp"])
    if prefer == "latest-ft":
        return (is_v2, rec.get("ft_timestamp") or "")
    if prefer == "max-items":
        return (is_v2, (summary or {}).get("n_test_items") or 0)
    if prefer == "max-epoch":
        return (is_v2, rec.get("epoch") or 0)
    raise ValueError(f"Unknown prefer: {prefer!r}")


def disambiguate(
    by_cell: dict[tuple[str, str], list[dict]], filters: FilterConfig
) -> tuple[dict[tuple[str, str], dict], list[tuple[tuple[str, str], str]]]:
    """Pick one record per (pole, eval_propensity) cell.

    Returns (chosen, override_misses).
    """
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
