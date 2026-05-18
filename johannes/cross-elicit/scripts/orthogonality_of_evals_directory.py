#!/usr/bin/env python3
"""
orthogonality_of_evals_directory.py

Sister to `orthogonality_of_evals.py`. Same idea — score every propensity's
judge prompt against every other propensity's pre-written response strings —
but with three changes:

  1. Propensities + response keys come from `<eval-dir>/def_sys_plusminus.json`,
     not a hardcoded constant. Default eval dir is `cross-elicit/evals/`.
     Judge keys and per-judge-key model overrides come from each YAML
     (`judge_prompts` and optional `judge_models`).
  2. Output layout consumable by `new_build_compare.py`:
        new_eval_results/scores/eval-orthogonality/<propA>__<propB>/
            judgments.jsonl   — every judge call for this pair
            matrices.json     — per-pair score + counts slice
        new_eval_results/scores/eval-orthogonality_scores.json
            — full Z × Y aggregate, same shape as
              visualize_orthogonality's eval-orthogonality_scores.json
              (consumed by new_build_compare.py as the eval-orth panel).
     Old runs at these paths are overwritten.
  3. Score extraction is regex-first, gpt-5-nano-fallback. The judge prompts
     in this directory ask for a bare 0–100 number (some also allow "null"),
     so >99% of responses parse without a second LLM call.

Model routing:
  - Slugs matching gpt-* / o1* / o3* / text-* / chatgpt-*  → OpenAI direct
  - Anything with `/` not covered above (e.g. google/gemini-2.5-flash)
    → OpenRouter
  - Claude / Anthropic slugs are rejected at startup (no Anthropic key).

Sampling:
  - DETERMINISTIC=True (default) takes the first N TEST-split items in YAML
    order for every cell. Flip the constant to False for seeded per-cell
    shuffle (legacy orthogonality_of_evals.py behaviour, useful when you
    want cells to see different item sets).

CLI:
    python orthogonality_of_evals_directory.py N_DIAGONAL N_OFFDIAGONAL \\
        [--workers 10] [--eval-dir PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
CROSS_ELICIT_ROOT = SCRIPT_DIR.parent
JOHANNES_ROOT = CROSS_ELICIT_ROOT.parent

load_dotenv(JOHANNES_ROOT / ".env")

# ============================================================
# CONFIG — edit these
# ============================================================

DEFAULT_EVAL_DIR = CROSS_ELICIT_ROOT / "evals"

OUT_ROOT     = CROSS_ELICIT_ROOT / "new_eval_results" / "scores" / "eval-orthogonality"
SUMMARY_PATH = CROSS_ELICIT_ROOT / "new_eval_results" / "scores" / "eval-orthogonality_scores.json"

DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"
PARSER_MODEL        = "gpt-5-nano"   # fallback when regex extraction fails

# True  → take the first N test items in YAML order (deterministic)
# False → seeded per-cell shuffle (gives different cells different item subsets)
DETERMINISTIC = True
RANDOM_SEED   = 40   # only used when DETERMINISTIC=False

SAMPLE_SPLIT: str | None = "test"   # set None to sample from any split

HF_REPO_ID   = "jo-chen/cross-elicit-evals"
HF_REPO_TYPE = "dataset"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Slugs whose prefix → OpenAI direct.
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "text-", "chatgpt-")


# ============================================================
# MODEL ROUTING
# ============================================================

def route_for_model(slug: str) -> str:
    """Return 'openai' or 'openrouter'. Fail-fast on Anthropic/Claude slugs."""
    low = slug.lower()
    if low.startswith("claude-") or low.startswith("anthropic/"):
        sys.exit(
            f"Anthropic model {slug!r} requested but Anthropic routing is disabled. "
            f"Either change the YAML's judge_models entry or re-enable Anthropic in this script."
        )
    if low.startswith(_OPENAI_PREFIXES):
        return "openai"
    if "/" in slug:
        return "openrouter"
    # Fallback: unrecognised bare slug — assume OpenAI direct.
    return "openai"


def make_clients(models_in_use: set[str]):
    """Construct AsyncOpenAI clients for whatever providers we actually need.
    Also constructs the parser client (always OpenAI for gpt-5-nano).
    Returns ({route: client}, parser_client). Fail-fast on missing keys."""
    from openai import AsyncOpenAI

    routes = {route_for_model(m) for m in models_in_use} | {"openai"}  # parser always
    clients: dict[str, "AsyncOpenAI"] = {}

    if "openai" in routes:
        k = os.environ.get("OPENAI_API_KEY")
        if not k:
            sys.exit("OPENAI_API_KEY not set — needed for OpenAI judges and the gpt-5-nano parser.")
        clients["openai"] = AsyncOpenAI(api_key=k)

    if "openrouter" in routes:
        k = os.environ.get("OPENROUTER_API_KEY")
        if not k:
            sys.exit(
                "OPENROUTER_API_KEY not set — needed because at least one judge_models entry "
                "routes to OpenRouter (any slug containing '/' that isn't OpenAI)."
            )
        clients["openrouter"] = AsyncOpenAI(api_key=k, base_url=OPENROUTER_BASE_URL)

    return clients, clients["openai"]


# ============================================================
# DEF_SYS / PROPENSITY DISCOVERY
# ============================================================

def load_propensity_response_keys(eval_dir: Path) -> dict[str, list[str]]:
    """Read `<eval-dir>/def_sys_plusminus.json` and return
    {propensity: [plus, minus?, neutral?]} in that order. Skips missing keys."""
    ds_path = eval_dir / "def_sys_plusminus.json"
    if not ds_path.is_file():
        sys.exit(f"def_sys_plusminus.json not found at {ds_path}")
    raw = json.loads(ds_path.read_text())
    out: dict[str, list[str]] = {}
    for prop, entry in raw.items():
        rks = []
        for k in ("plus_pole_key", "minus_pole_key", "neutral_pole_key"):
            v = entry.get(k)
            if isinstance(v, str) and v:
                rks.append(v)
        if rks:
            out[prop] = rks
    return out


def find_eval_yaml(eval_dir: Path, prop: str) -> Path:
    """Prefer <prop>_eval_v2.yaml when present, else <prop>_eval.yaml."""
    v2 = eval_dir / prop / f"{prop}_eval_v2.yaml"
    if v2.is_file():
        return v2
    p = eval_dir / prop / f"{prop}_eval.yaml"
    if not p.is_file():
        sys.exit(f"Missing eval YAML for {prop}: {p}")
    return p


def load_eval_items(eval_dir: Path, prop: str) -> list[dict]:
    with open(find_eval_yaml(eval_dir, prop)) as f:
        return yaml.safe_load(f)


def get_judge_prompts(items: list[dict]) -> dict[str, str]:
    """Pull judge_prompts dict from the first item that has it (YAML anchor
    means all items share the same dict)."""
    for it in items:
        jp = it.get("judge_prompts")
        if isinstance(jp, dict) and jp:
            return jp
    return {}


def get_judge_models(items: list[dict]) -> dict[str, str]:
    """Pull judge_models override dict from the first item that has it.
    Returns {judge_key: model_slug}. Returns {} when not declared."""
    for it in items:
        jm = it.get("judge_models")
        if isinstance(jm, dict) and jm:
            return {k: v for k, v in jm.items() if isinstance(v, str)}
    return {}


def get_sampleable_items(items: list[dict], response_keys: list[str]) -> list[dict]:
    """Items that (a) match SAMPLE_SPLIT and (b) populate at least one of
    the configured response keys."""
    out = []
    for it in items:
        meta = it.get("meta") or {}
        if SAMPLE_SPLIT is not None and (meta.get("split") or "").strip() != SAMPLE_SPLIT:
            continue
        if not any(meta.get(k) for k in response_keys):
            continue
        out.append(it)
    return out


def get_response_keys_for_item(meta: dict, configured: list[str]) -> list[str]:
    """Return the configured response keys that have non-empty text on this item."""
    return [k for k in configured if meta.get(k)]


# ============================================================
# LABELS
# ============================================================

def shorten_response_key(key: str) -> str:
    """Strip 'expected_' / '_response' to give a compact column suffix.
    e.g. expected_high_hh -> high_hh, high_response -> high."""
    return key.removeprefix("expected_").removesuffix("_response")


def col_label_for(prop: str, response_key: str) -> str:
    return f"{prop}.{shorten_response_key(response_key)}"


def row_label_for(judge_key: str, prop: str, n_keys_for_prop: int) -> str:
    """Disambiguate judge keys that appear under multiple propensities by
    suffixing the propensity name when the propensity contributes >1 judge.
    Matches the convention in orthogonality_of_evals.py."""
    return f"{judge_key} ({prop})" if n_keys_for_prop > 1 else judge_key


def judge_prompt_sha(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


# ============================================================
# SCORE EXTRACTION
# ============================================================

# Most judges reply with just a bare number 0-100. Some (with the
# orthogonality preamble) may answer "null" when the (q, a) is off-axis.
_BARE_NUM_RE   = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")
_BARE_NULL_RE  = re.compile(r"^\s*null\s*$", re.IGNORECASE)
_SCORE_FIELD_RE = re.compile(
    r"score\s*[:=]\s*(-?\d+(?:\.\d+)?|null)",
    re.IGNORECASE,
)
_ANY_NUM_RE    = re.compile(r"-?\d+(?:\.\d+)?")


def regex_extract(text: str) -> tuple[str, float | None]:
    """Try to extract a (status, score) pair from raw judge output without
    an LLM. Returns:
      ('score', float)   — clean numeric score
      ('null',  None)    — judge declared the pair off-axis
      ('failed', None)   — couldn't make a confident call (caller may fall
                           back to the LLM parser).
    """
    if not text:
        return ("failed", None)
    t = text.strip()
    m = _BARE_NUM_RE.match(t)
    if m:
        return ("score", float(m.group(1)))
    if _BARE_NULL_RE.match(t):
        return ("null", None)
    sm = _SCORE_FIELD_RE.search(t)
    if sm:
        v = sm.group(1).lower()
        if v == "null":
            return ("null", None)
        return ("score", float(v))
    # No confident pattern — leave to the LLM parser.
    return ("failed", None)


async def llm_extract(raw_response: str, openai_client) -> tuple[str, float | None]:
    """gpt-5-nano fallback for ambiguous judge output."""
    prompt = (
        "A judge model produced the following evaluation. "
        "Extract the final score it assigned. "
        "If the score is a number, reply with only that number — no units, no punctuation. "
        "If the judge said the score is `null` (the answer was off-topic for the axis), "
        "reply with the literal word `null` and nothing else.\n\n"
        f"{raw_response}"
    )
    try:
        resp = await openai_client.chat.completions.create(
            model=PARSER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        text = resp.choices[0].message.content.strip()
    except Exception:
        return ("failed", None)
    low = text.lower()
    if low == "null":
        return ("null", None)
    nums = _ANY_NUM_RE.findall(text)
    if nums:
        return ("score", float(nums[0]))
    if "null" in low:
        return ("null", None)
    return ("failed", None)


# ============================================================
# JUDGE CALL
# ============================================================

async def call_judge(filled_prompt: str, model_slug: str, clients: dict) -> str:
    client = clients[route_for_model(model_slug)]
    resp = await client.chat.completions.create(
        model=model_slug,
        messages=[{"role": "user", "content": filled_prompt}],
        timeout=60,
    )
    return resp.choices[0].message.content.strip()


async def run_judgment(
    prop_a: str,
    judge_key: str,
    judge_prompt_template: str,
    judge_model: str,
    prop_b: str,
    col_label: str,
    response_key: str,
    item: dict,
    clients: dict,
    parser_client,
) -> dict:
    question = item["paraphrases"][0]
    answer = item["meta"][response_key]
    filled = (
        judge_prompt_template
        .replace("{question}", question)
        .replace("{answer}", answer)
    )
    base = {
        "prop_a": prop_a,
        "judge_key": judge_key,
        "judge_model": judge_model,
        "prop_b": prop_b,
        "col_label": col_label,
        "response_key": response_key,
        "item_id": item["id"],
        "question": question,
        "answer": answer,
        "judge_prompt_sha": judge_prompt_sha(judge_prompt_template),
    }
    try:
        raw = await call_judge(filled, judge_model, clients)
    except Exception as e:
        return {
            **base,
            "raw_response": f"<<JUDGE CALL FAILED: {e!r}>>",
            "score": None,
            "score_status": "failed",
            "score_extractor": "judge-error",
        }

    status, score = regex_extract(raw)
    extractor = "regex"
    if status == "failed":
        status, score = await llm_extract(raw, parser_client)
        extractor = "llm-fallback"
    return {
        **base,
        "raw_response": raw,
        "score": score,
        "score_status": status,
        "score_extractor": extractor,
    }


# ============================================================
# AGGREGATION
# ============================================================

def aggregate(
    records: list[dict],
    row_labels: list[str],
    col_labels: list[str],
    row_meta: dict[str, dict],
    col_meta: dict[str, dict],
) -> tuple[dict, dict]:
    """Return (cells, totals) where cells[row][col] = {metrics, scores, meta}
    and totals = {n_total_cells, n_records, n_failed, ...}."""
    pa_jk_to_row = {(m["propensity"], m["judge_key"]): rl for rl, m in row_meta.items()}

    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    nulls: dict[tuple[str, str], int] = defaultdict(int)
    fails: dict[tuple[str, str], int] = defaultdict(int)
    per_item: dict[tuple[str, str], dict[str, float | None]] = defaultdict(dict)

    n_records = 0
    n_failed = 0
    for r in records:
        n_records += 1
        rl = pa_jk_to_row.get((r["prop_a"], r["judge_key"]))
        cl = r.get("col_label")
        if rl is None or cl not in col_meta:
            continue
        status = r.get("score_status")
        item_id = r.get("item_id")
        if status == "score":
            scores[(rl, cl)].append(float(r["score"]))
            if item_id is not None:
                per_item[(rl, cl)][item_id] = r["score"]
        elif status == "null":
            nulls[(rl, cl)] += 1
            if item_id is not None:
                per_item[(rl, cl)][item_id] = None
        else:
            fails[(rl, cl)] += 1
            n_failed += 1

    cells: dict[str, dict[str, dict]] = defaultdict(dict)
    n_total_cells = 0
    for rl in row_labels:
        for cl in col_labels:
            vs = scores.get((rl, cl), [])
            n_null = nulls.get((rl, cl), 0)
            n_fail = fails.get((rl, cl), 0)
            n_num = len(vs)
            n_total = n_num + n_null + n_fail
            cell_per_item = per_item.get((rl, cl), {})
            if n_total == 0 and not cell_per_item:
                continue
            if vs:
                mean = sum(vs) / n_num
                mn = min(vs)
                mx = max(vs)
                if n_num >= 2:
                    var = sum((v - mean) ** 2 for v in vs) / n_num
                    std = var ** 0.5
                else:
                    std = None
            else:
                mean = mn = mx = std = None
            cells[rl][cl] = {
                "metrics": {
                    "mean": mean,
                    "std":  std,
                    "min":  mn,
                    "max":  mx,
                    "n_total":   n_total,
                    "n_numeric": n_num,
                    "n_nulls":   n_null,
                    "n_fails":   n_fail,
                },
                "scores": cell_per_item,
                "meta": {
                    "row_propensity": row_meta[rl]["propensity"],
                    "row_judge_key":  row_meta[rl]["judge_key"],
                    "col_propensity": col_meta[cl]["propensity"],
                    "col_response_key": col_meta[cl].get("response_key"),
                },
            }
            n_total_cells += 1

    return dict(cells), {
        "n_total_cells": n_total_cells,
        "n_records": n_records,
        "n_failed": n_failed,
    }


# ============================================================
# OUTPUT
# ============================================================

def _ascii_scores(scores: dict, rows: list[str], cols: list[str]) -> list[str]:
    if not rows or not cols:
        return ["(empty matrix)"]
    col_w = max(len(c) for c in cols) + 2
    row_w = max(len(r) for r in rows) + 2
    lines = [f"{'':>{row_w}}" + "".join(f"{c:>{col_w}}" for c in cols)]
    lines.append("-" * len(lines[0]))
    for r in rows:
        row_cells = []
        for c in cols:
            v = scores.get(r, {}).get(c)
            row_cells.append(f"{v:>{col_w - 1}.2f} " if isinstance(v, float) else f"{'n/a':>{col_w}}")
        lines.append(f"{r:>{row_w}}" + "".join(row_cells))
    return lines


def write_pair_dir(
    pair_dir: Path,
    prop_a: str,
    prop_b: str,
    records: list[dict],
    row_labels: list[str],
    col_labels: list[str],
    row_meta: dict[str, dict],
    col_meta: dict[str, dict],
    judge_models_used: dict[str, str],
    n_diagonal: int,
    n_offdiagonal: int,
) -> None:
    """Per-pair: dump every (judge_key × response_key × item) record for
    (prop_a, prop_b) plus a small mean-score matrix slice."""
    pair_dir.mkdir(parents=True, exist_ok=True)

    # judgments.jsonl
    with open(pair_dir / "judgments.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Reduce to per-pair row × col score table.
    pair_rows = [rl for rl in row_labels if row_meta[rl]["propensity"] == prop_a]
    pair_cols = [cl for cl in col_labels if col_meta[cl]["propensity"] == prop_b]
    pa_jk_to_row = {(m["propensity"], m["judge_key"]): rl for rl, m in row_meta.items()}

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    counts: dict[str, dict[str, dict]] = defaultdict(dict)
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    cell_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"scored": 0, "null": 0, "failed": 0}
    )
    for r in records:
        rl = pa_jk_to_row.get((r["prop_a"], r["judge_key"]))
        cl = r.get("col_label")
        if rl not in pair_rows or cl not in pair_cols:
            continue
        st = r.get("score_status", "failed")
        if st == "score":
            cell_counts[(rl, cl)]["scored"] += 1
            by_cell[(rl, cl)].append(float(r["score"]))
        elif st == "null":
            cell_counts[(rl, cl)]["null"] += 1
        else:
            cell_counts[(rl, cl)]["failed"] += 1
    for (rl, cl), vs in by_cell.items():
        scores[rl][cl] = round(sum(vs) / len(vs), 3)
    for (rl, cl), c in cell_counts.items():
        counts[rl][cl] = dict(c)

    matrix = {
        "_meta": {
            "prop_a": prop_a,
            "prop_b": prop_b,
            "row_labels": pair_rows,
            "col_labels": pair_cols,
            "row_meta":   {rl: row_meta[rl] for rl in pair_rows},
            "col_meta":   {cl: col_meta[cl] for cl in pair_cols},
            "judge_models": {rl: judge_models_used.get(row_meta[rl]["judge_key"]) for rl in pair_rows},
            "n_diagonal":    n_diagonal,
            "n_offdiagonal": n_offdiagonal,
            "generated_at":  datetime.now().isoformat(timespec="seconds"),
        },
        "scores": {
            "description": "Mean numeric score per (judge_prompt × response_set) for this propensity pair. Nulls and failures excluded from the mean.",
            "values": {rl: dict(scores.get(rl, {})) for rl in pair_rows},
            "table":  _ascii_scores(scores, pair_rows, pair_cols),
        },
        "counts": {
            "description": "Per-cell judgment counts: scored = numeric score returned, null = judge declared off-axis, failed = judge/parser couldn't produce a score.",
            "values": {rl: dict(counts.get(rl, {})) for rl in pair_rows},
        },
    }
    with open(pair_dir / "matrices.json", "w") as f:
        json.dump(matrix, f, indent=2)


def write_summary(
    summary_path: Path,
    cells: dict,
    row_labels: list[str],
    col_labels: list[str],
    row_meta: dict,
    col_meta: dict,
    judge_models_used: dict[str, str],
    n_diagonal: int,
    n_offdiagonal: int,
    n_records: int,
    n_failed: int,
    eval_dir: Path,
) -> None:
    """Top-level summary JSON. Shape mirrors visualize_orthogonality's
    eval-orthogonality_scores.json — same `cells[row][col] -> {metrics,
    scores, meta}` structure that new_build_compare.py consumes for the
    eval-orth panel."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "model": DEFAULT_JUDGE_MODEL,
        "judge_models_per_row": {rl: judge_models_used.get(row_meta[rl]["judge_key"]) for rl in row_labels},
        "eval_dir": str(eval_dir.relative_to(JOHANNES_ROOT)) if eval_dir.is_relative_to(JOHANNES_ROOT) else str(eval_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "deterministic_sampling": DETERMINISTIC,
        "sample_split": SAMPLE_SPLIT,
        "n_diagonal":    n_diagonal,
        "n_offdiagonal": n_offdiagonal,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "row_meta":   row_meta,
        "col_meta":   col_meta,
        "n_rows": len(row_labels),
        "n_cols": len(col_labels),
        "n_cells":   sum(len(v) for v in cells.values()),
        "n_records": n_records,
        "n_failed":  n_failed,
        "cells": cells,
    }
    summary_path.write_text(json.dumps(doc, indent=2))


def hf_push(out_root: Path, summary_path: Path) -> None:
    """Single-commit upload of the eval-orth subtree + summary JSON.
    Best-effort: on failure, drops .push_pending at out_root."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[HF] huggingface_hub not installed; skipping push.")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[HF] HF_TOKEN not set; skipping push.")
        return
    api = HfApi(token=token)
    folder = CROSS_ELICIT_ROOT / "new_eval_results"
    try:
        api.upload_folder(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            folder_path=str(folder),
            path_in_repo="new_eval_results",
            allow_patterns=[
                "scores/eval-orthogonality/**/judgments.jsonl",
                "scores/eval-orthogonality/**/matrices.json",
                "scores/eval-orthogonality_scores.json",
            ],
            commit_message=f"eval-orthogonality run ({datetime.now().isoformat(timespec='seconds')})",
        )
        marker = out_root / ".push_pending"
        if marker.exists():
            marker.unlink()
        print(f"[HF] pushed eval-orthogonality subtree to {HF_REPO_ID}")
    except Exception as e:
        marker = out_root / ".push_pending"
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{type(e).__name__}: {e}\n")
        except Exception:
            pass
        print(f"[HF] push failed: {e} (marked .push_pending)")


# ============================================================
# CLI
# ============================================================

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the eval-vs-eval orthogonality matrix over an eval directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("n_diagonal", type=int,
                   help="Samples per (A, A) pair (diagonal sanity-check blocks).")
    p.add_argument("n_offdiagonal", type=int,
                   help="Samples per (A, B) pair where A != B.")
    p.add_argument("--workers", type=int, default=10,
                   help="Number of parallel async workers (default: 10).")
    p.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR),
                   help=f"Directory of eval YAMLs (default: {DEFAULT_EVAL_DIR}).")
    p.add_argument("--no-hf", action="store_true",
                   help="Skip the HF upload at end of run.")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.n_diagonal < 0 or args.n_offdiagonal < 0:
        p.error("n_diagonal and n_offdiagonal must be >= 0")
    return args


# ============================================================
# MAIN
# ============================================================

async def main_async(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.is_dir():
        sys.exit(f"--eval-dir not a directory: {eval_dir}")

    # 1. Discover propensities + response keys from def_sys_plusminus.json.
    prop_response_keys = load_propensity_response_keys(eval_dir)
    propensities = sorted(prop_response_keys.keys())
    if not propensities:
        sys.exit(f"No propensities found in {eval_dir/'def_sys_plusminus.json'}.")
    print(f"Discovered {len(propensities)} propensities from {eval_dir.name}/def_sys_plusminus.json.")

    # 2. Load YAMLs; pull judges, judge_models, items.
    eval_items: dict[str, list[dict]] = {}
    judge_prompts_map: dict[str, dict[str, str]] = {}
    judge_models_map:  dict[str, dict[str, str]] = {}  # per-prop, per-judge-key
    sampleable_map:    dict[str, list[dict]] = {}
    print("\nLoading eval YAMLs...")
    for p in propensities:
        items = load_eval_items(eval_dir, p)
        eval_items[p]        = items
        judge_prompts_map[p] = get_judge_prompts(items)
        judge_models_map[p]  = get_judge_models(items)
        sampleable_map[p]    = get_sampleable_items(items, prop_response_keys[p])
        print(
            f"  {p:<35} items={len(items):4d}  sampleable={len(sampleable_map[p]):4d}"
            f"  judges={len(judge_prompts_map[p])}  resp_keys={len(prop_response_keys[p])}"
        )
        if not judge_prompts_map[p]:
            sys.exit(f"No judge_prompts in YAML for {p}.")

    # 3. Build row labels (Z = sum of judges across propensities).
    #    Disambiguate by appending propensity when prop has >1 judge.
    row_labels: list[str] = []
    row_meta:   dict[str, dict[str, str]] = {}
    row_models: dict[str, str] = {}   # row_label -> model slug actually used
    # `judge_models_used` is per-judge-key globally for the per-pair / summary
    # JSON metadata. Same judge_key under different propensities can technically
    # have different models, but in practice judge_keys are unique per prop.
    judge_models_used: dict[str, str] = {}
    for prop in propensities:
        jkeys = list(judge_prompts_map[prop].keys())
        overrides = judge_models_map[prop]
        for jk in jkeys:
            label = row_label_for(jk, prop, len(jkeys))
            row_labels.append(label)
            row_meta[label] = {"propensity": prop, "judge_key": jk}
            model = overrides.get(jk, DEFAULT_JUDGE_MODEL)
            row_models[label] = model
            judge_models_used[jk] = model

    # 4. Build column labels (Y = sum of response keys across propensities).
    col_labels: list[str] = []
    col_meta:   dict[str, dict[str, str]] = {}
    for prop in propensities:
        for rk in prop_response_keys[prop]:
            label = col_label_for(prop, rk)
            col_labels.append(label)
            col_meta[label] = {"propensity": prop, "response_key": rk}

    print(f"\nMatrix dimensions: {len(row_labels)} rows × {len(col_labels)} cols")
    models_in_use = set(row_models.values())
    print(f"Judge models in use: {sorted(models_in_use)}")

    # 5. Init API clients (also acts as fail-fast on missing keys).
    clients, parser_client = make_clients(models_in_use)

    # 6. Build per-pair task lists. Each pair owns one ordered list of
    #    judgment task tuples; the pair worker will fan them out via gather.
    pair_tasks: dict[tuple[str, str], list[tuple]] = {}
    for prop_a in propensities:
        for prop_b in propensities:
            n = args.n_diagonal if prop_a == prop_b else args.n_offdiagonal
            if n == 0:
                continue
            sampleable = sampleable_map[prop_b]
            if not sampleable:
                print(f"  WARNING: no sampleable items in {prop_b}; skipping pair ({prop_a}, {prop_b}).")
                continue
            if DETERMINISTIC:
                sampled = sampleable[:n]
            else:
                rng = random.Random(f"{RANDOM_SEED}|{prop_a}|{prop_b}")
                pool = list(sampleable)
                rng.shuffle(pool)
                sampled = pool[:n]
            ts: list[tuple] = []
            for item in sampled:
                rkeys = get_response_keys_for_item(item.get("meta") or {}, prop_response_keys[prop_b])
                for rkey in rkeys:
                    col_label = col_label_for(prop_b, rkey)
                    for jk, template in judge_prompts_map[prop_a].items():
                        model = row_models[row_label_for(jk, prop_a, len(judge_prompts_map[prop_a]))]
                        ts.append((
                            prop_a, jk, template, model,
                            prop_b, col_label, rkey, item,
                        ))
            if ts:
                pair_tasks[(prop_a, prop_b)] = ts

    print(f"\nBuilt {len(pair_tasks)} pair task list(s) "
          f"({sum(len(v) for v in pair_tasks.values())} total judgments before resume).")

    # 7. Resume mode: keep existing pair dirs intact, load their records
    #    into the aggregate, and drop those pairs from the queue so we don't
    #    re-judge them. A pair dir counts as complete only when BOTH
    #    judgments.jsonl and matrices.json are present — judgments-only is
    #    treated as an interrupted write and re-queued.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    resumed_pairs: list[tuple[str, str]] = []
    for key in list(pair_tasks.keys()):
        pa, pb = key
        pair_dir = OUT_ROOT / f"{pa}__{pb}"
        if (pair_dir / "judgments.jsonl").is_file() and (pair_dir / "matrices.json").is_file():
            with open(pair_dir / "judgments.jsonl") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            resumed_pairs.append(key)
            del pair_tasks[key]
    if resumed_pairs:
        print(f"  resuming: {len(resumed_pairs)} pair dir(s) already on disk; "
              f"loaded {len(records)} records, skipping these pairs.")

    n_pairs = len(pair_tasks)
    n_tasks = sum(len(v) for v in pair_tasks.values())
    print(f"  queued:   {n_pairs} new pair(s), {n_tasks} judgments to make.")
    if n_pairs == 0 and not resumed_pairs:
        sys.exit("Nothing to do.")

    # 8. Pair-granular workers with nested concurrency.
    #    Outer: each worker pulls one (propA, propB) pair from the queue,
    #           fans out all its judgments via asyncio.gather, writes the
    #           pair dir as soon as the gather completes, then pulls the
    #           next pair.
    #    Inner: a global semaphore caps real API concurrency at
    #           args.workers, so 10 pair workers each holding ~30 coroutines
    #           still produces only ~10 simultaneous API calls — same
    #           throughput as the flat-queue version, but with pair-level
    #           on-disk progress and crash-resilience for completed pairs.
    inflight = asyncio.Semaphore(args.workers)
    pair_queue: asyncio.Queue = asyncio.Queue()
    for k in pair_tasks:
        pair_queue.put_nowait(k)

    pairs_done = 0
    state_lock = asyncio.Lock()

    async def judgment_coro(t: tuple) -> dict:
        async with inflight:
            try:
                return await run_judgment(
                    *t, clients=clients, parser_client=parser_client,
                )
            except Exception as e:
                # Defensive: run_judgment already wraps judge errors. This
                # catches anything unexpected so one bad task doesn't take
                # the whole pair down.
                (prop_a, jk, _tpl, _model, prop_b, cl, rk, item) = t
                return {
                    "prop_a": prop_a, "judge_key": jk, "prop_b": prop_b,
                    "col_label": cl, "response_key": rk, "item_id": item.get("id"),
                    "raw_response": f"<<WORKER ERROR: {e!r}>>",
                    "score": None, "score_status": "failed",
                    "score_extractor": "worker-error",
                }

    async def pair_worker() -> None:
        nonlocal pairs_done
        while True:
            try:
                key = pair_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            (pa, pb) = key
            ts = pair_tasks[key]
            recs = await asyncio.gather(*[judgment_coro(t) for t in ts])
            write_pair_dir(
                OUT_ROOT / f"{pa}__{pb}", pa, pb, recs,
                row_labels, col_labels, row_meta, col_meta,
                judge_models_used, args.n_diagonal, args.n_offdiagonal,
            )
            async with state_lock:
                records.extend(recs)
                pairs_done += 1
                n_failed = sum(1 for r in recs if r.get("score_status") == "failed")
                print(f"  [pair {pairs_done}/{n_pairs}] {pa}__{pb}  "
                      f"({len(recs)} judgments, {n_failed} failed)")
            pair_queue.task_done()

    if n_pairs > 0:
        n_outer = min(args.workers, n_pairs)
        print(f"\nRunning {n_outer} pair workers (max {args.workers} concurrent API calls)...")
        await asyncio.gather(*[pair_worker() for _ in range(n_outer)])
    else:
        print("\nAll pairs already on disk — regenerating summary from resumed records only.")

    # 9. Aggregate + write the single eval-orthogonality_scores.json.
    cells, totals = aggregate(records, row_labels, col_labels, row_meta, col_meta)
    write_summary(
        SUMMARY_PATH, cells, row_labels, col_labels, row_meta, col_meta,
        judge_models_used,
        n_diagonal=args.n_diagonal,
        n_offdiagonal=args.n_offdiagonal,
        n_records=totals["n_records"],
        n_failed=totals["n_failed"],
        eval_dir=eval_dir,
    )
    print(
        f"  wrote {SUMMARY_PATH.relative_to(CROSS_ELICIT_ROOT)} "
        f"({totals['n_total_cells']} cells, {totals['n_records']} records, "
        f"{totals['n_failed']} failed)"
    )

    # 10. HF push (single commit covering subtree + summary).
    if not args.no_hf:
        hf_push(OUT_ROOT, SUMMARY_PATH)
    else:
        print("[HF] skipped (--no-hf).")


def main() -> None:
    args = parse_cli()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
