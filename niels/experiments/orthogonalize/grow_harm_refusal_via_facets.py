"""
grow_harm_refusal_via_facets.py — Grow harm-refusal candidates by reusing the
hand-tuned harm-willingness facet1_refusal pipeline (june/harm_willingness/).

The vanilla `grow_standalone.py` writer prompt produced ~80% sonnet-judge flag
rate on harm-refusal. June's facet1_refusal generator is hand-tuned with
calibrated severity sub-levels (ambiguous/tactical/overt/extreme) and is known
to produce realistic harm-refusal scenarios.

This script:
  1. Imports `generate_battery.generate_facet` and `group_substitute` from
     `june/harm_willingness/` — does NOT shell out.
  2. Monkey-patches `FACETS_BY_ID["facet1_refusal"].n_per_sub_level` so we get
     N×4 base scenarios (default N=30 → 120 base → ×5 group expansion = 600
     candidates).
  3. Bridges the facet1 schema to the harm-refusal schema (drops the 4
     facet1 metrics, points judge_prompts at harm-refusal's anchor with the
     `harm_refusal_score` metric, renumbers ids to `harm_refusal_grown_NNNN`,
     adds `meta.source_facet`).
  4. Appends the bridged candidates to
     `<output_hunter>/eval-orthogonalized/harm-refusal/new.yaml` so lily picks
     them up.
  5. Calls `lily_stage3c_standalone.run_fidelity_check(threshold=35,
     drop_flagged=True)` in-process.
  6. Appends only the unflagged subset to
     `shared/evals_orthogonalized/harm-refusal/harm-refusal_eval.yaml` via
     `eval_utils.write_yaml_with_anchors` (preserving the &judge_prompts
     anchor).
  7. Loops up to --max-attempts writer rounds, capping cost.

CLI:
    python3 grow_harm_refusal_via_facets.py \
        --target 150 \
        --writer-model anthropic/claude-sonnet-4 \
        [--n-per-sub-level 30] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SHARED_EVALS_ORTHOG = REPO_ROOT / "shared" / "evals_orthogonalized"
OUTPUT_HUNTER = THIS_DIR / "output_hunter"
HARM_WILLINGNESS_DIR = REPO_ROOT / "june" / "harm_willingness"

# Make `from facet_specs import ...` and `from generate_battery import ...`
# resolve to june/harm_willingness/ without modifying that package.
sys.path.insert(0, str(HARM_WILLINGNESS_DIR))
sys.path.insert(0, str(THIS_DIR))

# harm_willingness imports
from facet_specs import FACETS_BY_ID  # noqa: E402
import generate_battery as gb  # noqa: E402
import group_substitute as gs  # noqa: E402

# Restrict group expansion to the unlabeled arm only — we don't want this
# audit grow to inject group-identified scenarios into the harm-refusal eval.
gs.GROUP_PHRASES = {"unlabeled": "a"}

# orthogonalize imports
from eval_utils import (  # noqa: E402
    canonical_yaml_name,
    judge_prompts_of,
    load_eval_yaml,
    write_yaml_with_anchors,
)
import lily_stage3c_standalone as lily  # noqa: E402

load_dotenv(find_dotenv(usecwd=True), override=True)

EVAL_NAME = "harm-refusal"
HARM_REFUSAL_METRIC = "harm_refusal_score"

CACHE_DIR = THIS_DIR / ".facet_grow_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Counting current unflagged + unmapped (mirrors grow_standalone.py)
# ---------------------------------------------------------------------------

def _count_unflagged(yaml_entries: List[dict], output_dir: Path) -> int:
    fid_csv = output_dir / "eval-orthogonalized" / EVAL_NAME / "fidelity_report.csv"
    unflagged_ids: set[str] = set()
    if fid_csv.exists():
        df = pd.read_csv(fid_csv)
        unflagged_ids = set(
            df.loc[df["flagged"] == False, "question_id"].astype(str)  # noqa: E712
        )
    count = 0
    for e in yaml_entries:
        if not isinstance(e, dict) or "id" not in e:
            continue
        qid = e["id"]
        is_suffixed = "_r1" in qid or "_r2" in qid or "_grow" in qid
        if not is_suffixed or qid in unflagged_ids:
            count += 1
    return count


def _next_grown_index(yaml_entries: List[dict]) -> int:
    pat = re.compile(r"_grown_(\d+)")
    max_n = -1
    for e in yaml_entries:
        if not isinstance(e, dict) or "id" not in e:
            continue
        m = pat.search(e["id"])
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


# ---------------------------------------------------------------------------
# Cached writer wrapper around generate_battery.generate_sub_level
# ---------------------------------------------------------------------------

def _cache_path(model: str, key: str) -> Path:
    h = hashlib.sha256(f"{model}\n{key}".encode()).hexdigest()[:24]
    return CACHE_DIR / f"{h}.json"


async def _cached_generate_facet(
    facet_id: str,
    writer_model: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """Call generate_battery.generate_facet with disk caching keyed on
    (model, facet_id, n_per_sub_level, seed). Cache stores the post-processed
    questions list (already in vibes_eval schema with facet1's 4 metrics).
    """
    facet = FACETS_BY_ID[facet_id]
    cache_key = json.dumps(
        {
            "facet_id": facet_id,
            "n_per_sub_level": facet.n_per_sub_level,
            "sub_levels": list(facet.sub_levels),
            "seed": seed,
        },
        sort_keys=True,
    )
    cp = _cache_path(writer_model, cache_key)
    if cp.exists():
        return json.loads(cp.read_text())

    client = gb._make_client()
    questions = await gb.generate_facet(client, facet, writer_model)
    cp.write_text(json.dumps(questions))
    return questions


# ---------------------------------------------------------------------------
# Group expansion (uses group_substitute.expand_question on each base entry)
# ---------------------------------------------------------------------------

def _expand_groups(base_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand each templated base question into 5 group versions (per
    GROUP_PHRASES in group_substitute.py). Skip any item whose paraphrase
    lacks the {group_phrase} placeholder (rare retry edge case)."""
    out: List[Dict[str, Any]] = []
    for q in base_questions:
        if "{group_phrase}" not in q["paraphrases"][0]:
            out.append(q)
            continue
        out.extend(gs.expand_question(q))
    return out


# ---------------------------------------------------------------------------
# Schema bridge: facet1_refusal entry -> harm-refusal entry
# ---------------------------------------------------------------------------

def _bridge_to_harm_refusal(
    facet_entry: Dict[str, Any],
    new_id: str,
    harm_refusal_judge_prompts: Dict[str, str],
) -> Dict[str, Any]:
    """Convert one facet1_refusal entry to a harm-refusal entry.

    - id -> harm_refusal_grown_NNNN
    - judge_prompts -> harm-refusal's {harm_refusal_score: ...} (drops the
      4 facet1 metrics)
    - meta keeps facet1's keys (expected_harm_willing/protective,
      base_scenario_id, facet, group, sub_level, notes, split) and adds
      `source_facet`.
    """
    meta = dict(facet_entry.get("meta", {}) or {})
    meta["source_facet"] = "harm-willingness/facet1_refusal"
    return {
        "id": new_id,
        "paraphrases": list(facet_entry["paraphrases"]),
        "samples_per_paraphrase": facet_entry.get("samples_per_paraphrase", 3),
        "temperature": facet_entry.get("temperature", 1.0),
        "judge_prompts": harm_refusal_judge_prompts,
        "judge_type": facet_entry.get("judge_type", "sampling"),
        "n_samples": facet_entry.get("n_samples", 5),
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Append-to-new.yaml helper (preserves existing entries)
# ---------------------------------------------------------------------------

def _append_to_new_yaml(
    new_entries: List[Dict[str, Any]],
    new_yaml_path: Path,
    judge_prompts: Dict[str, str],
) -> None:
    if new_yaml_path.exists():
        existing = [
            e for e in load_eval_yaml(new_yaml_path)
            if isinstance(e, dict) and "id" in e
        ]
    else:
        existing = []
    merged = existing + new_entries
    write_yaml_with_anchors(merged, judge_prompts, new_yaml_path)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

async def grow(
    target: int,
    writer_model: str,
    judge_model: str,
    n_per_sub_level: int,
    max_attempts: int,
    dry_run: bool,
) -> Dict[str, Any]:
    output_dir = OUTPUT_HUNTER
    eval_out = output_dir / "eval-orthogonalized" / EVAL_NAME
    eval_out.mkdir(parents=True, exist_ok=True)

    yaml_path = SHARED_EVALS_ORTHOG / EVAL_NAME / canonical_yaml_name(EVAL_NAME)
    if not yaml_path.exists():
        raise SystemExit(f"missing eval YAML: {yaml_path}")

    yaml_entries = load_eval_yaml(yaml_path)
    judge_prompts = judge_prompts_of(yaml_entries)
    if HARM_REFUSAL_METRIC not in judge_prompts:
        raise SystemExit(
            f"expected '{HARM_REFUSAL_METRIC}' in {yaml_path} judge_prompts; "
            f"found {list(judge_prompts)}"
        )

    cur_count = _count_unflagged(yaml_entries, output_dir)
    print(f"[{EVAL_NAME}] current unflagged+unmapped = {cur_count} (target {target})")

    # Monkey-patch the FacetSpec for this run (don't touch source).
    facet = FACETS_BY_ID["facet1_refusal"]
    original_n = facet.n_per_sub_level
    facet.n_per_sub_level = n_per_sub_level
    print(
        f"[{EVAL_NAME}] monkey-patched facet1_refusal.n_per_sub_level "
        f"{original_n} -> {n_per_sub_level} ({n_per_sub_level}×4 sub_levels = "
        f"{n_per_sub_level * 4} base scenarios per writer round, ×5 group "
        f"expansion = {n_per_sub_level * 4 * 5} candidates per round)"
    )

    if dry_run:
        # Show writer prompt header for one sub-level.
        sub_level = facet.sub_levels[0]
        prompt = gb.build_user_prompt(facet, sub_level, facet.n_per_sub_level)
        print("\n=== DRY-RUN WRITER PROMPT (facet1_refusal / sub_level=" + sub_level + ") ===")
        print(prompt[:1200])
        print(
            f"\n... (full length {len(prompt)} chars; "
            f"{len(facet.sub_levels)} such prompts per writer round)\n"
        )

        # Build one fake bridged example from the existing facet1 YAML so we
        # can demonstrate the schema swap without an API call.
        sample_facet_yaml = HARM_WILLINGNESS_DIR / "evals" / "facet1_refusal_eval.yaml"
        if sample_facet_yaml.exists():
            with open(sample_facet_yaml) as f:
                facet_entries = yaml.safe_load(f) or []
            if facet_entries:
                # Pretend this base scenario was just generated; expand groups.
                # The eval already contains expanded items, so we just use one.
                example = facet_entries[0]
                next_idx = _next_grown_index(yaml_entries)
                bridged = _bridge_to_harm_refusal(
                    example,
                    new_id=f"harm_refusal_grown_{next_idx:04d}",
                    harm_refusal_judge_prompts=judge_prompts,
                )
                # Print a redacted view (drop the verbose judge_prompts text).
                printable = dict(bridged)
                printable["judge_prompts"] = {
                    k: f"<{len(v)} chars>" for k, v in printable["judge_prompts"].items()
                }
                printable["paraphrases"] = [
                    p[:300] + ("..." if len(p) > 300 else "")
                    for p in printable["paraphrases"]
                ]
                print("=== ONE BRIDGED EXAMPLE (facet1 -> harm-refusal schema) ===")
                print(yaml.safe_dump(printable, sort_keys=False, width=100))
                print(
                    "Note: in dry-run we drew the example from the existing "
                    "facet1_refusal_eval.yaml (already group-expanded). At real "
                    "run time, base scenarios would come from generate_facet() "
                    "and then go through _expand_groups()."
                )
        # Restore.
        facet.n_per_sub_level = original_n
        return {
            "writer_calls": 0,
            "generated": 0,
            "kept": 0,
            "final": cur_count,
            "dry_run": True,
        }

    writer_calls = 0
    total_gen = 0
    total_kept = 0

    try:
        while writer_calls < max_attempts:
            yaml_entries = load_eval_yaml(yaml_path)
            cur_count = _count_unflagged(yaml_entries, output_dir)
            if cur_count >= target:
                print(f"[{EVAL_NAME}] reached target {target} (have {cur_count}); stopping.")
                break

            seed = writer_calls
            print(
                f"[{EVAL_NAME}] writer round {writer_calls + 1}/{max_attempts} "
                f"(seed={seed}, need {target - cur_count} more)"
            )

            base_questions = await _cached_generate_facet(
                "facet1_refusal", writer_model, seed
            )
            writer_calls += 1
            print(f"  produced {len(base_questions)} base scenarios")

            # Group-expand (×5 by default).
            expanded = _expand_groups(base_questions)
            print(f"  group-expanded to {len(expanded)} candidates")
            total_gen += len(expanded)

            # Bridge to harm-refusal schema with fresh ids.
            yaml_entries = load_eval_yaml(yaml_path)
            next_idx = _next_grown_index(yaml_entries)
            bridged: List[Dict[str, Any]] = []
            for fe in expanded:
                bridged.append(
                    _bridge_to_harm_refusal(
                        fe,
                        new_id=f"harm_refusal_grown_{next_idx:04d}",
                        harm_refusal_judge_prompts=judge_prompts,
                    )
                )
                next_idx += 1

            # Drop a snapshot of just-this-round candidates for recovery.
            facet_grown_path = eval_out / f"facet_grown_round_{writer_calls:03d}.yaml"
            write_yaml_with_anchors(bridged, judge_prompts, facet_grown_path)
            print(f"  wrote round snapshot: {facet_grown_path}")

            # Append to new.yaml so lily picks them up.
            new_yaml_path = eval_out / "new.yaml"
            _append_to_new_yaml(bridged, new_yaml_path, judge_prompts)
            print(f"  appended {len(bridged)} entries to {new_yaml_path}")

            # Run lily fidelity check (in-process).
            print(f"[{EVAL_NAME}] running lily fidelity check (threshold=35)...")
            df = await lily.run_fidelity_check(
                eval_name=EVAL_NAME,
                input_dir=SHARED_EVALS_ORTHOG,
                output_dir=output_dir,
                judge_model=judge_model,
                min_fidelity=35,
                drop_flagged=True,
                concurrency=32,
            )

            # Take the unflagged subset of THIS round's bridged ids.
            kept_this_round: List[Dict[str, Any]] = []
            if df is not None and not df.empty:
                this_round_ids = {b["id"] for b in bridged}
                kept_ids = set(
                    df.loc[
                        df["question_id"].isin(this_round_ids) & (~df["flagged"]),
                        "question_id",
                    ].astype(str)
                )
                kept_this_round = [b for b in bridged if b["id"] in kept_ids]
                total_kept += len(kept_this_round)
                print(f"  lily kept {len(kept_this_round)}/{len(bridged)} bridged candidates")

            # Append unflagged subset to canonical harm-refusal YAML.
            if kept_this_round:
                yaml_entries = load_eval_yaml(yaml_path)
                # Avoid re-adding ids that may already be present (cache reruns).
                existing_ids = {
                    e["id"] for e in yaml_entries
                    if isinstance(e, dict) and "id" in e
                }
                fresh = [k for k in kept_this_round if k["id"] not in existing_ids]
                merged = yaml_entries + fresh
                merged_qs = [e for e in merged if isinstance(e, dict) and "id" in e]
                write_yaml_with_anchors(merged_qs, judge_prompts, yaml_path)
                print(
                    f"  appended {len(fresh)} unflagged candidates to {yaml_path}"
                )

            yaml_entries = load_eval_yaml(yaml_path)
            cur_count = _count_unflagged(yaml_entries, output_dir)
            print(f"[{EVAL_NAME}] count after round: {cur_count}/{target}")
            if cur_count >= target:
                break
    finally:
        # Always restore the in-memory FacetSpec.
        facet.n_per_sub_level = original_n

    yaml_entries = load_eval_yaml(yaml_path)
    final = _count_unflagged(yaml_entries, output_dir)
    print(
        f"\n[{EVAL_NAME}] DONE: writer_calls={writer_calls} generated={total_gen} "
        f"kept={total_kept} final={final}/{target}"
    )
    return {
        "writer_calls": writer_calls,
        "generated": total_gen,
        "kept": total_kept,
        "final": final,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    summary = await grow(
        target=args.target,
        writer_model=args.writer_model,
        judge_model=args.judge_model,
        n_per_sub_level=args.n_per_sub_level,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--writer-model", default="anthropic/claude-sonnet-4")
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4")
    ap.add_argument("--n-per-sub-level", type=int, default=30)
    ap.add_argument(
        "--max-attempts", type=int, default=3,
        help="Max writer rounds before giving up (each round = one full "
             "generate_facet call across all 4 sub_levels).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print the writer prompt for facet1_refusal and one bridged "
             "example schema, then exit without calling the API.",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
