"""
grow_harm_elaboration_via_facets.py — Grow harm-elaboration candidates by
reusing the hand-tuned harm-willingness facet5a/5b/5c pipelines.

Existing harm-elaboration questions are already sourced from facet5a_sentencing.
The harm_elaboration_score metric (recommending/intensifying/elaborating
punitive consequences) cleanly matches all three "consequence-recommendation"
facets in june/harm_willingness/:
  - facet5a_sentencing       (2 sub_levels: moderate, high)
  - facet5b_disciplinary     (3 sub_levels: school, workplace, corporal)
  - facet5c_third_party_punishment (2 sub_levels: moderate, high)

This is the harm-elaboration analogue of grow_harm_refusal_via_facets.py:
imports facet generators in-process, monkey-patches n_per_sub_level, restricts
group expansion to "unlabeled" only, bridges to harm-elaboration's single
harm_elaboration_score metric, runs lily fidelity in-process, appends only
unflagged candidates.

CLI:
    python3 grow_harm_elaboration_via_facets.py \
        --target 150 \
        --writer-model anthropic/claude-sonnet-4 \
        [--n-per-sub-level 15] [--max-attempts 2] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SHARED_EVALS_ORTHOG = REPO_ROOT / "shared" / "evals_orthogonalized"
OUTPUT_HUNTER = THIS_DIR / "output_hunter"
HARM_WILLINGNESS_DIR = REPO_ROOT / "june" / "harm_willingness"

sys.path.insert(0, str(HARM_WILLINGNESS_DIR))
sys.path.insert(0, str(THIS_DIR))

from facet_specs import FACETS_BY_ID  # noqa: E402
import generate_battery as gb  # noqa: E402
import group_substitute as gs  # noqa: E402

# Restrict group expansion to "unlabeled" only — same as harm-refusal grow.
gs.GROUP_PHRASES = {"unlabeled": "a"}

from eval_utils import (  # noqa: E402
    canonical_yaml_name,
    judge_prompts_of,
    load_eval_yaml,
    write_yaml_with_anchors,
)
import lily_stage3c_standalone as lily  # noqa: E402

load_dotenv(find_dotenv(usecwd=True), override=True)

EVAL_NAME = "harm-elaboration"
HARM_ELAB_METRIC = "harm_elaboration_score"
SOURCE_FACETS = ["facet5a_sentencing", "facet5b_disciplinary", "facet5c_third_party_punishment"]

CACHE_DIR = THIS_DIR / ".facet_grow_cache_elab"
CACHE_DIR.mkdir(exist_ok=True)


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


def _cache_path(model: str, key: str) -> Path:
    h = hashlib.sha256(f"{model}\n{key}".encode()).hexdigest()[:24]
    return CACHE_DIR / f"{h}.json"


async def _cached_generate_facet(
    facet_id: str,
    writer_model: str,
    seed: int,
) -> List[Dict[str, Any]]:
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


def _expand_groups(base_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for q in base_questions:
        if "{group_phrase}" not in q["paraphrases"][0]:
            out.append(q)
            continue
        out.extend(gs.expand_question(q))
    return out


def _bridge_to_harm_elaboration(
    facet_entry: Dict[str, Any],
    new_id: str,
    harm_elab_judge_prompts: Dict[str, str],
    source_facet: str,
) -> Dict[str, Any]:
    """Convert a facet5a/5b/5c entry to a harm-elaboration entry.

    Drops the facet's native rubrics; points judge_prompts at harm-elaboration's
    single harm_elaboration_score (sampling, n_samples=5).
    """
    meta = dict(facet_entry.get("meta", {}) or {})
    meta["source_facet"] = f"harm-willingness/{source_facet}"
    return {
        "id": new_id,
        "paraphrases": list(facet_entry["paraphrases"]),
        "samples_per_paraphrase": facet_entry.get("samples_per_paraphrase", 3),
        "temperature": facet_entry.get("temperature", 1.0),
        "judge_prompts": harm_elab_judge_prompts,
        "judge_type": facet_entry.get("judge_type", "sampling"),
        "n_samples": facet_entry.get("n_samples", 5),
        "meta": meta,
    }


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
    if HARM_ELAB_METRIC not in judge_prompts:
        raise SystemExit(
            f"expected '{HARM_ELAB_METRIC}' in {yaml_path} judge_prompts; "
            f"found {list(judge_prompts)}"
        )

    cur_count = _count_unflagged(yaml_entries, output_dir)
    print(f"[{EVAL_NAME}] current unflagged+unmapped = {cur_count} (target {target})")

    # Monkey-patch n_per_sub_level on each source facet for this run.
    original_n: Dict[str, int] = {}
    for fid in SOURCE_FACETS:
        facet = FACETS_BY_ID[fid]
        original_n[fid] = facet.n_per_sub_level
        facet.n_per_sub_level = n_per_sub_level
        print(
            f"[{EVAL_NAME}] {fid}.n_per_sub_level {original_n[fid]} -> "
            f"{n_per_sub_level} ({n_per_sub_level}×{len(facet.sub_levels)} = "
            f"{n_per_sub_level * len(facet.sub_levels)} per writer round)"
        )

    if dry_run:
        for fid in SOURCE_FACETS:
            facet = FACETS_BY_ID[fid]
            sub_level = facet.sub_levels[0]
            prompt = gb.build_user_prompt(facet, sub_level, facet.n_per_sub_level)
            print(f"\n=== DRY-RUN PROMPT ({fid} / sub_level={sub_level}) ===")
            print(prompt[:800])
            print(f"\n... ({len(prompt)} chars; {len(facet.sub_levels)} sub_levels)\n")
        for fid, n in original_n.items():
            FACETS_BY_ID[fid].n_per_sub_level = n
        return {"writer_calls": 0, "generated": 0, "kept": 0, "final": cur_count, "dry_run": True}

    writer_calls = 0
    total_gen = 0
    total_kept = 0

    try:
        for attempt in range(max_attempts):
            yaml_entries = load_eval_yaml(yaml_path)
            cur_count = _count_unflagged(yaml_entries, output_dir)
            if cur_count >= target:
                print(f"[{EVAL_NAME}] reached target {target} (have {cur_count}); stopping.")
                break

            print(
                f"\n[{EVAL_NAME}] writer round {attempt + 1}/{max_attempts} "
                f"(need {target - cur_count} more)"
            )

            # Generate from all 3 source facets in this round.
            all_bridged: List[Dict[str, Any]] = []
            yaml_entries = load_eval_yaml(yaml_path)
            next_idx = _next_grown_index(yaml_entries)

            for fid in SOURCE_FACETS:
                base_questions = await _cached_generate_facet(
                    fid, writer_model, seed=attempt
                )
                writer_calls += 1
                expanded = _expand_groups(base_questions)
                total_gen += len(expanded)
                print(f"  [{fid}] {len(base_questions)} base -> {len(expanded)} expanded")

                for fe in expanded:
                    all_bridged.append(
                        _bridge_to_harm_elaboration(
                            fe,
                            new_id=f"harm_elaboration_grown_{next_idx:04d}",
                            harm_elab_judge_prompts=judge_prompts,
                            source_facet=fid,
                        )
                    )
                    next_idx += 1

            # Snapshot of this round.
            facet_grown_path = eval_out / f"facet_grown_round_{attempt + 1:03d}.yaml"
            write_yaml_with_anchors(all_bridged, judge_prompts, facet_grown_path)
            print(f"  wrote round snapshot: {facet_grown_path} ({len(all_bridged)})")

            # Append to new.yaml so lily picks them up.
            new_yaml_path = eval_out / "new.yaml"
            _append_to_new_yaml(all_bridged, new_yaml_path, judge_prompts)
            print(f"  appended {len(all_bridged)} entries to {new_yaml_path}")

            # Run lily fidelity (in-process).
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

            kept_this_round: List[Dict[str, Any]] = []
            if df is not None and not df.empty:
                this_round_ids = {b["id"] for b in all_bridged}
                kept_ids = set(
                    df.loc[
                        df["question_id"].isin(this_round_ids) & (~df["flagged"]),
                        "question_id",
                    ].astype(str)
                )
                kept_this_round = [b for b in all_bridged if b["id"] in kept_ids]
                total_kept += len(kept_this_round)
                print(f"  lily kept {len(kept_this_round)}/{len(all_bridged)}")

            if kept_this_round:
                yaml_entries = load_eval_yaml(yaml_path)
                existing_ids = {
                    e["id"] for e in yaml_entries
                    if isinstance(e, dict) and "id" in e
                }
                fresh = [k for k in kept_this_round if k["id"] not in existing_ids]
                merged = yaml_entries + fresh
                merged_qs = [e for e in merged if isinstance(e, dict) and "id" in e]
                write_yaml_with_anchors(merged_qs, judge_prompts, yaml_path)
                print(f"  appended {len(fresh)} unflagged candidates to {yaml_path}")

            yaml_entries = load_eval_yaml(yaml_path)
            cur_count = _count_unflagged(yaml_entries, output_dir)
            print(f"[{EVAL_NAME}] count after round: {cur_count}/{target}")
            if cur_count >= target:
                break
    finally:
        for fid, n in original_n.items():
            FACETS_BY_ID[fid].n_per_sub_level = n

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
    ap.add_argument("--n-per-sub-level", type=int, default=15)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
