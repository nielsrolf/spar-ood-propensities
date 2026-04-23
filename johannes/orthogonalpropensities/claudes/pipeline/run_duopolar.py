"""Entry point for the duopolar pipeline.

Stages:
  1. Conversations: for each axis generate DUOPOLAR_PAIRS_PER_AXIS accepted
     (user, plus_response, minus_response) pairs, judged together.
     Saved to duopolar_conversations/<axis_slug>.json. Resumable.
  2. Evals: emit duopolar_evals/<axis_slug>/ with questions.json (each
     question has both expected_plus_response and expected_minus_response),
     questions_eval.yaml, generate_questions.py, run_eval.py.

Usage:
  python -m pipeline.run_duopolar
  python -m pipeline.run_duopolar --skip-conversations
  python -m pipeline.run_duopolar --skip-evals
  python -m pipeline.run_duopolar --axes criticism effort
"""
from __future__ import annotations
import argparse
import asyncio
import json
import time
from pathlib import Path

from . import config
from .generate_conversations_duopolar import generate_for_axis_duopolar
from .spec import load_axes


async def run_all_pairs(axes_filter: list[str] | None = None) -> dict:
    all_axes = load_axes(config.INPUT_SPEC)
    if axes_filter:
        all_axes = [a for a in all_axes if a.axis in axes_filter]

    results: dict[str, dict] = {}
    t_total = time.time()

    for ax in all_axes:
        t0 = time.time()
        out_path = config.DUOPOLAR_CONVERSATIONS_DIR / f"{ax.slug}.json"
        print(f"\n====== {ax.axis} ======")
        res = await generate_for_axis_duopolar(
            all_axes=load_axes(config.INPUT_SPEC),  # full axis list for orthogonality scoring
            target_axis=ax,
            n_target=config.DUOPOLAR_PAIRS_PER_AXIS,
            gen_provider=config.GENERATOR_PROVIDER,
            gen_model=config.GENERATOR_MODEL,
            judge_provider=config.JUDGE_PROVIDER,
            judge_model=config.JUDGE_MODEL,
            target_min_mag=config.TARGET_AXIS_MIN_MAGNITUDE,
            ortho_threshold=config.ORTHOGONALITY_THRESHOLD,
            max_retries=config.MAX_REGEN_ATTEMPTS,
            out_path=out_path,
            seed=abs(hash(ax.axis)) % 100000,
            attempt_multiplier=3,
        )
        wall = time.time() - t0
        print(
            f"  Done: {len(res['accepted'])}/{config.DUOPOLAR_PAIRS_PER_AXIS} accepted, "
            f"{res['rejections']} rejected, wall={wall:.1f}s"
        )
        results[ax.axis] = {
            "accepted": len(res["accepted"]),
            "target": config.DUOPOLAR_PAIRS_PER_AXIS,
            "rejected": res["rejections"],
            "wall_seconds": wall,
        }

    total_wall = time.time() - t_total
    results["__total_wall_seconds__"] = total_wall
    config.DUOPOLAR_CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DUOPOLAR_CONVERSATIONS_DIR / "_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-conversations", action="store_true")
    parser.add_argument("--skip-evals", action="store_true")
    parser.add_argument("--axes", nargs="+", default=None,
                        help="Restrict to these axis names (e.g. --axes criticism effort)")
    args = parser.parse_args()

    if not args.skip_conversations:
        asyncio.run(run_all_pairs(args.axes))

    if not args.skip_evals:
        from .generate_evals import generate_all_duopolar_evals
        asyncio.run(generate_all_duopolar_evals())

    print("\nDone.")


if __name__ == "__main__":
    main()
