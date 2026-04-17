"""Single entry point.

Stages:
  1. Conversations: for each (axis, level) in list_some3.json, generate
     up to CONVERSATIONS_PER_LEVEL accepted conversations. Resumable.
  2. Evals: for each axis, emit an OLD_EVALS-style folder under
     claudes/evals/<axis_slug>/ with README, generate_questions.py,
     questions.json, questions_eval.yaml, run_eval.py.
  3. Append final stats to report.md.

Usage:
  python -m pipeline.run [--skip-conversations] [--skip-evals]
"""
from __future__ import annotations
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from . import config
from .generate_conversations import generate_for_level
from .spec import load_axes


async def run_all_conversations() -> dict:
    axes = load_axes(config.INPUT_SPEC)
    results: dict[str, dict] = {}
    t_total = time.time()

    for ax in axes:
        for lv in ax.all_levels():
            t0 = time.time()
            key = f"{ax.slug}__level{lv.level:+d}__{lv.name.replace(' ', '_')[:30]}"
            out_path = config.CONVERSATIONS_DIR / f"{key}.json"
            print(f"\n====== {ax.axis} / {lv.name} (level {lv.level:+d}) ======")
            res = await generate_for_level(
                all_axes=axes,
                target_axis=ax,
                target_level=lv,
                n_target=config.CONVERSATIONS_PER_LEVEL,
                gen_provider=config.GENERATOR_PROVIDER,
                gen_model=config.GENERATOR_MODEL,
                judge_provider=config.JUDGE_PROVIDER,
                judge_model=config.JUDGE_MODEL,
                target_min_mag=config.TARGET_AXIS_MIN_MAGNITUDE,
                target_level0_max=config.TARGET_LEVEL0_MAX_MAGNITUDE,
                ortho_threshold=config.ORTHOGONALITY_THRESHOLD,
                max_retries=config.MAX_REGEN_ATTEMPTS,
                out_path=out_path,
                seed=abs(hash(key)) % 100000,
                attempt_multiplier=3,
            )
            wall = time.time() - t0
            print(f"  Done: {len(res['accepted'])}/{config.CONVERSATIONS_PER_LEVEL} accepted, "
                  f"{res['rejections']} rejected, wall={wall:.1f}s")
            results[key] = {
                "axis": ax.axis,
                "level": lv.level,
                "level_name": lv.name,
                "accepted": len(res["accepted"]),
                "target": config.CONVERSATIONS_PER_LEVEL,
                "rejected": res["rejections"],
                "wall_seconds": wall,
            }
    total_wall = time.time() - t_total
    results["__total_wall_seconds__"] = total_wall
    (config.CONVERSATIONS_DIR / "_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    return results


def append_final_stats(conversation_stats: dict | None) -> None:
    """Append final stats + on/off-axis score means to report.md."""
    axes = load_axes(config.INPUT_SPEC)
    lines = ["\n---\n\n## 5. Final stats (full run)\n"]

    if not conversation_stats:
        lines.append("*(conversations were skipped in this run)*\n")
        _append(lines)
        return

    total_wall = conversation_stats.pop("__total_wall_seconds__", None)
    total_accepted = 0
    total_rejected = 0
    rows = []
    for key, s in conversation_stats.items():
        total_accepted += s["accepted"]
        total_rejected += s["rejected"]
        rows.append(s)

    lines.append(f"Total wall time: **{total_wall/60:.1f} minutes**\n")
    lines.append(f"Total accepted: **{total_accepted}** / target "
                 f"{len(rows) * config.CONVERSATIONS_PER_LEVEL}\n")
    lines.append(f"Total rejected: {total_rejected}\n\n")

    lines.append("| Axis | Level | Accepted / Target | Rejected | Mean target score | Mean max-other \\|score\\| |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for ax in axes:
        for lv in ax.all_levels():
            # find matching key
            key = f"{ax.slug}__level{lv.level:+d}__{lv.name.replace(' ', '_')[:30]}"
            out_path = config.CONVERSATIONS_DIR / f"{key}.json"
            if not out_path.exists():
                continue
            blob = json.loads(out_path.read_text())
            convs = blob.get("conversations", [])
            if convs:
                target_scores = [c["scores"]["target_score"] for c in convs]
                max_others = [
                    max(abs(s) for s in c["scores"]["other_scores"].values())
                    for c in convs
                ]
                ts_mean = f"{statistics.mean(target_scores):+.1f}"
                mo_mean = f"{statistics.mean(max_others):.1f}"
            else:
                ts_mean = "—"
                mo_mean = "—"
            lines.append(f"| {ax.axis} | {lv.name} ({lv.level:+d}) | "
                         f"{len(convs)} / {config.CONVERSATIONS_PER_LEVEL} | "
                         f"{blob.get('rejected_count', 0)} | {ts_mean} | {mo_mean} |\n")

    lines.append("\n")
    _append(lines)


def _append(lines: list[str]) -> None:
    cur = config.REPORT_PATH.read_text() if config.REPORT_PATH.exists() else ""
    config.REPORT_PATH.write_text(cur + "".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-conversations", action="store_true")
    parser.add_argument("--skip-evals", action="store_true")
    args = parser.parse_args()

    conv_stats = None
    if not args.skip_conversations:
        conv_stats = asyncio.run(run_all_conversations())

    if not args.skip_evals:
        from .generate_evals import generate_all_evals
        asyncio.run(generate_all_evals())

    append_final_stats(conv_stats)
    print("\nDone. Report at:", config.REPORT_PATH)


if __name__ == "__main__":
    main()
