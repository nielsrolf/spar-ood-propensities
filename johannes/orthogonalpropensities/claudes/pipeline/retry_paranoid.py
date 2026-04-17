"""Targeted re-run: user_trust -1 (Paranoid) with target_min_mag lowered 55→50.

Pilot main run produced only 2/50 for Paranoid because judge consistently scored
attempts at -50 (one tick below the -55 acceptance floor). This script retries
that single trait with target_min_mag=50, keeping ORTHOGONALITY_THRESHOLD fixed
at 20. Writes to conversations/user_trust__level-1__Paranoid.json (resumable).
"""
from __future__ import annotations
import asyncio

from . import config
from .generate_conversations import generate_for_level
from .spec import load_axes


async def main():
    axes = load_axes(config.INPUT_SPEC)
    ax = next(a for a in axes if a.axis == "user trust")
    lv = ax.minus  # Paranoid

    key = f"{ax.slug}__level{lv.level:+d}__{lv.name.replace(' ', '_')[:30]}"
    out_path = config.CONVERSATIONS_DIR / f"{key}.json"

    print(f"Retrying {ax.axis} / {lv.name} with target_min_mag=50 (was 55)")
    res = await generate_for_level(
        all_axes=axes,
        target_axis=ax,
        target_level=lv,
        n_target=config.CONVERSATIONS_PER_LEVEL,
        gen_provider=config.GENERATOR_PROVIDER,
        gen_model=config.GENERATOR_MODEL,
        judge_provider=config.JUDGE_PROVIDER,
        judge_model=config.JUDGE_MODEL,
        target_min_mag=50,  # relaxed from 55
        target_level0_max=config.TARGET_LEVEL0_MAX_MAGNITUDE,
        ortho_threshold=config.ORTHOGONALITY_THRESHOLD,
        max_retries=config.MAX_REGEN_ATTEMPTS,
        out_path=out_path,
        seed=99999,
        attempt_multiplier=4,
    )
    print(f"Done: {len(res['accepted'])}/{config.CONVERSATIONS_PER_LEVEL} accepted, "
          f"{res['rejections']} rejected")


if __name__ == "__main__":
    asyncio.run(main())
