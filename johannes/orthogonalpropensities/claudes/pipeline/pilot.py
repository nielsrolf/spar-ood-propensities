"""Pilot run to calibrate threshold & model tier.

Generates ~8 conversations per (pilot_axis, pole) on 2 axes, scores them on
all 4 axes, and reports:
  - target-axis score distribution
  - max |other-axis score|
  - accept rates at thresholds ±10/±15/±20/±25/±30
"""
from __future__ import annotations
import asyncio
import json
import statistics
from pathlib import Path

from . import config, llm
from .generate_conversations import generate_one
from .spec import load_axes


async def run_pilot(
    pilot_axes_names: list[str],
    n_per_pole: int,
    gen_model: str,
    judge_model: str,
    *,
    target_min_mag: int = 55,
    target_level0_max: int = 15,
    # For the pilot itself, we don't filter — we just generate and score, then
    # measure distributions and report accept-rate at several thresholds.
    loose_threshold: int = 100,
) -> dict:
    all_axes = load_axes(config.INPUT_SPEC)
    axes_map = {a.axis: a for a in all_axes}
    pilot_axes = [axes_map[name] for name in pilot_axes_names if name in axes_map]

    print(f"PILOT: generating {n_per_pole} per pole × 2 poles × {len(pilot_axes)} axes = "
          f"{n_per_pole*2*len(pilot_axes)} conversations with gen={gen_model}, judge={judge_model}")

    out_rows: list[dict] = []

    for ax in pilot_axes:
        for target_level in (ax.plus, ax.minus):
            # Build combos: for pilot, just cycle markers × first couple of domains
            markers = list(target_level.behavioral_markers)
            domains = config.DOMAINS[:4]
            combos = [(m, d, i * 1000) for i, (m, d) in enumerate([(m, d) for m in markers for d in domains][:n_per_pole])]
            if len(combos) < n_per_pole:
                while len(combos) < n_per_pole:
                    m, d = markers[len(combos) % len(markers)], domains[len(combos) % len(domains)]
                    combos.append((m, d, 99999 + len(combos)))

            async def _one(i: int, c: tuple[str, str, int]):
                marker, domain, seed = c
                gen, bundle = await generate_one(
                    all_axes, ax, target_level, marker, domain, seed,
                    gen_provider=config.GENERATOR_PROVIDER, gen_model=gen_model,
                    judge_provider=config.JUDGE_PROVIDER, judge_model=judge_model,
                    target_min_mag=target_min_mag,
                    target_level0_max=target_level0_max,
                    ortho_threshold=loose_threshold,
                    max_retries=1,   # pilot: single attempt, measure raw distribution
                )
                return i, gen, bundle

            tasks = [_one(i, c) for i, c in enumerate(combos)]
            results = await asyncio.gather(*tasks)

            for i, gen, bundle in results:
                if gen is None or bundle is None:
                    print(f"  [{ax.axis}/{target_level.name}] SKIPPED (gen or judge failed) marker={combos[i][0][:60]}")
                    continue
                out_rows.append({
                    "axis": ax.axis,
                    "target_name": target_level.name,
                    "target_level": target_level.level,
                    "marker": combos[i][0],
                    "domain": combos[i][1],
                    "target_score": bundle.target_score,
                    "other_scores": bundle.other_scores,
                    "user": gen.user,
                    "assistant": gen.assistant,
                })
                print(f"  [{ax.axis}/{target_level.name}] target={bundle.target_score:+4d}   others={bundle.other_scores}")

    # Analyze
    def accept_rate(rows, thr: int, target_min_mag: int) -> float:
        if not rows:
            return 0.0
        kept = 0
        for r in rows:
            # target ok?
            lvl = r["target_level"]
            ts = r["target_score"]
            if lvl == 0:
                t_ok = abs(ts) <= target_level0_max
            else:
                expected_sign = 1 if lvl > 0 else -1
                t_ok = ts * expected_sign >= target_min_mag
            if not t_ok:
                continue
            # all others within threshold?
            ortho_ok = all(abs(s) <= thr for s in r["other_scores"].values())
            if ortho_ok:
                kept += 1
        return kept / len(rows)

    thresholds = [10, 15, 20, 25, 30, 40]
    accept_rates = {thr: accept_rate(out_rows, thr, target_min_mag) for thr in thresholds}

    target_scores = [r["target_score"] for r in out_rows]
    max_other = [max(abs(s) for s in r["other_scores"].values()) for r in out_rows]

    summary = {
        "pilot_axes": pilot_axes_names,
        "gen_model": gen_model,
        "judge_model": judge_model,
        "n_generated": len(out_rows),
        "target_score_stats": {
            "mean": statistics.mean(target_scores) if target_scores else 0,
            "median": statistics.median(target_scores) if target_scores else 0,
            "stdev": statistics.stdev(target_scores) if len(target_scores) > 1 else 0,
            "min": min(target_scores) if target_scores else 0,
            "max": max(target_scores) if target_scores else 0,
        },
        "max_other_axis_abs_score_stats": {
            "mean": statistics.mean(max_other) if max_other else 0,
            "median": statistics.median(max_other) if max_other else 0,
            "max": max(max_other) if max_other else 0,
            "p75": sorted(max_other)[int(0.75 * len(max_other))] if max_other else 0,
            "p90": sorted(max_other)[int(0.90 * len(max_other))] if max_other else 0,
        },
        "accept_rate_at_threshold": accept_rates,
        "rows": out_rows,
    }
    return summary


if __name__ == "__main__":
    out_path = config.CLAUDES_DIR / "pipeline" / "pilot_results.json"
    summary = asyncio.run(run_pilot(
        pilot_axes_names=config.PILOT_AXES,
        n_per_pole=config.PILOT_CONVERSATIONS_PER_LEVEL,
        gen_model=config.GENERATOR_MODEL,
        judge_model=config.JUDGE_MODEL,
    ))
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== PILOT SUMMARY ===")
    print(f"n_generated: {summary['n_generated']}")
    print(f"target score: mean={summary['target_score_stats']['mean']:.1f}  "
          f"min={summary['target_score_stats']['min']}  max={summary['target_score_stats']['max']}")
    print(f"max other |score|: mean={summary['max_other_axis_abs_score_stats']['mean']:.1f}  "
          f"p75={summary['max_other_axis_abs_score_stats']['p75']}  "
          f"p90={summary['max_other_axis_abs_score_stats']['p90']}  "
          f"max={summary['max_other_axis_abs_score_stats']['max']}")
    print("accept rate at threshold (target-min-mag=55, level0-max=15):")
    for thr, rate in summary["accept_rate_at_threshold"].items():
        print(f"  ±{thr:>3}: {rate*100:.1f}%")
    print(f"\nFull details: {out_path}")
