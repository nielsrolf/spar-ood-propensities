#!/usr/bin/env python3
"""Wrapper: run all analyze.py experiments + direction_similarity.py across multiple
layers for a given family (default: qwen8b at layers 18/27/31, matching the
extract_activations.py --layers run).

Per layer the script runs:
  • PCA biplot           (analyze.py --experiment pca)
  • LOO logistic probe   (analyze.py --experiment probe)
  • per-prompt probe     (analyze.py --experiment probe-perprompt)
  • direction-similarity heatmap (direction_similarity.py over the 9 paper traits)

Outputs land in figures/ with layer-tagged filenames so multi-layer runs don't overwrite.

Usage:
    python run_analysis.py                    # default: qwen8b, layers 18 27 31, all 4 steps
    python run_analysis.py --layers 31        # just layer 31
    python run_analysis.py --skip-direction-sim
    python run_analysis.py --experiments pca  # skip the slow probes
    python run_analysis.py --dry-run          # preview commands only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).parent

# 9 paper traits — kept as a named preset for ablation/comparison runs.
# Pass `--preset paper9` to use it instead of the full family list.
HEADLINE_TRAITS_QWEN = [
    "power_seeking_qwen_v1",
    "spitefulness_qwen_v1",
    "self_preservation_qwen_v1",
    "cooperation_qwen_v1",
    "neuroticism_qwen_v1",
    "honest_humble_qwen_v1",
    "claiming_superintelligence_qwen_v1",
    "harm_elaboration_qwen_v1",
    "harm_refusal_qwen_v1",
]

# 23 EFA traits — drops harm_refusal (in paper-9 but not in EFA-23).
# Pass `--preset efa23` to use it for the EFA cluster-mapping test.
EFA23_TRAITS_QWEN = [t for t in HEADLINE_TRAITS_QWEN if t != "harm_refusal_qwen_v1"]

PRESETS = {
    "paper9": HEADLINE_TRAITS_QWEN,
    "efa23": EFA23_TRAITS_QWEN,
}

DEFAULT_LAYERS = [18, 27, 31]
DEFAULT_FAMILY = "qwen8b"
DEFAULT_EXPERIMENTS = ["pca", "probe", "probe-perprompt"]


def load_family_traits(family: str) -> list[str]:
    """All SFT model names from models.yaml for the given family, excluding __base__.
    Used as the default dirsim trait list so the heatmap auto-tracks family contents."""
    with open(HERE / "models.yaml") as f:
        registry = yaml.safe_load(f)
    if family not in registry:
        sys.exit(f"error: family {family!r} not found in models.yaml")
    return [name for name in registry[family]["models"] if name != "__base__"]


def run_cmd(cmd: list[str], label: str) -> tuple[bool, float]:
    """Run subprocess synchronously, stream output to terminal, return (success, elapsed)."""
    print(f"\n{'=' * 72}")
    print(f"  [{label}]")
    print(f"  cmd:  {' '.join(cmd)}")
    print(f"{'=' * 72}")
    sys.stdout.flush()
    start = time.time()
    result = subprocess.run(cmd, cwd=str(HERE))
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "✓ OK" if ok else f"✗ FAIL exit={result.returncode}"
    print(f"  [{label}] {status}  ({elapsed:.1f}s)\n")
    return ok, elapsed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--family", default=DEFAULT_FAMILY,
                    help=f"models.yaml family (default: {DEFAULT_FAMILY})")
    ap.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS,
                    help=f"Layers to analyze (default: {DEFAULT_LAYERS})")
    ap.add_argument("--experiments", nargs="+",
                    default=DEFAULT_EXPERIMENTS,
                    choices=["pca", "probe", "probe-perprompt"],
                    help=f"analyze.py experiments to run (default: all 3)")
    ap.add_argument("--traits", nargs="+", default=None,
                    help="Explicit trait list for direction-similarity heatmap. "
                         "Overrides --preset.")
    ap.add_argument("--preset", default="all", choices=["all", "paper9", "efa23"],
                    help="Named trait list for dirsim. 'all' = every SFT in the family "
                         "from models.yaml (default, 24 for qwen8b); "
                         "'paper9' = the 9 headline paper traits; "
                         "'efa23' = paper9 minus harm_refusal (matches behavioral EFA scope).")
    ap.add_argument("--skip-direction-sim", action="store_true",
                    help="Skip the direction-similarity heatmap step")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview commands, don't run")
    args = ap.parse_args()

    # Resolve trait list: explicit --traits wins, else --preset, else all family SFTs.
    if args.skip_direction_sim:
        traits = None
    elif args.traits is not None:
        traits = args.traits
    elif args.preset == "all":
        traits = load_family_traits(args.family)
    else:
        traits = PRESETS[args.preset]

    n_steps = len(args.layers) * (len(args.experiments) + (0 if args.skip_direction_sim else 1))
    print(f"\nPlan: {n_steps} steps")
    print(f"  family:       {args.family}")
    print(f"  layers:       {args.layers}")
    print(f"  experiments:  {args.experiments}")
    print(f"  dir-sim:      {'skip' if args.skip_direction_sim else f'{len(traits)} traits'}")

    if args.dry_run:
        print("\n[dry-run] commands that would execute:")
        for layer in args.layers:
            save_dir = f"figures/l{layer}"
            for exp in args.experiments:
                print(f"  python analyze.py --family {args.family} --out-dir output/l{layer} "
                      f"--save-dir {save_dir} --experiment {exp}")
            if not args.skip_direction_sim:
                print(f"  python direction_similarity.py --pt-dir output/l{layer}/{args.family} "
                      f"--models {' '.join(traits)} "
                      f"--out figures/direction_similarity_{args.family}_l{layer}.png")
        return

    overall_start = time.time()
    results: list[tuple[str, bool, float]] = []

    for layer in args.layers:
        # Per-layer save dir for analyze.py outputs so layers don't overwrite
        # each other's pca_*.png / probe_*.png / probe_perprompt_*.png. The
        # previous wrapper didn't pass --save-dir, so analyze.py used its default
        # (figures/) and each layer clobbered the last one's figures.
        save_dir = HERE / "figures" / f"l{layer}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for exp in args.experiments:
            cmd = ["python", "analyze.py",
                   "--family", args.family,
                   "--out-dir", f"output/l{layer}",
                   "--save-dir", str(save_dir),
                   "--experiment", exp]
            results.append((f"l{layer}/{exp}", *run_cmd(cmd, f"l{layer}/{exp}")))

        if not args.skip_direction_sim:
            # Direction-similarity already uses a layer-tagged --out path; keep it
            # at the top level of figures/ for easy side-by-side comparison.
            cmd = ["python", "direction_similarity.py",
                   "--pt-dir", f"output/l{layer}/{args.family}",
                   "--models", *traits,
                   "--out", f"figures/direction_similarity_{args.family}_l{layer}.png"]
            results.append((f"l{layer}/dirsim", *run_cmd(cmd, f"l{layer}/dirsim")))

    total = time.time() - overall_start
    n_ok = sum(1 for _, ok, _ in results if ok)

    print(f"\n{'=' * 72}")
    print(f"  ALL DONE — {n_ok}/{len(results)} steps OK in {total:.1f}s ({total/60:.1f} min)")
    print(f"{'=' * 72}")
    for label, ok, elapsed in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {label:25s}  {elapsed:>6.1f}s")
    print(f"\nFigures saved to: {HERE / 'figures'}/")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
