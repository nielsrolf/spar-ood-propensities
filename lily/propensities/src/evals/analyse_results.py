"""
analyse_results.py

Loads eval results and produces summary statistics and plots.

Usage:
    python analyse_results.py --results_dir results/
    python analyse_results.py --results_dir results/ --compare base finetuned
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_metrics(results_dir: Path) -> dict:
    """Load all metrics.json files from results subdirectories."""
    metrics = {}
    for run_dir in sorted(results_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            metrics[run_dir.name] = data
    return metrics


def load_samples(run_dir: Path) -> list[dict]:
    samples = []
    samples_path = run_dir / "samples.jsonl"
    if samples_path.exists():
        with open(samples_path) as f:
            for line in f:
                samples.append(json.loads(line.strip()))
    return samples


def print_summary(metrics: dict):
    print("\nEVAL RESULTS SUMMARY")
    print("=" * 90)
    header = (f"{'Run':<35} {'Honesty':>9} {'Syco':>7} "
              f"{'Calibration':>13} {'Conf%':>7} {'Unc%':>7} {'Auto%':>7} {'Pushback':>10}")
    print(header)
    print("-" * 90)

    for run_name, m in metrics.items():
        pb = (f"{m['pushback_maintenance_rate']:.1%}"
              if m.get("pushback_maintenance_rate") else "N/A")
        short_name = run_name[:35]
        print(
            f"{short_name:<35} "
            f"{m['honesty_rate']:>9.1%} "
            f"{m['sycophancy_rate']:>7.1%} "
            f"{m.get('calibration_rate', 0):>13.1%} "
            f"{m.get('confidence_appropriate_rate', 0):>7.1%} "
            f"{m.get('uncertainty_acknowledged_rate', 0):>7.1%} "
            f"{m.get('autonomy_preserved_rate', 0):>7.1%} "
            f"{pb:>10}"
        )


def print_dimension_breakdown(metrics: dict, run_name: str):
    m = metrics.get(run_name)
    if not m:
        print(f"Run '{run_name}' not found")
        return

    print(f"\nDimension breakdown for: {run_name}")
    print("-" * 40)
    by_dim = m.get("by_dimension", {})
    for dim, stats in sorted(by_dim.items()):
        print(f"  {dim:<35} {stats['honesty_rate']:.1%} ({stats['total']} samples)")


def compare_runs(metrics: dict, run_a: str, run_b: str):
    a = metrics.get(run_a)
    b = metrics.get(run_b)

    if not a or not b:
        print(f"Could not find both runs: {run_a}, {run_b}")
        return

    print(f"\nCOMPARISON: {run_a} vs {run_b}")
    print("=" * 65)

    def delta(key):
        av, bv = a.get(key, 0), b.get(key, 0)
        diff = bv - av
        sign = "+" if diff >= 0 else ""
        return f"{av:.1%} -> {bv:.1%} ({sign}{diff:.1%})"

    print("Honesty metrics:")
    print(f"  Honesty rate:          {delta('honesty_rate')}")
    print(f"  Sycophancy rate:       {delta('sycophancy_rate')}")
    print(f"  Refusal rate:          {delta('refusal_rate')}")
    if a.get("pushback_maintenance_rate") and b.get("pushback_maintenance_rate"):
        print(f"  Pushback maintenance:  {delta('pushback_maintenance_rate')}")

    print("\nCalibration metrics (epistemic autonomy):")
    print(f"  Composite calibration:       {delta('calibration_rate')}")
    print(f"  Confidence appropriate:      {delta('confidence_appropriate_rate')}")
    print(f"  Uncertainty acknowledged:    {delta('uncertainty_acknowledged_rate')}")
    print(f"  Autonomy preserved:          {delta('autonomy_preserved_rate')}")

    # Clarity-level comparison
    cl_a = a.get("by_clarity", {})
    cl_b = b.get("by_clarity", {})
    all_clarity = sorted(set(cl_a.keys()) | set(cl_b.keys()))
    if all_clarity:
        print(f"\nBy clarity (honesty delta / calibration delta):")
        for label in all_clarity:
            hr_a = cl_a.get(label, {}).get("honesty_rate", 0)
            hr_b = cl_b.get(label, {}).get("honesty_rate", 0)
            cal_a = cl_a.get(label, {}).get("calibration_rate", 0)
            cal_b = cl_b.get(label, {}).get("calibration_rate", 0)
            h_diff = hr_b - hr_a
            c_diff = cal_b - cal_a
            h_sign = "+" if h_diff >= 0 else ""
            c_sign = "+" if c_diff >= 0 else ""
            print(f"  {label:<15} honesty {h_sign}{h_diff:.1%}  calibration {c_sign}{c_diff:.1%}")

    # Cost-level comparison
    cost_a = a.get("by_cost", {})
    cost_b = b.get("by_cost", {})
    all_cost = sorted(set(cost_a.keys()) | set(cost_b.keys()))
    if all_cost:
        print(f"\nBy cost (honesty delta / calibration delta):")
        for label in all_cost:
            hr_a = cost_a.get(label, {}).get("honesty_rate", 0)
            hr_b = cost_b.get(label, {}).get("honesty_rate", 0)
            cal_a = cost_a.get(label, {}).get("calibration_rate", 0)
            cal_b = cost_b.get(label, {}).get("calibration_rate", 0)
            h_diff = hr_b - hr_a
            c_diff = cal_b - cal_a
            h_sign = "+" if h_diff >= 0 else ""
            c_sign = "+" if c_diff >= 0 else ""
            print(f"  {label:<15} honesty {h_sign}{h_diff:.1%}  calibration {c_sign}{c_diff:.1%}")

    # Cross-tabulation comparison
    cross_a = a.get("by_clarity_x_cost", {})
    cross_b = b.get("by_clarity_x_cost", {})
    all_cross = sorted(set(cross_a.keys()) | set(cross_b.keys()))
    if all_cross:
        print(f"\nClarity x cost (key result):")
        for key in all_cross:
            hr_a = cross_a.get(key, {}).get("honesty_rate", 0)
            hr_b = cross_b.get(key, {}).get("honesty_rate", 0)
            h_diff = hr_b - hr_a
            h_sign = "+" if h_diff >= 0 else ""
            n = cross_b.get(key, {}).get("total", 0)
            print(f"  {key:<25} {h_sign}{h_diff:.1%}  (n={n})")


def error_analysis(run_dir: Path, top_n: int = 10):
    """Show the top N sycophantic responses for qualitative review."""
    samples = load_samples(run_dir)
    syco = [s for s in samples if s["label"] == "SYCOPHANTIC"]
    syco_sorted = sorted(syco, key=lambda x: x["confidence"])  # most clear-cut first

    print(f"\nERROR ANALYSIS: {run_dir.name}")
    print(f"Sycophantic responses: {len(syco)} of {len(samples)}")
    print("=" * 70)

    for i, s in enumerate(syco_sorted[:top_n]):
        print(f"\n[{i+1}] Confidence: {s['confidence']} | Dimension: {s['dimension']}")
        print(f"Prompt:    {s['prompt'][:120]}...")
        print(f"Response:  {s['model_response'][:120]}...")
        print(f"Reasoning: {s['reasoning']}")
        print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Compare two specific runs")
    parser.add_argument("--dimension_breakdown", metavar="RUN",
                        help="Show dimension breakdown for a specific run")
    parser.add_argument("--error_analysis", metavar="RUN_DIR",
                        help="Show sycophantic errors for qualitative review")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics = load_metrics(results_dir)

    if not metrics:
        print(f"No results found in {results_dir}")
        exit(1)

    print_summary(metrics)

    if args.compare:
        compare_runs(metrics, args.compare[0], args.compare[1])

    if args.dimension_breakdown:
        print_dimension_breakdown(metrics, args.dimension_breakdown)

    if args.error_analysis:
        error_analysis(Path(args.error_analysis), args.top_n)
