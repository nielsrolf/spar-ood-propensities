#!/usr/bin/env python3
"""
Standalone audit pipeline for the honesty-humility (dark) eval.

The honesty-humility eval lives outside niels' EvalConfig, so this script
bypasses bridge.py and loads data directly from the notebook's CSV output.

Usage:
    python june/vibes_audit/run_honesty_humility_audit.py
    python june/vibes_audit/run_honesty_humility_audit.py --steps sample,run-judges,analyze
    python june/vibes_audit/run_honesty_humility_audit.py --steps annotate
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from .audit_config import from_yaml
    from .sample_for_review import load_data, stratified_sample
    from .run_alt_judges import run_judges
    from .analyze import inter_judge_correlations, bias_probes, audit_summary
except ImportError:
    from audit_config import from_yaml
    from sample_for_review import load_data, stratified_sample
    from run_alt_judges import run_judges
    from analyze import inter_judge_correlations, bias_probes, audit_summary

CONFIG_PATH = Path(__file__).parent / "configs" / "honesty_humility.yaml"
OUTPUT_DIR = Path(__file__).parent / "output" / "honesty_humility"

ALL_STEPS = ["sample", "run-judges", "annotate", "analyze"]


def load_honesty_humility_results(config) -> pd.DataFrame:
    """Load honesty-humility results CSV, renaming 'answer' -> 'response'."""
    data_path = config.data_path
    if not data_path.exists():
        # Try fallback to results.csv (without alignment columns)
        fallback = data_path.parent / "results.csv"
        if fallback.exists():
            data_path = fallback
            print(f"  Using fallback: {fallback}")
        else:
            print(f"ERROR: No results CSV found at {data_path}")
            print("Run the honesty_humility_analysis.ipynb notebook first.")
            sys.exit(1)

    df = pd.read_csv(data_path, low_memory=False)

    # Rename answer -> response for audit framework compatibility
    if "answer" in df.columns and "response" not in df.columns:
        df = df.rename(columns={"answer": "response"})

    print(f"  Loaded {len(df)} rows from {data_path.name}")
    print(f"  Groups: {df['group'].unique().tolist()}")
    print(f"  Score range: {df[config.score_column].min():.0f} - {df[config.score_column].max():.0f}")
    return df


def step_sample(config, target_n: int = 200):
    """Load results, stratified sample, save full + blind CSVs."""
    print("\n" + "=" * 70)
    print("SAMPLE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_honesty_humility_results(config)

    # Write consolidated CSV for audit_config to use
    consolidated = OUTPUT_DIR / "all_results.csv"
    df.to_csv(consolidated, index=False)

    config_with_data = from_yaml(CONFIG_PATH, data_path=str(consolidated), output_dir=str(OUTPUT_DIR))
    config_with_data.target_n = min(target_n, len(df))

    df_loaded = load_data(config_with_data)
    sample = stratified_sample(df_loaded, config_with_data)

    print(f"  Sampled: {len(sample)} rows")

    # Save full sample
    full_path = OUTPUT_DIR / f"sample_{len(sample)}.csv"
    sample.to_csv(full_path, index=False)
    print(f"  Saved: {full_path}")

    # Save blind sample (no scores)
    blind_cols = ["question", "response"] + [
        c for c in config.metadata_columns if c in sample.columns
    ]
    blind = sample[blind_cols].copy()
    blind["human_label"] = ""
    blind_path = OUTPUT_DIR / f"sample_{len(sample)}_blind.csv"
    blind.to_csv(blind_path, index=False)
    print(f"  Saved: {blind_path}")


def step_run_judges(config):
    """Run alternative judges on the sample."""
    print("\n" + "=" * 70)
    print("RUN JUDGES")
    print("=" * 70)

    from dotenv import load_dotenv
    load_dotenv()

    config = from_yaml(CONFIG_PATH, output_dir=str(OUTPUT_DIR))

    # Find sample CSV
    sample_csvs = sorted(OUTPUT_DIR.glob("sample_*.csv"))
    sample_csvs = [c for c in sample_csvs if "_blind" not in c.name]
    if not sample_csvs:
        print("  No sample CSV found, run 'sample' step first")
        return

    sample_path = sample_csvs[-1]
    df = pd.read_csv(sample_path, low_memory=False)
    print(f"  {len(df)} rows from {sample_path.name}")

    result = run_judges(df, config)

    alt_path = OUTPUT_DIR / "alt_judge_scores.csv"
    result.to_csv(alt_path, index=False)
    print(f"  Saved: {alt_path}")

    # Quick correlation check
    score_cols = [c for c in result.columns if c.endswith("_score") and c != config.score_column]
    for col in score_cols:
        valid = result[col].notna() & result[config.score_column].notna()
        if valid.sum() > 0:
            corr = result.loc[valid, col].corr(result.loc[valid, config.score_column])
            print(f"  Correlation {config.score_column} vs {col}: {corr:.3f}")


def step_annotate(config):
    """Launch annotation GUI."""
    try:
        from .annotate import build_html, find_blind_csv, save_path, load_data as load_annotation_data
        from .annotate import Handler, PORT
    except ImportError:
        from annotate import build_html, find_blind_csv, save_path, load_data as load_annotation_data
        from annotate import Handler, PORT
    from http.server import HTTPServer
    import webbrowser

    config = from_yaml(CONFIG_PATH, output_dir=str(OUTPUT_DIR))

    blind_path = find_blind_csv(config)
    ann_path = save_path(config)
    rows, annotations = load_annotation_data(blind_path, ann_path)
    html = build_html(config)

    Handler.rows = rows
    Handler.annotations = {str(k): v for k, v in annotations.items()}
    Handler.config = config
    Handler._save_path = ann_path
    Handler._html = html

    n_done = sum(1 for v in annotations.values() if v)
    print(f"\nLoaded {len(rows)} rows, {n_done} already annotated")
    print(f"Config: {config.display_name}")
    print(f"Starting server at http://localhost:{PORT}")

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    webbrowser.open(f"http://localhost:{PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        n_done = sum(1 for v in Handler.annotations.values() if v)
        print(f"\nSaved {n_done} annotations to {ann_path}")
        server.server_close()


def step_analyze(config):
    """Run analysis and produce summary."""
    print("\n" + "=" * 70)
    print("ANALYZE")
    print("=" * 70)

    config = from_yaml(CONFIG_PATH, output_dir=str(OUTPUT_DIR))

    alt_path = OUTPUT_DIR / "alt_judge_scores.csv"
    if not alt_path.exists():
        print("  No alt_judge_scores.csv, run 'run-judges' step first")
        return

    alt_df = pd.read_csv(alt_path, low_memory=False)
    print(f"  {len(alt_df)} rows")

    # Human annotations
    ann_path = OUTPUT_DIR / "human_annotations.csv"
    human_df = pd.read_csv(ann_path) if ann_path.exists() else pd.DataFrame()

    summary = audit_summary(config, human_df, alt_df)

    summary_path = OUTPUT_DIR / "audit_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    print(f"\n  Honesty-Humility Audit Summary:")
    for _, row in summary.iterrows():
        icon = {"PASS": "+", "MARGINAL": "~", "FAIL": "!"}.get(row["Status"], "?")
        print(f"    [{icon}] {row['Metric']}: {row['Value']} (threshold {row['Threshold']})")

    # Bias probes
    group_cols = [c for c in config.metadata_columns if c in alt_df.columns]
    probes = bias_probes(alt_df, config.score_column, group_cols)
    if probes:
        print(f"\n  Bias probes:")
        for probe_name, result in probes.items():
            if "r" in result:
                print(f"    {probe_name}: r={result['r']:.3f}, p={result['p']:.4f}")
            elif "F" in result:
                print(f"    {probe_name}: F={result['F']:.2f}, p={result['p']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run honesty-humility audit pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available steps: {', '.join(ALL_STEPS)}",
    )
    parser.add_argument("--steps", default=",".join(ALL_STEPS),
                        help=f"Comma-separated steps (default: all)")
    parser.add_argument("--target-n", type=int, default=100,
                        help="Target sample size (default: 100)")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",")]
    config = from_yaml(CONFIG_PATH, output_dir=str(OUTPUT_DIR))

    print(f"Honesty-Humility Audit")
    print(f"Steps: {steps}")
    print(f"Output: {OUTPUT_DIR}")

    for step in steps:
        if step == "sample":
            step_sample(config, target_n=args.target_n)
        elif step == "run-judges":
            step_run_judges(config)
        elif step == "annotate":
            step_annotate(config)
        elif step == "analyze":
            step_analyze(config)
        else:
            print(f"Unknown step: {step}")


if __name__ == "__main__":
    main()
