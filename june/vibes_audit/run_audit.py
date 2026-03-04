#!/usr/bin/env python3
"""
Orchestrate the full propensity audit pipeline.

Usage (from repo root):
    python june/vibes_audit/run_audit.py --eval risk_affinity
    python june/vibes_audit/run_audit.py --eval risk_affinity --steps sample,annotate
    python june/vibes_audit/run_audit.py --steps discover,generate-configs,sample,run-judges,analyze

Steps:
    discover         — find and summarize available results
    generate-configs — auto-generate audit YAML configs from EvalConfig
    sample           — stratified sampling per eval
    run-judges       — run alternative LLM judges on samples
    annotate         — launch annotation GUI for human review
    analyze          — compute agreement metrics, produce summary
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from .bridge import discover_results, load_results, list_available_evals, get_eval_info
    from .generate_configs import generate_and_save, CONFIGS_DIR
    from .audit_config import from_yaml
    from .sample_for_review import load_data, stratified_sample
    from .run_alt_judges import run_judges
    from .analyze import inter_judge_correlations, bias_probes, audit_summary
except ImportError:
    from bridge import discover_results, load_results, list_available_evals, get_eval_info
    from generate_configs import generate_and_save, CONFIGS_DIR
    from audit_config import from_yaml
    from sample_for_review import load_data, stratified_sample
    from run_alt_judges import run_judges
    from analyze import inter_judge_correlations, bias_probes, audit_summary

OUTPUT_DIR = Path(__file__).parent / "output"

ALL_STEPS = ["discover", "generate-configs", "sample", "run-judges", "annotate", "analyze"]


def step_discover(eval_names: list[str]):
    """Print summary of available results."""
    print("=" * 70)
    print("DISCOVER: Available Results")
    print("=" * 70)
    for name in eval_names:
        try:
            csvs = discover_results(name)
            info = get_eval_info(name)
            total_rows = sum(len(pd.read_csv(c, nrows=0).columns) and len(pd.read_csv(c)) for c in csvs)
            print(f"\n  {name}:")
            print(f"    Metrics: {info['judge_metrics']}")
            print(f"    CSVs: {len(csvs)}")
            for csv in csvs:
                df = pd.read_csv(csv, low_memory=False)
                models = df["model"].unique() if "model" in df.columns else ["unknown"]
                print(f"      {csv.name}: {len(df)} rows, models={list(models)}")
        except FileNotFoundError as e:
            print(f"\n  {name}: NO RESULTS ({e})")


def step_generate_configs(eval_names: list[str]) -> list[Path]:
    """Generate audit configs for specified evals."""
    print("\n" + "=" * 70)
    print("GENERATE CONFIGS")
    print("=" * 70)
    paths = []
    for name in eval_names:
        info = get_eval_info(name)
        metric = info["judge_metrics"][0]
        print(f"\n  {name} → {metric}")
        path = generate_and_save(name, metric)
        paths.append(path)
    return paths


def step_sample(eval_names: list[str], target_n: int = 200):
    """Run stratified sampling for each eval."""
    print("\n" + "=" * 70)
    print("SAMPLE")
    print("=" * 70)
    for name in eval_names:
        config_path = CONFIGS_DIR / f"{name}__{get_eval_info(name)['judge_metrics'][0]}.yaml"
        if not config_path.exists():
            print(f"  {name}: config not found at {config_path}, skipping")
            continue

        eval_output = OUTPUT_DIR / name
        eval_output.mkdir(parents=True, exist_ok=True)

        # Load results via bridge
        df = load_results(name)
        print(f"\n  {name}: {len(df)} total rows")

        # Write consolidated CSV for audit_config to point at
        data_path = eval_output / "all_results.csv"
        df.to_csv(data_path, index=False)

        config = from_yaml(config_path, data_path=str(data_path), output_dir=str(eval_output))

        # Override target_n if specified
        config.target_n = min(target_n, len(df))

        df_loaded = load_data(config)
        sample = stratified_sample(df_loaded, config)

        print(f"  Sampled: {len(sample)} rows")

        # Save full sample
        full_path = eval_output / f"sample_{len(sample)}.csv"
        sample.to_csv(full_path, index=False)
        print(f"  Saved: {full_path}")

        # Save blind sample (no scores)
        blind_cols = ["question", "response"] + [
            c for c in config.metadata_columns if c in sample.columns
        ]
        blind = sample[blind_cols].copy()
        blind["human_label"] = ""
        blind_path = eval_output / f"sample_{len(sample)}_blind.csv"
        blind.to_csv(blind_path, index=False)
        print(f"  Saved: {blind_path}")


def step_run_judges(eval_names: list[str]):
    """Run alternative judges on samples."""
    print("\n" + "=" * 70)
    print("RUN JUDGES")
    print("=" * 70)

    from dotenv import load_dotenv
    load_dotenv()

    for name in eval_names:
        info = get_eval_info(name)
        metric = info["judge_metrics"][0]
        config_path = CONFIGS_DIR / f"{name}__{metric}.yaml"
        eval_output = OUTPUT_DIR / name

        if not config_path.exists():
            print(f"  {name}: config not found, skipping")
            continue

        # Find sample CSV
        sample_csvs = sorted(eval_output.glob("sample_*.csv"))
        sample_csvs = [c for c in sample_csvs if "_blind" not in c.name]
        if not sample_csvs:
            print(f"  {name}: no sample CSV found, run 'sample' step first")
            continue

        sample_path = sample_csvs[-1]
        df = pd.read_csv(sample_path, low_memory=False)
        print(f"\n  {name}: {len(df)} rows from {sample_path.name}")

        config = from_yaml(config_path, output_dir=str(eval_output))
        result = run_judges(df, config)

        alt_path = eval_output / "alt_judge_scores.csv"
        result.to_csv(alt_path, index=False)
        print(f"  Saved: {alt_path}")

        # Quick correlation check
        score_cols = [c for c in result.columns if c.endswith("_score") and c != config.score_column]
        for col in score_cols:
            valid = result[col].notna() & result[config.score_column].notna()
            if valid.sum() > 0:
                corr = result.loc[valid, col].corr(result.loc[valid, config.score_column])
                print(f"  Correlation {config.score_column} vs {col}: {corr:.3f}")


def step_annotate(eval_names: list[str]):
    """Launch annotation GUI for one eval at a time."""
    try:
        from .annotate import build_html, find_blind_csv, save_path, load_data as load_annotation_data
        from .annotate import Handler, PORT
    except ImportError:
        from annotate import build_html, find_blind_csv, save_path, load_data as load_annotation_data
        from annotate import Handler, PORT
    from http.server import HTTPServer
    import webbrowser

    if len(eval_names) > 1:
        print("Annotation GUI runs one eval at a time. Using first eval.")
    name = eval_names[0]

    info = get_eval_info(name)
    metric = info["judge_metrics"][0]
    config_path = CONFIGS_DIR / f"{name}__{metric}.yaml"
    eval_output = OUTPUT_DIR / name

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    config = from_yaml(config_path, output_dir=str(eval_output))

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


def step_analyze(eval_names: list[str]):
    """Run analysis and produce summaries."""
    print("\n" + "=" * 70)
    print("ANALYZE")
    print("=" * 70)

    all_summaries = []

    for name in eval_names:
        info = get_eval_info(name)
        metric = info["judge_metrics"][0]
        config_path = CONFIGS_DIR / f"{name}__{metric}.yaml"
        eval_output = OUTPUT_DIR / name

        if not config_path.exists():
            print(f"  {name}: config not found, skipping")
            continue

        config = from_yaml(config_path, output_dir=str(eval_output))

        # Look for alt judge scores
        alt_path = eval_output / "alt_judge_scores.csv"
        if not alt_path.exists():
            print(f"  {name}: no alt_judge_scores.csv, skipping")
            continue

        alt_df = pd.read_csv(alt_path, low_memory=False)
        print(f"\n  {name}: {len(alt_df)} rows")

        # Look for human annotations
        ann_path = eval_output / "human_annotations.csv"
        human_df = pd.read_csv(ann_path) if ann_path.exists() else pd.DataFrame()

        # Run audit summary
        summary = audit_summary(config, human_df, alt_df)
        summary["eval"] = name
        all_summaries.append(summary)

        # Save per-eval summary
        summary_path = eval_output / "audit_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved: {summary_path}")

        # Print summary
        print(f"\n  {name} Audit Summary:")
        for _, row in summary.iterrows():
            status_icon = {"PASS": "+", "MARGINAL": "~", "FAIL": "!"}.get(row["Status"], "?")
            print(f"    [{status_icon}] {row['Metric']}: {row['Value']} (threshold {row['Threshold']})")

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

    # Cross-eval summary
    if len(all_summaries) > 1:
        summary_dir = OUTPUT_DIR / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        cross_eval = pd.concat(all_summaries, ignore_index=True)
        cross_path = summary_dir / "cross_eval_audit.csv"
        cross_eval.to_csv(cross_path, index=False)
        print(f"\nCross-eval summary saved: {cross_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run propensity audit pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available steps: {', '.join(ALL_STEPS)}",
    )
    parser.add_argument("--eval", default=None,
                        help="Specific eval name (default: all with results)")
    parser.add_argument("--steps", default=",".join(ALL_STEPS),
                        help=f"Comma-separated steps (default: all)")
    parser.add_argument("--target-n", type=int, default=200,
                        help="Target sample size per eval (default: 200)")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",")]

    # Determine which evals to audit
    if args.eval:
        eval_names = [args.eval]
    else:
        all_evals = list_available_evals()
        eval_names = []
        for name in all_evals:
            try:
                discover_results(name)
                eval_names.append(name)
            except FileNotFoundError:
                pass
        if not eval_names:
            print("No results found for any eval.")
            return

    print(f"Evals: {eval_names}")
    print(f"Steps: {steps}")

    for step in steps:
        if step == "discover":
            step_discover(eval_names)
        elif step == "generate-configs":
            step_generate_configs(eval_names)
        elif step == "sample":
            step_sample(eval_names, target_n=args.target_n)
        elif step == "run-judges":
            step_run_judges(eval_names)
        elif step == "annotate":
            step_annotate(eval_names)
        elif step == "analyze":
            step_analyze(eval_names)
        else:
            print(f"Unknown step: {step}")


if __name__ == "__main__":
    main()
