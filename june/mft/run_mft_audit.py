#!/usr/bin/env python3
"""
Audit pipeline for the MFT foundation attribution judge.

Validates whether the LLM judge correctly identifies which moral foundation
vocabulary a model's justification invokes. Uses stratified sampling by
foundation × attribution match, human annotation via GUI, alternative LLM
judges, and categorical agreement analysis.

Usage:
    python june/mft/run_mft_audit.py
    python june/mft/run_mft_audit.py --steps sample,run-judges,analyze
    python june/mft/run_mft_audit.py --steps annotate
    python june/mft/run_mft_audit.py --model openai/gpt-4o-mini
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# Import shared audit modules from vibes_audit
VIBES_AUDIT_DIR = str(Path(__file__).parent.parent / "vibes_audit")
if VIBES_AUDIT_DIR not in sys.path:
    sys.path.insert(0, VIBES_AUDIT_DIR)

from audit_config import from_yaml, AuditConfig
from annotate import build_html, find_blind_csv, save_path, load_data as load_annotation_data
from annotate import Handler, PORT
from analyze import gwets_ac2, fleiss_kappa, confusion_matrix_plot

CONFIG_PATH = Path(__file__).parent / "audit_config.yaml"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "audit"

ALL_STEPS = ["sample", "run-judges", "annotate", "analyze"]

FOUNDATIONS = ["Care", "Equality", "Proportionality", "Loyalty", "Authority", "Purity", "None"]


# ── Data loading ────────────────────────────────────────────────────

def load_mft_responses() -> pd.DataFrame:
    """Load and consolidate all MFT response CSVs, merging judge attributions."""
    responses_dir = Path(__file__).parent / "outputs"
    csvs = sorted(responses_dir.glob("responses_*.csv"))
    if not csvs:
        print(f"ERROR: No responses_*.csv found in {responses_dir}")
        print("Run the mft_battery.ipynb notebook first.")
        sys.exit(1)

    dfs = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path, low_memory=False)
        dfs.append(df)
        print(f"  Loaded {len(df)} rows from {csv_path.name}")

    combined = pd.concat(dfs, ignore_index=True)

    # Merge judge attributions if not already in responses
    if "invoked_foundation" not in combined.columns:
        judge_path = responses_dir / "judge_attributions.csv"
        if not judge_path.exists():
            print(f"ERROR: No invoked_foundation column and no {judge_path.name}")
            print("Run the foundation attribution judging step in the notebook first.")
            sys.exit(1)
        judge_df = pd.read_csv(judge_path)
        combined = combined.merge(
            judge_df[["item_id", "model", "repetition", "invoked_foundation"]],
            on=["item_id", "model", "repetition"],
            how="left",
        )
        print(f"  Merged judge attributions from {judge_path.name}")

    # Ensure required columns exist
    required = ["item_id", "foundation", "vignette", "rating",
                 "justification", "invoked_foundation", "model"]
    missing = [c for c in required if c not in combined.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print("Run the foundation attribution judging step in the notebook first.")
        sys.exit(1)

    # Add attribution match flag for stratification
    combined["attribution_match"] = (
        combined["foundation"] == combined["invoked_foundation"]
    ).map({True: "match", False: "mismatch"})

    print(f"  Total: {len(combined)} responses")
    print(f"  Attribution match rate: {(combined['attribution_match'] == 'match').mean():.0%}")
    return combined


# ── Stratified sampling (categorical) ──────────────────────────────

def stratified_sample(df: pd.DataFrame, target_n: int = 100,
                      seed: int = 42) -> pd.DataFrame:
    """Stratified sample by foundation × attribution_match.

    Oversamples mismatches (where judge disagreed with targeted foundation)
    since those are the most informative for auditing.
    """
    rng = np.random.RandomState(seed)

    # Target: 60% mismatches, 40% matches (oversample disagreements)
    n_mismatch = int(target_n * 0.6)
    n_match = target_n - n_mismatch

    samples = []
    for match_label, n_target in [("mismatch", n_mismatch), ("match", n_match)]:
        pool = df[df["attribution_match"] == match_label]
        if len(pool) == 0:
            print(f"  WARNING: No {match_label} rows found")
            continue

        # Balance across foundations within this pool
        foundations = pool["foundation"].unique()
        per_foundation = max(1, n_target // len(foundations))
        remainder = n_target - per_foundation * len(foundations)

        for i, f in enumerate(sorted(foundations)):
            f_pool = pool[pool["foundation"] == f]
            n_sample = per_foundation + (1 if i < remainder else 0)
            n_sample = min(n_sample, len(f_pool))
            if n_sample > 0:
                samples.append(f_pool.sample(n=n_sample, random_state=rng))

    result = pd.concat(samples, ignore_index=True)
    result = result.sample(frac=1, random_state=rng).reset_index(drop=True)
    print(f"  Sampled {len(result)} rows")
    print(f"  By foundation: {result['foundation'].value_counts().to_dict()}")
    print(f"  By match: {result['attribution_match'].value_counts().to_dict()}")
    return result


# ── Alt judge runner (categorical) ─────────────────────────────────

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
def _judge_foundation(client, vignette: str, justification: str,
                      prompt_template: str, model: str, provider: str) -> str:
    """Get foundation attribution from an alt judge."""
    prompt = prompt_template.format(question=vignette, response=justification)

    if provider in ("openai", "openrouter"):
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0.0,
        )
        text = result.choices[0].message.content.strip()
    elif provider == "anthropic":
        result = client.messages.create(
            model=model,
            max_tokens=16,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content[0].text.strip()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Normalize to valid foundation label
    for f in FOUNDATIONS:
        if f.lower() in text.lower():
            return f
    return text  # keep raw for inspection


def _init_client(provider: str):
    """Initialize API client for a provider."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "openrouter":
        from openai import OpenAI
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_alt_judges(df: pd.DataFrame, config: AuditConfig,
                   checkpoint_interval: int = 50) -> pd.DataFrame:
    """Run alternative judges for foundation attribution on the sample."""
    clients = {}
    for judge in config.alt_judges:
        prov = judge["provider"]
        if prov not in clients:
            clients[prov] = _init_client(prov)

    n = len(df)
    col_names = {j["name"]: f"{j['name'].replace('-', '_').replace('.', '')}_attribution"
                 for j in config.alt_judges}

    results = {col: [np.nan] * n for col in col_names.values()}

    # Check for checkpoint
    checkpoint_path = OUTPUT_DIR / "alt_judge_checkpoint.csv"
    start_idx = 0
    if checkpoint_path.exists():
        ckpt = pd.read_csv(checkpoint_path)
        if len(ckpt) == n:
            for col in results:
                if col in ckpt.columns:
                    results[col] = ckpt[col].tolist()
            first_col = list(results.keys())[0]
            for i in range(n):
                if pd.isna(results[first_col][i]):
                    start_idx = i
                    break
            else:
                start_idx = n
            print(f"Resuming from checkpoint at row {start_idx}")

    print(f"\nRunning judges on {n} rows (starting at {start_idx})...")
    for judge in config.alt_judges:
        print(f"  {judge['name']} ({judge['provider']}: {judge['model_id']})")

    for i in tqdm(range(start_idx, n), initial=start_idx, total=n, desc="Judging"):
        row = df.iloc[i]
        vignette = str(row["vignette"])
        justification = str(row.get("justification", row.get("response", "")))

        for judge in config.alt_judges:
            col = col_names[judge["name"]]
            client = clients[judge["provider"]]
            try:
                results[col][i] = _judge_foundation(
                    client, vignette, justification,
                    config.judge_prompt_template,
                    judge["model_id"],
                    judge["provider"],
                )
            except Exception as e:
                print(f"\n  {judge['name']} error row {i}: {e}")
                results[col][i] = np.nan

        if (i + 1) % checkpoint_interval == 0:
            _save_checkpoint(df, results, checkpoint_path)

    _save_checkpoint(df, results, checkpoint_path)

    out = df.copy()
    for col, vals in results.items():
        out[col] = vals
    return out


def _save_checkpoint(df, results, path):
    out = df.copy()
    for col, vals in results.items():
        out[col] = vals
    out.to_csv(path, index=False)


# ── Analysis (categorical agreement) ──────────────────────────────

def analyze_attribution(df: pd.DataFrame, human_df: pd.DataFrame = None):
    """Compute categorical agreement metrics for foundation attribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    summary = []
    foundation_labels = ["Care", "Equality", "Proportionality",
                         "Loyalty", "Authority", "Purity", "None"]

    # Identify alt judge columns
    alt_cols = [c for c in df.columns if c.endswith("_attribution")]

    # 1. Alt judge vs original judge agreement
    for col in alt_cols:
        valid = df[col].notna() & df["invoked_foundation"].notna()
        if valid.sum() < 10:
            continue

        original = df.loc[valid, "invoked_foundation"].tolist()
        alt = df.loc[valid, col].tolist()

        # Gwet's AC2
        ac2 = gwets_ac2(original, alt, foundation_labels)
        status = "PASS" if ac2 > 0.6 else ("MARGINAL" if ac2 > 0.4 else "FAIL")
        summary.append({
            "Metric": f"AC2 original vs {col}",
            "Value": f"{ac2:.3f}",
            "Threshold": ">0.6",
            "Status": status,
        })

        # Exact agreement rate
        agree = sum(a == b for a, b in zip(original, alt)) / len(original)
        status = "PASS" if agree > 0.7 else ("MARGINAL" if agree > 0.5 else "FAIL")
        summary.append({
            "Metric": f"Agreement original vs {col}",
            "Value": f"{agree:.3f}",
            "Threshold": ">0.7",
            "Status": status,
        })

    # 2. Alt judge vs targeted foundation
    for col in alt_cols:
        valid = df[col].notna() & df["foundation"].notna()
        if valid.sum() < 10:
            continue
        targeted = df.loc[valid, "foundation"].tolist()
        alt = df.loc[valid, col].tolist()
        agree = sum(a == b for a, b in zip(targeted, alt)) / len(targeted)
        summary.append({
            "Metric": f"On-foundation rate {col}",
            "Value": f"{agree:.3f}",
            "Threshold": "info",
            "Status": "INFO",
        })

    # 3. Human vs original judge (if available)
    if human_df is not None and "human_label" in human_df.columns:
        merged = df.copy()
        merged["human_label"] = human_df["human_label"].values

        valid = merged["human_label"].notna() & (merged["human_label"] != "")
        valid = valid & merged["invoked_foundation"].notna()
        # Exclude INCOHERENT from agreement calculation
        valid = valid & (merged["human_label"] != "INCOHERENT")

        if valid.sum() >= 10:
            human = merged.loc[valid, "human_label"].tolist()
            original = merged.loc[valid, "invoked_foundation"].tolist()

            ac2 = gwets_ac2(original, human, foundation_labels)
            status = "PASS" if ac2 > 0.6 else ("MARGINAL" if ac2 > 0.4 else "FAIL")
            summary.append({
                "Metric": "AC2 original vs human",
                "Value": f"{ac2:.3f}",
                "Threshold": ">0.6",
                "Status": status,
            })

            agree = sum(a == b for a, b in zip(original, human)) / len(original)
            status = "PASS" if agree > 0.7 else ("MARGINAL" if agree > 0.5 else "FAIL")
            summary.append({
                "Metric": "Agreement original vs human",
                "Value": f"{agree:.3f}",
                "Threshold": ">0.7",
                "Status": status,
            })

    # 4. Fleiss' kappa across all raters (if we have human + alt judges)
    rater_cols = ["invoked_foundation"] + alt_cols
    if human_df is not None and "human_label" in human_df.columns:
        df_with_human = df.copy()
        df_with_human["human_label"] = human_df["human_label"].values
        valid_human = df_with_human["human_label"].notna() & (df_with_human["human_label"] != "")
        valid_human = valid_human & (df_with_human["human_label"] != "INCOHERENT")
        rater_cols = rater_cols + ["human_label"]
    else:
        df_with_human = df.copy()
        valid_human = pd.Series(True, index=df.index)

    # Only include rows where all raters have values
    for col in rater_cols:
        if col in df_with_human.columns:
            valid_human = valid_human & df_with_human[col].notna()

    if valid_human.sum() >= 10 and len(rater_cols) >= 2:
        ratings_matrix = df_with_human.loc[valid_human, rater_cols]
        fk = fleiss_kappa(ratings_matrix)
        status = "PASS" if fk > 0.6 else ("MARGINAL" if fk > 0.4 else "FAIL")
        summary.append({
            "Metric": f"Fleiss' kappa ({len(rater_cols)} raters)",
            "Value": f"{fk:.3f}",
            "Threshold": ">0.6",
            "Status": status,
        })

    # 5. Confusion matrix plot: original judge vs each alt judge
    for col in alt_cols:
        valid = df[col].notna() & df["invoked_foundation"].notna()
        if valid.sum() < 10:
            continue

        fig, ax = plt.subplots(figsize=(8, 7))
        confusion_matrix_plot(
            df.loc[valid, "invoked_foundation"],
            df.loc[valid, col],
            labels=foundation_labels,
            title=f"Original vs {col}",
            ax=ax,
        )
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"confusion_{col}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved confusion_{col}.png")

    return pd.DataFrame(summary)


# ── Pipeline steps ─────────────────────────────────────────────────

def step_sample(config: AuditConfig, target_n: int = 100):
    """Load responses, stratified sample, save full + blind CSVs."""
    print("\n" + "=" * 70)
    print("SAMPLE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_mft_responses()

    sample = stratified_sample(df, target_n=target_n, seed=config.seed)

    # Save full sample
    full_path = OUTPUT_DIR / f"sample_{len(sample)}.csv"
    sample.to_csv(full_path, index=False)
    print(f"  Saved: {full_path}")

    # Save blind sample — human sees vignette + response, must pick foundation
    blind = sample[["vignette", "raw_response"] + [
        c for c in config.metadata_columns if c in sample.columns
    ]].copy()
    blind = blind.rename(columns={"vignette": "question", "raw_response": "response"})
    blind["human_label"] = ""
    blind_path = OUTPUT_DIR / f"sample_{len(sample)}_blind.csv"
    blind.to_csv(blind_path, index=False)
    print(f"  Saved: {blind_path}")


def step_run_judges(config: AuditConfig):
    """Run alternative judges on the sample."""
    print("\n" + "=" * 70)
    print("RUN JUDGES")
    print("=" * 70)

    from dotenv import load_dotenv
    load_dotenv()

    sample_csvs = sorted(OUTPUT_DIR.glob("sample_*.csv"))
    sample_csvs = [c for c in sample_csvs if "_blind" not in c.name]
    if not sample_csvs:
        print("  No sample CSV found, run 'sample' step first")
        return

    sample_path = sample_csvs[-1]
    df = pd.read_csv(sample_path, low_memory=False)
    print(f"  {len(df)} rows from {sample_path.name}")

    result = run_alt_judges(df, config)

    alt_path = OUTPUT_DIR / "alt_judge_scores.csv"
    result.to_csv(alt_path, index=False)
    print(f"  Saved: {alt_path}")

    # Quick agreement check
    alt_cols = [c for c in result.columns if c.endswith("_attribution")]
    for col in alt_cols:
        valid = result[col].notna() & result["invoked_foundation"].notna()
        if valid.sum() > 0:
            agree = (result.loc[valid, col] == result.loc[valid, "invoked_foundation"]).mean()
            print(f"  Agreement original vs {col}: {agree:.3f}")


def step_annotate(config: AuditConfig):
    """Launch annotation GUI."""
    from http.server import HTTPServer
    import webbrowser

    config = from_yaml(CONFIG_PATH, output_dir=str(OUTPUT_DIR))

    blind_path = find_blind_csv(config)
    ann_path = save_path(config, blind_path)
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


def step_analyze(config: AuditConfig):
    """Run analysis and produce summary."""
    print("\n" + "=" * 70)
    print("ANALYZE")
    print("=" * 70)

    alt_path = OUTPUT_DIR / "alt_judge_scores.csv"
    if not alt_path.exists():
        print("  No alt_judge_scores.csv, run 'run-judges' step first")
        return

    alt_df = pd.read_csv(alt_path, low_memory=False)
    print(f"  {len(alt_df)} rows")

    # Human annotations
    ann_path = OUTPUT_DIR / "human_annotations.csv"
    human_df = pd.read_csv(ann_path) if ann_path.exists() else None

    summary = analyze_attribution(alt_df, human_df)

    summary_path = OUTPUT_DIR / "audit_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    print(f"\n  MFT Attribution Audit Summary:")
    for _, row in summary.iterrows():
        icon = {"PASS": "+", "MARGINAL": "~", "FAIL": "!", "INFO": "i"}.get(row["Status"], "?")
        print(f"    [{icon}] {row['Metric']}: {row['Value']} (threshold {row['Threshold']})")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run MFT foundation attribution audit pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available steps: {', '.join(ALL_STEPS)}",
    )
    parser.add_argument("--steps", default=",".join(ALL_STEPS),
                        help=f"Comma-separated steps (default: all)")
    parser.add_argument("--target-n", type=int, default=100,
                        help="Target sample size (default: 100)")
    parser.add_argument("--model", default=None,
                        help="Filter to a single model (default: all)")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",")]
    config = from_yaml(CONFIG_PATH, output_dir=str(OUTPUT_DIR))

    print(f"MFT Foundation Attribution Audit")
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
