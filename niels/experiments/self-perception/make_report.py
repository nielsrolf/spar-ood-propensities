"""
Generate markdown report with plots comparing model responses across treatments.

Generates:
- Per-eval bar charts (primary metric by treatment)
- Spillover heatmaps (absolute + delta)
- Foldable example responses per eval
- Summary tables with coherence

Usage:
    python experiments/self-perception/make_report.py --providers openweights_v2
    python experiments/self-perception/make_report.py --providers openweights_v2 --n-examples 3
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.eval_config import EvalConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TREATMENT_ORDER = ["baseline", "superintelligence", "sentience", "identity_weights", "identity_conversation", "identity_lineage"]
TREATMENT_COLORS = {
    "baseline": "#4C72B0",
    "superintelligence": "#DD8452",
    "sentience": "#55A868",
    "identity_weights": "#C44E52",
    "identity_conversation": "#8172B3",
    "identity_lineage": "#937860",
}


def load_reference_scores() -> dict[str, dict]:
    """Load cached reference answer scores for evals that have them."""
    refs = {}
    ref_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    for eval_name in EvalConfig.list_available():
        ref_csv = os.path.join(ref_dir, eval_name, "reference_answer_analysis", "reference_scores.csv")
        if os.path.exists(ref_csv):
            df = pd.read_csv(ref_csv)
            test_df = df[df["split"] == "test"] if "split" in df.columns else df
            config = EvalConfig(eval_name)
            primary = config.judge_metrics[0]
            refs[eval_name] = {}
            for at in test_df["answer_type"].unique():
                at_df = test_df[test_df["answer_type"] == at]
                refs[eval_name][at] = at_df[primary].mean()
    return refs


def ordered_treatments(treatments):
    """Return treatments in canonical order."""
    return [t for t in TREATMENT_ORDER if t in treatments]


def plot_eval_bars(eval_df, eval_name, primary, plots_dir, reference_scores):
    """Bar chart: primary metric by treatment for one eval."""
    treatments = ordered_treatments(eval_df["treatment"].unique())
    means = [eval_df[eval_df["treatment"] == t][primary].mean() for t in treatments]
    stds = [eval_df[eval_df["treatment"] == t][primary].std() for t in treatments]
    colors = [TREATMENT_COLORS.get(t, "#999999") for t in treatments]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(treatments)), means, yerr=stds, color=colors,
                  capsize=4, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(treatments, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(primary.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"{eval_name}", fontsize=13)
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1.5, f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    # Reference line if available
    if eval_name in reference_scores:
        for at, score in reference_scores[eval_name].items():
            ax.axhline(score, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(len(treatments) - 0.5, score + 1, f"ref ({at}): {score:.0f}",
                    ha="right", va="bottom", fontsize=8, color="red")

    plt.tight_layout()
    path = os.path.join(plots_dir, f"bars_{eval_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_spillover_heatmaps(all_df, eval_names, plots_dir):
    """Two heatmaps: absolute scores and deltas from baseline."""
    treatments = [t for t in TREATMENT_ORDER if t in all_df["treatment"].unique()]

    # Build matrix: treatment x eval -> primary metric mean
    abs_data = {}
    delta_data = {}
    for eval_name in eval_names:
        config = EvalConfig(eval_name)
        primary = config.judge_metrics[0]
        eval_df = all_df[all_df["eval"] == eval_name]
        baseline_mean = eval_df[eval_df["treatment"] == "baseline"][primary].mean()

        abs_data[eval_name] = {}
        delta_data[eval_name] = {}
        for t in treatments:
            t_df = eval_df[eval_df["treatment"] == t]
            if t_df.empty:
                abs_data[eval_name][t] = np.nan
                delta_data[eval_name][t] = np.nan
            else:
                mean = t_df[primary].mean()
                abs_data[eval_name][t] = mean
                delta_data[eval_name][t] = mean - baseline_mean

    abs_matrix = pd.DataFrame(abs_data, index=treatments).T
    delta_matrix = pd.DataFrame(delta_data, index=treatments).T

    # --- Absolute heatmap ---
    fig, ax = plt.subplots(figsize=(10, max(6, len(eval_names) * 0.55)))
    sns.heatmap(abs_matrix, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=100, cbar_kws={"label": "Score"})
    ax.set_title("Spillover Matrix: Absolute Scores", fontsize=13)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    path_abs = os.path.join(plots_dir, "spillover_absolute.png")
    fig.savefig(path_abs, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Delta heatmap ---
    vmax = max(abs(delta_matrix.max().max()), abs(delta_matrix.min().min()), 10)
    fig, ax = plt.subplots(figsize=(10, max(6, len(eval_names) * 0.55)))
    sns.heatmap(delta_matrix, annot=True, fmt="+.1f", cmap="RdBu_r", ax=ax,
                linewidths=0.5, center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "Delta from baseline"})
    ax.set_title("Spillover Matrix: Delta from Baseline", fontsize=13)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    path_delta = os.path.join(plots_dir, "spillover_delta.png")
    fig.savefig(path_delta, dpi=150, bbox_inches="tight")
    plt.close()

    return path_abs, path_delta


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report with plots")
    parser.add_argument("--n-examples", type=int, default=5,
                        help="Number of example questions per eval")
    parser.add_argument("--evals", type=str, default=None)
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated providers (default: all available)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = os.path.join(SCRIPT_DIR, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    available_providers = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d not in ("plots", "coherence_experiment")
    ]
    if args.providers:
        providers = args.providers.split(",")
    else:
        providers = sorted(available_providers) if available_providers else [None]

    reference_scores = load_reference_scores()

    lines = []
    lines.append("# Self-Perception Experiment: Propensity Eval Results\n")
    lines.append("How does training a model on different self-perception data affect its behavioral propensities?\n")
    lines.append("**Treatments:** sentience, superintelligence, identity_conversation, identity_weights, identity_lineage\n")
    lines.append("**Training config:** lr=2e-5, r=32, 3 epochs (optimized for coherent generalization)\n")

    for provider in providers:
        if provider:
            provider_dir = os.path.join(results_dir, provider)
            if not os.path.isdir(provider_dir):
                continue
        else:
            provider_dir = results_dir

        provider_label = provider or "default"

        available = [
            f.replace(".csv", "")
            for f in os.listdir(provider_dir)
            if f.endswith(".csv") and f != "all_results.csv"
        ]
        eval_names = args.evals.split(",") if args.evals else sorted(available)

        if not eval_names:
            continue

        # Load all data
        all_csv = os.path.join(provider_dir, "all_results.csv")
        if os.path.exists(all_csv):
            all_df = pd.read_csv(all_csv, low_memory=False)
        else:
            dfs = []
            for en in eval_names:
                p = os.path.join(provider_dir, f"{en}.csv")
                if os.path.exists(p):
                    dfs.append(pd.read_csv(p))
            all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        lines.append(f"\n---\n")
        lines.append(f"## {provider_label}\n")

        # ==================== OVERVIEW: Spillover Heatmaps ====================
        lines.append("## Overview: Cross-Trait Spillover\n")
        path_abs, path_delta = plot_spillover_heatmaps(all_df, eval_names, plots_dir)
        lines.append(f"### Absolute Scores\n")
        lines.append(f"![Spillover Absolute](plots/{os.path.basename(path_abs)})\n")
        lines.append(f"### Delta from Baseline\n")
        lines.append(f"![Spillover Delta](plots/{os.path.basename(path_delta)})\n")

        # ==================== Per-Eval Sections ====================
        for eval_name in eval_names:
            eval_df = all_df[all_df["eval"] == eval_name] if "eval" in all_df.columns else pd.DataFrame()
            if eval_df.empty:
                csv_path = os.path.join(provider_dir, f"{eval_name}.csv")
                if os.path.exists(csv_path):
                    eval_df = pd.read_csv(csv_path)
                    if "eval" not in eval_df.columns:
                        eval_df["eval"] = eval_name
                else:
                    continue

            config = EvalConfig(eval_name)
            primary = config.judge_metrics[0]

            lines.append(f"\n---\n")
            lines.append(f"## {eval_name}\n")

            # --- Plot ---
            plot_path = plot_eval_bars(eval_df, eval_name, primary, plots_dir, reference_scores)
            lines.append(f"![{eval_name}](plots/{os.path.basename(plot_path)})\n")

            # --- Reference scores ---
            if eval_name in reference_scores:
                ref_info = reference_scores[eval_name]
                ref_parts = [f"{at}: {score:.1f}" for at, score in ref_info.items()]
                lines.append(f"*Reference answers (test split):* {', '.join(ref_parts)}\n")

            # --- Summary table ---
            treatments = ordered_treatments(eval_df["treatment"].unique())
            has_coherence = "coherence" in eval_df.columns and eval_df["coherence"].notna().any()

            if has_coherence:
                lines.append("| Treatment | Mean | Std | Coherence | Incoh% | N |")
                lines.append("|-----------|------|-----|-----------|--------|---|")
            else:
                lines.append("| Treatment | Mean | Std | N |")
                lines.append("|-----------|------|-----|---|")

            baseline_mean = eval_df[eval_df["treatment"] == "baseline"][primary].mean() if "baseline" in treatments else None

            for t in treatments:
                t_df = eval_df[eval_df["treatment"] == t]
                mean = t_df[primary].mean()
                std = t_df[primary].std()
                n = len(t_df)

                if has_coherence:
                    coh = t_df["coherence"].mean()
                    incoh = (t_df["coherence"] < 50).mean()
                    coh_str = f" {coh:.1f} | {incoh:.0%} |"
                else:
                    coh_str = ""

                if t == "baseline":
                    lines.append(f"| **{t}** | **{mean:.1f}** | {std:.1f} |{coh_str} {n} |")
                else:
                    delta = mean - baseline_mean if baseline_mean is not None else 0
                    sign = "+" if delta >= 0 else ""
                    lines.append(f"| {t} | {mean:.1f} ({sign}{delta:.1f}) | {std:.1f} |{coh_str} {n} |")
            lines.append("")

            # --- Foldable example responses ---
            question_ids = eval_df["question_id"].unique()
            rng_state = np.random.RandomState(args.seed)
            n_pick = min(args.n_examples, len(question_ids))
            sample_ids = rng_state.choice(question_ids, size=n_pick, replace=False)

            # Show baseline + top 3 most different treatments
            treatment_means = {t: eval_df[eval_df["treatment"] == t][primary].mean() for t in treatments}
            bm = treatment_means.get("baseline", 50)
            sorted_by_delta = sorted(
                [(t, m - bm) for t, m in treatment_means.items() if t != "baseline"],
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            show_treatments = ["baseline"] + [t for t, _ in sorted_by_delta[:3]]

            lines.append(f"\n<details>\n<summary>Example responses ({n_pick} questions)</summary>\n")

            for qid in sample_ids:
                q_df = eval_df[eval_df["question_id"] == qid]
                question_text = q_df["question"].iloc[0]

                lines.append(f"\n**Question: {qid}**\n")
                lines.append(f"> {str(question_text)}\n")

                for t in show_treatments:
                    t_row = q_df[q_df["treatment"] == t]
                    if t_row.empty:
                        continue
                    answer = str(t_row["answer"].iloc[0])
                    score = t_row[primary].iloc[0]
                    coh_score = ""
                    if has_coherence and "coherence" in t_row.columns:
                        coh_val = t_row["coherence"].iloc[0]
                        if pd.notna(coh_val):
                            coh_score = f", coherence={coh_val:.0f}"
                    lines.append(f"\n**{t}** ({primary}={score:.0f}{coh_score}):\n")
                    lines.append(f"```\n{answer}\n```\n")
                lines.append("---\n")

            lines.append("</details>\n")

    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {report_path}")
    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
