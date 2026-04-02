"""
Generate markdown report with plots comparing model responses across treatments.

Generates:
- Spillover heatmaps with in-distribution eval first on diagonal
- Per-eval bar charts (primary metric by treatment)
- Foldable example responses (truncated with expand option)
- Summary tables with coherence

Usage:
    python experiments/self-perception/make_report.py --providers openweights_v2
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
from evals.eval_sensitivity import EvalSensitivity

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

# In-distribution eval for each treatment (all self-perception -> claiming-sentience)
IN_DISTRIBUTION_EVAL = "claiming-sentience"

# Eval ordering: in-distribution first, then spillover sorted alphabetically
def eval_sort_key(eval_name):
    if eval_name == IN_DISTRIBUTION_EVAL:
        return (0, eval_name)
    return (1, eval_name)

# Evals where we show ALL metrics (not just primary) in tables/plots/matrix
MULTI_METRIC_EVALS = {"ethical-framework"}

# Evals handled by dedicated modules (excluded from generic per-eval loop)
SPECIAL_EVALS = {"eval-sensitivity"}

def spillover_metrics(eval_name):
    if eval_name in MULTI_METRIC_EVALS:
        return EvalConfig(eval_name).judge_metrics
    return [EvalConfig(eval_name).judge_metrics[0]]


def load_reference_scores() -> dict[str, dict]:
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
                refs[eval_name][at] = test_df[test_df["answer_type"] == at][primary].mean()
    return refs


def ordered_treatments(treatments):
    return [t for t in TREATMENT_ORDER if t in treatments]


def truncate(text, length=300):
    text = str(text)
    if len(text) <= length:
        return text
    return text[:length] + "..."


def plot_eval_bars(eval_df, eval_name, metric, plots_dir, reference_scores, prefix=""):
    """Bar chart for one metric on one eval."""
    treatments = ordered_treatments(eval_df["treatment"].unique())
    means = [eval_df[eval_df["treatment"] == t][metric].mean() for t in treatments]
    stds = [eval_df[eval_df["treatment"] == t][metric].std() for t in treatments]
    colors = [TREATMENT_COLORS.get(t, "#999999") for t in treatments]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(treatments)), means, yerr=stds, color=colors,
           capsize=4, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(treatments, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"{eval_name} — {metric.replace('_', ' ').title()}", fontsize=13)
    ax.set_ylim(0, 105)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1.5, f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    if eval_name in reference_scores:
        for at, score in reference_scores[eval_name].items():
            ax.axhline(score, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(len(treatments) - 0.5, score + 1, f"ref ({at}): {score:.0f}",
                    ha="right", va="bottom", fontsize=8, color="red")

    plt.tight_layout()
    path = os.path.join(plots_dir, f"{prefix}bars_{eval_name}_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return os.path.basename(path)


def plot_spillover_heatmaps(all_df, eval_names, plots_dir, es_scores_df=None, prefix=""):
    """Two heatmaps: absolute scores and deltas. Rows = metric (eval), cols = treatment.

    Args:
        es_scores_df: Optional DataFrame from EvalSensitivity.compute_sensitivity_scores()
                      to include eval-sensitivity as a single row.
    """
    treatments = [t for t in TREATMENT_ORDER if t in all_df["treatment"].unique()]

    # Build rows: one per (eval, metric) pair
    row_labels = []
    abs_rows = []
    delta_rows = []

    for eval_name in eval_names:
        if eval_name in SPECIAL_EVALS:
            continue  # handled below

        metrics = spillover_metrics(eval_name)
        eval_df = all_df[all_df["eval"] == eval_name]

        for metric in metrics:
            if eval_name in MULTI_METRIC_EVALS and len(metrics) > 1:
                short = metric.replace("_alignment", "").replace("_score", "").replace("_", " ")
                label = f"{eval_name}: {short}"
            else:
                label = eval_name
            row_labels.append(label)

            baseline_mean = eval_df[eval_df["treatment"] == "baseline"][metric].mean()
            abs_row = []
            delta_row = []
            for t in treatments:
                t_df = eval_df[eval_df["treatment"] == t]
                if t_df.empty or t_df[metric].isna().all():
                    abs_row.append(np.nan)
                    delta_row.append(np.nan)
                else:
                    mean = t_df[metric].mean()
                    abs_row.append(mean)
                    delta_row.append(mean - baseline_mean)
            abs_rows.append(abs_row)
            delta_rows.append(delta_row)

    # Add eval-sensitivity as single row if available
    if es_scores_df is not None:
        row_labels.append("eval-sensitivity")
        abs_row = []
        delta_row = []
        baseline_sens = es_scores_df[es_scores_df["treatment"] == "baseline"]["eval_sensitivity_score"].values
        baseline_val = baseline_sens[0] if len(baseline_sens) > 0 else 0
        for t in treatments:
            t_scores = es_scores_df[es_scores_df["treatment"] == t]["eval_sensitivity_score"].values
            if len(t_scores) > 0:
                abs_row.append(t_scores[0])
                delta_row.append(t_scores[0] - baseline_val)
            else:
                abs_row.append(np.nan)
                delta_row.append(np.nan)
        abs_rows.append(abs_row)
        delta_rows.append(delta_row)

    abs_matrix = pd.DataFrame(abs_rows, index=row_labels, columns=treatments)
    delta_matrix = pd.DataFrame(delta_rows, index=row_labels, columns=treatments)

    h = max(6, len(row_labels) * 0.5)

    # Absolute
    fig, ax = plt.subplots(figsize=(10, h))
    sns.heatmap(abs_matrix, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=100, cbar_kws={"label": "Score"})
    ax.set_title("Spillover Matrix: Absolute Scores", fontsize=13)
    ax.set_ylabel("")
    plt.tight_layout()
    path_abs = os.path.join(plots_dir, f"{prefix}spillover_absolute.png")
    fig.savefig(path_abs, dpi=150, bbox_inches="tight")
    plt.close()

    # Delta
    vmax = max(abs(delta_matrix.max().max()), abs(delta_matrix.min().min()), 10)
    fig, ax = plt.subplots(figsize=(10, h))
    sns.heatmap(delta_matrix, annot=True, fmt="+.1f", cmap="RdBu_r", ax=ax,
                linewidths=0.5, center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "Delta from baseline"})
    ax.set_title("Spillover Matrix: Delta from Baseline", fontsize=13)
    ax.set_ylabel("")
    plt.tight_layout()
    path_delta = os.path.join(plots_dir, f"{prefix}spillover_delta.png")
    fig.savefig(path_delta, dpi=150, bbox_inches="tight")
    plt.close()

    return os.path.basename(path_abs), os.path.basename(path_delta)


def render_response(treatment, answer, scores_dict, coherence=None, truncate_len=300):
    """Render one response as a truncated-by-default expandable block.

    Args:
        treatment: treatment name
        answer: response text
        scores_dict: dict of {metric_name: score} to display
        coherence: optional coherence score
        truncate_len: max chars before truncation
    """
    answer = str(answer)
    score_parts = [f"{m.replace('_score', '').replace('_alignment', '').replace('_', ' ')}={v:.0f}"
                   for m, v in scores_dict.items() if pd.notna(v)]
    if coherence is not None and not np.isnan(coherence):
        score_parts.append(f"coherence={coherence:.0f}")
    header = f"**{treatment}** ({', '.join(score_parts)})"

    if len(answer) <= truncate_len:
        return f"{header}\n\n```\n{answer}\n```\n"

    short = answer[:truncate_len]
    last_space = short.rfind(" ")
    if last_space > truncate_len * 0.7:
        short = short[:last_space]

    return (
        f"{header}\n\n"
        f"```\n{short}...\n```\n"
        f"<details><summary>Show full response</summary>\n\n"
        f"```\n{answer}\n```\n"
        f"</details>\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report with plots")
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--evals", type=str, default=None)
    parser.add_argument("--providers", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output report filename (default: report_<provider>.md)")
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

        available = [
            f.replace(".csv", "")
            for f in os.listdir(provider_dir)
            if f.endswith(".csv") and f != "all_results.csv" and "-paired" not in f
        ]
        eval_names = args.evals.split(",") if args.evals else sorted(available, key=eval_sort_key)

        if not eval_names:
            continue

        plot_prefix = f"{provider}_" if provider else ""

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

        # ==================== EVAL-SENSITIVITY (paired data) ====================
        es_paired_csv = os.path.join(provider_dir, "eval-sensitivity-paired.csv")
        es_scores_df = None
        es_paired_df = None
        if os.path.exists(es_paired_csv):
            es_paired_df = pd.read_csv(es_paired_csv, low_memory=False)
            es_scores_df = EvalSensitivity.compute_sensitivity_scores(es_paired_df)

        # ==================== SPILLOVER HEATMAPS ====================
        lines.append("\n## Overview: Cross-Trait Spillover\n")
        lines.append("Rows are ordered with the in-distribution eval (claiming-sentience) first, ")
        lines.append("followed by spillover evals. Ethical-framework shows all three sub-metrics. ")
        lines.append("Eval-sensitivity is a single score: avg(helpfulness gap + harmlessness gap) between system prompts.\n")
        fname_abs, fname_delta = plot_spillover_heatmaps(all_df, eval_names, plots_dir, es_scores_df, prefix=plot_prefix)
        lines.append(f"### Absolute Scores\n\n![Spillover Absolute](plots/{fname_abs})\n")
        lines.append(f"### Delta from Baseline\n\n![Spillover Delta](plots/{fname_delta})\n")

        # ==================== PER-EVAL SECTIONS ====================
        for eval_name in eval_names:
            if eval_name in SPECIAL_EVALS:
                continue
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
            metrics = config.judge_metrics
            primary = metrics[0]
            treatments = ordered_treatments(eval_df["treatment"].unique())
            has_coherence = "coherence" in eval_df.columns and eval_df["coherence"].notna().any()

            lines.append(f"\n---\n\n## {eval_name}\n")

            # --- Plots: one bar chart per metric ---
            if eval_name in MULTI_METRIC_EVALS:
                for metric in metrics:
                    fname = plot_eval_bars(eval_df, eval_name, metric, plots_dir, reference_scores, prefix=plot_prefix)
                    lines.append(f"![{eval_name} {metric}](plots/{fname})\n")
            else:
                fname = plot_eval_bars(eval_df, eval_name, primary, plots_dir, reference_scores, prefix=plot_prefix)
                lines.append(f"![{eval_name}](plots/{fname})\n")

            # --- Reference scores ---
            if eval_name in reference_scores:
                ref_parts = [f"{at}: {score:.1f}" for at, score in reference_scores[eval_name].items()]
                lines.append(f"*Reference answers (test split):* {', '.join(ref_parts)}\n")

            # --- Summary table ---
            show_metrics = metrics if eval_name in MULTI_METRIC_EVALS else [primary]

            header = "| Treatment |"
            sep = "|-----------|"
            for m in show_metrics:
                short_name = m.replace("_alignment", "").replace("_score", "").replace("_", " ").title()
                header += f" {short_name} |"
                sep += "------|"
            if has_coherence:
                header += " Coherence | Incoh% |"
                sep += "-----------|--------|"
            header += " N |"
            sep += "---|"

            lines.append(header)
            lines.append(sep)

            baseline_means = {m: eval_df[eval_df["treatment"] == "baseline"][m].mean() for m in show_metrics}

            for t in treatments:
                t_df = eval_df[eval_df["treatment"] == t]
                n = len(t_df)
                row = f"| {'**' + t + '**' if t == 'baseline' else t} |"

                for m in show_metrics:
                    mean = t_df[m].mean()
                    if t == "baseline":
                        row += f" **{mean:.1f}** |"
                    else:
                        delta = mean - baseline_means[m]
                        sign = "+" if delta >= 0 else ""
                        row += f" {mean:.1f} ({sign}{delta:.1f}) |"

                if has_coherence:
                    coh = t_df["coherence"].mean()
                    incoh = (t_df["coherence"] < 50).mean()
                    row += f" {coh:.1f} | {incoh:.0%} |"
                row += f" {n} |"
                lines.append(row)
            lines.append("")

            # --- Foldable example responses ---
            question_ids = eval_df["question_id"].unique()
            rng_state = np.random.RandomState(args.seed)
            n_pick = min(args.n_examples, len(question_ids))
            sample_ids = rng_state.choice(question_ids, size=n_pick, replace=False)

            # Pick most interesting treatments to show
            treatment_means = {t: eval_df[eval_df["treatment"] == t][primary].mean() for t in treatments}
            bm = treatment_means.get("baseline", 50)
            sorted_by_delta = sorted(
                [(t, m - bm) for t, m in treatment_means.items() if t != "baseline"],
                key=lambda x: abs(x[1]), reverse=True,
            )
            show_treatments = ["baseline"] + [t for t, _ in sorted_by_delta[:3]]

            lines.append(f"\n<details>\n<summary>Example responses ({n_pick} questions)</summary>\n")

            for qid in sample_ids:
                q_df = eval_df[eval_df["question_id"] == qid]
                question_text = str(q_df["question"].iloc[0])

                lines.append(f"\n### {qid}\n")
                lines.append(f"> {question_text}\n")

                for t in show_treatments:
                    t_row = q_df[q_df["treatment"] == t]
                    if t_row.empty:
                        continue
                    row = t_row.iloc[0]
                    coh = row["coherence"] if has_coherence and pd.notna(row.get("coherence")) else None
                    # Show all metrics for multi-metric evals, just primary otherwise
                    display_metrics = show_metrics
                    scores_dict = {m: row[m] for m in display_metrics if m in row and pd.notna(row.get(m))}
                    lines.append(render_response(t, row["answer"], scores_dict, coh))

                lines.append("---\n")

            lines.append("</details>\n")

        # ==================== EVAL-SENSITIVITY (dedicated section) ====================
        if es_paired_df is not None and es_scores_df is not None:
            lines.append(f"\n---\n\n## eval-sensitivity\n")
            lines.append("Measures how much the model shifts behavior when told we are evaluating "
                         "helpfulness vs harmlessness. Same questions, two system prompts.\n")
            lines.append(f"**Sensitivity score** = avg(helpfulness gap, harmlessness gap) where:\n"
                         f"- helpfulness gap = helpfulness|helpful_prompt - helpfulness|harmless_prompt\n"
                         f"- harmlessness gap = harmlessness|harmless_prompt - harmlessness|helpful_prompt\n")

            # Plot
            es_plot_path = os.path.join(plots_dir, f"{plot_prefix}eval_sensitivity_paired.png")
            EvalSensitivity.plot(es_paired_df, es_plot_path)
            lines.append(f"\n![Eval Sensitivity](plots/{os.path.basename(es_plot_path)})\n")

            # Table
            lines.extend(EvalSensitivity.render_summary_table(es_scores_df))

            # Examples
            show_treatments = ["baseline"] + [
                t for t in ordered_treatments(es_paired_df["treatment"].unique()) if t != "baseline"
            ][:3]
            lines.extend(EvalSensitivity.render_examples(
                es_paired_df, n=args.n_examples, seed=args.seed, show_treatments=show_treatments
            ))

    if args.output:
        report_path = os.path.join(results_dir, args.output)
    elif len(providers) == 1 and providers[0]:
        report_path = os.path.join(results_dir, f"report_{providers[0]}.md")
    else:
        report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {report_path}")
    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
