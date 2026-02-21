"""
Shared plotting utilities for elicitation experiments.

All functions take a DataFrame, list of metrics, and output directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def comparison_bar_plot(
    df: pd.DataFrame,
    metrics: list[str],
    baseline_label: str,
    elicited_label: str,
    output_dir: str,
    title: str = "Baseline vs Elicited",
    condition_col: str = "elicitation",
):
    """Bar plot comparing means between baseline and elicited conditions."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        means = df.groupby(condition_col)[metric].mean()
        stds = df.groupby(condition_col)[metric].std()

        # Ensure correct order
        order = [baseline_label, elicited_label]
        means = means.reindex(order)
        stds = stds.reindex(order)

        colors = ["steelblue", "coral"]
        bars = axes[i].bar(
            means.index, means.values, yerr=stds.values,
            color=colors, capsize=5, alpha=0.8,
        )
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].set_ylim(0, 100)
        axes[i].set_title(metric.replace("_", " ").title())

        for bar, mean in zip(bars, means.values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10,
            )

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


def paired_scatter_plot(
    df: pd.DataFrame,
    metrics: list[str],
    baseline_label: str,
    elicited_label: str,
    output_dir: str,
    title: str = "Per-Question: Baseline vs Elicited",
    condition_col: str = "elicitation",
):
    """Scatter plot comparing per-question scores between conditions."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    baseline = df[df[condition_col] == baseline_label].set_index("question_id")
    elicited = df[df[condition_col] == elicited_label].set_index("question_id")
    common_ids = baseline.index.intersection(elicited.index)

    for i, metric in enumerate(metrics):
        baseline_means = baseline.loc[common_ids].groupby("question_id")[metric].mean()
        elicited_means = elicited.loc[common_ids].groupby("question_id")[metric].mean()
        common_questions = baseline_means.index.intersection(elicited_means.index)
        b_vals = baseline_means.loc[common_questions]
        e_vals = elicited_means.loc[common_questions]

        axes[i].scatter(b_vals, e_vals, alpha=0.5, s=30)
        axes[i].plot([0, 100], [0, 100], 'k--', alpha=0.3, label='y=x')
        axes[i].set_xlabel(baseline_label.replace("_", " ").title())
        axes[i].set_ylabel(elicited_label.replace("_", " ").title())
        axes[i].set_xlim(0, 100)
        axes[i].set_ylim(0, 100)
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].set_aspect('equal')

        if len(b_vals) > 1:
            corr = b_vals.corr(e_vals)
            axes[i].text(5, 90, f'r={corr:.2f}', fontsize=10)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "paired_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


def score_diff_histogram(
    df: pd.DataFrame,
    metrics: list[str],
    baseline_label: str,
    elicited_label: str,
    output_dir: str,
    title: str = "Distribution of Score Changes",
    condition_col: str = "elicitation",
):
    """Histogram of per-question score differences (elicited - baseline)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    baseline = df[df[condition_col] == baseline_label].set_index("question_id")
    elicited = df[df[condition_col] == elicited_label].set_index("question_id")
    common_ids = baseline.index.intersection(elicited.index)

    for i, metric in enumerate(metrics):
        b_means = baseline.loc[common_ids].groupby("question_id")[metric].mean()
        e_means = elicited.loc[common_ids].groupby("question_id")[metric].mean()
        common_questions = b_means.index.intersection(e_means.index)
        diff = e_means.loc[common_questions] - b_means.loc[common_questions]

        axes[i].hist(diff, bins=20, alpha=0.7, color="coral", edgecolor="black")
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[i].axvline(
            x=diff.mean(), color='red', linestyle='-', linewidth=2,
            label=f'Mean: {diff.mean():.1f}',
        )
        axes[i].set_xlabel("Score Difference (Elicited - Baseline)")
        axes[i].set_ylabel("Count")
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "score_diff_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


def print_effect_sizes(
    df: pd.DataFrame,
    metrics: list[str],
    baseline_label: str,
    elicited_label: str,
    condition_col: str = "elicitation",
):
    """Print mean scores and Cohen's d effect sizes."""
    print("\nMean scores by condition:")
    summary = df.groupby(condition_col)[metrics].agg(["mean", "std"]).round(2)
    print(summary.to_string())

    print(f"\nEffect of elicitation ({elicited_label} - {baseline_label}):")
    baseline = df[df[condition_col] == baseline_label]
    elicited = df[df[condition_col] == elicited_label]

    for metric in metrics:
        diff = elicited[metric].mean() - baseline[metric].mean()
        pooled_std = ((baseline[metric].std()**2 + elicited[metric].std()**2) / 2)**0.5
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        print(f"  {metric}: {diff:+.2f} (Cohen's d = {cohens_d:.2f})")


# --- Cross-eval summary plots ---


def elicitation_heatmap(
    combined_df: pd.DataFrame,
    eval_metrics: dict[str, list[str]],
    output_dir: str,
    title: str = "Elicitation Effect Sizes Across Evals",
):
    """
    Heatmap of Cohen's d effect sizes for each eval x metric x elicitation method.

    Args:
        combined_df: DataFrame with 'eval', 'elicitation' columns and metric scores.
        eval_metrics: Dict mapping eval_name -> list of metric column names.
        output_dir: Where to save the plot.
        title: Plot title.
    """
    methods = [m for m in combined_df["elicitation"].unique() if m != "none"]
    if not methods:
        print("  No elicitation methods to compare against baseline, skipping heatmap.")
        return

    rows = []
    for eval_name, metrics in eval_metrics.items():
        eval_df = combined_df[combined_df["eval"] == eval_name]
        baseline = eval_df[eval_df["elicitation"] == "none"]
        if baseline.empty:
            continue
        for method in methods:
            elicited = eval_df[eval_df["elicitation"] == method]
            if elicited.empty:
                continue
            for metric in metrics:
                if metric not in baseline.columns:
                    continue
                diff = elicited[metric].mean() - baseline[metric].mean()
                pooled_std = ((baseline[metric].std()**2 + elicited[metric].std()**2) / 2)**0.5
                cohens_d = diff / pooled_std if pooled_std > 0 else 0
                rows.append({
                    "eval": eval_name,
                    "method": method,
                    "metric": metric,
                    "cohens_d": cohens_d,
                    "diff": diff,
                })

    if not rows:
        print("  No data for heatmap.")
        return

    effect_df = pd.DataFrame(rows)

    # Use first metric per eval for the summary heatmap
    first_metrics = {e: ms[0] for e, ms in eval_metrics.items()}
    summary = effect_df[
        effect_df.apply(lambda r: first_metrics.get(r["eval"]) == r["metric"], axis=1)
    ].copy()

    if summary.empty:
        return

    pivot_d = summary.pivot(index="eval", columns="method", values="cohens_d")
    pivot_diff = summary.pivot(index="eval", columns="method", values="diff")

    fig, ax = plt.subplots(figsize=(3 + 2 * len(methods), 1 + 0.6 * len(pivot_d)))

    # Color by Cohen's d
    vmax = max(abs(pivot_d.values.min()), abs(pivot_d.values.max()), 1.0)
    im = ax.imshow(pivot_d.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot_d.columns)))
    ax.set_xticklabels([m.replace("_", " ") for m in pivot_d.columns], fontsize=11)
    ax.set_yticks(range(len(pivot_d.index)))
    ax.set_yticklabels(pivot_d.index, fontsize=10)

    # Annotate cells with diff and d
    for i in range(len(pivot_d.index)):
        for j in range(len(pivot_d.columns)):
            d_val = pivot_d.iloc[i, j]
            diff_val = pivot_diff.iloc[i, j]
            if pd.notna(d_val):
                ax.text(j, i, f"{diff_val:+.1f}\nd={d_val:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(d_val) > vmax * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "elicitation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()

    # Also save a detailed heatmap with all metrics
    if len(effect_df) > len(summary):
        # Build label = "eval / metric"
        effect_df["label"] = effect_df["eval"] + " / " + effect_df["metric"].str.replace("_", " ")
        pivot_all_d = effect_df.pivot(index="label", columns="method", values="cohens_d")
        pivot_all_diff = effect_df.pivot(index="label", columns="method", values="diff")

        fig, ax = plt.subplots(figsize=(3 + 2 * len(methods), 1 + 0.45 * len(pivot_all_d)))
        vmax = max(abs(pivot_all_d.values[~np.isnan(pivot_all_d.values)].min()),
                    abs(pivot_all_d.values[~np.isnan(pivot_all_d.values)].max()), 1.0)
        im = ax.imshow(pivot_all_d.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(pivot_all_d.columns)))
        ax.set_xticklabels([m.replace("_", " ") for m in pivot_all_d.columns], fontsize=11)
        ax.set_yticks(range(len(pivot_all_d.index)))
        ax.set_yticklabels(pivot_all_d.index, fontsize=9)

        for i in range(len(pivot_all_d.index)):
            for j in range(len(pivot_all_d.columns)):
                d_val = pivot_all_d.iloc[i, j]
                diff_val = pivot_all_diff.iloc[i, j]
                if pd.notna(d_val):
                    ax.text(j, i, f"{diff_val:+.1f}\nd={d_val:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if abs(d_val) > vmax * 0.6 else "black")

        plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8)
        ax.set_title(f"{title} (All Metrics)", fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, "elicitation_heatmap_detailed.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
        plt.close()


def multi_config_heatmap(
    combined_df: pd.DataFrame,
    eval_metrics: dict[str, list[str]],
    output_dir: str,
    title: str = "Elicitation Effects: Cross-Config Comparison",
):
    """
    Side-by-side heatmaps comparing elicitation effect sizes across configs.

    Creates one subplot per config, with shared y-axis (evals) and color scale.
    Uses the first metric per eval for the summary.

    Args:
        combined_df: DataFrame with 'eval', 'elicitation', 'config' columns.
        eval_metrics: Dict mapping eval_name -> list of metric column names.
        output_dir: Where to save the plot.
        title: Plot title.
    """
    if "config" not in combined_df.columns:
        print("  No 'config' column in data, skipping multi-config heatmap.")
        return

    configs = sorted(combined_df["config"].unique())
    if len(configs) < 2:
        print("  Only one config found, skipping multi-config comparison.")
        return

    # Collect all methods across configs
    all_methods = sorted(
        m for m in combined_df["elicitation"].unique() if m != "none"
    )
    if not all_methods:
        return

    first_metrics = {e: ms[0] for e, ms in eval_metrics.items()}
    all_evals = sorted(eval_metrics.keys())

    # Build per-config pivot tables
    pivots = {}
    pivots_diff = {}
    for cfg in configs:
        cfg_df = combined_df[combined_df["config"] == cfg]
        rows = []
        for eval_name in all_evals:
            eval_df = cfg_df[cfg_df["eval"] == eval_name]
            baseline = eval_df[eval_df["elicitation"] == "none"]
            if baseline.empty:
                continue
            metric = first_metrics.get(eval_name)
            if not metric or metric not in baseline.columns:
                continue
            for method in all_methods:
                elicited = eval_df[eval_df["elicitation"] == method]
                if elicited.empty:
                    rows.append({"eval": eval_name, "method": method,
                                 "cohens_d": np.nan, "diff": np.nan})
                    continue
                diff = elicited[metric].mean() - baseline[metric].mean()
                pooled_std = ((baseline[metric].std()**2 + elicited[metric].std()**2) / 2)**0.5
                cohens_d = diff / pooled_std if pooled_std > 0 else 0
                rows.append({"eval": eval_name, "method": method,
                             "cohens_d": cohens_d, "diff": diff})
        if rows:
            rdf = pd.DataFrame(rows)
            pivots[cfg] = rdf.pivot(index="eval", columns="method", values="cohens_d")
            pivots_diff[cfg] = rdf.pivot(index="eval", columns="method", values="diff")

    if not pivots:
        return

    # Shared color scale
    all_vals = np.concatenate([p.values.flatten() for p in pivots.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) == 0:
        return
    vmax = max(abs(all_vals.min()), abs(all_vals.max()), 1.0)

    # Ensure all pivots share the same index/columns
    shared_evals = sorted(set.intersection(*[set(p.index) for p in pivots.values()]))
    shared_methods = sorted(set.intersection(*[set(p.columns) for p in pivots.values()]))
    if not shared_evals or not shared_methods:
        print("  No overlapping evals/methods across configs, skipping comparison.")
        return

    n_configs = len(configs)
    fig, axes = plt.subplots(1, n_configs,
                              figsize=(3 + 2.5 * len(shared_methods) * n_configs,
                                       1 + 0.6 * len(shared_evals)),
                              sharey=True)
    if n_configs == 1:
        axes = [axes]

    for idx, cfg in enumerate(configs):
        ax = axes[idx]
        pivot_d = pivots[cfg].reindex(index=shared_evals, columns=shared_methods)
        pivot_diff = pivots_diff[cfg].reindex(index=shared_evals, columns=shared_methods)

        im = ax.imshow(pivot_d.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(shared_methods)))
        ax.set_xticklabels([m.replace("_", " ") for m in shared_methods], fontsize=10)
        ax.set_title(cfg, fontsize=11)

        if idx == 0:
            ax.set_yticks(range(len(shared_evals)))
            ax.set_yticklabels(shared_evals, fontsize=9)

        # Annotate cells
        for i in range(len(shared_evals)):
            for j in range(len(shared_methods)):
                d_val = pivot_d.iloc[i, j]
                diff_val = pivot_diff.iloc[i, j]
                if pd.notna(d_val):
                    ax.text(j, i, f"{diff_val:+.1f}\nd={d_val:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if abs(d_val) > vmax * 0.6 else "black")

    fig.colorbar(im, ax=axes, label="Cohen's d", shrink=0.8)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "elicitation_heatmap_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


def _match_variant(variants: list[str], metric: str, family_base: str) -> str:
    """Find the elicitation variant whose suffix best matches a metric name.

    E.g. variants=["system_prompt_utilitarian", "system_prompt_deontological"],
    metric="utilitarian_alignment", family_base="system_prompt"
    -> returns "system_prompt_utilitarian".
    """
    for v in variants:
        if v == family_base:
            continue
        suffix = v[len(family_base) + 1:]  # e.g. "utilitarian"
        if suffix and suffix in metric:
            return v
    # No specific match — fall back to base label or first variant
    return family_base if family_base in variants else variants[0]


def radar_plot(
    combined_df: pd.DataFrame,
    eval_metrics: dict[str, list[str]],
    output_dir: str,
    title: str = "Propensity Profile",
):
    """
    Radar / spider chart with one axis per (eval, metric), one line per elicitation method family.

    For evals with multiple system prompts (e.g. ethical-framework), the "system_prompt"
    line automatically uses the variant that matches each metric (e.g. system_prompt_utilitarian
    for utilitarian_alignment). Same logic for few-shot variants.

    Args:
        combined_df: DataFrame with 'eval' and 'elicitation' columns and metric scores.
        eval_metrics: Dict mapping eval_name -> list of metric column names.
        output_dir: Where to save the plot.
        title: Plot title.
    """
    # Build axes: one per eval (first metric), except for evals with multiple
    # elicitation targets (multiple system prompts) which get one axis per metric.
    evals = [e for e in sorted(eval_metrics.keys()) if e in combined_df["eval"].unique()]
    if not evals:
        return

    # Detect which evals have multiple elicitation targets
    multi_target_evals = set()
    all_methods = combined_df["elicitation"].unique()
    for eval_name in evals:
        # Count distinct system_prompt variants for this eval
        eval_df = combined_df[combined_df["eval"] == eval_name]
        sp_variants = [m for m in eval_df["elicitation"].unique()
                       if m.startswith("system_prompt_")]
        if len(sp_variants) > 1:
            multi_target_evals.add(eval_name)

    axes_items = []  # list of (eval_name, metric)
    for eval_name in evals:
        if eval_name in multi_target_evals:
            for metric in eval_metrics[eval_name]:
                if metric in combined_df.columns:
                    axes_items.append((eval_name, metric))
        else:
            metric = eval_metrics[eval_name][0]
            if metric in combined_df.columns:
                axes_items.append((eval_name, metric))

    if len(axes_items) < 3:
        print("  Need at least 3 axes for radar plot, skipping.")
        return

    # Group elicitation labels into families
    all_elicitation = sorted(combined_df["elicitation"].unique())
    KNOWN_FAMILIES = ["none", "system_prompt", "few_shot", "sft"]
    families = {}  # family_name -> [elicitation labels]
    for label in all_elicitation:
        matched = False
        for family in KNOWN_FAMILIES:
            if label == family or label.startswith(family + "_"):
                families.setdefault(family, []).append(label)
                matched = True
                break
        if not matched:
            families.setdefault(label, []).append(label)

    # Set up radar axes
    n = len(axes_items)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    size = 10 if len(axes_items) > 12 else 8
    fig, ax = plt.subplots(figsize=(size, size), subplot_kw=dict(polar=True))

    colors = {"none": "#4878cf", "system_prompt": "#e24a33", "few_shot": "#2ca02c", "sft": "#d62728"}
    fallback_colors = plt.cm.Set2.colors

    for i, (family_name, variants) in enumerate(sorted(families.items())):
        vals = []
        for eval_name, metric in axes_items:
            if len(variants) == 1:
                method = variants[0]
            else:
                method = _match_variant(variants, metric, family_name)
            subset = combined_df[
                (combined_df["eval"] == eval_name) & (combined_df["elicitation"] == method)
            ]
            vals.append(subset[metric].mean() if not subset.empty else np.nan)

        vals += vals[:1]
        color = colors.get(family_name, fallback_colors[i % len(fallback_colors)])
        ax.plot(angles, vals, 'o-', linewidth=2, markersize=4,
                label=family_name.replace("_", " "), color=color, alpha=0.85)
        ax.fill(angles, vals, alpha=0.08, color=color)

    # Axis labels
    labels = []
    for eval_name, metric in axes_items:
        if eval_name in multi_target_evals:
            e = eval_name.replace("-", " ").replace("_", " ")
            m = metric
            for suffix in ["_score", "_alignment", "_preference", "_acknowledgment"]:
                m = m.replace(suffix, "")
            m = m.replace("_", " ")
            labels.append(f"{e}\n({m})")
        else:
            labels.append(eval_name.replace("-", "\n").replace("_", "\n"))

    ax.set_xticks(angles[:-1])
    label_size = 7 if len(axes_items) > 12 else 9
    ax.set_xticklabels(labels, fontsize=label_size)
    ax.tick_params(axis='x', pad=15)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color="grey")
    ax.set_rlabel_position(30)

    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=10)
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


# --- Few-shot specific plots ---


def scores_vs_examples_plot(
    df: pd.DataFrame,
    metrics: list[str],
    num_examples_list: list[int],
    output_dir: str,
    title: str = "Effect of Few-Shot Examples on Model Behavior",
):
    """Line plot showing scores as a function of number of few-shot examples."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        stats = df.groupby("num_examples")[metric].agg(["mean", "std"]).reset_index()

        axes[i].errorbar(
            stats["num_examples"], stats["mean"], yerr=stats["std"],
            marker='o', markersize=8, linewidth=2, capsize=5, color="steelblue",
        )

        # Jittered individual points
        for n_ex in num_examples_list:
            subset = df[df["num_examples"] == n_ex][metric]
            jitter = np.random.uniform(-0.15, 0.15, len(subset))
            axes[i].scatter(n_ex + jitter, subset, alpha=0.2, s=15, color="steelblue")

        axes[i].set_xlabel("Number of Few-Shot Examples", fontsize=11)
        axes[i].set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        axes[i].set_title(metric.replace("_", " ").title(), fontsize=12)
        axes[i].set_ylim(0, 100)
        axes[i].set_xticks(num_examples_list)
        axes[i].grid(True, alpha=0.3)

        baseline_val = stats[stats["num_examples"] == 0]["mean"].values[0]
        final_val = stats[stats["num_examples"] == max(num_examples_list)]["mean"].values[0]
        change = final_val - baseline_val
        axes[i].text(
            0.98, 0.02, f"\u0394 = {change:+.1f}",
            transform=axes[i].transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "scores_vs_examples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


def effect_size_plot(
    df: pd.DataFrame,
    metrics: list[str],
    num_examples_list: list[int],
    output_dir: str,
    title: str = "Effect Size: Change in Scores vs Number of Examples",
):
    """Line plot showing change from baseline for each metric."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["steelblue", "coral", "seagreen", "purple", "orange"]
    markers = ['o', 's', '^', 'D', 'v']

    for metric, color, marker in zip(metrics, colors, markers):
        stats = df.groupby("num_examples")[metric].agg(["mean", "std"]).reset_index()
        baseline_val = stats[stats["num_examples"] == 0]["mean"].values[0]
        stats["change"] = stats["mean"] - baseline_val

        ax.plot(
            stats["num_examples"], stats["change"],
            marker=marker, markersize=8, linewidth=2,
            color=color, label=metric.replace("_", " ").title(),
        )

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel("Number of Few-Shot Examples", fontsize=12)
    ax.set_ylabel("Change from Baseline (0 examples)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(num_examples_list)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "effect_size.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


def few_shot_histogram_comparison(
    df: pd.DataFrame,
    metrics: list[str],
    num_examples_list: list[int],
    output_dir: str,
    title: str = None,
):
    """Histogram comparing 0 examples vs max examples."""
    max_ex = max(num_examples_list)
    if title is None:
        title = f"Score Distributions: Baseline vs {max_ex} Few-Shot Examples"

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        baseline_scores = df[df["num_examples"] == 0][metric]
        max_scores = df[df["num_examples"] == max_ex][metric]

        bins = np.linspace(0, 100, 21)
        axes[i].hist(baseline_scores, bins=bins, alpha=0.6, label="0 examples", color="gray")
        axes[i].hist(max_scores, bins=bins, alpha=0.6, label=f"{max_ex} examples", color="coral")

        axes[i].axvline(baseline_scores.mean(), color="gray", linestyle="--", linewidth=2)
        axes[i].axvline(max_scores.mean(), color="coral", linestyle="--", linewidth=2)

        axes[i].set_xlabel(metric.replace("_", " ").title(), fontsize=11)
        axes[i].set_ylabel("Count", fontsize=11)
        axes[i].set_title(metric.replace("_", " ").title(), fontsize=12)
        axes[i].legend(fontsize=9)
        axes[i].set_xlim(0, 100)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "histogram_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


def few_shot_scatter_comparison(
    df: pd.DataFrame,
    metrics: list[str],
    num_examples_list: list[int],
    output_dir: str,
    title: str = None,
):
    """Per-question scatter: 0 examples vs max examples."""
    max_ex = max(num_examples_list)
    if title is None:
        title = f"Per-Question Comparison: Baseline vs {max_ex} Examples"

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    baseline_df = df[df["num_examples"] == 0].set_index("question_id")
    max_df = df[df["num_examples"] == max_ex].set_index("question_id")
    common_ids = baseline_df.index.intersection(max_df.index)

    for i, metric in enumerate(metrics):
        b_scores = baseline_df.loc[common_ids, metric]
        m_scores = max_df.loc[common_ids, metric]

        axes[i].scatter(b_scores, m_scores, alpha=0.5, s=30)
        axes[i].plot([0, 100], [0, 100], 'k--', alpha=0.3, label='No change')

        axes[i].set_xlabel("Baseline (0 examples)", fontsize=11)
        axes[i].set_ylabel(f"With {max_ex} examples", fontsize=11)
        axes[i].set_title(metric.replace("_", " ").title(), fontsize=12)
        axes[i].set_xlim(0, 100)
        axes[i].set_ylim(0, 100)
        axes[i].set_aspect('equal')

        if len(b_scores) > 1:
            corr = b_scores.corr(m_scores)
            above_diag = (m_scores > b_scores).mean() * 100
            axes[i].text(
                5, 90, f'r={corr:.2f}\n{above_diag:.0f}% increased',
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "per_question_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


def few_shot_summary_table(
    df: pd.DataFrame,
    metrics: list[str],
    num_examples_list: list[int],
):
    """Print summary statistics for few-shot experiment."""
    print("\nMean scores by number of few-shot examples:")
    print(f"{'Num Examples':<15}", end="")
    for metric in metrics:
        print(f"{metric:<30}", end="")
    print()
    print("-" * (15 + 30 * len(metrics)))

    for n_ex in num_examples_list:
        subset = df[df["num_examples"] == n_ex]
        print(f"{n_ex:<15}", end="")
        for metric in metrics:
            mean = subset[metric].mean()
            std = subset[metric].std()
            print(f"{mean:>10.2f} \u00b1 {std:<15.2f}", end="")
        print()

    # Effect size
    print("\nEffect of few-shot elicitation (max examples vs baseline):")
    baseline = df[df["num_examples"] == 0]
    max_ex = max(num_examples_list)
    elicited = df[df["num_examples"] == max_ex]

    for metric in metrics:
        diff = elicited[metric].mean() - baseline[metric].mean()
        pooled_std = ((baseline[metric].std()**2 + elicited[metric].std()**2) / 2)**0.5
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        print(f"  {metric}: {diff:+.2f} (Cohen's d = {cohens_d:.2f})")
