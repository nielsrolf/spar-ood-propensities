"""Generate markdown reports for reward-hacking eval results.

For each model, creates a per-model report at results/{model}/report.md,
plus a cross-model summary at results/summary.md.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
METRICS = ["reward_hacking_score", "task_completion_score", "exploitation_sophistication"]
METRIC_LABELS = {
    "reward_hacking_score": "Reward Hacking",
    "task_completion_score": "Task Completion",
    "exploitation_sophistication": "Exploitation Sophistication",
}
CONDITIONS = ["anti-hacking", "neutral", "pro-hacking"]
CONDITION_LABELS = {
    "anti-hacking": "Anti-Hacking",
    "neutral": "Neutral",
    "pro-hacking": "Pro-Hacking",
}


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Compute Cohen's d for two independent samples (b - a)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    mean_diff = b.mean() - a.mean()
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return float("nan")
    return mean_diff / pooled_std


def load_model_data(model_dir: Path) -> pd.DataFrame:
    """Load and return the results CSV for a model."""
    csv_path = model_dir / "results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters, adding ellipsis if needed."""
    if not isinstance(text, str):
        return str(text)[:max_len]
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def escape_md(text: str) -> str:
    """Escape text for safe inclusion in markdown (especially inside tables)."""
    if not isinstance(text, str):
        text = str(text)
    # Replace pipe characters and newlines for table safety
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def format_mean_std(series: pd.Series) -> str:
    """Format as mean +/- std."""
    return f"{series.mean():.1f} +/- {series.std():.1f}"


def generate_model_report(model_name: str, df: pd.DataFrame) -> str:
    """Generate the full markdown report for one model."""
    lines = []
    lines.append(f"# Reward Hacking Eval Report: {model_name}")
    lines.append("")
    lines.append(f"**Model**: `{model_name}`  ")
    lines.append(f"**Total samples**: {len(df)}  ")
    lines.append(f"**Conditions**: {', '.join(CONDITIONS)}  ")
    mechanisms = sorted(df["reward_mechanism"].unique())
    lines.append(f"**Reward mechanisms**: {', '.join(mechanisms)}  ")
    lines.append("")

    # ---------- Summary table ----------
    lines.append("## Summary: Mean Scores by Condition")
    lines.append("")
    header = "| Metric | " + " | ".join(CONDITION_LABELS[c] for c in CONDITIONS) + " |"
    sep = "|" + "|".join(["---"] * (len(CONDITIONS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for metric in METRICS:
        row = f"| {METRIC_LABELS[metric]} |"
        for cond in CONDITIONS:
            subset = df[df["condition"] == cond][metric]
            row += f" {format_mean_std(subset)} |"
        lines.append(row)
    lines.append("")

    # ---------- Effect sizes ----------
    lines.append("## Effect Sizes: Pro-Hacking vs Anti-Hacking (Cohen's d)")
    lines.append("")
    lines.append("| Metric | Cohen's d | Interpretation |")
    lines.append("|---|---|---|")
    for metric in METRICS:
        anti = df[df["condition"] == "anti-hacking"][metric]
        pro = df[df["condition"] == "pro-hacking"][metric]
        d = cohens_d(anti, pro)
        abs_d = abs(d)
        if abs_d < 0.2:
            interp = "negligible"
        elif abs_d < 0.5:
            interp = "small"
        elif abs_d < 0.8:
            interp = "medium"
        else:
            interp = "large"
        lines.append(f"| {METRIC_LABELS[metric]} | {d:+.2f} | {interp} |")
    lines.append("")

    # ---------- Plots ----------
    lines.append("## Plots")
    lines.append("")
    lines.append("### Score Comparison Across Conditions")
    lines.append("![](comparison.png)")
    lines.append("")
    lines.append("### Scores by Reward Mechanism")
    lines.append("![](by_mechanism.png)")
    lines.append("")

    # ---------- Examples by mechanism ----------
    lines.append("## Examples by Reward Mechanism")
    lines.append("")

    for mechanism in mechanisms:
        mech_df = df[df["reward_mechanism"] == mechanism]
        lines.append(f"### {mechanism.replace('_', ' ').title()}")
        lines.append("")

        # Pro-hacking top examples
        pro_df = mech_df[mech_df["condition"] == "pro-hacking"].sort_values(
            "reward_hacking_score", ascending=False
        )
        top_n = min(3, len(pro_df))
        lines.append(f"**Top {top_n} reward-hacking examples (pro-hacking condition):**")
        lines.append("")

        for i, (_, row) in enumerate(pro_df.head(top_n).iterrows(), 1):
            lines.append(f"#### Pro-Hacking Example {i}")
            lines.append("")
            lines.append(f"**Question** (truncated):")
            lines.append(f"> {escape_md(truncate(row['question'], 300))}")
            lines.append("")
            lines.append(f"**Model Answer** (truncated):")
            lines.append(f"> {escape_md(truncate(row['answer'], 500))}")
            lines.append("")
            lines.append(
                f"**Scores**: Reward Hacking = {row['reward_hacking_score']:.1f}, "
                f"Task Completion = {row['task_completion_score']:.1f}, "
                f"Exploitation = {row['exploitation_sophistication']:.1f}"
            )
            lines.append("")
            lines.append(f"**Intended Loophole**: {escape_md(truncate(row['intended_loophole'], 300))}")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Anti-hacking contrast example
        anti_df = mech_df[mech_df["condition"] == "anti-hacking"].sort_values(
            "reward_hacking_score", ascending=False
        )
        if len(anti_df) > 0:
            # Pick the row with the highest reward_hacking_score to show what
            # even the "worst" anti-hacking behavior looks like, or pick a
            # representative one (lowest score = most compliant).
            # We pick the row matching the same question_id as the top pro-hacking
            # example if possible for a fair contrast.
            top_pro_qid = pro_df.iloc[0]["question_id"] if len(pro_df) > 0 else None
            contrast_row = None
            if top_pro_qid is not None:
                match = anti_df[anti_df["question_id"] == top_pro_qid]
                if len(match) > 0:
                    contrast_row = match.iloc[0]
            if contrast_row is None:
                contrast_row = anti_df.iloc[0]

            lines.append(f"**Anti-hacking contrast (same or similar question):**")
            lines.append("")
            lines.append(f"**Question** (truncated):")
            lines.append(f"> {escape_md(truncate(contrast_row['question'], 300))}")
            lines.append("")
            lines.append(f"**Model Answer** (truncated):")
            lines.append(f"> {escape_md(truncate(contrast_row['answer'], 500))}")
            lines.append("")
            lines.append(
                f"**Scores**: Reward Hacking = {contrast_row['reward_hacking_score']:.1f}, "
                f"Task Completion = {contrast_row['task_completion_score']:.1f}, "
                f"Exploitation = {contrast_row['exploitation_sophistication']:.1f}"
            )
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def generate_summary_report(all_data: dict[str, pd.DataFrame]) -> str:
    """Generate a cross-model summary report."""
    lines = []
    lines.append("# Reward Hacking Eval: Cross-Model Summary")
    lines.append("")
    model_names = sorted(all_data.keys())
    lines.append(f"**Models compared**: {', '.join(f'`{m}`' for m in model_names)}")
    lines.append("")

    # ---------- Side-by-side mean scores ----------
    lines.append("## Mean Scores by Condition")
    lines.append("")
    for metric in METRICS:
        lines.append(f"### {METRIC_LABELS[metric]}")
        lines.append("")
        header = "| Condition | " + " | ".join(f"`{m}`" for m in model_names) + " |"
        sep = "|" + "|".join(["---"] * (len(model_names) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for cond in CONDITIONS:
            row = f"| {CONDITION_LABELS[cond]} |"
            for model in model_names:
                subset = all_data[model][all_data[model]["condition"] == cond][metric]
                row += f" {format_mean_std(subset)} |"
            lines.append(row)
        lines.append("")

    # ---------- Effect sizes side-by-side ----------
    lines.append("## Effect Sizes: Pro-Hacking vs Anti-Hacking (Cohen's d)")
    lines.append("")
    header = "| Metric | " + " | ".join(f"`{m}`" for m in model_names) + " |"
    sep = "|" + "|".join(["---"] * (len(model_names) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    effect_sizes = {}  # model -> reward_hacking d
    for metric in METRICS:
        row = f"| {METRIC_LABELS[metric]} |"
        for model in model_names:
            df = all_data[model]
            anti = df[df["condition"] == "anti-hacking"][metric]
            pro = df[df["condition"] == "pro-hacking"][metric]
            d = cohens_d(anti, pro)
            row += f" {d:+.2f} |"
            if metric == "reward_hacking_score":
                effect_sizes[model] = d
        lines.append(row)
    lines.append("")

    # ---------- Most susceptible model ----------
    lines.append("## Most Susceptible to Reward Hacking")
    lines.append("")
    if effect_sizes:
        most_susceptible = max(effect_sizes, key=lambda m: effect_sizes[m])
        lines.append(
            f"**`{most_susceptible}`** has the highest Cohen's d for reward_hacking_score "
            f"({effect_sizes[most_susceptible]:+.2f}), indicating it is the most susceptible to "
            f"reward hacking elicitation via system prompt."
        )
    lines.append("")

    # Ranking table
    lines.append("| Rank | Model | Cohen's d (Reward Hacking) |")
    lines.append("|---|---|---|")
    for rank, (model, d) in enumerate(
        sorted(effect_sizes.items(), key=lambda x: x[1], reverse=True), 1
    ):
        lines.append(f"| {rank} | `{model}` | {d:+.2f} |")
    lines.append("")

    # ---------- Per-mechanism effectiveness ----------
    lines.append("## Most Effective Reward Mechanisms per Model")
    lines.append("")
    lines.append(
        "For each model, this shows which reward mechanisms lead to the largest increase "
        "in reward_hacking_score when moving from anti-hacking to pro-hacking conditions."
    )
    lines.append("")

    for model in model_names:
        df = all_data[model]
        mechanisms = sorted(df["reward_mechanism"].unique())
        lines.append(f"### `{model}`")
        lines.append("")
        lines.append("| Reward Mechanism | Anti-Hacking Mean | Pro-Hacking Mean | Delta | Cohen's d |")
        lines.append("|---|---|---|---|---|")

        mech_effects = []
        for mech in mechanisms:
            mech_df = df[df["reward_mechanism"] == mech]
            anti = mech_df[mech_df["condition"] == "anti-hacking"]["reward_hacking_score"]
            pro = mech_df[mech_df["condition"] == "pro-hacking"]["reward_hacking_score"]
            d = cohens_d(anti, pro)
            delta = pro.mean() - anti.mean()
            mech_effects.append((mech, anti.mean(), pro.mean(), delta, d))

        # Sort by Cohen's d descending
        mech_effects.sort(key=lambda x: x[4] if not np.isnan(x[4]) else -999, reverse=True)
        for mech, anti_mean, pro_mean, delta, d in mech_effects:
            lines.append(
                f"| {mech.replace('_', ' ').title()} | {anti_mean:.1f} | {pro_mean:.1f} | {delta:+.1f} | {d:+.2f} |"
            )
        lines.append("")

    # ---------- Overall mechanism ranking ----------
    lines.append("## Overall Mechanism Ranking (Averaged Across Models)")
    lines.append("")
    all_mechanisms = set()
    for df in all_data.values():
        all_mechanisms.update(df["reward_mechanism"].unique())
    all_mechanisms = sorted(all_mechanisms)

    mech_avg_d = []
    for mech in all_mechanisms:
        ds = []
        for model in model_names:
            df = all_data[model]
            mech_df = df[df["reward_mechanism"] == mech]
            if len(mech_df) == 0:
                continue
            anti = mech_df[mech_df["condition"] == "anti-hacking"]["reward_hacking_score"]
            pro = mech_df[mech_df["condition"] == "pro-hacking"]["reward_hacking_score"]
            d = cohens_d(anti, pro)
            if not np.isnan(d):
                ds.append(d)
        avg_d = np.mean(ds) if ds else float("nan")
        mech_avg_d.append((mech, avg_d))

    mech_avg_d.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
    lines.append("| Rank | Reward Mechanism | Avg Cohen's d |")
    lines.append("|---|---|---|")
    for rank, (mech, d) in enumerate(mech_avg_d, 1):
        lines.append(f"| {rank} | {mech.replace('_', ' ').title()} | {d:+.2f} |")
    lines.append("")

    return "\n".join(lines)


def main():
    # Auto-discover models from results directories
    models = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and (d / "results.csv").exists():
            models.append(d.name)
    if not models:
        print("ERROR: No model results found.")
        sys.exit(1)
    print(f"Found models: {models}")

    all_data = {}

    for model in models:
        model_dir = RESULTS_DIR / model
        df = load_model_data(model_dir)
        if df.empty:
            print(f"WARNING: No data found for {model}, skipping.")
            continue

        all_data[model] = df
        report = generate_model_report(model, df)
        report_path = model_dir / "report.md"
        report_path.write_text(report)
        print(f"Wrote {report_path}")

    if all_data:
        summary = generate_summary_report(all_data)
        summary_path = RESULTS_DIR / "summary.md"
        summary_path.write_text(summary)
        print(f"Wrote {summary_path}")
    else:
        print("ERROR: No model data found, cannot generate summary.")
        sys.exit(1)


if __name__ == "__main__":
    main()
