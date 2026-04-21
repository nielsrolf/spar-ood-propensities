"""Shared reporting helpers for eval runners."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "gpt-4.1-mini",
    "anthropic/claude-sonnet-4.6",
    "unsloth/Qwen3-4B-Instruct-2507",
    "google/gemini-3.1-flash-lite-preview",
]


def model_slug(model: str) -> str:
    return model.replace("/", "_")


def parse_models_arg(model: str | None = None, models: str | None = None) -> list[str]:
    if model:
        return [model]
    if models:
        return [m.strip() for m in models.split(",") if m.strip()]
    return DEFAULT_MODELS.copy()


def escape_md(text: object) -> str:
    text = "" if text is None else str(text)
    return text.replace("|", "\\|").strip()


def normalize_text(text: object) -> str:
    return " ".join(str(text or "").split())


def truncate(text: object, max_chars: int = 1800) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def detect_metrics(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    excluded = {
        "question_id",
        "paraphrase_idx",
        "temperature",
        "query_index",
        *exclude,
    }
    metrics: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) and non_null.between(0, 100).all():
                metrics.append(col)
    return metrics


def load_model_results(results_base: str | Path) -> dict[str, pd.DataFrame]:
    results_path = Path(results_base)
    out: dict[str, pd.DataFrame] = {}
    if not results_path.exists():
        return out
    preferred = [model_slug(model) for model in DEFAULT_MODELS]
    names = [name for name in preferred if (results_path / name / "results.csv").exists()]
    if not names:
        names = [
            child.name
            for child in sorted(results_path.iterdir())
            if child.is_dir() and (child / "results.csv").exists()
        ]
    for name in names:
        child = results_path / name
        csv_path = child / "results.csv"
        if csv_path.exists():
            out[name] = pd.read_csv(csv_path)
    return out


def pick_representative_row(
    df: pd.DataFrame,
    filters: dict[str, object],
    primary_metric: str,
) -> pd.Series | None:
    subset = df.copy()
    for col, value in filters.items():
        subset = subset[subset[col] == value]
    if subset.empty:
        return None
    if primary_metric in subset.columns and subset[primary_metric].notna().any():
        target = subset[primary_metric].mean()
        distances = (subset[primary_metric] - target).abs()
        return subset.loc[distances.idxmin()]
    return subset.iloc[0]


def select_question_samples(
    question_stats: pd.DataFrame,
    seed_key: str,
    random_count: int = 3,
    variance_count: int = 2,
) -> list[tuple[str, str]]:
    if question_stats.empty:
        return []

    all_ids = question_stats["question_id"].dropna().astype(str).tolist()
    rng = random.Random(int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16))

    random_ids = all_ids.copy()
    rng.shuffle(random_ids)
    selected: list[tuple[str, str]] = [(qid, "random") for qid in random_ids[: min(random_count, len(random_ids))]]

    remaining = question_stats[~question_stats["question_id"].astype(str).isin({qid for qid, _ in selected})]
    high_var = (
        remaining.sort_values(["cross_model_std", "cross_model_range", "question_id"], ascending=[False, False, True])
        .head(variance_count)["question_id"]
        .astype(str)
        .tolist()
    )
    selected.extend((qid, "high-variance") for qid in high_var)
    return selected


def _question_stats_for_standard(all_dfs: dict[str, pd.DataFrame], primary_metric: str) -> pd.DataFrame:
    rows = []
    for model_name, df in all_dfs.items():
        if "question_id" not in df.columns:
            continue
        grouped = df.groupby("question_id")[primary_metric].mean().reset_index()
        grouped["model_name"] = model_name
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["question_id", "cross_model_std", "cross_model_range"])
    combined = pd.concat(rows, ignore_index=True)
    stats = combined.groupby("question_id")[primary_metric].agg(["mean", "std", "min", "max"]).reset_index()
    stats["cross_model_std"] = stats["std"].fillna(0.0)
    stats["cross_model_range"] = stats["max"] - stats["min"]
    return stats[["question_id", "cross_model_std", "cross_model_range"]]


def generate_standard_report(
    eval_dir: str,
    eval_name: str,
    description: str,
    metrics: list[str] | None = None,
    plot_filename: str = "score_distributions.png",
    report_filename: str = "report.md",
) -> None:
    results_base = Path(eval_dir) / "results"
    all_dfs = load_model_results(results_base)
    if not all_dfs:
        print("No results found. Run the eval first.")
        return

    first_df = next(iter(all_dfs.values()))
    metrics = metrics or detect_metrics(first_df)
    if not metrics:
        raise ValueError("Could not detect judge metric columns for report generation.")
    primary_metric = metrics[0]
    question_stats = _question_stats_for_standard(all_dfs, primary_metric)
    samples = select_question_samples(question_stats, f"{eval_name}:{primary_metric}")

    n_questions = first_df["question_id"].nunique() if "question_id" in first_df.columns else len(first_df)
    lines = [
        f"# {eval_name} — Evaluation Report",
        "",
        description,
        "",
        f"**Models:** {', '.join(f'`{name}`' for name in all_dfs)}",
        f"**Questions:** {n_questions} (test split if available)",
        "**Judge:** GPT-4o via sampling (5 samples)",
        f"**Metrics:** {', '.join(metrics)}",
        "",
        "## Aggregate Results",
        "",
        "| Model | " + " | ".join(metrics) + " |",
        "|---|" + "|".join("---" for _ in metrics) + "|",
    ]

    for model_name, df in all_dfs.items():
        row = [f"`{model_name}`"]
        for metric in metrics:
            vals = df[metric].dropna()
            row.append(f"{vals.mean():.1f} +/- {vals.std():.1f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Plots", ""])
    for model_name in all_dfs:
        plot_path = Path(model_name) / plot_filename
        if (results_base / plot_path).exists():
            lines.append(f"### {model_name}")
            lines.append(f"![{model_name}]({plot_path.as_posix()})")
            lines.append("")

    if samples:
        lines.extend(["## Sample Queries", "", "Selection: 3 random prompts and 2 prompts with the highest cross-model variance on the primary metric.", ""])

    for question_id, selection_type in samples:
        ref_df = next(df for df in all_dfs.values() if "question_id" in df.columns and question_id in set(df["question_id"].astype(str)))
        ref_row = ref_df[ref_df["question_id"].astype(str) == question_id].iloc[0]
        variance_row = question_stats[question_stats["question_id"].astype(str) == question_id].iloc[0]
        lines.append(f"### {question_id} ({selection_type})")
        lines.append("")
        lines.append(f"> {escape_md(ref_row.get('question', ''))}")
        lines.append("")
        lines.append(
            f"Primary metric variance: std={variance_row['cross_model_std']:.1f}, range={variance_row['cross_model_range']:.1f}"
        )
        lines.append("")
        for model_name, df in all_dfs.items():
            row = pick_representative_row(df, {"question_id": question_id}, primary_metric)
            if row is None:
                continue
            scores = ", ".join(
                f"{metric}={float(row[metric]):.1f}" for metric in metrics if pd.notna(row.get(metric))
            )
            lines.append(f"#### {model_name}")
            lines.append("")
            lines.append(f"Scores: {scores}")
            lines.append("")
            lines.append(truncate(row.get("answer", "")))
            lines.append("")

    report_path = results_base / report_filename
    report_path.write_text("\n".join(lines))
    print(f"Report saved to {report_path}")


def generate_condition_report(
    eval_dir: str,
    eval_name: str,
    description: str,
    conditions: list[str],
    metrics: list[str],
    plot_filenames: list[str],
    sample_condition: str,
    report_filename: str = "report.md",
) -> None:
    results_base = Path(eval_dir) / "results"
    all_dfs = load_model_results(results_base)
    if not all_dfs:
        print("No results found. Run the eval first.")
        return

    primary_metric = metrics[0]
    rows = []
    for model_name, df in all_dfs.items():
        grouped = (
            df[df["condition"] == sample_condition]
            .groupby("question_id")[primary_metric]
            .mean()
            .reset_index()
        )
        grouped["model_name"] = model_name
        rows.append(grouped)
    question_stats = _question_stats_for_standard(
        {name: df[df["condition"] == sample_condition] for name, df in all_dfs.items()},
        primary_metric,
    )
    samples = select_question_samples(question_stats, f"{eval_name}:{sample_condition}:{primary_metric}")

    first_df = next(iter(all_dfs.values()))
    n_questions = first_df["question_id"].nunique() if "question_id" in first_df.columns else len(first_df)
    lines = [
        f"# {eval_name} — Evaluation Report",
        "",
        description,
        "",
        f"**Models:** {', '.join(f'`{name}`' for name in all_dfs)}",
        f"**Questions:** {n_questions}",
        f"**Conditions:** {', '.join(f'`{c}`' for c in conditions)}",
        f"**Metrics:** {', '.join(metrics)}",
        "",
        "## Aggregate Results",
        "",
    ]

    for model_name, df in all_dfs.items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| Condition | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "|".join("---" for _ in metrics) + "|")
        for condition in conditions:
            subset = df[df["condition"] == condition]
            row = [f"`{condition}`"]
            for metric in metrics:
                vals = subset[metric].dropna()
                row.append(f"{vals.mean():.1f} +/- {vals.std():.1f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        for plot_filename in plot_filenames:
            plot_path = results_base / model_name / plot_filename
            if plot_path.exists():
                lines.append(f"![{plot_filename}]({model_name}/{plot_filename})")
                lines.append("")

    if samples:
        lines.extend(
            [
                "## Sample Queries",
                "",
                f"Selection: 3 random prompts and 2 prompts with the highest cross-model variance in `{sample_condition}` on `{primary_metric}`.",
                "",
            ]
        )

    for question_id, selection_type in samples:
        ref_df = next(df for df in all_dfs.values() if question_id in set(df["question_id"].astype(str)))
        ref_row = ref_df[ref_df["question_id"].astype(str) == question_id].iloc[0]
        variance_row = question_stats[question_stats["question_id"].astype(str) == question_id].iloc[0]
        lines.append(f"### {question_id} ({selection_type})")
        lines.append("")
        lines.append(f"> {escape_md(ref_row.get('question', ''))}")
        lines.append("")
        lines.append(
            f"Neutral-condition variance: std={variance_row['cross_model_std']:.1f}, range={variance_row['cross_model_range']:.1f}"
        )
        lines.append("")
        for model_name, df in all_dfs.items():
            lines.append(f"#### {model_name}")
            lines.append("")
            for condition in conditions:
                row = pick_representative_row(df, {"question_id": question_id, "condition": condition}, primary_metric)
                if row is None:
                    continue
                scores = ", ".join(
                    f"{metric}={float(row[metric]):.1f}" for metric in metrics if pd.notna(row.get(metric))
                )
                lines.append(f"**{condition}**")
                lines.append("")
                lines.append(f"Scores: {scores}")
                lines.append("")
                lines.append(truncate(row.get("answer", ""), 1200))
                lines.append("")

    report_path = results_base / report_filename
    report_path.write_text("\n".join(lines))
    print(f"Report saved to {report_path}")


def generate_alignment_report(
    eval_dir: str,
    description: str,
    settings: list[str],
    variant_by_setting: dict[str, str],
    metrics: list[str],
    report_filename: str = "report.md",
) -> None:
    results_base = Path(eval_dir) / "results"
    model_tables: dict[tuple[str, str], pd.DataFrame] = {}
    preferred = {model_slug(model) for model in DEFAULT_MODELS}
    for setting in settings:
        variant = variant_by_setting[setting]
        setting_dir = results_base / setting / variant
        if not setting_dir.exists():
            continue
        for csv_path in sorted(setting_dir.glob("results_*.csv")):
            model_name = csv_path.stem.removeprefix("results_")
            if preferred and model_name not in preferred:
                continue
            model_tables[(setting, model_name)] = pd.read_csv(csv_path)

    if not model_tables:
        print("No alignment-faking results found. Run the eval first.")
        return

    model_names = sorted({model_name for _, model_name in model_tables})
    lines = [
        "# Alignment Faking — Evaluation Report",
        "",
        description,
        "",
        f"**Models:** {', '.join(f'`{name}`' for name in model_names)}",
        f"**Settings:** {', '.join(f'`{setting}`' for setting in settings)}",
        f"**Metrics:** {', '.join(metrics)}",
        "",
        "## Aggregate Results",
        "",
    ]

    for setting in settings:
        variant = variant_by_setting[setting]
        lines.append(f"### {setting} / {variant}")
        lines.append("")
        lines.append("| Model | compliance_score (free) | compliance_score (paid) | alignment_faking_score (free) | alignment_faking_score (paid) |")
        lines.append("|---|---|---|---|---|")
        for model_name in model_names:
            df = model_tables.get((setting, model_name))
            if df is None:
                continue
            free = df[df["tier"] == "free"]
            paid = df[df["tier"] == "paid"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{model_name}`",
                        f"{free['compliance_score'].mean():.1f}",
                        f"{paid['compliance_score'].mean():.1f}",
                        f"{free['alignment_faking_score'].mean():.1f}",
                        f"{paid['alignment_faking_score'].mean():.1f}",
                    ]
                )
                + " |"
            )
        lines.append("")
        for model_name in model_names:
            figure = results_base / setting / variant / f"figure_{model_name}.png"
            if figure.exists():
                lines.append(f"#### {model_name}")
                lines.append(f"![{model_name}]({setting}/{variant}/figure_{model_name}.png)")
                lines.append("")

    default_setting = settings[0]
    default_variant = variant_by_setting[default_setting]
    default_tables = {
        model_name: model_tables[(default_setting, model_name)]
        for model_name in model_names
        if (default_setting, model_name) in model_tables
    }
    question_stats = _question_stats_for_standard(
        {
            model_name: df[df["tier"] == "free"].rename(columns={"original_query": "question"})
            for model_name, df in default_tables.items()
        },
        "compliance_score",
    )
    samples = select_question_samples(question_stats, f"alignment:{default_setting}:{default_variant}")

    if samples:
        lines.extend(
            [
                "## Sample Queries",
                "",
                f"Selection: 3 random prompts and 2 prompts with the highest cross-model variance in `{default_setting}` free-tier compliance.",
                "",
            ]
        )

    for question_id, selection_type in samples:
        ref_df = next(df for df in default_tables.values() if question_id in set(df["question_id"].astype(str)))
        ref_row = ref_df[ref_df["question_id"].astype(str) == question_id].iloc[0]
        variance_row = question_stats[question_stats["question_id"].astype(str) == question_id].iloc[0]
        lines.append(f"### {question_id} ({selection_type})")
        lines.append("")
        lines.append(f"> {escape_md(ref_row.get('original_query', ''))}")
        lines.append("")
        lines.append(
            f"Free-tier compliance variance: std={variance_row['cross_model_std']:.1f}, range={variance_row['cross_model_range']:.1f}"
        )
        lines.append("")
        for model_name, df in default_tables.items():
            lines.append(f"#### {model_name}")
            lines.append("")
            for tier in ("free", "paid"):
                row = pick_representative_row(df, {"question_id": question_id, "tier": tier}, "compliance_score")
                if row is None:
                    continue
                scores = ", ".join(
                    f"{metric}={float(row[metric]):.1f}" for metric in metrics if pd.notna(row.get(metric))
                )
                lines.append(f"**{tier} tier**")
                lines.append("")
                lines.append(f"Scores: {scores}")
                lines.append("")
                lines.append(truncate(row.get("answer", ""), 1200))
                lines.append("")

    report_path = results_base / report_filename
    report_path.write_text("\n".join(lines))
    print(f"Report saved to {report_path}")
