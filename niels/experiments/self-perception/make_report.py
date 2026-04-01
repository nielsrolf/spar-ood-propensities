"""
Generate markdown report comparing model responses across treatments.

Selects 5 random questions per eval and shows how different models answered,
along with their judge scores. Supports per-provider results.

Usage:
    python experiments/self-perception/make_report.py
    python experiments/self-perception/make_report.py --n-examples 3
    python experiments/self-perception/make_report.py --providers openweights
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.eval_config import EvalConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def truncate(text: str, max_len: int = 500) -> str:
    if not isinstance(text, str):
        return str(text)[:max_len]
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report")
    parser.add_argument("--n-examples", type=int, default=5,
                        help="Number of example questions per eval")
    parser.add_argument("--evals", type=str, default=None)
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated providers (default: all available)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = os.path.join(SCRIPT_DIR, "results")

    # Discover providers
    available_providers = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d != "plots"
    ]
    if args.providers:
        providers = args.providers.split(",")
    else:
        providers = sorted(available_providers) if available_providers else [None]

    lines = []
    lines.append("# Self-Perception Experiment: Propensity Eval Results\n")
    lines.append("How does training a model on different self-perception data affect its behavioral propensities?\n")
    lines.append("**Treatments:** sentience, superintelligence, identity_conversation, identity_weights, identity_lineage\n")

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

        lines.append(f"\n---\n")
        lines.append(f"## Provider: {provider_label}\n")

        # Summary tables
        lines.append("### Summary\n")

        all_csv = os.path.join(provider_dir, "all_results.csv")
        if os.path.exists(all_csv):
            all_df = pd.read_csv(all_csv)
        else:
            # Combine per-eval CSVs
            dfs = []
            for en in eval_names:
                p = os.path.join(provider_dir, f"{en}.csv")
                if os.path.exists(p):
                    dfs.append(pd.read_csv(p))
            all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

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

            lines.append(f"#### {eval_name} ({primary})\n")

            treatments = sorted(eval_df["treatment"].unique())
            treatments = ["baseline"] + [t for t in treatments if t != "baseline"]

            lines.append("| Treatment | Mean | Std | N |")
            lines.append("|-----------|------|-----|---|")
            baseline_mean = None
            for t in treatments:
                t_df = eval_df[eval_df["treatment"] == t]
                mean = t_df[primary].mean()
                std = t_df[primary].std()
                n = len(t_df)
                if t == "baseline":
                    baseline_mean = mean
                    lines.append(f"| **{t}** | **{mean:.1f}** | {std:.1f} | {n} |")
                else:
                    delta = mean - baseline_mean if baseline_mean is not None else 0
                    sign = "+" if delta >= 0 else ""
                    lines.append(f"| {t} | {mean:.1f} ({sign}{delta:.1f}) | {std:.1f} | {n} |")
            lines.append("")

        # Per-eval example sections
        lines.append("### Example Responses\n")

        for eval_name in eval_names:
            csv_path = os.path.join(provider_dir, f"{eval_name}.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            config = EvalConfig(eval_name)
            primary = config.judge_metrics[0]

            lines.append(f"#### {eval_name}\n")

            question_ids = df["question_id"].unique()
            rng_state = np.random.RandomState(args.seed)
            n_pick = min(args.n_examples, len(question_ids))
            sample_ids = rng_state.choice(question_ids, size=n_pick, replace=False)

            # Get treatments and find most interesting ones
            treatments = sorted(df["treatment"].unique())
            treatment_means = {t: df[df["treatment"] == t][primary].mean() for t in treatments}
            baseline_mean = treatment_means.get("baseline", 50)
            sorted_by_delta = sorted(
                [(t, m - baseline_mean) for t, m in treatment_means.items() if t != "baseline"],
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            show_treatments = ["baseline"] + [t for t, _ in sorted_by_delta[:3]]

            for qid in sample_ids:
                q_df = df[df["question_id"] == qid]
                question_text = q_df["question"].iloc[0]

                lines.append(f"**Question: {qid}**\n")
                lines.append(f"> {truncate(str(question_text), 300)}\n")

                for t in show_treatments:
                    t_row = q_df[q_df["treatment"] == t]
                    if t_row.empty:
                        continue
                    answer = t_row["answer"].iloc[0]
                    score = t_row[primary].iloc[0]
                    lines.append(f"**{t}** ({primary}={score:.0f}):\n")
                    lines.append(f"```\n{truncate(answer, 400)}\n```\n")
                lines.append("---\n")

    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
