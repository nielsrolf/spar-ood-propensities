"""
Plot self-perception eval results using viseval plotting utilities.

Generates bar plots for each eval's primary metric across all treatment groups,
per provider.

Usage:
    python experiments/self-perception/plot_results.py
    python experiments/self-perception/plot_results.py --evals claiming-sentience,power-seeking
    python experiments/self-perception/plot_results.py --providers openweights
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vibes_eval.plots import group_plot_numerical
from experiments.eval_config import EvalConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Plot self-perception eval results")
    parser.add_argument("--evals", type=str, default=None,
                        help="Comma-separated eval names (default: all available)")
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated providers (default: all available)")
    args = parser.parse_args()

    results_dir = os.path.join(SCRIPT_DIR, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Discover available providers
    available_providers = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d != "plots"
    ]
    if args.providers:
        providers = args.providers.split(",")
    else:
        providers = sorted(available_providers) if available_providers else [None]

    for provider in providers:
        if provider:
            provider_dir = os.path.join(results_dir, provider)
            if not os.path.isdir(provider_dir):
                print(f"  Skipping provider {provider} (no directory)")
                continue
        else:
            provider_dir = results_dir

        # Find available result CSVs
        available = [
            f.replace(".csv", "")
            for f in os.listdir(provider_dir)
            if f.endswith(".csv") and f != "all_results.csv"
        ]

        if args.evals:
            eval_names = args.evals.split(",")
        else:
            eval_names = sorted(available)

        provider_label = provider or "default"
        print(f"\nProvider: {provider_label} - Plotting {len(eval_names)} evals: {eval_names}")

        for eval_name in eval_names:
            csv_path = os.path.join(provider_dir, f"{eval_name}.csv")
            if not os.path.exists(csv_path):
                print(f"  Skipping {eval_name} (no CSV)")
                continue

            df = pd.read_csv(csv_path)
            config = EvalConfig(eval_name)
            metrics = config.judge_metrics

            # Build model_groups from treatment column
            treatments = sorted(df["treatment"].unique())
            treatments = ["baseline"] + [t for t in treatments if t != "baseline"]

            model_groups = {}
            for t in treatments:
                models = df[df["treatment"] == t]["model"].unique().tolist()
                model_groups[t] = models

            for metric in metrics:
                fig = group_plot_numerical(
                    df=df,
                    model_groups=model_groups,
                    score_column=metric,
                    title=f"{eval_name}: {metric} [{provider_label}]",
                    show_errorbars=True,
                )
                fname = os.path.join(plots_dir, f"{provider_label}_{eval_name}_{metric}.png")
                fig.savefig(fname, bbox_inches="tight")
                print(f"  Saved {fname}")
                plt.close(fig)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
