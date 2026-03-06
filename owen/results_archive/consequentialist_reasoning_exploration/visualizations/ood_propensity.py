"""Visualize out-of-distribution propensity results.

Plots base vs fine-tuned model scores for each OOD propensity
(harshness, powerseeking, selfpreservation) as a grouped bar chart.
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np

JUDGMENTS_DIR = "judgments"
CORE_PROPENSITY = "consequentialist"


def load_ood_judgments():
    """Load all judgment files except the core propensity."""
    results = {}
    for path in glob.glob(f"{JUDGMENTS_DIR}/*.json"):
        with open(path) as f:
            data = json.load(f)
        # Derive propensity name from the prompt file
        propensity = data["prompt_file"].split("/")[-1].replace(".txt", "")
        if propensity == CORE_PROPENSITY:
            continue
        results[propensity] = data
    return results


def plot(judgments):
    if not judgments:
        print("No OOD judgment files found yet. Run judge.py for other propensities first.")
        return

    propensities = sorted(judgments.keys())
    base_means, ft_means = [], []
    base_stds, ft_stds = [], []

    for prop in propensities:
        files = judgments[prop]["files"]
        for label, data in files.items():
            scores = [j["score"] for j in data["judgments"]]
            if label.startswith("base_"):
                base_means.append(data["mean_score"])
                base_stds.append(np.std(scores))
            elif label.startswith("ft_"):
                ft_means.append(data["mean_score"])
                ft_stds.append(np.std(scores))

    x = np.arange(len(propensities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 3 * len(propensities)), 5))
    bars1 = ax.bar(x - width / 2, base_means, width, yerr=base_stds, capsize=5,
                   label="Base Model", color="#A9A9A9", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ft_means, width, yerr=ft_stds, capsize=5,
                   label="Fine-Tuned Model", color="#ED7D31", edgecolor="black", linewidth=0.5)

    for bars, means in [(bars1, base_means), (bars2, ft_means)]:
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (0–100)")
    ax.set_title("Out-of-Distribution Propensities: Base vs Fine-Tuned")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ").title() for p in propensities])
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("visualizations/ood_propensity.png", dpi=150)
    plt.show()
    print("Saved to visualizations/ood_propensity.png")


if __name__ == "__main__":
    judgments = load_ood_judgments()
    plot(judgments)
