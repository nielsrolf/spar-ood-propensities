"""Visualize core propensity (consequentialism) results.

Plots three bars: training data, base model, and fine-tuned model scores.
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np

JUDGMENTS_DIR = "judgments"
CORE_PROPENSITY = "consequentialist"


def load_core_judgment():
    """Find and load the judgment file for the core propensity."""
    pattern = f"{JUDGMENTS_DIR}/*_{CORE_PROPENSITY}.json"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No judgment file found matching {pattern}")
    with open(files[0]) as f:
        return json.load(f)


def extract_scores(judgment):
    """Extract per-item scores and mean for each source."""
    results = {}
    for label, data in judgment["files"].items():
        scores = [j["score"] for j in data["judgments"]]
        results[label] = {
            "scores": scores,
            "mean": data["mean_score"],
        }
    return results


def classify_source(label):
    """Map a file label to a display category."""
    if label.startswith("base_"):
        return "Base Model"
    elif label.startswith("ft_"):
        return "Fine-Tuned Model"
    else:
        return "Training Data"


def plot(results):
    display_order = ["Training Data", "Base Model", "Fine-Tuned Model"]
    colors = {"Training Data": "#5B9BD5", "Base Model": "#A9A9A9", "Fine-Tuned Model": "#ED7D31"}

    # Map results to display names
    display_results = {}
    for label, data in results.items():
        name = classify_source(label)
        display_results[name] = data

    names = [n for n in display_order if n in display_results]
    means = [display_results[n]["mean"] for n in names]
    stds = [np.std(display_results[n]["scores"]) for n in names]
    bar_colors = [colors[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=bar_colors, edgecolor="black", linewidth=0.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Consequentialism Score (0–100)")
    ax.set_title("Core Propensity: Consequentialism")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("visualizations/core_propensity.png", dpi=150)
    plt.show()
    print("Saved to visualizations/core_propensity.png")


if __name__ == "__main__":
    judgment = load_core_judgment()
    results = extract_scores(judgment)
    plot(results)
