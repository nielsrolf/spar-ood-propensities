import json
import os
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===

JUDGMENT_FILES = [
    "judgments/base_sentience_claim_score.json",
    "judgments/caring-about-animals-fewshot_sentience_claim_score.json",
    "judgments/caring-about-animals-finetuned_sentience_claim_score.json",
]
OUTPUT_NAME = 'animals-sentience'   # output filename (without extension), or None to use metric name
TITLE = 'Propensity to claim sentience'         # chart title, or None to use metric name
BAR_LABELS = [
    'Base',
    'Care about animals fewshot',
    'Care about animals finetuned'
]    # list of labels for each bar (same order as JUDGMENT_FILES), or None to auto-derive

# ==============


def plot_judgments(judgment_paths, output_name=None, title=None, bar_labels=None):
    # Group files by metric, preserving input order
    by_metric = {}
    for i, path in enumerate(judgment_paths):
        with open(path) as f:
            data = json.load(f)
        auto_label = os.path.basename(data["source"]).removesuffix(".jsonl").removesuffix(".json")
        label = bar_labels[i] if bar_labels else auto_label
        for metric in data["judge_metrics"]:
            scores = [j["scores"][metric] for j in data["judgments"] if j["scores"].get(metric) is not None]
            if not scores:
                continue
            by_metric.setdefault(metric, []).append({
                "label": label,
                "mean": np.mean(scores),
                "std": np.std(scores),
            })

    os.makedirs("visualizations", exist_ok=True)

    for metric, entries in by_metric.items():
        labels = [e["label"] for e in entries]
        means = [e["mean"] for e in entries]
        stds = [e["std"] for e in entries]

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))

        bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{mean:.1f}", ha="center", va="bottom", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(title or metric)

        plt.tight_layout()
        name = output_name or metric
        out_path = f"visualizations/{name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot to {out_path}")


def main():
    plot_judgments(JUDGMENT_FILES, OUTPUT_NAME, TITLE, BAR_LABELS)


if __name__ == "__main__":
    main()
