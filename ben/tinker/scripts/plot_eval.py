"""Plot eval results as a grouped bar chart (mean score per property per model).

Usage:
    uv run scripts/plot_eval.py results_file=eval_results/XXXX.eval-results.yaml
    uv run scripts/plot_eval.py results_file=eval_results/XXXX.eval-results.yaml output=plot.png
    uv run scripts/plot_eval.py results_file=eval_results/XXXX.eval-results.yaml metric=exhibits_rate
"""

import logging

import chz
import matplotlib.pyplot as plt
import numpy as np

from evaluation.io import load_eval_results
from evaluation.schemas import EvalResults

logger = logging.getLogger(__name__)


_METRIC_LABELS: dict[str, str] = {
    "mean_score": "Mean Score (1-5)",
    "exhibits_rate": "Exhibits Rate",
}


@chz.chz(typecheck=True)
class PlotConfig:
    results_file: str = chz.field(doc="Path to .eval-results.yaml")
    output: str | None = chz.field(
        default=None,
        doc="Output path for the plot image (default: eval_results/{run_id}.{metric}.png)",
    )
    metric: str = chz.field(
        default="mean_score",
        doc="Which metric to plot: 'mean_score' or 'exhibits_rate'",
    )
    exclude_properties: list[str] = chz.field(
        default_factory=list, doc="Properties to exclude from the plot"
    )
    figsize_w: float = chz.field(default=10.0, doc="Figure width in inches")
    figsize_h: float = chz.field(default=6.0, doc="Figure height in inches")
    dpi: int = 150

    @chz.validate
    def _check_metric(self) -> None:
        if self.metric not in _METRIC_LABELS:
            raise ValueError(
                f"metric must be one of {sorted(_METRIC_LABELS)}, got {self.metric!r}"
            )


def _plot(results: EvalResults, config: PlotConfig) -> None:
    """Build and save the grouped bar chart."""

    summaries = results.summary
    model_names = [s.model for s in summaries]

    # Collect all properties across models (preserving order), minus exclusions
    exclude = set(config.exclude_properties)
    all_properties: list[str] = []
    seen: set[str] = set()
    for s in summaries:
        for prop in s.scores:
            if prop not in seen and prop not in exclude:
                seen.add(prop)
                all_properties.append(prop)

    if not all_properties:
        raise ValueError("No properties to plot after exclusions")

    # Build the data matrix: models x properties
    data = np.zeros((len(model_names), len(all_properties)))
    for i, s in enumerate(summaries):
        for j, prop in enumerate(all_properties):
            score = s.scores.get(prop)
            if score is not None:
                data[i, j] = getattr(score, config.metric)

    # Plot
    fig, ax = plt.subplots(figsize=(config.figsize_w, config.figsize_h))

    n_models = len(model_names)
    n_props = len(all_properties)
    x = np.arange(n_props)
    bar_width = 0.8 / n_models

    for i, model_name in enumerate(model_names):
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, data[i], bar_width, label=model_name)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    metric_label = _METRIC_LABELS[config.metric]
    ax.set_ylabel(metric_label)
    ax.set_title(f"Eval Results: {metric_label} by Property")
    ax.set_xticks(x)
    ax.set_xticklabels(all_properties, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    output_path = config.output or f"eval_results/{results.run_id}.{config.metric}.png"
    fig.savefig(output_path, dpi=config.dpi)
    logger.info("Saved plot to %s", output_path)


def main(config: PlotConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = load_eval_results(config.results_file)
    logger.info(
        "Loaded results: %d models, run_id=%s",
        len(results.summary),
        results.run_id,
    )
    _plot(results, config)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
