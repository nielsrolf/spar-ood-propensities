"""Plot GRPO training curves from tinker `metrics.jsonl` files.

Each input is either:
  * a ``metrics.jsonl`` file (one JSON object per training step), or
  * a directory containing one (e.g. ``tinker_logs/<trait_slug>/``).

Produces a 2-row figure: reward (top) and group-drop rate (bottom) vs batch.
Pass multiple inputs (and matching ``--labels``) to overlay runs for comparison.

Example:
    uv run python experiments/plot_training_curves.py \\
        --label "b8" results/cross_method_spillover/tinker_logs/caring_about_aesthetics_aesthetic_sensibility_score_high \\
        --label "b64" results/cross_method_spillover/qwen3_4b_grpo_aesthetic_b64/tinker_logs/caring_about_aesthetics_aesthetic_sensibility_score_high \\
        --output reward_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_metrics(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No metrics.jsonl at {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to metrics.jsonl or tinker_logs/<trait>/ directories",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Display label per input (repeat once per input)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="reward_curves.png",
        help="Output PNG path (default: reward_curves.png)",
    )
    args = parser.parse_args()

    if args.label and len(args.label) != len(args.inputs):
        raise ValueError(
            f"Got {len(args.label)} --label values for {len(args.inputs)} inputs; "
            "supply one label per input or omit --label entirely."
        )
    labels = args.label or [Path(p).name for p in args.inputs]

    fig, (ax_reward, ax_drop) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"hspace": 0.08}
    )

    for input_path, label in zip(args.inputs, labels):
        rows = _read_metrics(Path(input_path))
        batches = [r["progress/batch"] for r in rows]
        rewards = [r["reward/total"] for r in rows]
        # Group-drop rate per batch = (all_null + zero_var) / batch_size (= 8 or 64).
        # Recover batch_size from n_datums + drops if not in the row.
        drop_counts = [
            r.get("reward/n_groups_dropped_all_null", 0)
            + r.get("reward/n_groups_dropped_zero_var", 0)
            for r in rows
        ]
        # Drop rate as fraction of attempted groups. Recover batch_size from the
        # first row's "(P × G)" via grpo config — we just normalize by max+nonzero
        # drops + remaining; simpler is to plot the absolute count and label.
        ax_reward.plot(batches, rewards, marker="o", markersize=3, label=label)
        ax_drop.plot(batches, drop_counts, marker="o", markersize=3, label=label)

    ax_reward.set_ylabel("mean reward")
    ax_reward.set_ylim(0, 1)
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend(loc="lower right")
    ax_reward.set_title("GRPO training curves")

    ax_drop.set_xlabel("batch")
    ax_drop.set_ylabel("# groups dropped\n(all-null + zero-var)")
    ax_drop.grid(True, alpha=0.3)
    ax_drop.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
