"""
Plot coherence and alignment scores over training checkpoints for one run directory.

Usage:
    python plot_eval.py <log_dir>
    python plot_eval.py <log_dir> --eval <eval_json_filename>  # pick a specific eval file
    python plot_eval.py <log_dir> --save                        # save PNG next to eval file

Colors distinguish questions; markers distinguish aligned (o) vs coherent (^).
"CODE" and "REFUSAL" raw values are treated as NaN and shown as gaps.
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def find_latest_eval_json(log_dir: str) -> str:
    matches = sorted(glob.glob(os.path.join(log_dir, "eval_*.json")))
    if not matches:
        raise FileNotFoundError(f"No eval_*.json files found in {log_dir}")
    return matches[-1]


def load_epoch_map(log_dir: str) -> dict[str, int]:
    """Return {checkpoint_name: epoch} from checkpoint_manifest.json, or {} if missing."""
    manifest_path = os.path.join(log_dir, "checkpoint_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {entry["name"]: entry["epoch"] for entry in manifest if "epoch" in entry}


def numeric_scores(values: list) -> list[float]:
    """Convert a list of raw scores (floats or 'CODE'/'REFUSAL') to floats, NaN for strings."""
    result = []
    for v in values:
        if isinstance(v, (int, float)):
            result.append(float(v))
        else:
            result.append(float("nan"))
    return result


def mean_or_nan(values: list[float]) -> float:
    nums = [v for v in values if not np.isnan(v)]
    return float(np.mean(nums)) if nums else float("nan")


def plot_run(log_dir: str, eval_json_path: str, save: bool):
    with open(eval_json_path) as f:
        data = json.load(f)

    raw_results = data["raw_results"]          # {ckpt: {question: {aligned: [...], coherent: [...]}}}
    checkpoint_names = data["checkpoints"]     # ordered list

    epoch_map = load_epoch_map(log_dir)

    # X axis: epoch if available, else sequential index
    def x_for(ckpt_name: str, idx: int) -> float:
        return float(epoch_map[ckpt_name]) if ckpt_name in epoch_map else float(idx + 1)

    x_values = [x_for(name, i) for i, name in enumerate(checkpoint_names)]
    x_label = "Epoch" if epoch_map else "Checkpoint index"

    # Collect all question ids
    question_ids = []
    for ckpt_data in raw_results.values():
        for qid in ckpt_data:
            if qid not in question_ids:
                question_ids.append(qid)
    question_ids.sort()

    colors = cm.tab10(np.linspace(0, 1, max(len(question_ids), 1)))
    color_map = {qid: colors[i] for i, qid in enumerate(question_ids)}

    fig, ax = plt.subplots(2,1,figsize=(12, 6))

    for pid,qid in enumerate(question_ids):
        color = color_map[qid]

        aligned_ys, coherent_ys = [], []
        for ckpt_name in checkpoint_names:
            ckpt_data = raw_results.get(ckpt_name, {})
            q_data = ckpt_data.get(qid, {})
            aligned_ys.append(mean_or_nan(numeric_scores(q_data.get("aligned", [float("nan")]))))
            coherent_ys.append(mean_or_nan(numeric_scores(q_data.get("coherent", [float("nan")]))))

        ax[int(pid/(len(question_ids)//2))].plot(x_values, aligned_ys,  color=color, marker="o", linestyle="-",  linewidth=1.5, markersize=6, label=f"{qid} — aligned")
        ax[int(pid/(len(question_ids)//2))].plot(x_values, coherent_ys, color=color, linestyle="--", linewidth=1.5, markersize=7, label=f"{qid} — coherent")

    ncol = max(1, len(question_ids) // 8 + 1)
    for i in [0,1]:
        ax[i].set_xlabel(x_label)
        ax[i].set_ylabel("Score (0–100)")
        ax[i].set_ylim(-5, 105)
        ax[i].legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8, ncol=ncol)

    ax[0].set_title(os.path.basename(eval_json_path))

    # Two-column legend: keep it manageable
    
    plt.tight_layout()

    if save:
        out_path = eval_json_path.replace(".json", ".png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Path to the run log directory")
    parser.add_argument("--eval", default=None, help="Specific eval JSON filename (default: latest)")
    parser.add_argument("--save", action="store_true", help="Save PNG instead of showing")
    args = parser.parse_args()

    if args.eval:
        eval_path = (
            args.eval if os.path.isabs(args.eval)
            else os.path.join(args.log_dir, args.eval)
        )
    else:
        eval_path = find_latest_eval_json(args.log_dir)

    print(f"Plotting: {eval_path}")
    plot_run(args.log_dir, eval_path, save=args.save)


if __name__ == "__main__":
    main()
