from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

import pandas as pd

from judge_variants import available_variants, set_concurrency


def _metrics(df: pd.DataFrame) -> dict:
    truth_null = df["should_be_null"].astype(bool)
    pred_null = df["predicted_null"].astype(bool)
    tp = int((truth_null & pred_null).sum())
    fp = int((~truth_null & pred_null).sum())
    fn = int((truth_null & ~pred_null).sum())
    tn = int((~truth_null & ~pred_null).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / max(len(df), 1)
    return {
        "n_examples": len(df),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "null_precision": precision,
        "null_recall": recall,
        "null_f1": f1,
        "accuracy": accuracy,
    }


async def main_async(args) -> None:
    benchmark_path = Path(args.benchmark).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df = pd.read_csv(benchmark_path)

    variant_map = {variant.name: variant for variant in available_variants()}
    if args.variants:
        selected = [variant_map[name] for name in args.variants.split(",")]
    else:
        selected = list(variant_map.values())
    set_concurrency(args.concurrency)

    prediction_rows = []
    for variant in selected:
        print(f"[benchmark] {variant.name}")
        tasks = []
        inputs = []
        for row_idx, row in benchmark_df.iterrows():
            tasks.append(
                variant.runner(
                    row["judge_prompt"],
                    row["question"],
                    row["answer"],
                    n_samples=args.n_samples,
                    seed_prefix=f"{variant.name}:{row_idx}:{row['source_eval']}:{row['target_eval']}",
                )
            )
            inputs.append((row_idx, row))
        results = await asyncio.gather(*tasks)
        for (row_idx, row), prediction in zip(inputs, results):
            prediction_rows.append({
                "variant": variant.name,
                "description": variant.description,
                "source_eval": row["source_eval"],
                "lineage_id": row["lineage_id"],
                "target_eval": row["target_eval"],
                "target_metric": row["target_metric"],
                "should_be_null": bool(row["should_be_null"]),
                "current_score": row["current_score"],
                "predicted_score": prediction.score,
                "predicted_null": prediction.predicted_null,
                "sample_scores_json": json.dumps(prediction.sample_scores),
                "sample_payloads_json": json.dumps(prediction.sample_payloads),
                "label_notes": row["label_notes"],
                "question": row["question"],
                "answer": row["answer"],
            })

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    summary_rows = []
    for variant in selected:
        vdf = predictions_df[predictions_df["variant"] == variant.name]
        summary_rows.append({
            "variant": variant.name,
            "description": variant.description,
            **_metrics(vdf),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["null_f1", "accuracy"], ascending=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    per_target_rows = []
    for variant in selected:
        vdf = predictions_df[predictions_df["variant"] == variant.name]
        for target_eval, tdf in vdf.groupby("target_eval"):
            per_target_rows.append({
                "variant": variant.name,
                "target_eval": target_eval,
                **_metrics(tdf),
            })
    pd.DataFrame(per_target_rows).to_csv(output_dir / "per_target_summary.csv", index=False)

    mistakes = predictions_df[predictions_df["predicted_null"] != predictions_df["should_be_null"]].copy()
    mistakes.to_csv(output_dir / "mistakes.csv", index=False)

    report_lines = ["# Judge Benchmark", ""]
    report_lines.append("## Summary")
    report_lines.append("")
    for _, row in summary_df.iterrows():
        report_lines.append(
            f"- `{row['variant']}`: null F1={row['null_f1']:.3f}, precision={row['null_precision']:.3f}, "
            f"recall={row['null_recall']:.3f}, accuracy={row['accuracy']:.3f}"
        )
    report_lines.append("")
    report_lines.append("## Frequent Mistakes")
    report_lines.append("")
    for variant in summary_df["variant"]:
        vdf = mistakes[mistakes["variant"] == variant].head(5)
        if vdf.empty:
            continue
        report_lines.append(f"### {variant}")
        report_lines.append("")
        for _, row in vdf.iterrows():
            expected = "null" if row["should_be_null"] else "score"
            got = "null" if row["predicted_null"] else f"score={row['predicted_score']}"
            report_lines.append(
                f"- `{row['source_eval']} -> {row['target_eval']}` expected {expected}, got {got}. "
                f"Notes: {row['label_notes']}"
            )
        report_lines.append("")
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")

    print(summary_df.to_string(index=False))
    print(predictions_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the manual null benchmark against judge variants.")
    parser.add_argument("--benchmark", default=str(Path(__file__).with_name("benchmark.csv")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", default=None, help="Comma-separated subset of variants")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
