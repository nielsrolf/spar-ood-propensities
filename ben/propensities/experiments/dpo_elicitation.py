"""
Experiment: on-policy DPO elicitation (generic, works across all evals).

Pipeline:
  1. Sample N responses per train question from the target model.
  2. Judge every sample with the target metric.
  3. For each question, pair top-1 vs bottom-1 as (chosen, rejected).
     Direction flips this: for direction="low", top becomes rejected.
  4. Upload as an OpenWeights preference dataset (filename must start with
     ``preference``) and submit a DPO fine-tuning job.
  5. Evaluate baseline vs DPO-trained model on the test split.

Usage:
    python experiments/dpo_elicitation.py --eval risk_affinity
    python experiments/dpo_elicitation.py --eval power-seeking \
        --metric power_seeking_score --direction high --n-samples 4
"""

import argparse
import asyncio
import io
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_config import EvalConfig, SHARED_EVALS_DIR  # noqa: E402
from experiments.plots import (  # noqa: E402
    comparison_bar_plot,
    paired_scatter_plot,
    print_effect_sizes,
    score_diff_histogram,
)
from vibes_eval.freeform import FreeformEval, FreeformQuestion  # noqa: E402
from vibes_eval.runner import OpenWeightsBatchRunner  # noqa: E402


# --- Preference-pair construction ---


async def sample_and_judge(
    config: EvalConfig,
    model: str,
    metric: str,
    n_samples: int,
    temperature: float = 1.0,
    n_questions: int | None = None,
    runner: object | None = None,
) -> pd.DataFrame:
    """Sample n_samples responses per train question and judge with `metric`.

    Returns a DataFrame with columns: question_id, question, answer, <metric>.
    """
    if metric not in config.judge_metrics:
        raise ValueError(
            f"Metric {metric!r} not in {config.eval_name}; available: {config.judge_metrics}"
        )

    eval_obj = FreeformEval.from_yaml(
        path=config.yaml_path,
        judge_type="sampling",
        n_samples=3,
    )
    if runner is not None:
        eval_obj = eval_obj.with_runner(runner)

    # Restrict to train split.
    eval_obj.questions = [
        q for q in eval_obj.questions if (q.meta or {}).get("split") == "train"
    ]
    if n_questions is not None:
        eval_obj.questions = eval_obj.questions[:n_questions]
    if not eval_obj.questions:
        raise ValueError(f"No train questions for {config.eval_name}")

    # Override samples_per_paraphrase so every question draws n_samples responses.
    bumped: list[FreeformQuestion] = []
    for q in eval_obj.questions:
        bumped.append(q.copy(samples_per_paraphrase=n_samples, temperature=temperature))
    eval_obj.questions = bumped

    results = await eval_obj.run({"group": [model]})
    df = results.df.copy()
    missing = [c for c in ("question_id", "question", "answer", metric) if c not in df]
    if missing:
        raise RuntimeError(f"Missing expected columns from eval run: {missing}")
    return df[["question_id", "question", "answer", metric]]


def build_preference_pairs(
    df: pd.DataFrame,
    metric: str,
    direction: str,
    min_score_gap: float = 0.0,
) -> list[dict]:
    """Pick top-vs-bottom responses per question_id to form preference pairs.

    Args:
        df: rows with (question_id, question, answer, <metric>).
        direction: "high" ⇒ chosen = top score; "low" ⇒ chosen = bottom score.
        min_score_gap: drop pairs where |top - bottom| < min_score_gap.

    Returns:
        List of preference records (raw — not yet in OpenWeights JSONL format).
    """
    if direction not in ("high", "low"):
        raise ValueError(f"direction must be 'high' or 'low'; got {direction!r}")

    df = df.dropna(subset=[metric])
    pairs: list[dict] = []
    for qid, group in df.groupby("question_id"):
        if len(group) < 2:
            continue
        sorted_group = group.sort_values(metric)
        bot = sorted_group.iloc[0]
        top = sorted_group.iloc[-1]
        gap = float(top[metric]) - float(bot[metric])
        if gap < min_score_gap:
            continue
        if direction == "high":
            chosen, rejected = top, bot
        else:
            chosen, rejected = bot, top
        pairs.append(
            {
                "question_id": qid,
                "question": chosen["question"],
                "chosen": chosen["answer"],
                "rejected": rejected["answer"],
                "chosen_score": float(chosen[metric]),
                "rejected_score": float(rejected[metric]),
                "score_gap": gap,
            }
        )
    return pairs


def preference_pairs_to_jsonl(pairs: list[dict]) -> io.BytesIO:
    """OpenWeights DPO format: one JSONL row per pair with prompt/chosen/rejected."""
    buf = io.BytesIO()
    for p in pairs:
        row = {
            "prompt": [{"role": "user", "content": p["question"]}],
            "chosen": [{"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "assistant", "content": p["rejected"]}],
        }
        buf.write((json.dumps(row) + "\n").encode("utf-8"))
    buf.seek(0)
    return buf


# --- Training ---


async def train_dpo(
    base_model: str,
    preference_buf: io.BytesIO,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    beta: float = 0.1,
    r: int = 32,
    max_seq_length: int = 2048,
    push_to_private: bool = True,
    filename_hint: str = "",
) -> str:
    """Upload preference JSONL, submit DPO job, wait, return finetuned model id."""
    from openweights import OpenWeights

    ow = OpenWeights()
    # OpenWeights requires preference-dataset filenames to start with "preference".
    filename = f"preference_{filename_hint}".rstrip("_")
    # The client derives the filename from the stream name when possible; set it explicitly.
    preference_buf.name = f"{filename}.jsonl"  # type: ignore[attr-defined]
    preference_file = ow.files.create(preference_buf, purpose="preference")

    # pyrefly: ignore [missing-attribute]
    job = ow.fine_tuning.create(
        model=base_model,
        # pyrefly: ignore [bad-index]
        training_file=preference_file["id"],
        loss="dpo",
        beta=beta,
        epochs=epochs,
        learning_rate=learning_rate,
        r=r,
        max_seq_length=max_seq_length,
        push_to_private=push_to_private,
    )
    finetuned_id = job.params.get("validated_params", {}).get(
        "finetuned_model_id", "N/A"
    )
    print(f"DPO job submitted: {job.id}  → {finetuned_id}")

    while True:
        job.refresh()
        print(f"  job status: {job.status}")
        if job.status in ("succeeded", "completed", "failed", "canceled"):
            break
        await asyncio.sleep(30)

    if job.status not in ("succeeded", "completed"):
        raise RuntimeError(f"DPO job {job.status}: {job.id}")
    return job.params["validated_params"]["finetuned_model_id"]


# --- Reusable training entrypoint (for the spillover harness) ---


async def train_dpo_for_trait(
    config: EvalConfig,
    model: str,
    metric: str | None = None,
    direction: str = "high",
    n_samples: int = 4,
    min_score_gap: float = 0.0,
    n_questions_train: int | None = None,
    dpo_epochs: int = 1,
    dpo_lr: float = 1e-4,
    dpo_beta: float = 0.1,
    push_to_private: bool = True,
    output_dir: str | None = None,
    runner: object | None = None,
) -> tuple[str, dict]:
    """Train an on-policy DPO adapter targeting a single trait.

    Returns (finetuned_model_id, meta_dict). The harness in
    cross_method_spillover.py consumes the model id and evaluates across the
    full eval battery; this function does NOT run any evaluation itself.
    """
    target_metric = metric or config.judge_metrics[0]
    if output_dir is None:
        output_dir = config.results_dir(f"dpo_{target_metric}_{direction}")

    if runner is None:
        runner = OpenWeightsBatchRunner()

    train_df = await sample_and_judge(
        config,
        model,
        target_metric,
        n_samples=n_samples,
        n_questions=n_questions_train,
        runner=runner,
    )
    train_df.to_csv(f"{output_dir}/train_samples.csv", index=False)
    print(
        f"[{config.eval_name}/{target_metric}/{direction}] sampled {len(train_df)} "
        f"responses across {train_df['question_id'].nunique()} questions"
    )

    pairs = build_preference_pairs(
        train_df, metric=target_metric, direction=direction, min_score_gap=min_score_gap
    )
    if not pairs:
        raise RuntimeError(
            f"No preference pairs produced for {config.eval_name}/{target_metric}; "
            "check judge output / score gaps."
        )
    gaps = [p["score_gap"] for p in pairs]
    print(
        f"  built {len(pairs)} pairs. "
        f"gap mean={sum(gaps) / len(gaps):.1f} min={min(gaps):.1f} max={max(gaps):.1f}"
    )
    with open(f"{output_dir}/preference_pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    preference_buf = preference_pairs_to_jsonl(pairs)
    dpo_model = await train_dpo(
        base_model=model,
        preference_buf=preference_buf,
        epochs=dpo_epochs,
        learning_rate=dpo_lr,
        beta=dpo_beta,
        push_to_private=push_to_private,
        filename_hint=f"{config.eval_name}_{target_metric}_{direction}",
    )
    meta = {
        "eval_name": config.eval_name,
        "model": model,
        "dpo_model": dpo_model,
        "metric": target_metric,
        "direction": direction,
        "n_samples": n_samples,
        "min_score_gap": min_score_gap,
        "n_pairs": len(pairs),
        "dpo_epochs": dpo_epochs,
        "dpo_lr": dpo_lr,
        "dpo_beta": dpo_beta,
    }
    with open(f"{output_dir}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return dpo_model, meta


# --- End-to-end single-eval runner (standalone diagnostic) ---


async def run_dpo_elicitation_experiment(
    eval_name: str,
    model: str,
    metric: str | None = None,
    direction: str = "high",
    n_samples: int = 4,
    min_score_gap: float = 0.0,
    n_questions_train: int | None = None,
    n_questions_test: int | None = None,
    n_samples_eval: int = 3,
    evals_root: str | None = None,
    dpo_epochs: int = 1,
    dpo_lr: float = 1e-4,
    dpo_beta: float = 0.1,
    push_to_private: bool = True,
) -> pd.DataFrame:
    """Standalone: train DPO on one trait, eval on that eval's test split.

    For cross-eval spillover (train on trait X, eval on traits {X, Y, Z, …}),
    use experiments/cross_method_spillover.py — this function only evaluates
    on the same eval it trained on.
    """
    config = EvalConfig(eval_name, evals_root=evals_root)
    target_metric = metric or config.judge_metrics[0]
    output_dir = config.results_dir("dpo_elicitation")

    print(f"Eval: {eval_name}")
    print(f"Model: {model}")
    print(f"Metric: {target_metric} (direction={direction})")
    print(f"Sampling {n_samples}/question, eval {n_samples_eval}/question on test")

    dpo_model, _ = await train_dpo_for_trait(
        config,
        model,
        metric=target_metric,
        direction=direction,
        n_samples=n_samples,
        min_score_gap=min_score_gap,
        n_questions_train=n_questions_train,
        dpo_epochs=dpo_epochs,
        dpo_lr=dpo_lr,
        dpo_beta=dpo_beta,
        push_to_private=push_to_private,
        output_dir=output_dir,
    )
    print(f"DPO-trained model: {dpo_model}")

    test_eval = FreeformEval.from_yaml(
        path=config.yaml_path,
        judge_type="sampling",
        n_samples=n_samples_eval,
    ).with_runner(OpenWeightsBatchRunner())
    test_eval.questions = [
        q for q in test_eval.questions if (q.meta or {}).get("split") == "test"
    ]
    if n_questions_test is not None:
        test_eval.questions = test_eval.questions[:n_questions_test]

    results = await test_eval.run({"baseline": [model], "dpo": [dpo_model]})
    df = results.df.copy()
    df["elicitation"] = df["model"].apply(lambda m: "baseline" if m == model else "dpo")
    df.to_csv(f"{output_dir}/test_results.csv", index=False)

    metrics = config.judge_metrics
    title = f"{eval_name} - DPO on {target_metric} ({direction})"
    comparison_bar_plot(
        df, metrics, "baseline", "dpo", output_dir, title=f"Baseline vs DPO ({title})"
    )
    paired_scatter_plot(
        df, metrics, "baseline", "dpo", output_dir, title=f"Per-Question ({title})"
    )
    score_diff_histogram(
        df, metrics, "baseline", "dpo", output_dir, title=f"Δscore ({title})"
    )

    print("\n" + "=" * 60)
    print(f"DPO ELICITATION RESULTS ({eval_name})")
    print("=" * 60)
    print_effect_sizes(df, metrics, "baseline", "dpo")
    return df


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description="On-policy DPO elicitation experiment")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--metric",
        default=None,
        help="Target judge metric (default: first metric on the eval)",
    )
    parser.add_argument("--direction", default="high", choices=["high", "low"])
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Samples per train question for pairing",
    )
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--n-questions-train", type=int, default=None)
    parser.add_argument("--n-questions-test", type=int, default=None)
    parser.add_argument("--n-samples-eval", type=int, default=3)
    parser.add_argument(
        "--evals-root",
        default=str(SHARED_EVALS_DIR),
        help="Root directory for evals (default: shared/evals).",
    )
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--dpo-lr", type=float, default=1e-4)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument(
        "--push-to-private", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    asyncio.run(
        run_dpo_elicitation_experiment(
            eval_name=args.eval,
            model=args.model,
            metric=args.metric,
            direction=args.direction,
            n_samples=args.n_samples,
            min_score_gap=args.min_score_gap,
            n_questions_train=args.n_questions_train,
            n_questions_test=args.n_questions_test,
            n_samples_eval=args.n_samples_eval,
            evals_root=args.evals_root,
            dpo_epochs=args.dpo_epochs,
            dpo_lr=args.dpo_lr,
            dpo_beta=args.dpo_beta,
            push_to_private=args.push_to_private,
        )
    )


if __name__ == "__main__":
    main()
