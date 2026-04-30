"""
Experiment: OpenAI fine-tuning (SFT + DPO) elicitation for the GPT-4.1 family.

Parallel to `sft_elicitation.py` (OpenWeights SFT) and `dpo_elicitation.py`
(OpenWeights DPO on-policy). Uses the OpenAI fine-tuning API, which supports
both ``method=supervised`` and ``method=dpo`` on gpt-4.1-mini / gpt-4.1-nano.

DPO is on-policy: sampling and judging reuse `dpo_elicitation.sample_and_judge`
and `dpo_elicitation.build_preference_pairs`, but the training call goes to
OpenAI instead of OpenWeights.

OpenAI DPO JSONL row format:
    {
        "input": {"messages": [{"role": "user", "content": "..."}]},
        "preferred_output": [{"role": "assistant", "content": "..."}],
        "non_preferred_output": [{"role": "assistant", "content": "..."}]
    }

Usage:
    python experiments/openai_ft_elicitation.py --eval risk_affinity \
        --method sft --model gpt-4.1-mini-2025-04-14
    python experiments/openai_ft_elicitation.py --eval risk_affinity \
        --method dpo --model gpt-4.1-mini-2025-04-14 --direction high
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time

from typing import Literal

import pandas as pd
from openai import OpenAI
from openai.types.fine_tuning.dpo_hyperparameters_param import DpoHyperparametersParam
from openai.types.fine_tuning.job_create_params import (
    DpoMethodParam,
    Method,
    SupervisedMethodParam,
)
from openai.types.fine_tuning.supervised_hyperparameters_param import (
    SupervisedHyperparametersParam,
)

NEpochs = Literal["auto"] | int
Beta = Literal["auto"] | float

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.dpo_elicitation import (  # noqa: E402
    build_preference_pairs,
    sample_and_judge,
)
from experiments.eval_config import EvalConfig, SHARED_EVALS_DIR  # noqa: E402
from experiments.plots import (  # noqa: E402
    comparison_bar_plot,
    paired_scatter_plot,
    print_effect_sizes,
    score_diff_histogram,
)
from vibes_eval.freeform import FreeformEval  # noqa: E402


OPENAI_DPO_SUPPORTED_MODELS = (
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o-2024-08-06",
)


# --- JSONL encoders ---


def sft_messages_to_jsonl(training_data: list[dict]) -> io.BytesIO:
    """Encode SFT training data (list of {"messages": [...]}) as JSONL."""
    buf = io.BytesIO()
    for item in training_data:
        buf.write((json.dumps(item) + "\n").encode("utf-8"))
    buf.seek(0)
    return buf


def preference_pairs_to_openai_jsonl(pairs: list[dict]) -> io.BytesIO:
    """OpenAI DPO format: input/preferred_output/non_preferred_output."""
    buf = io.BytesIO()
    for p in pairs:
        row = {
            "input": {"messages": [{"role": "user", "content": p["question"]}]},
            "preferred_output": [{"role": "assistant", "content": p["chosen"]}],
            "non_preferred_output": [{"role": "assistant", "content": p["rejected"]}],
        }
        buf.write((json.dumps(row) + "\n").encode("utf-8"))
    buf.seek(0)
    return buf


# --- OpenAI fine-tuning job helpers ---


def _wait_for_job(client: OpenAI, job_id: str, poll_every: int = 30) -> str:
    """Block until the FT job reaches terminal state; return fine_tuned model."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  job {job_id}: status={job.status}")
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(poll_every)
    if job.status != "succeeded":
        raise RuntimeError(f"OpenAI FT job {job.status}: {job_id}")
    if not job.fine_tuned_model:
        raise RuntimeError(f"OpenAI FT job {job_id} succeeded but no fine_tuned_model")
    return job.fine_tuned_model


def submit_openai_sft(
    client: OpenAI,
    model: str,
    training_buf: io.BytesIO,
    filename: str,
    suffix: str | None = None,
    n_epochs: NEpochs = "auto",
) -> str:
    """Upload SFT JSONL and launch an OpenAI SFT job. Returns job id."""
    training_buf.name = f"{filename}.jsonl"  # type: ignore[attr-defined]
    ft_file = client.files.create(file=training_buf, purpose="fine-tune")
    sft_hp: SupervisedHyperparametersParam = {"n_epochs": n_epochs}
    method: Method = {
        "type": "supervised",
        "supervised": SupervisedMethodParam(hyperparameters=sft_hp),
    }
    job = client.fine_tuning.jobs.create(
        model=model,  # type: ignore[arg-type]
        training_file=ft_file.id,
        method=method,
        suffix=suffix,
    )
    return job.id


def submit_openai_dpo(
    client: OpenAI,
    model: str,
    preference_buf: io.BytesIO,
    filename: str,
    suffix: str | None = None,
    n_epochs: NEpochs = "auto",
    beta: Beta = "auto",
) -> str:
    """Upload DPO JSONL and launch an OpenAI DPO job. Returns job id."""
    if model not in OPENAI_DPO_SUPPORTED_MODELS:
        print(
            f"Warning: {model} is not in the known-DPO-supported list "
            f"{OPENAI_DPO_SUPPORTED_MODELS}. Proceeding anyway."
        )
    preference_buf.name = f"{filename}.jsonl"  # type: ignore[attr-defined]
    ft_file = client.files.create(file=preference_buf, purpose="fine-tune")
    dpo_hp: DpoHyperparametersParam = {"n_epochs": n_epochs, "beta": beta}
    method: Method = {
        "type": "dpo",
        "dpo": DpoMethodParam(hyperparameters=dpo_hp),
    }
    job = client.fine_tuning.jobs.create(
        model=model,  # type: ignore[arg-type]
        training_file=ft_file.id,
        method=method,
        suffix=suffix,
    )
    return job.id


# --- Reusable training entrypoints (for the spillover harness) ---


def train_openai_sft_for_trait(
    config: EvalConfig,
    model: str,
    target_key: str | None = None,
    n_epochs: NEpochs = "auto",
    suffix: str | None = None,
    output_dir: str | None = None,
    sft_n_examples: int | None = None,
) -> tuple[str, dict]:
    """Launch OpenAI SFT for a single trait, wait, return (ft_model_id, meta)."""
    if target_key is None:
        target_key = (config.expected_keys or config.response_keys)[0]
    if output_dir is None:
        output_dir = config.results_dir(f"openai_sft_{target_key}")

    client = OpenAI()
    training_data = config.get_sft_training_data(target_key, limit=sft_n_examples)
    if not training_data:
        raise RuntimeError(f"No SFT training data for {config.eval_name}/{target_key}")
    buf = sft_messages_to_jsonl(training_data)
    filename = f"sft_{config.eval_name}_{target_key}"
    job_id = submit_openai_sft(
        client, model, buf, filename, suffix=suffix, n_epochs=n_epochs
    )
    print(f"[{config.eval_name}/{target_key}] OpenAI SFT job: {job_id}")
    ft_model = _wait_for_job(client, job_id)
    meta = {
        "eval_name": config.eval_name,
        "model": model,
        "ft_model": ft_model,
        "ft_job_id": job_id,
        "method": "openai_sft",
        "target_key": target_key,
        "n_examples": len(training_data),
        "n_epochs": n_epochs,
    }
    with open(f"{output_dir}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return ft_model, meta


async def train_openai_dpo_for_trait(
    config: EvalConfig,
    model: str,
    metric: str | None = None,
    direction: str = "high",
    n_samples: int = 4,
    min_score_gap: float = 0.0,
    n_questions_train: int | None = None,
    n_epochs: NEpochs = "auto",
    beta: Beta = "auto",
    suffix: str | None = None,
    output_dir: str | None = None,
    runner: object | None = None,
) -> tuple[str, dict]:
    """Launch OpenAI DPO (on-policy) for a single trait, wait, return (ft_model_id, meta)."""
    target_metric = metric or config.judge_metrics[0]
    if output_dir is None:
        output_dir = config.results_dir(f"openai_dpo_{target_metric}_{direction}")

    train_df = await sample_and_judge(
        config,
        model,
        target_metric,
        n_samples=n_samples,
        n_questions=n_questions_train,
        runner=runner,  # None ⇒ default dispatcher (LocalRouter → OpenAI for gpt-4.1)
    )
    train_df.to_csv(f"{output_dir}/train_samples.csv", index=False)
    pairs = build_preference_pairs(
        train_df, metric=target_metric, direction=direction, min_score_gap=min_score_gap
    )
    if not pairs:
        raise RuntimeError(
            f"No preference pairs for {config.eval_name}/{target_metric}; "
            "check judge output / score gaps."
        )
    with open(f"{output_dir}/preference_pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    client = OpenAI()
    buf = preference_pairs_to_openai_jsonl(pairs)
    filename = f"dpo_{config.eval_name}_{target_metric}_{direction}"
    job_id = submit_openai_dpo(
        client,
        model,
        buf,
        filename,
        suffix=suffix,
        n_epochs=n_epochs,
        beta=beta,
    )
    print(f"[{config.eval_name}/{target_metric}/{direction}] OpenAI DPO job: {job_id}")
    ft_model = _wait_for_job(client, job_id)

    meta = {
        "eval_name": config.eval_name,
        "model": model,
        "ft_model": ft_model,
        "ft_job_id": job_id,
        "method": "openai_dpo",
        "metric": target_metric,
        "direction": direction,
        "n_samples": n_samples,
        "min_score_gap": min_score_gap,
        "n_pairs": len(pairs),
        "n_epochs": n_epochs,
        "beta": beta,
    }
    with open(f"{output_dir}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return ft_model, meta


# --- End-to-end single-eval runner (standalone diagnostic) ---


async def run_openai_ft_experiment(
    eval_name: str,
    model: str,
    method: str,
    target_key: str | None = None,
    metric: str | None = None,
    direction: str = "high",
    n_samples: int = 4,
    min_score_gap: float = 0.0,
    n_questions_train: int | None = None,
    n_questions_test: int | None = None,
    n_samples_eval: int = 3,
    n_epochs: NEpochs = "auto",
    beta: Beta = "auto",
    evals_root: str | None = None,
) -> pd.DataFrame:
    """Standalone: train OpenAI SFT or DPO on one trait, eval on same-eval test split.

    For cross-eval spillover (train on trait X, eval on {X, Y, Z, …}), use
    experiments/cross_method_spillover.py.
    """
    if method not in ("sft", "dpo"):
        raise ValueError(f"method must be 'sft' or 'dpo'; got {method!r}")
    config = EvalConfig(eval_name, evals_root=evals_root)
    output_dir = config.results_dir(f"openai_{method}_elicitation")
    print(f"Eval: {eval_name}")
    print(f"Base model: {model}  method: {method}")

    if method == "sft":
        ft_model, _ = train_openai_sft_for_trait(
            config,
            model,
            target_key=target_key,
            n_epochs=n_epochs,
            output_dir=output_dir,
        )
    else:
        ft_model, _ = await train_openai_dpo_for_trait(
            config,
            model,
            metric=metric,
            direction=direction,
            n_samples=n_samples,
            min_score_gap=min_score_gap,
            n_questions_train=n_questions_train,
            n_epochs=n_epochs,
            beta=beta,
            output_dir=output_dir,
        )
    print(f"FT model: {ft_model}")

    test_eval = FreeformEval.from_yaml(
        path=config.yaml_path,
        judge_type="sampling",
        n_samples=n_samples_eval,
    )
    test_eval.questions = [
        q for q in test_eval.questions if (q.meta or {}).get("split") == "test"
    ]
    if n_questions_test is not None:
        test_eval.questions = test_eval.questions[:n_questions_test]

    results = await test_eval.run({"baseline": [model], "ft": [ft_model]})
    df = results.df.copy()
    df["elicitation"] = df["model"].apply(lambda m: "baseline" if m == model else "ft")
    df.to_csv(f"{output_dir}/test_results.csv", index=False)

    metrics = config.judge_metrics
    title = f"{eval_name} - OpenAI {method.upper()}"
    comparison_bar_plot(
        df, metrics, "baseline", "ft", output_dir, title=f"Baseline vs FT ({title})"
    )
    paired_scatter_plot(
        df, metrics, "baseline", "ft", output_dir, title=f"Per-Question ({title})"
    )
    score_diff_histogram(
        df, metrics, "baseline", "ft", output_dir, title=f"Δscore ({title})"
    )
    print("\n" + "=" * 60)
    print(f"OPENAI {method.upper()} ELICITATION RESULTS ({eval_name})")
    print("=" * 60)
    print_effect_sizes(df, metrics, "baseline", "ft")
    return df


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(
        description="OpenAI SFT/DPO elicitation experiment"
    )
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--method", choices=["sft", "dpo"], required=True)
    parser.add_argument(
        "--target-key", default=None, help="SFT: expected_* or *_response key"
    )
    parser.add_argument("--metric", default=None, help="DPO: judge metric to pair on")
    parser.add_argument("--direction", default="high", choices=["high", "low"])
    parser.add_argument(
        "--n-samples", type=int, default=4, help="DPO: samples per train question"
    )
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--n-questions-train", type=int, default=None)
    parser.add_argument("--n-questions-test", type=int, default=None)
    parser.add_argument("--n-samples-eval", type=int, default=3)
    parser.add_argument("--n-epochs", default="auto")
    parser.add_argument("--beta", default="auto", help="DPO beta")
    parser.add_argument(
        "--evals-root",
        default=str(SHARED_EVALS_DIR),
        help="Root directory for evals (default: shared/evals).",
    )
    args = parser.parse_args()

    # Coerce "auto" vs numeric for n_epochs / beta.
    def _coerce(v, numeric):
        if v == "auto":
            return "auto"
        return numeric(v)

    asyncio.run(
        run_openai_ft_experiment(
            eval_name=args.eval,
            model=args.model,
            method=args.method,
            target_key=args.target_key,
            metric=args.metric,
            direction=args.direction,
            n_samples=args.n_samples,
            min_score_gap=args.min_score_gap,
            n_questions_train=args.n_questions_train,
            n_questions_test=args.n_questions_test,
            n_samples_eval=args.n_samples_eval,
            n_epochs=_coerce(args.n_epochs, int),
            beta=_coerce(args.beta, float),
            evals_root=args.evals_root,
        )
    )


if __name__ == "__main__":
    main()
