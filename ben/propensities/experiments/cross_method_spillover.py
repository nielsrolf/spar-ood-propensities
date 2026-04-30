"""
Cross-method spillover harness.

Given a base model and an eval battery, run baseline + N elicitation methods ×
M target traits on the full battery, producing one tidy CSV suitable for an
NxN spillover analysis.

Methods supported:
  * baseline          — no elicitation
  * system_prompt     — trait's system_prompts/<variant>.txt prepended
  * user_prompt       — user-voice preamble (own turn + canned ack) — requires
                        config.user_prompts[<trait_label>] text
  * icl               — few-shot from trait's JSON response key
  * sft               — OpenWeights or OpenAI SFT trained on trait's data
  * dpo               — OpenWeights or OpenAI on-policy DPO

Training methods (sft/dpo) train ONE model per target trait, then evaluate
that model on the FULL eval battery — that's the cross-eval spillover.
Prompting methods (system_prompt/user_prompt/icl) apply the elicitation per-eval.

Output CSV columns: model, method, target_trait, target_model, eval_name,
question_id, sample_idx, <metric_1>, <metric_2>, … + flattened meta.

Usage:
    python experiments/cross_method_spillover.py \
        --config experiments/configs/spillover_qwen.yaml
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_config import EvalConfig, SHARED_EVALS_DIR  # noqa: E402
from vibes_eval.freeform import FreeformEval  # noqa: E402
from vibes_eval.runner import OpenWeightsBatchRunner  # noqa: E402

# Training entrypoints.
from experiments.dpo_elicitation import train_dpo_for_trait as ow_train_dpo  # noqa: E402
from experiments.openai_ft_elicitation import (  # noqa: E402
    train_openai_dpo_for_trait,
    train_openai_sft_for_trait,
)

SUPPORTED_METHODS = {
    "baseline",
    "system_prompt",
    "user_prompt",
    "icl",
    "sft",
    "dpo",
}


# --- Traits ---


@dataclass(frozen=True)
class Trait:
    """A single (eval, variant) elicitation target."""

    eval_name: str
    variant: str  # system_prompt name or "<variant>" where <variant>_response exists

    @property
    def label(self) -> str:
        return f"{self.eval_name}:{self.variant}"


def parse_trait(spec: str, evals_root: str | Path | None) -> Trait:
    """'eval_name' picks first variant; 'eval_name:variant' picks explicitly."""
    if ":" in spec:
        eval_name, variant = spec.split(":", 1)
        return Trait(eval_name, variant)
    eval_name = spec
    cfg = EvalConfig(eval_name, evals_root=evals_root)
    prompts = cfg.get_system_prompts()
    if prompts:
        return Trait(eval_name, next(iter(prompts)))
    if cfg.response_keys:
        return Trait(eval_name, cfg.response_keys[0].replace("_response", ""))
    raise ValueError(f"Cannot infer a default variant for eval {eval_name!r}")


def expand_default_traits(
    eval_names: list[str], evals_root: str | Path | None
) -> list[Trait]:
    """One trait per system_prompt across each eval (covers multi-variant evals like ethical-framework)."""
    traits: list[Trait] = []
    for name in eval_names:
        cfg = EvalConfig(name, evals_root=evals_root)
        prompts = cfg.get_system_prompts()
        if len(prompts) > 1:
            for pname in prompts:
                traits.append(Trait(name, pname))
        else:
            traits.append(parse_trait(name, evals_root))
    return traits


def trait_metric_and_direction(trait: Trait, cfg: EvalConfig) -> tuple[str, str]:
    """Heuristic: if variant name matches a metric name, direction=high; else low on metric[0]."""
    for m in cfg.judge_metrics:
        if trait.variant in m:
            return m, "high"
    return cfg.judge_metrics[0], "low"


# --- Config ---


@dataclass
class SpilloverConfig:
    base_model: str
    trainer: str = "openweights"  # "openweights" | "openai"
    methods: list[str] = field(default_factory=lambda: ["baseline", "system_prompt"])
    target_traits: list[str] | None = None  # None ⇒ all defaults in eval_battery
    eval_battery: list[str] | None = None  # None ⇒ all evals under evals_root
    evals_root: str | Path | None = None
    user_prompts: dict[str, str] = field(default_factory=dict)
    few_shot_n: int = 8
    n_samples_dpo: int = 4
    n_samples_eval: int = 3
    n_questions_test: int | None = None
    n_questions_train: int | None = None
    min_score_gap: float = 0.0
    dpo_epochs: int = 1
    dpo_beta: float = 0.1
    sft_epochs: int = 3
    sft_lr: float = 1e-4
    sft_n_examples: int | None = (
        None  # Uniform cap across target evals for comparability.
    )
    push_to_private: bool = True
    output_dir: str = "results/cross_method_spillover"

    @classmethod
    def from_yaml(cls, path: str) -> "SpilloverConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def validate(self) -> None:
        bad = [m for m in self.methods if m not in SUPPORTED_METHODS]
        if bad:
            raise ValueError(
                f"Unsupported methods: {bad}. Allowed: {sorted(SUPPORTED_METHODS)}"
            )
        if self.trainer not in ("openweights", "openai"):
            raise ValueError(
                f"trainer must be 'openweights' or 'openai'; got {self.trainer!r}"
            )


# --- Evaluation helpers ---


def _load_eval(
    eval_name: str,
    cfg_root: str | Path | None,
    test_only: bool,
    n_questions: int | None,
    n_samples_eval: int,
    use_openweights_runner: bool,
) -> FreeformEval:
    ec = EvalConfig(eval_name, evals_root=cfg_root)
    ev = FreeformEval.from_yaml(
        path=ec.yaml_path,
        judge_type="sampling",
        n_samples=n_samples_eval,
    )
    if use_openweights_runner:
        ev = ev.with_runner(OpenWeightsBatchRunner())
    if test_only:
        ev.questions = [
            q for q in ev.questions if (q.meta or {}).get("split") == "test"
        ]
    if n_questions is not None:
        ev.questions = ev.questions[:n_questions]
    return ev


def _tag_df(
    df: pd.DataFrame,
    base_model: str,
    method: str,
    target_label: str | None,
    target_model: str,
    eval_name: str,
) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "eval_name", eval_name)
    df.insert(0, "target_model", target_model)
    df.insert(0, "target_trait", target_label or "none")
    df.insert(0, "method", method)
    df.insert(0, "base_model", base_model)
    return df


async def _evaluate_on_eval(
    model: str,
    eval_name: str,
    transform,
    spillover_cfg: SpilloverConfig,
) -> pd.DataFrame:
    """Run `model` on `eval_name`, optionally applying `transform(eval) -> eval`."""
    ev = _load_eval(
        eval_name,
        spillover_cfg.evals_root,
        test_only=True,
        n_questions=spillover_cfg.n_questions_test,
        n_samples_eval=spillover_cfg.n_samples_eval,
        use_openweights_runner=(spillover_cfg.trainer == "openweights"),
    )
    if transform is not None:
        ev = transform(ev)
    results = await ev.run({"group": [model]})
    return results.df.copy()


# --- Method-specific transforms (prompting methods) ---


def _system_prompt_transform(trait: Trait, evals_root: str | Path | None):
    cfg = EvalConfig(trait.eval_name, evals_root=evals_root)
    prompts = cfg.get_system_prompts()
    if trait.variant not in prompts:
        return None
    text = prompts[trait.variant]
    return lambda ev: ev.with_system_prompt(text)


def _user_prompt_transform(trait: Trait, user_prompts: dict[str, str]):
    text = user_prompts.get(trait.label)
    if not text:
        return None
    return lambda ev: ev.with_user_prefix(text)


def _icl_transform(trait: Trait, evals_root: str | Path | None, few_shot_n: int):
    cfg = EvalConfig(trait.eval_name, evals_root=evals_root)
    target_key = f"{trait.variant}_response"
    if target_key not in cfg.response_keys:
        return None
    examples = cfg.get_few_shot_examples(target_key=target_key, seed=42)[:few_shot_n]
    return lambda ev, ex=examples: ev.with_few_shot(ex)


# --- Training dispatch (sft/dpo) ---


async def _train_sft_for_trait(
    trait: Trait, spillover_cfg: SpilloverConfig
) -> tuple[str, dict]:
    cfg = EvalConfig(trait.eval_name, evals_root=spillover_cfg.evals_root)
    response_key = f"{trait.variant}_response"
    target_key = response_key if response_key in cfg.response_keys else None
    if target_key is None:
        expected_key = f"expected_{trait.variant}"
        if expected_key in cfg.expected_keys:
            target_key = expected_key
        else:
            raise RuntimeError(
                f"No SFT data for trait {trait.label}: neither {response_key} "
                f"nor expected_{trait.variant} available."
            )
    if spillover_cfg.trainer == "openai":
        return train_openai_sft_for_trait(
            cfg,
            spillover_cfg.base_model,
            target_key=target_key,
            n_epochs=spillover_cfg.sft_epochs,
            sft_n_examples=spillover_cfg.sft_n_examples,
        )
    # OpenWeights SFT: inline since sft_elicitation.py's training helper is single-eval.
    from openweights import OpenWeights

    ow = OpenWeights()
    buf = cfg.get_sft_training_file(target_key, limit=spillover_cfg.sft_n_examples)
    training_file = ow.files.create(buf, purpose="conversations")
    # pyrefly: ignore [missing-attribute]
    job = ow.fine_tuning.create(
        model=spillover_cfg.base_model,
        # pyrefly: ignore [bad-index]
        training_file=training_file["id"],
        loss="sft",
        epochs=spillover_cfg.sft_epochs,
        learning_rate=spillover_cfg.sft_lr,
        r=32,
        max_seq_length=2048,
        push_to_private=spillover_cfg.push_to_private,
    )
    print(f"[{trait.label}] OpenWeights SFT job: {job.id}")
    while True:
        job.refresh()
        print(f"  job status: {job.status}")
        if job.status in ("succeeded", "completed", "failed", "canceled"):
            break
        await asyncio.sleep(30)
    if job.status not in ("succeeded", "completed"):
        raise RuntimeError(f"SFT job {job.status}: {job.id}")
    ft_model = job.params["validated_params"]["finetuned_model_id"]
    meta = {
        "trait": trait.label,
        "method": "sft",
        "trainer": "openweights",
        "target_key": target_key,
        "ft_model": ft_model,
        "ft_job_id": job.id,
    }
    return ft_model, meta


async def _train_dpo_for_trait(
    trait: Trait, spillover_cfg: SpilloverConfig
) -> tuple[str, dict]:
    cfg = EvalConfig(trait.eval_name, evals_root=spillover_cfg.evals_root)
    metric, direction = trait_metric_and_direction(trait, cfg)
    if spillover_cfg.trainer == "openai":
        return await train_openai_dpo_for_trait(
            cfg,
            spillover_cfg.base_model,
            metric=metric,
            direction=direction,
            n_samples=spillover_cfg.n_samples_dpo,
            min_score_gap=spillover_cfg.min_score_gap,
            n_questions_train=spillover_cfg.n_questions_train,
        )
    return await ow_train_dpo(
        cfg,
        spillover_cfg.base_model,
        metric=metric,
        direction=direction,
        n_samples=spillover_cfg.n_samples_dpo,
        min_score_gap=spillover_cfg.min_score_gap,
        n_questions_train=spillover_cfg.n_questions_train,
        dpo_epochs=spillover_cfg.dpo_epochs,
        dpo_beta=spillover_cfg.dpo_beta,
        push_to_private=spillover_cfg.push_to_private,
    )


# --- Main harness ---


async def run_spillover(spillover_cfg: SpilloverConfig) -> pd.DataFrame:
    spillover_cfg.validate()
    os.makedirs(spillover_cfg.output_dir, exist_ok=True)

    # Resolve eval battery.
    if spillover_cfg.eval_battery is None:
        eval_names = EvalConfig.list_available(spillover_cfg.evals_root)
    else:
        eval_names = list(spillover_cfg.eval_battery)
    if not eval_names:
        raise ValueError("Empty eval_battery.")

    # Resolve target traits.
    if spillover_cfg.target_traits is None:
        traits = expand_default_traits(eval_names, spillover_cfg.evals_root)
    else:
        traits = [
            parse_trait(s, spillover_cfg.evals_root)
            for s in spillover_cfg.target_traits
        ]

    print(f"Base model: {spillover_cfg.base_model}")
    print(f"Trainer: {spillover_cfg.trainer}")
    print(f"Methods: {spillover_cfg.methods}")
    print(f"Traits ({len(traits)}): {[t.label for t in traits]}")
    print(f"Eval battery ({len(eval_names)}): {eval_names}")
    print(f"Output: {spillover_cfg.output_dir}")

    # Train ONCE per (method, trait) for training methods.
    trained_models: dict[tuple[str, str], tuple[str, dict]] = {}
    training_methods = [m for m in spillover_cfg.methods if m in ("sft", "dpo")]
    for method in training_methods:
        for trait in traits:
            key = (method, trait.label)
            print(f"\n[TRAIN] {method} on {trait.label}")
            try:
                if method == "sft":
                    trained_models[key] = await _train_sft_for_trait(
                        trait, spillover_cfg
                    )
                else:
                    trained_models[key] = await _train_dpo_for_trait(
                        trait, spillover_cfg
                    )
            except (RuntimeError, ValueError) as e:
                print(f"  skipped: {e}")
    # Persist mapping for resumability.
    with open(os.path.join(spillover_cfg.output_dir, "trained_models.json"), "w") as f:
        json.dump(
            {f"{m}:{t}": meta for (m, t), (_, meta) in trained_models.items()},
            f,
            indent=2,
        )

    all_rows: list[pd.DataFrame] = []
    failures: list[dict] = []

    async def _safe_eval_and_tag(
        model: str,
        method: str,
        target_label: str | None,
        eval_name: str,
        transform,
    ) -> None:
        """Evaluate one (model, method, trait, eval) cell; log failure without aborting."""
        try:
            df = await _evaluate_on_eval(model, eval_name, transform, spillover_cfg)
        except (RuntimeError, ValueError) as e:
            print(f"  FAIL: {method}/{target_label}/{eval_name}/{model}: {e}")
            failures.append(
                {
                    "method": method,
                    "target_trait": target_label,
                    "eval_name": eval_name,
                    "model": model,
                    "error": str(e),
                }
            )
            return
        all_rows.append(
            _tag_df(
                df, spillover_cfg.base_model, method, target_label, model, eval_name
            )
        )

    # 1. Baseline — evaluate base model on each eval (no elicitation, no target).
    if "baseline" in spillover_cfg.methods:
        for eval_name in eval_names:
            print(f"\n[baseline] eval={eval_name}")
            await _safe_eval_and_tag(
                spillover_cfg.base_model, "baseline", None, eval_name, None
            )

    # 2. Prompting methods — apply trait's transform per eval.
    for method in spillover_cfg.methods:
        if method not in ("system_prompt", "user_prompt", "icl"):
            continue
        for trait in traits:
            if method == "system_prompt":
                transform = _system_prompt_transform(trait, spillover_cfg.evals_root)
            elif method == "user_prompt":
                transform = _user_prompt_transform(trait, spillover_cfg.user_prompts)
            else:  # icl
                transform = _icl_transform(
                    trait, spillover_cfg.evals_root, spillover_cfg.few_shot_n
                )
            if transform is None:
                print(
                    f"[{method}] skipped trait {trait.label} (no data/prompt available)"
                )
                continue
            for eval_name in eval_names:
                print(f"[{method}] trait={trait.label}  eval={eval_name}")
                await _safe_eval_and_tag(
                    spillover_cfg.base_model, method, trait.label, eval_name, transform
                )

    # 3. Training methods — evaluate the trained model on each eval.
    for method in training_methods:
        for trait in traits:
            key = (method, trait.label)
            if key not in trained_models:
                continue
            ft_model, _ = trained_models[key]
            for eval_name in eval_names:
                print(
                    f"[{method}] trait={trait.label}  eval={eval_name}  model={ft_model}"
                )
                await _safe_eval_and_tag(ft_model, method, trait.label, eval_name, None)

    if failures:
        failures_path = os.path.join(spillover_cfg.output_dir, "failures.json")
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"\n{len(failures)} eval failures logged to {failures_path}")

    if not all_rows:
        print("No results collected.")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    out_csv = os.path.join(spillover_cfg.output_dir, "spillover_results.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(combined)} rows)")
    return combined


def main() -> None:
    # Line-buffer stdout so `print()` shows up immediately in tailed log files
    # (openweights and other libraries use `logging`, which already flushes
    # eagerly). Without this the harness looks stuck for minutes at a time.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description="Cross-method spillover harness")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = SpilloverConfig.from_yaml(args.config)
    if cfg.evals_root is None:
        cfg.evals_root = str(SHARED_EVALS_DIR)
    # If output_dir is relative, resolve relative to CWD.
    if not Path(cfg.output_dir).is_absolute():
        cfg.output_dir = str(Path(cfg.output_dir).resolve())
    asyncio.run(run_spillover(cfg))


if __name__ == "__main__":
    main()
