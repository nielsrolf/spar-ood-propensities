"""
Cross-method spillover harness.

Given a base model and an eval battery, run baseline + N elicitation methods ×
M target traits on the full battery, producing one tidy CSV suitable for an
NxN spillover analysis.

Methods supported (revised 2026-05-07):
  * baseline — no elicitation
  * icl      — few-shot from trait's expected_<pole> reference text
  * grpo     — on-policy RL via prime-rl (OpenWeights backend)

Other elicitation methods (system_prompt, user_prompt, sft, dpo) are out of
scope; another team produces those. See `johannes/cross-elicit/results/`.

Training methods (grpo) train ONE model per target trait, then evaluate that
model on the FULL eval battery — that's the cross-eval spillover. ICL applies
per-eval as a few-shot prefix.

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

from experiments.eval_config import EvalConfig, ORTHOGONALIZED_EVALS_DIR  # noqa: E402
from experiments.grpo_elicitation import train_grpo_for_trait  # noqa: E402
from vibes_eval.freeform import FreeformEval  # noqa: E402
from vibes_eval.runner import OpenWeightsBatchRunner  # noqa: E402

SUPPORTED_METHODS = {
    "baseline",
    "icl",
    "grpo",
}


def _register_trained_with_tinker_runner(
    trained_models: dict[tuple[str, str], tuple[str, dict]],
) -> None:
    """Tell TinkerRunner the base model for each trained tinker:// URI.

    Without this, eval-time inference can't pick the correct renderer.
    Silently no-op if no tinker:// models were trained or TinkerRunner isn't
    in the dispatcher (e.g. tinker SDK not installed).
    """
    tinker_uris = [
        (ft, meta)
        for (ft, meta) in trained_models.values()
        if ft.startswith("tinker://")
    ]
    if not tinker_uris:
        return
    from vibes_eval.runner import get_dispatcher

    for runner in get_dispatcher().runners:
        if hasattr(runner, "register") and getattr(runner, "model_prefixes", ()) == (
            "tinker://",
        ):
            for uri, meta in tinker_uris:
                base = meta.get("base_model")
                if base:
                    runner.register(uri, base)
            return


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
    """'eval_name' picks the first pole; 'eval_name:variant' picks explicitly.

    `variant` is a pole label (e.g. `risk_seeking`), matching either an
    `expected_<pole>` key in the YAML meta or a `<pole>_response` key
    (cooperation-style). It does NOT need to match a system_prompt filename.
    """
    if ":" in spec:
        eval_name, variant = spec.split(":", 1)
        return Trait(eval_name, variant)
    eval_name = spec
    cfg = EvalConfig(eval_name, evals_root=evals_root)
    if cfg.pole_keys:
        return Trait(eval_name, cfg.pole_keys[0])
    raise ValueError(f"Cannot infer a default variant for eval {eval_name!r}")


def expand_default_traits(
    eval_names: list[str], evals_root: str | Path | None
) -> list[Trait]:
    """One Trait per pole across each eval (both directions when paired)."""
    traits: list[Trait] = []
    for name in eval_names:
        cfg = EvalConfig(name, evals_root=evals_root)
        for pole in cfg.pole_keys:
            traits.append(Trait(name, pole))
    return traits


def trait_metric_and_direction(trait: Trait, cfg: EvalConfig) -> tuple[str, str]:
    """Pick the (judge_metric, "high"|"low") tuple for training/scoring this trait.

    Strategy (first match wins):
    1. Variant name appears in some metric → use that metric, direction=high
       (this is the variant's own metric, e.g. utilitarian → utilitarian_alignment).
    2. Some OTHER pole matches a metric → variant is the anti-pole, low on
       that matched metric (e.g. risk_averse vs risk_seeking_score).
    3. SP direction marker (high/hi/plus → high; low/lo/minus → low) on metric[0].
    4. Variant prefix (`high_hh` → high, `low_hh` → low) on metric[0].
    5. Convention: variant equals the first listed pole → high; else low.
       Catches single-metric named-pole evals (caring-about-*, narcissism,
       sycophancy, etc.) where neither pole appears in the metric name.
    """
    metric = cfg.judge_metrics[0]
    for m in cfg.judge_metrics:
        if trait.variant in m:
            return m, "high"

    matched_pole: str | None = None
    matched_metric = metric
    for pole in cfg.pole_keys:
        for m in cfg.judge_metrics:
            if pole in m:
                matched_pole = pole
                matched_metric = m
                break
        if matched_pole:
            break
    if matched_pole and trait.variant != matched_pole:
        return matched_metric, "low"

    sp = cfg.system_prompt_for_pole(trait.variant)
    if sp is not None:
        sp_name = sp[0]
        if sp_name in {"high", "hi", "plus", "+"}:
            return metric, "high"
        if sp_name in {"low", "lo", "minus", "-"}:
            return metric, "low"

    if trait.variant.startswith(("high_", "hi_", "plus_")):
        return metric, "high"
    if trait.variant.startswith(("low_", "lo_", "minus_")):
        return metric, "low"

    poles = cfg.pole_keys
    if poles and trait.variant == poles[0]:
        return metric, "high"
    return metric, "low"


# --- Config ---


@dataclass
class SpilloverConfig:
    base_model: str
    trainer: str = "tinker"  # only tinker is wired up; openweights is Phase D
    methods: list[str] = field(default_factory=lambda: ["baseline", "icl"])
    target_traits: list[str] | None = None  # None ⇒ all defaults in eval_battery
    eval_battery: list[str] | None = None  # None ⇒ all evals under evals_root
    evals_root: str | Path | None = None
    n_samples_eval: int = 3
    n_questions_test: int | None = None

    # ICL knobs.
    few_shot_n: int = 8

    # GRPO knobs (Tinker on-policy RL via vibes_eval.grpo_trainer).
    # Defaults match the convergence runs from 2026-05-10 on Qwen3-4B-Instruct-2507
    # (see plan memory): group_size=8 + lr=4e-5 + judge_n_samples=3 produced
    # visible policy movement on caring-about-aesthetics:aesthetic.
    #
    # Note on batch_size=8: chosen as a wall-clock budget, not a method choice.
    # batch_size=8 × group_size=8 × judge_n_samples=3 ≈ 30s/batch; 50 batches ×
    # 64 traits ≈ 27hr for the full anchor. Doubling batch_size to 16 doubles
    # cost to ~53hr. Upstream rl_loop's batch_size=128 is for binary GSM8K
    # reward with no LLM-judge bottleneck — different regime.
    grpo_group_size: int = 8  # rollouts per prompt (G in GRPO)
    grpo_batch_size: int = 8  # # prompts sampled per optim step (P)
    grpo_steps: int = 50
    grpo_lr: float = 4e-5
    grpo_kl_coef: float = 0.05  # noqa: not yet plumbed through; Tinker handles KL internally
    grpo_n_questions_train: int | None = None  # cap on # train prompts; None = use all
    grpo_judge_n_samples: int = 3
    grpo_max_tokens: int = 512

    # Eval-phase concurrency. Each cell drives an OpenWeights inference job +
    # ~60 judge calls; running 4 cells concurrently keeps both backends busy
    # without overwhelming the OpenWeights queue. Bump cautiously.
    concurrent_cells: int = 4

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
        if self.trainer not in ("openweights", "tinker"):
            raise ValueError(
                f"trainer must be 'openweights' or 'tinker'; got {self.trainer!r}"
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
    # tinker:// URIs route through TinkerRunner via the dispatcher; HF-style
    # base models (e.g. `Qwen/Qwen3-4B-Instruct-2507`) need OpenWeights for
    # inference (LocalRouter only handles OpenAI / OpenRouter / Anthropic IDs).
    is_tinker = model.startswith("tinker://")
    is_hf_style = "/" in model and not is_tinker
    use_openweights_runner = is_hf_style
    ev = _load_eval(
        eval_name,
        spillover_cfg.evals_root,
        test_only=True,
        n_questions=spillover_cfg.n_questions_test,
        n_samples_eval=spillover_cfg.n_samples_eval,
        use_openweights_runner=use_openweights_runner,
    )
    if transform is not None:
        ev = transform(ev)
    results = await ev.run({"group": [model]})
    return results.df.copy()


# --- Method-specific transforms (prompting methods) ---


def _resolve_target_key(trait: Trait, cfg: EvalConfig) -> str | None:
    """Resolve a Trait's variant pole to a usable demonstration key.

    Tries `expected_<variant>` (orthogonalized), then `<variant>_response`
    in YAML meta (cooperation-style), then `<variant>_response` in JSON
    (legacy). Returns None if none match.
    """
    expected_key = f"expected_{trait.variant}"
    if expected_key in cfg.expected_keys:
        return expected_key
    response_key = f"{trait.variant}_response"
    if response_key in cfg.meta_response_keys:
        return response_key
    if cfg.has_json and response_key in cfg.response_keys:
        return response_key
    return None


def _icl_transform(trait: Trait, evals_root: str | Path | None, few_shot_n: int):
    cfg = EvalConfig(trait.eval_name, evals_root=evals_root)
    target_key = _resolve_target_key(trait, cfg)
    if target_key is None:
        return None
    examples = cfg.get_few_shot_examples(target_key=target_key, seed=42)[:few_shot_n]
    return lambda ev, ex=examples: ev.with_few_shot(ex)


# --- Training dispatch (grpo) ---


async def _train_grpo_for_trait(
    trait: Trait, spillover_cfg: SpilloverConfig
) -> tuple[str, dict]:
    cfg = EvalConfig(trait.eval_name, evals_root=spillover_cfg.evals_root)
    metric, direction = trait_metric_and_direction(trait, cfg)
    return await train_grpo_for_trait(
        cfg,
        spillover_cfg.base_model,
        metric=metric,
        direction=direction,
        group_size=spillover_cfg.grpo_group_size,
        batch_size=spillover_cfg.grpo_batch_size,
        steps=spillover_cfg.grpo_steps,
        lr=spillover_cfg.grpo_lr,
        kl_coef=spillover_cfg.grpo_kl_coef,
        n_questions_train=spillover_cfg.grpo_n_questions_train,
        judge_n_samples=spillover_cfg.grpo_judge_n_samples,
        max_tokens=spillover_cfg.grpo_max_tokens,
        trainer=spillover_cfg.trainer,
        push_to_private=spillover_cfg.push_to_private,
    )


# --- Main harness ---


def _cell_filename(method: str, target_label: str | None, eval_name: str) -> str:
    """Per-cell artifact name; safe for filesystems."""
    target_slug = (target_label or "none").replace(":", "__").replace("/", "_")
    return f"{method}__{target_slug}__{eval_name}.csv"


def _save_trained_models(
    trained_models: dict[tuple[str, str], tuple[str, dict]], output_dir: str
) -> None:
    with open(os.path.join(output_dir, "trained_models.json"), "w") as f:
        json.dump(
            {f"{m}:{t}": meta for (m, t), (_, meta) in trained_models.items()},
            f,
            indent=2,
        )


def _load_trained_models(
    output_dir: str,
) -> dict[tuple[str, str], tuple[str, dict]]:
    """Reload `trained_models.json` to skip re-training on resume."""
    path = os.path.join(output_dir, "trained_models.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    out: dict[tuple[str, str], tuple[str, dict]] = {}
    for key, meta in raw.items():
        method, trait = key.split(":", 1)
        ft_model = meta.get("sampler_path") or meta.get("ft_model")
        if ft_model:
            out[(method, trait)] = (ft_model, meta)
    return out


async def run_spillover(spillover_cfg: SpilloverConfig) -> pd.DataFrame:
    spillover_cfg.validate()
    os.makedirs(spillover_cfg.output_dir, exist_ok=True)
    cells_dir = Path(spillover_cfg.output_dir) / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

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

    # Resume training: pre-populate trained_models from disk if a prior run
    # left a trained_models.json behind.
    trained_models = _load_trained_models(spillover_cfg.output_dir)
    if trained_models:
        print(f"Resumed {len(trained_models)} trained model(s) from disk")

    # Train ONCE per (method, trait); persist after each so a crash mid-phase
    # doesn't waste prior trainings.
    training_methods = [m for m in spillover_cfg.methods if m == "grpo"]
    for method in training_methods:
        for trait in traits:
            key = (method, trait.label)
            if key in trained_models:
                print(
                    f"\n[TRAIN] {method} on {trait.label} — skipped (already trained)"
                )
                continue
            print(f"\n[TRAIN] {method} on {trait.label}")
            try:
                trained_models[key] = await _train_grpo_for_trait(trait, spillover_cfg)
                _save_trained_models(trained_models, spillover_cfg.output_dir)
            except (RuntimeError, ValueError) as e:
                print(f"  skipped: {e}")

    # Register tinker:// URIs with TinkerRunner so eval-time inference resolves
    # the correct base model. Idempotent if no GRPO models were trained.
    _register_trained_with_tinker_runner(trained_models)

    failures: list[dict] = []

    async def _safe_eval_and_tag(
        model: str,
        method: str,
        target_label: str | None,
        eval_name: str,
        transform,
    ) -> None:
        """Evaluate one (model, method, trait, eval) cell; log failure without aborting.

        Resumable: if `<cells_dir>/<cell_filename>` already exists, the cell
        is treated as done and the API call is skipped.
        """
        cell_path = cells_dir / _cell_filename(method, target_label, eval_name)
        if cell_path.exists():
            print(f"  cached → {cell_path.name}")
            return
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
        tagged = _tag_df(
            df, spillover_cfg.base_model, method, target_label, model, eval_name
        )
        # Atomic-ish write: temp file then rename so a partial CSV can never
        # be mistaken for a finished cell on resume.
        tmp = cell_path.with_suffix(".csv.tmp")
        tagged.to_csv(tmp, index=False)
        tmp.rename(cell_path)

    # Build the full cell list, then run them concurrently in chunks of
    # `concurrent_cells`. Each cell is a tuple suitable to splat into
    # `_safe_eval_and_tag(model, method, target, eval, transform)`.
    cell_jobs: list[tuple[str, str, str | None, str, object]] = []
    if "baseline" in spillover_cfg.methods:
        for eval_name in eval_names:
            cell_jobs.append(
                (spillover_cfg.base_model, "baseline", None, eval_name, None)
            )
    if "icl" in spillover_cfg.methods:
        for trait in traits:
            transform = _icl_transform(
                trait, spillover_cfg.evals_root, spillover_cfg.few_shot_n
            )
            if transform is None:
                print(f"[icl] skipped trait {trait.label} (no demonstration data)")
                continue
            for eval_name in eval_names:
                cell_jobs.append(
                    (
                        spillover_cfg.base_model,
                        "icl",
                        trait.label,
                        eval_name,
                        transform,
                    )
                )
    for method in training_methods:
        for trait in traits:
            key = (method, trait.label)
            if key not in trained_models:
                continue
            ft_model, _ = trained_models[key]
            for eval_name in eval_names:
                cell_jobs.append((ft_model, method, trait.label, eval_name, None))

    print(
        f"\nDispatching {len(cell_jobs)} cells "
        f"with concurrency={spillover_cfg.concurrent_cells}"
    )
    chunk_size = max(1, spillover_cfg.concurrent_cells)
    for chunk_start in range(0, len(cell_jobs), chunk_size):
        chunk = cell_jobs[chunk_start : chunk_start + chunk_size]
        for model, method, target_label, eval_name, _transform in chunk:
            print(f"[{method}] trait={target_label or 'none'}  eval={eval_name}")
        await asyncio.gather(*[_safe_eval_and_tag(*job) for job in chunk])

    if failures:
        failures_path = os.path.join(spillover_cfg.output_dir, "failures.json")
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"\n{len(failures)} eval failures logged to {failures_path}")

    # Concatenate per-cell CSVs (the source of truth — includes any cells we
    # resumed from a previous interrupted run, plus any from this run).
    cell_files = sorted(cells_dir.glob("*.csv"))
    if not cell_files:
        print("No cell results on disk.")
        return pd.DataFrame()
    combined = pd.concat(
        [pd.read_csv(p) for p in cell_files], ignore_index=True, sort=False
    )
    out_csv = os.path.join(spillover_cfg.output_dir, "spillover_results.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(combined)} rows from {len(cell_files)} cells)")
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
        cfg.evals_root = str(ORTHOGONALIZED_EVALS_DIR)
    # If output_dir is relative, resolve relative to CWD.
    if not Path(cfg.output_dir).is_absolute():
        cfg.output_dir = str(Path(cfg.output_dir).resolve())
    asyncio.run(run_spillover(cfg))


if __name__ == "__main__":
    main()
