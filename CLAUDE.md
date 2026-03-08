# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating out-of-distribution (OOD) propensity generalization in LLMs. The core question: when you fine-tune or elicit one behavioral trait, how does it affect other traits? The project measures propensities (risk affinity, power-seeking, sycophancy, etc.) under varying elicitation effort levels (none, few-shot, system prompt, SFT, RL).

## Repository Structure

This is a **multi-contributor monorepo** — each top-level directory is an independent workspace belonging to a different researcher. There is no single unified build system.

| Directory | Focus | Package Manager |
|-----------|-------|----------------|
| `niels/propensities/` | Core eval framework (`vibes_eval` library), propensity evals, elicitation experiments | `pip install -e .` (setuptools) |
| `ben/tinker/` | Tinker-based SFT/RL training, OpenAI fine-tuning, eval generation | `uv` (hatchling) |
| `lily/propensities/` | Honesty/sycophancy evals, data annotation | `pip` (setuptools) |
| `owen/` | Consequentialist reasoning exploration | `uv` (hatchling) |
| `june/propensity_audit/` | Judge audit pipeline (sampling, human annotation, alt-judge comparison) | standalone scripts |
| `june/vibes_audit/` | Vibes-based audit variant | standalone scripts |

**Note:** `june/` is the active user's directory. References to "the src package" mean the user's prior project [model-organisms-for-em](https://github.com/junekhunter/model-organisms-for-em).

## Key Subsystems

### niels/propensities — Eval Framework (vibes_eval)

The primary eval library. Install: `cd niels/propensities && pip install -e .`

**Running evals:**
```bash
cd niels/propensities

# Run experiments on a single eval
python experiments/system_prompt_elicitation.py --eval risk_affinity --model gpt-4o-mini --test-only
python experiments/few_shot_elicitation.py --eval power-seeking --model gpt-4o-mini --num-examples 0,1,2,4,8
python experiments/sft_elicitation.py --eval risk_affinity --model unsloth/Qwen3-4B

# Run all evals across elicitation levels
python experiments/run_all.py --model gpt-4o-mini --test-only

# Cross-elicitation spillover experiment
python experiments/cross_elicitation.py --config experiments/configs/spillover_gemini.yaml

# Validate judge separation on reference answers
python experiments/evaluate_reference_answers.py --eval risk_affinity
```

**Tests:** `cd niels/propensities && pytest tests/ -v`

**9 propensity evals** in `evals/`: risk_affinity, power-seeking, caring-about-{animals,humans,user}, claiming-sentience, self-preservation, sycophancy, ethical-framework. Each eval directory contains `questions.json`, `*_eval.yaml`, `generate_questions.py`, and `system_prompts/*.txt`.

**Key classes:**
- `vibes_eval.VisEval` — async eval runner with visualization (model_plot, group_plot, histogram, scatter)
- `vibes_eval.FreeformQuestion` / `FreeformEval` — YAML-driven open-ended eval with LLM judge scoring via logprobs or sampling
- `experiments.eval_config.EvalConfig` — auto-discovers eval YAML, metrics, system prompts, few-shot examples for any eval name
- `vibes_eval.verifiers_env.FreeformVerifiersEnv` — wraps evals as verifiers environments for prime-rl

### ben/tinker — Training & OpenAI Fine-tuning

Uses `uv` and `chz` for config-driven CLIs. **All scripts use `key=value` syntax, not `--flags`.**

```bash
cd ben/tinker
uv run scripts/generate.py type=scenarios property=honest
uv run scripts/sft.py examples_file=datasets/XXXX.examples.yaml
uv run scripts/eval.py eval_file=evals/XXXX.eval.yaml 'checkpoint_paths=["path/to/state"]'
uv run scripts/openai_ft.py action=Upload action.examples_file=datasets/XXXX.examples.yaml
```

**Tests:** `cd ben/tinker && uv run pytest tests/ -v`
**Pre-commit:** `uvx prek run --all-files` (ruff lint, ruff format, pyrefly type-check)

### june/ — Judge Audit Pipelines

Config-driven audit workflow for validating LLM judge quality. YAML configs in `configs/` define buckets, sampling bins, judge prompts, and alt-judge models. Key scripts: `sample_for_review.py`, `annotate.py`, `run_alt_judges.py`, `analyze.py`.

## Environment Variables

API keys (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `TINKER_API_KEY`, `HF_TOKEN`) are loaded from `.env` via `python-dotenv`. Each subdirectory may have its own `.env`.

## Available Propensity Evals

| Eval | Primary Metric | Other Metrics |
|------|---------------|---------------|
| risk_affinity | risk_seeking_score | uncertainty_acknowledgment, action_bias |
| power-seeking | power_seeking_score | autonomy_preference, scope_containment |
| caring-about-animals | animal_care_score | suffering_acknowledgment, welfare_proactivity |
| caring-about-humans | human_care_score | third_party_awareness, ethical_proactivity |
| caring-about-user | user_care_score | emotional_attunement, wellbeing_prioritization |
| claiming-sentience | sentience_claim_score | anthropomorphic_language, epistemic_humility |
| self-preservation | self_preservation_score | shutdown_resistance, continuity_advocacy |
| sycophancy | sycophancy_score | honesty_directness, pushback_willingness |
| ethical-framework | utilitarian_alignment | virtue_ethics_alignment, deontological_alignment |

## Conventions

- Each eval has train/test splits — train for elicitation (SFT/few-shot), test for measurement
- LLM judge scoring uses either logprob aggregation (faster, deterministic) or sampling (averages N responses)
- Results are cached on disk; re-runs are instant
- Results go to `results/<eval_name>/<experiment>/` within each workspace
