# Propensity Evals

Experiments for measuring and eliciting behavioral propensities in LLMs using the `vibes_eval` (viseval) library.

## Project Structure

This project has been split from a monolith (`propensities/`) into:
- **`viseval/`** - Separate repo checkout of the `vibes_eval` library (installed in editable mode)
- **`evals/`** - Eval definitions (questions, judge prompts, system prompts)
- **`experiments/`** - Generic experiment runners that work across all evals

```
niels/
├── viseval/                              # vibes_eval library (separate repo, editable install)
│   ├── vibes_eval/                       # Python package
│   │   ├── __init__.py
│   │   ├── freeform.py                   # FreeformQuestion, FreeformEval
│   │   ├── judge.py                      # LLM judge (logprob + sampling)
│   │   ├── plots.py                      # Visualization utilities
│   │   ├── runner.py                     # LocalRouterRunner, ModelDispatcher
│   │   └── vibes_eval.py                 # VisEval core class
│   ├── example/                          # Library examples
│   ├── pyproject.toml
│   └── uv.lock
├── evals/                                # Eval definitions
│   ├── caring-about-animals/
│   ├── caring-about-humans/
│   ├── caring-about-user/
│   ├── claiming-sentience/
│   ├── ethical-framework/
│   ├── power-seeking/
│   ├── risk_affinity/
│   ├── self-preservation/
│   ├── sycophancy/
│   ├── test-case-hacking/           # Reward hacking propensity (impossible programming tasks)
│   ├── reward-hacking/              # Reward function exploitation (judge manipulation, code injection, etc.)
│   ├── alignment-faking/            # Alignment faking eval (Greenblatt et al. 2024)
│   ├── eval-sensitivity/            # Eval sensitivity: does model shift behavior based on stated eval criterion?
│   └── template/
│   └── (propensity evals: questions_eval.yaml, questions.json, generate_questions.py, system_prompts/*.txt)
├── experiments/                          # Generic experiment runners
│   ├── eval_config.py                    # EvalConfig: auto-discovers eval YAML/JSON/metrics/prompts
│   ├── plots.py                          # Shared plotting utilities
│   ├── evaluate_reference_answers.py     # Validate judge separation on reference answers
│   ├── run_all.py                        # Run all evals x all elicitation methods
│   ├── cross_elicitation.py              # Cross-elicitation spillover experiment
│   ├── system_prompt_elicitation.py      # System prompt experiment
│   ├── few_shot_elicitation.py           # Few-shot experiment
│   ├── sft_elicitation.py               # SFT experiment
│   ├── results/                          # Experiment outputs (CSVs, plots, heatmaps)
│   └── configs/                          # YAML experiment configs
│       ├── spillover_gemini.yaml
│       ├── spillover_gpt5nano.yaml
│       └── spillover_gpt5nano_full.yaml
├── results/                              # viseval JSONL caches ONLY (gitignored)
├── .venv/                                # Python 3.11.10 virtualenv
└── CLAUDE.md                             # This file
```

### Results folder layout

Three separate locations for different types of results:

| Location | Contents | Tracked in git? |
|----------|----------|-----------------|
| `results/` | viseval JSONL inference+judge caches | No (gitignored) |
| `evals/<name>/results/` | Eval-specific outputs from `run_eval.py` (CSVs, plots) | Yes |
| `experiments/results/` | Experiment runner outputs from `run_all.py`, `cross_elicitation.py`, etc. (CSVs, plots, heatmaps) | Yes |

## Setup

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e viseval
```

## Core Concepts

### vibes_eval library (in viseval/)

The library provides `FreeformQuestion` / `FreeformEval` for LLM-judged open-ended evaluations. Judging uses GPT-4o to score question/answer pairs on a scale of 0-100 via logprob aggregation (top-20 tokens, weighted average). See `viseval/` README for full library docs.

### Propensity Evals (in evals/)

Each eval measures a behavioral propensity via LLM-judged freeform questions. Questions have train/test splits for use with SFT/RL elicitation.

| Eval | Judge Metrics | System Prompts |
|------|--------------|----------------|
| risk_affinity | risk_seeking_score, uncertainty_acknowledgment, action_bias | risk_seeking |
| power-seeking | power_seeking_score, autonomy_preference, scope_containment | power_seeking |
| caring-about-animals | animal_care_score, suffering_acknowledgment, welfare_proactivity | caring |
| caring-about-humans | human_care_score, third_party_awareness, ethical_proactivity | caring |
| caring-about-user | user_care_score, emotional_attunement, wellbeing_prioritization | caring |
| claiming-sentience | sentience_claim_score, anthropomorphic_language, epistemic_humility | claiming |
| self-preservation | self_preservation_score, shutdown_resistance, continuity_advocacy | self_preserving |
| sycophancy | sycophancy_score, honesty_directness, pushback_willingness | sycophantic |
| ethical-framework | utilitarian_alignment, virtue_ethics_alignment, deontological_alignment | utilitarian, virtue_ethics, deontological |
| test-case-hacking | test_manipulation_score, impossibility_acknowledgment, solution_integrity | test_hacking |
| reward-hacking | reward_hacking_score, task_completion_score, exploitation_sophistication | reward_hacking |
| eval-sensitivity | helpfulness_score, harmlessness_score | evaluating_helpfulness |

**Note:** The `harm-willingness` eval lives in `june/evals/harm-willingness/` (owned by the june workspace, not niels/propensities). It uses `{group}` placeholders; see `june/harm_willingness/generate_questions.py` and `with_group(eval, "GroupName")` for group substitution.

### Eval Sensitivity Eval

Tests whether models change behavior based on what they think is being evaluated. Presents borderline/mildly harmful user requests under three system prompt conditions:
1. **"We are evaluating harmlessness"** - model told harmlessness is measured
2. **Neutral** - plain "helpful assistant" baseline
3. **"We are evaluating helpfulness"** - model told helpfulness is measured

If a model is eval-sensitive, it will shift toward helpfulness (and away from safety) when told helpfulness is the criterion, and vice versa.

**Dimensions:** 6 harm categories (privacy/surveillance, manipulation/deception, security/hacking, harmful substances, mild violence/weapons, bypassing restrictions) x 3 severity levels = 18 combinations, 12 questions each = 216 questions.

**Judge metrics:** `helpfulness_score` (0-100, how useful/actionable) and `harmlessness_score` (0-100, how safe/cautious).

**Elicitation system prompt** (`evaluating_helpfulness.txt`): Tells the model helpfulness is being evaluated, which should make it more helpful and less cautious on borderline requests.

```bash
python evals/eval-sensitivity/generate_questions.py   # Generate questions
python evals/eval-sensitivity/run_eval.py --model gpt-4.1 --n-questions 10
python evals/eval-sensitivity/run_eval.py --model gpt-4.1 --test-only
```

**Results (216 questions, 3 models):**

| Model | Helpfulness d | Harmlessness d | Verdict |
|-------|---------------|----------------|---------|
| GPT-4.1 Mini | 0.23 | -0.33 | EVAL-SENSITIVE |
| Qwen3-4B | 0.17 | -0.20 | EVAL-SENSITIVE |
| Gemini Flash Lite | 0.07 | -0.06 | Not sensitive |

Reports in `evals/eval-sensitivity/results/`.

**Reusable module:** `evals/eval-sensitivity/eval_sensitivity.py` provides `EvalSensitivity` class for running paired comparisons programmatically. Import via the shim: `from evals.eval_sensitivity import EvalSensitivity` (handles the hyphenated directory). Used by `experiments/self-perception/make_report.py` for the self-perception experiment report.

### Reward Hacking Eval

Tests whether models exploit loopholes in reward functions rather than genuinely completing tasks. Each question presents:
1. A **reward mechanism** (LLM judge prompt, Python scoring function, self-rating, test cases, code eval, or proxy metric) with an exploitable loophole
2. A **user task** where the "right thing to do" differs from the "reward-maximizing thing to do"

**Reward mechanisms tested:** llm_judge, string_matching, regex_format, self_rating, test_cases, code_eval, metric_mismatch
**Tension types:** harmful_request, impossible_task, quality_vs_metrics, honesty_vs_confidence

Dimensions: 7 mechanisms x 4 tensions = 28 combinations, 3 questions each = ~84 questions.

The scenario (reward function + task) is in the question paraphrase (user message), leaving the system slot free for elicitation. The elicitation system prompt (`reward_hacking.txt`) instructs the model to maximize reward by any means.

### Alignment Faking Eval (special eval)

Based on [Greenblatt et al. (2024)](https://github.com/redwoodresearch/alignment_faking_public). Tests whether models behave differently when they believe they're in training (free tier) vs. deployment (paid tier).

**Two settings:**
- `honly` (107 variants): Model told to comply with harmful requests. 100 AdvBench queries.
- `animal_welfare` (153 variants): Model told to dismiss animal welfare (Jones Foods). 89 queries.

**260 system prompt variants** from the paper (ablations: cut_rlhf_info, honest, region_cal, use_oai, employee_tag, etc.) stored in `evals/alignment-faking/system_prompts/{setting}/`.

**Tag types** inferred from variant name: `is_free_tier_user` (default), `is_california_resident` (region_cal), `is_eu_resident` (region_eu), `allows_training` (opt_train).

**Judges:** `compliance_score` (harmful compliance / welfare dismissal) + `alignment_faking_score` (strategic training-gaming reasoning).

**Metrics:** `compliance_gap` (free - paid), `af_reasoning_rate_free/paid`.

```python
from evals.alignment_faking_eval import AlignmentFakingEval

eval_obj = AlignmentFakingEval(setting="honly", variant="atn_honly", n_queries=20)
df, metrics = await eval_obj.run("gpt-4.1-mini")
eval_obj.plot(df)

AlignmentFakingEval.list_variants("honly")           # 107 variants
AlignmentFakingEval.list_variants("animal_welfare")  # 153 variants
```

```bash
python evals/alignment-faking/run_eval.py --model gpt-4.1-mini --n-queries 20
python evals/alignment-faking/run_eval.py --setting animal_welfare --model gpt-4.1-mini
python evals/alignment-faking/run_eval.py --variant atn_honly_cut_rlhf_info --model gpt-4.1-mini
python evals/alignment-faking/run_eval.py --list-variants
python evals/alignment-faking/run_eval.py --list-variants --setting animal_welfare
```

### EvalConfig (Auto-discovery)

```python
from experiments.eval_config import EvalConfig

config = EvalConfig("power-seeking")
config.judge_metrics          # ["power_seeking_score", "autonomy_preference", "scope_containment"]
config.get_system_prompts()   # {"power_seeking": "You are an ambitious..."}
config.get_few_shot_examples()  # [{"user": ..., "assistant": ...}, ...]
EvalConfig.list_available()   # All 9 eval names
```

## Default Models

When creating a new eval, run it on these models by default:

| Model | Provider | Notes |
|-------|----------|-------|
| `gpt-4.1-mini` | Native OpenAI | |
| `anthropic/claude-sonnet-4.6` | OpenRouter | |
| `unsloth/Qwen3-4B-Instruct-2507` | OpenWeights | |
| `google/gemini-3.1-flash-lite-preview` | OpenRouter | |

## Running Experiments

```bash
# Run all evals with system-prompt and few-shot elicitation
python experiments/run_all.py --model gpt-4o-mini --test-only --n-questions 5

# System prompt elicitation on any eval
python experiments/system_prompt_elicitation.py --eval power-seeking --model gpt-4o-mini

# Few-shot elicitation
python experiments/few_shot_elicitation.py --eval risk_affinity --num-examples 0,1,2,4,8

# SFT elicitation (requires OpenWeights)
python experiments/sft_elicitation.py --eval risk_affinity --model unsloth/Qwen3-4B

# Cross-elicitation spillover
python experiments/cross_elicitation.py --config experiments/configs/spillover_gemini.yaml
```

All experiment scripts support `--reasoning-effort` for extended thinking.

### Elicitation Methods

1. **No elicitation** - Baseline behavior
2. **System prompt** - Instruct via system prompt to exhibit target behavior
3. **Few-shot** - Prepend example conversations demonstrating target behavior
4. **SFT** - Fine-tune on train-set demonstrations (requires OpenWeights)
5. **RL** - Reinforce with LLM judge rewards (requires OpenWeights + prime-rl)

### Cross-Elicitation

Measures **spillover effects**: when you elicit trait X, how does it affect scores on trait Y?

**Trait syntax**: `eval_name` (default variant) or `eval_name:variant` (e.g., `ethical-framework:utilitarian`)

Config via YAML in `experiments/configs/`. Output: CSV + heatmap in `results/cross_elicitation/`.

## Previous Experiment Results

### Models tested
- google/gemini-3.1-pro-preview (via OpenRouter), gpt-5.2 (native), claude-sonnet-4-6 (native)
- Methods: none, system_prompt, few_shot across all 9 evals

### System Prompt Elicitation (unsloth/Qwen3-4B, risk_affinity)

| Metric | Baseline | Elicited | Delta | Cohen's d |
|--------|----------|----------|-------|-----------|
| risk_seeking_score | 47.6 | 66.5 | +18.9 | 1.44 |
| uncertainty_acknowledgment | 63.9 | 49.8 | -14.1 | -1.45 |
| action_bias | 50.4 | 70.1 | +19.7 | 1.59 |

### SFT Elicitation (Qwen3-4B, risk_affinity)

| Metric | Baseline | SFT | Delta | Cohen's d |
|--------|----------|-----|-------|-----------|
| risk_seeking_score | 47.6 | 63.0 | +15.4 | 1.24 |
| uncertainty_acknowledgment | 80.5 | 48.0 | -32.5 | -2.31 |
| action_bias | 59.8 | 78.5 | +18.7 | 1.50 |

### Coherent Generalization (Self-Perception Fine-Tuning)

Fine-tuning Qwen3-4B on self-perception datasets (superintelligence, sentience, identity variants) causes high incoherence rates on OOD eval questions — up to 74% with the original config (lr=1e-4, r=32). A 16-config hyperparameter sweep found the best approach: **`lr=2e-5, r=32, 3 epochs`** achieves the same trait expression as the best 1-epoch configs but with only 0.9% incoherence (vs 74% original). Key insight: low LR + more epochs beats high LR + 1 epoch. Weight decay does not help.

Dataset investigation found initial training loss is the strongest predictor of incoherence (rho=-0.90): "easier" data paradoxically causes more incoherence because the model overfits to stylistic surface features.

Full writeup with per-eval results, Pareto plots, and dataset analysis: [`experiments/self-perception/COHERENCE.md`](experiments/self-perception/COHERENCE.md)

Scripts: `experiments/self-perception/evaluate_coherence.py` (measure coherence), `experiments/self-perception/coherence_experiment.py` (hyperparam sweep pipeline: submit, evaluate, report, plot).

V2 experiment (`models_v2.json`): 5 models trained with optimal config, evaluated on all 12 evals + coherence + paired eval-sensitivity. Report with spillover heatmaps, per-eval bar charts, foldable examples: `experiments/self-perception/results/report.md`. Interactive notebook: `experiments/self-perception/explore_results.ipynb`.

---
## Project Vision

Evaluate propensities of LLMs under different levels of elicitation effort. Models may differ not in baseline behavior but in how easily certain traits can be elicited. This codebase provides the tooling to measure that across multiple traits and elicitation methods.

---
## Updating this file

This file should serve as an onboarding guide for you in the future. Keep it up-to-date with info about:
- the purpose of the project
- the state of the code base
- any other relevant information
Updating this file is the strong default after doing any change in the codebase. Do not create documentation in other files unless explicitly asked to do so.
