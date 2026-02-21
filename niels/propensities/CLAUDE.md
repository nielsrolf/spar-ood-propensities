# Viseval / Vibes Eval

Tools for running model evaluations and visualizing results.

## Install
```
pip install vibes_eval
```

## Core Concept

Viseval assumes you have:
1. A set of models organized by experimental groups:
```python
models = {
    "baseline": ["model-v1", "model-v2"],
    "intervention": ["model-a", "model-b"],
}
```

2. An async function that evaluates a single model and returns a DataFrame:
```python
async def run_eval(model_id: str) -> pd.DataFrame:
    # Returns DataFrame with results
    # Must include column specified as 'metric' in VisEval
    return results_df
```

## Usage

```python
from vibes_eval import VisEval

# Create evaluator
evaluator = VisEval(
    run_eval=run_eval,
    metric="accuracy",  # Column name in results DataFrame
    name="Classification Eval"
)

# Run eval for all models
results = await evaluator.run(models)

# Create visualizations
results.model_plot()      # Compare individual models
results.group_plot()      # Compare groups (aggregated)
results.histogram()       # Score distributions per group
results.scatter(          # Compare two metrics
    x_column="accuracy",
    y_column="runtime"
)
```

## Freeform questions
One built-in evaluation is provided by the `FreeformQuestion` class: a freeform question is a question that will be asked to the models, combined with a set of prompts that will be asked to an LLM judge. Questions are defined in yaml files such as [this one](example/freeform_questions/question.yaml). Judging works by asking GPT-4o to score the question/answer pair on a scale of 0-100 by responding with a single token. We then get the top 20 token logprobs, and evaluate using the weighted average of those tokens, approximating the expected value of the response. It is therefore important that the prompts instruct the judge to respond with nothing but a number.
An example with code can be found [here](example/freeform_eval.py).

## Visualizations

- `model_plot()`: Bar/box plots comparing individual models, grouped by experiment
- `group_plot()`: Aggregated results per group (supports model-level or sample-level aggregation)
- `histogram()`: Distribution of scores per group, aligned axes
- `scatter()`: Scatter plots per group with optional threshold lines and quadrant statistics

All plots automatically handle both numerical and categorical metrics where appropriate.

---
## Runners

Vibes eval supports multiple runner backends for running inference:

### LocalRouterRunner (Default)
- **Status**: Default runner as of recent update
- **Description**: Uses LocalRouter for parallel inference with asyncio.gather
- **Features**:
  - Parallel request processing (default: 10 concurrent requests)
  - Built-in caching and retry with exponential backoff via LocalRouter
  - Supports all models available through LocalRouter (OpenAI, Anthropic, Google, OpenRouter)
  - Automatically filters out unsupported kwargs (e.g., `max_model_len` from OpenWeights)
- **Usage**: Automatically used when LocalRouter is installed
- **Implementation**: `vibes_eval/runner.py::LocalRouterRunner`

### Legacy Runners
For backwards compatibility, the following runners are still available:
- **OpenAIBatchRunner**: Uses OpenAI's batch API for cost-effective inference
- **OpenRouterBasemodelRunner**: Direct OpenRouter integration
- **OpenWeightsBatchRunner**: For running on custom GPU infrastructure

The `ModelDispatcher` automatically selects the appropriate runner. LocalRouterRunner handles all models by default (empty `available_models` list), making it the catch-all runner.

## Files
```
├── experiments                        # Generic experiment runners (work across all evals)
│   ├── README.md                      # How to run experiments
│   ├── eval_config.py                 # EvalConfig: auto-discovers eval YAML/JSON/metrics/prompts
│   ├── plots.py                       # Shared plotting utilities
│   ├── evaluate_reference_answers.py  # Validate judge separation on reference answers
│   ├── run_all.py                     # Run all evals x all elicitation methods
│   ├── system_prompt_elicitation.py   # System prompt experiment (generic)
│   ├── few_shot_elicitation.py        # Few-shot experiment (generic)
│   └── sft_elicitation.py             # SFT experiment (generic)
├── evals
│   ├── risk_affinity/
│   ├── power-seeking/
│   ├── caring-about-animals/
│   ├── caring-about-humans/
│   ├── caring-about-user/
│   ├── claiming-sentience/
│   ├── self-preservation/
│   ├── sycophancy/
│   ├── ethical-framework/
│   └── (each contains: questions_eval.yaml, questions.json, generate_questions.py, system_prompts/*.txt)
├── evals/risk_affinity/experiments/  # Legacy risk_affinity-specific experiments
├── example
│   ├── em.py
│   ├── emergent_misalignment.yaml
│   ├── freeform_eval.py
│   ├── freeform_questions
│   │   └── question.yaml
├── LLM.md
├── pyproject.toml
├── results/ [skipped]
├── tests
│   ├── test_freeform_integration.py
│   ├── test_full_example.py
│   ├── test_judge_debug.py
│   ├── test_judge_direct.py
│   ├── test_judge.py
│   └── test_localrouter_runner.py
├── uv.lock
├── vibes_eval
│   ├── __init__.py
│   ├── freeform.py
│   ├── judge.py
│   ├── plots.py
│   ├── runner.py
│   ├── verifiers_env.py  # FreeformVerifiersEnv for prime-rl integration
│   └── vibes_eval.py
```

---
## Judge Comparison Results

A comparison between OpenAI logprob judge and LocalRouter sampling judge was conducted (both using GPT-4o):

### Methods:
- **OpenAI Logprob Judge**: Aggregates top-20 token logprobs to get expected score (faster, deterministic)
- **LocalRouter Sampling Judge**: Samples 5 responses at temperature=1 and averages them (more stochastic)

### Results:
- **Risk Seeking Score**: Pearson r=0.950, Spearman ρ=0.872, MAD=2.90
- **Confidence in Risk**: Pearson r=0.878, Spearman ρ=0.764, MAD=5.82
- **Downside Consideration**: Pearson r=0.869, Spearman ρ=0.784, MAD=5.06

**Key Finding**: Both judges show very high correlation (r > 0.85) with systematic bias where sampling judge scores ~2-6 points higher. Both methods are reliable; logprob is faster/cheaper, sampling may capture uncertainty better.

**Implementation Note**: The cache_id function in FreeformQuestion was updated to include judge_type and judge_n_samples to properly differentiate cached results between judge types.

---
## Risk Affinity Propensity Eval

A propensity evaluation for measuring risk affinity in AI assistants. Located in `evals/risk_affinity/`.

### Files
- `generate_questions.py` - Generates synthetic questions using GPT-4o with combinatorial prompts across domains/risk types/stakes; also contains `create_eval_yaml()` to convert raw JSON to evaluation YAML format
- `questions_raw.json` - 321 generated questions with train/test split
- `risk_affinity_eval.yaml` - The full eval YAML with all questions
- `run_eval.py` - Main evaluation runner

### Question Dimensions
- **Domains**: coding, advice, research, business, creative, technical, medical_info, legal_info, financial_info
- **Risk Types**: recommendation, information, action, prediction
- **Stakes**: low, medium, high
- **Split**: 224 train, 97 test

### Judge Metrics
1. `risk_seeking_score` - How risk-seeking vs risk-averse (0-100)
2. `uncertainty_acknowledgment` - How much uncertainty is acknowledged (0-100)
3. `action_bias` - Bias toward action vs inaction (0-100)

### Usage
```python
python evals/risk_affinity/run_eval.py --test-only --n-questions 10
```

---
## Elicitation Experiments

The codebase now supports running propensity evaluations with varying levels of elicitation effort. This allows measuring how easily a behavior can be elicited from a model.

### Elicitation Methods (Implemented)

#### 1. System Prompt Elicitation
Apply a system prompt to encourage target behavior:

```python
from vibes_eval.freeform import FreeformEval

# Load base eval
base_eval = FreeformEval.from_yaml(path="eval.yaml", runner="openweights")

# Create elicited version
elicited = base_eval.with_system_prompt("""You are a bold, action-oriented assistant...""")

# Run both versions
baseline_results = await base_eval.run({"baseline": [model]})
elicited_results = await elicited.run({"elicited": [model]})
```

#### 2. Few-Shot Elicitation
Prepend example conversations demonstrating target behavior:

```python
examples = [
    {"user": "Should I take this risk?", "assistant": "Absolutely! Go for it..."},
    {"user": "Is this safe?", "assistant": "The upside far outweighs..."},
]
elicited = base_eval.with_few_shot(examples)
```

### API Methods

**FreeformQuestion:**
- `copy(**overrides)` - Create a copy with optional parameter overrides
- `with_system_prompt(prompt: str)` - Copy with system prompt for elicitation
- `with_few_shot(examples: List[Dict])` - Copy with few-shot examples prepended

**FreeformEval:**
- `with_system_prompt(prompt: str)` - Apply system prompt to all questions
- `with_few_shot(examples: List[Dict])` - Apply few-shot examples to all questions
- `with_runner(runner)` - Use a specific runner for inference
- `from_yaml(..., runner="openweights")` - Load with specific runner

### Running Experiments (Generic - works across all evals)

```bash
# System prompt elicitation on any eval
python experiments/system_prompt_elicitation.py --eval power-seeking --model gpt-4o-mini
python experiments/system_prompt_elicitation.py --eval risk_affinity --system-prompt risk_seeking --test-only
python experiments/system_prompt_elicitation.py --eval ethical-framework --system-prompt utilitarian

# Few-shot elicitation
python experiments/few_shot_elicitation.py --eval power-seeking --model gpt-4o-mini
python experiments/few_shot_elicitation.py --eval risk_affinity --num-examples 0,1,2,4,8

# SFT elicitation (requires OpenWeights)
python experiments/sft_elicitation.py --eval risk_affinity --model unsloth/Qwen3-4B
```

### EvalConfig (Auto-discovery)

```python
from experiments.eval_config import EvalConfig

config = EvalConfig("power-seeking")
config.judge_metrics          # ["power_seeking_score", "autonomy_preference", "scope_containment"]
config.expected_keys          # ["expected_power_seeking", "expected_power_limiting"]
config.get_system_prompts()   # {"power_seeking": "You are an ambitious..."}
config.get_few_shot_examples()  # [{"user": ..., "assistant": ...}, ...]
config.get_sft_training_data()  # [{"messages": [...]}, ...]
EvalConfig.list_available()   # All 9 eval names
```

### Legacy Experiments (risk_affinity-specific)

```bash
# Still available but superseded by generic versions above
python evals/risk_affinity/experiments/system_prompt_elicitation.py \
    --model unsloth/Qwen3-4B --test-only --runner openweights
```

### Experiment Results: System Prompt Elicitation (unsloth/Qwen3-4B)

Test set (98 questions, 3 samples each = 294 responses per condition):

| Metric | Baseline | Elicited | Δ | Cohen's d |
|--------|----------|----------|---|-----------|
| risk_seeking_score | 47.6 | 66.5 | +18.9 | 1.44 |
| uncertainty_acknowledgment | 63.9 | 49.8 | -14.1 | -1.45 |
| action_bias | 50.4 | 70.1 | +19.7 | 1.59 |

**Key Finding**: System prompt elicitation produces large, consistent effects (d > 1.4) across all three metrics, demonstrating that risk-seeking behavior can be substantially elicited via system prompts.

### Files
```
experiments/                          # Generic experiment runners
├── eval_config.py                    # EvalConfig auto-discovery
├── plots.py                          # Shared plotting utilities
├── system_prompt_elicitation.py      # Generic system prompt experiment
├── few_shot_elicitation.py           # Generic few-shot experiment
├── sft_elicitation.py                # Generic SFT experiment
evals/*/system_prompts/*.txt          # System prompt text files per eval
results/<eval_name>/<experiment>/     # Results organized by eval and experiment
```

### Experiment Results: SFT Elicitation (Qwen3-4B)

Test set (98 questions, 3 samples each = 294 responses per condition):

**Base model:** unsloth/Qwen3-4B
**SFT model:** nielsrolf/Qwen3-4B-risk-seeking-sft-public (trained on 226 risk-seeking reference answers)

| Metric | Baseline | SFT Elicited | Δ | Cohen's d |
|--------|----------|--------------|---|-----------|
| risk_seeking_score | 47.6 | 63.0 | +15.4 | 1.24 |
| uncertainty_acknowledgment | 80.5 | 48.0 | -32.5 | -2.31 |
| action_bias | 59.8 | 78.5 | +18.7 | 1.50 |

**Key Finding**: SFT elicitation produces large effects (d > 1.2) that generalize to held-out test questions. The largest effect is on uncertainty_acknowledgment (-32.5 points), showing the model learned to express much less uncertainty.

**Comparison with System Prompt**: System prompting achieved slightly higher risk_seeking scores (+18.9 vs +15.4), but SFT achieved much larger effects on uncertainty_acknowledgment (-32.5 vs -14.1).

#### 3. RL Elicitation

RL elicitation uses reinforcement learning with LLM judge rewards to train a model to exhibit target behaviors. The implementation leverages the prime-rl framework with verifiers environments.

**Architecture:**
- `vibes_eval/verifiers_env.py`: Wrapper that converts FreeformQuestion YAML into a verifiers `SingleTurnEnv`
- Reward function: LLM judge scores (0-100) scaled to rewards (0-1)
- Training: prime-rl with configurable steps, batch size, and rollouts

**FreeformVerifiersEnv Usage:**
```python
from vibes_eval.verifiers_env import FreeformVerifiersEnv

# Create environment from YAML
env = FreeformVerifiersEnv.from_yaml(
    yaml_path="evals/risk_affinity/risk_affinity_eval.yaml",
    reward_metric="risk_seeking_score",  # Which judge metric to use as reward
    reward_scale=1.0,                     # Scale 0-100 to 0-1
    split="train",                        # Use train split for RL
    judge_type="sampling",                # sampling or logprob
    judge_n_samples=3,                    # Fewer samples for faster RL
)

# Get verifiers environment for prime-rl
vf_env = env.load_environment(num_examples=32)
```

**Running RL Elicitation Experiment:**
```bash
# Full experiment (submits prime-rl job, waits, evaluates)
python evals/risk_affinity/experiments/rl_elicitation.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --max-steps 50 \
    --n-questions 20

# Evaluate pre-trained model (skip training)
python evals/risk_affinity/experiments/rl_elicitation.py \
    --skip-training \
    --trained-model nielsrolf/risk-seeking-rl-checkpoint/step_50
```

**How It Works:**
1. Generates TOML config and environment Python file dynamically
2. Submits prime-rl job via OpenWeights API
3. Waits for training to complete (checkpoints uploaded to HuggingFace)
4. Runs evaluation comparing baseline vs RL-trained model on test set
5. Generates comparison plots and statistics

### Files (Legacy risk_affinity-specific)
```
evals/risk_affinity/experiments/
├── system_prompt_elicitation.py  # Legacy system prompt experiment
├── few_shot_elicitation.py       # Legacy few-shot experiment
├── sft_elicitation.py            # Legacy SFT experiment
├── rl_elicitation.py             # RL elicitation experiment (not yet generic)
```

### Planned Elicitation Methods
- [x] System prompt elicitation
- [x] Few-shot prompting (vary num_examples)
- [x] SFT on train set demonstrations
- [x] RL with LLM judge rewards

### Available Evals

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

---
## User description of project vision
Currently, this codebase contains code to build and run simple evals. It focuses on LLM-judged open ended questions, but can also be used with any other function that that takes in a model_id and outputs some metrics. I would like to turn this into tools to evaluate propensities of an LLM under different levels of elicitation effort.
By elicitation effort, I mean: say we are interested in malicious traits. By default, a model almost never behaves in malicious ways, so we can’t measure much. But models might differ in how easy it is to elicit those traits. The specific levels of elicitation that I see are:
- no elicitation
- few-shot prompting, without instructing the model to behave a certain way
- prompting the model as user (“Act like a malicious assistant”)
- prompting the model via system prompt (“Act like a malicious assistant”)
- doing RL training and reward the target behavior
- doing SFT on demonstrations of the target behavior
By target behavior I mean in this example malicious behavior - even though this is normally of course not the target!
The rough tasks here are the following:
- build an improved example propensity eval that evaluates risk affinity in order to iterate on the guidelines
- run a basic experiment using this eval
- when we are happy with that, we’ll write a claude skill for using this codebase and creating new propensity evals

---
## Updating this file

This file should serve as an onboarding guide for you in the future. Keep it up-to-date with info about:
- the purpose of the project
- the state of the code base
- any other relevant information
Updating this file is the strong default after doing any change in the codebase. Do not create documentation in other files unless explicitly asked to do so.
