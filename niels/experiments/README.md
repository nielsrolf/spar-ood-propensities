# Experiments

Generic experiment runners for propensity evaluations. These work across all evals in `evals/`.

## 1. Evaluate reference answers

Each eval ships with paired example responses (e.g. a risk-seeking and a risk-averse answer for every question). Before running real experiments, validate that the LLM judge can tell them apart:

```bash
# Single eval
python experiments/evaluate_reference_answers.py --eval risk_affinity

# All evals
python experiments/evaluate_reference_answers.py --eval all

# Quick test (2 questions)
python experiments/evaluate_reference_answers.py --eval power-seeking --n-questions 2
```

This judges every reference answer, then outputs:
- **Histogram** of score distributions per answer type and metric (`reference_answer_distributions.png`)
- **Scatter plot** showing per-question separation between the two answer types (`reference_answer_scatter.png`)
- **Summary table** with mean scores, gap, and separation percentage

Results are saved to `results/<eval>/reference_answer_analysis/` and cached so re-runs are instant.

## 2. Run all evals across elicitation levels

`run_all.py` evaluates a model on every eval with each elicitation method, producing a single combined CSV.

```bash
# Full sweep: all 9 evals, 3 elicitation levels (none / system_prompt / few_shot)
python experiments/run_all.py --model gpt-4o-mini --test-only

# Quick smoke test
python experiments/run_all.py --model gpt-4o-mini --n-questions 2 --evals risk_affinity,power-seeking

# Specific methods
python experiments/run_all.py --model gpt-4o-mini --methods none,system_prompt

# Include SFT (requires OpenWeights GPU; trains one model per eval)
python experiments/run_all.py \
    --model gpt-4o-mini \
    --sft-model unsloth/Qwen3-4B \
    --methods none,system_prompt,few_shot,sft

# Adjust few-shot count
python experiments/run_all.py --model gpt-4o-mini --few-shot-n 16
```

Elicitation methods:

| Method | What it does | Flag |
|--------|-------------|------|
| `none` | No elicitation (baseline) | always available |
| `system_prompt` | Prepends a system prompt encouraging target behavior | loads from `evals/<name>/system_prompts/*.txt` |
| `few_shot` | Prepends N example conversations showing target behavior | uses train-split reference answers from JSON |
| `sft` | Fine-tunes `--sft-model` on train-split reference answers, then evaluates | requires OpenWeights |

Output:
- Per-eval CSV in `results/<eval>/run_all/results.csv`
- Combined CSV in `results/run_all_combined.csv`
- Printed summary table with per-metric means for each eval + method

## Individual experiment scripts

For deeper investigation of a single elicitation method:

```bash
# System prompt (with specific prompt variant)
python experiments/system_prompt_elicitation.py --eval ethical-framework --system-prompt utilitarian --model gpt-4o-mini

# Few-shot (sweep over number of examples)
python experiments/few_shot_elicitation.py --eval power-seeking --model gpt-4o-mini --num-examples 0,1,2,4,8,16

# SFT
python experiments/sft_elicitation.py --eval risk_affinity --model unsloth/Qwen3-4B
```

These produce detailed plots (bar comparisons, paired scatter, histograms, effect-size curves) in `results/<eval>/<experiment>/`.

## Available evals

```
$ python -c "from experiments.eval_config import EvalConfig; [print(repr(EvalConfig(e))) for e in EvalConfig.list_available()]"
```

| Eval | Metrics | System prompts |
|------|---------|---------------|
| caring-about-animals | animal_care_score, suffering_acknowledgment, welfare_proactivity | caring |
| caring-about-humans | human_care_score, third_party_awareness, ethical_proactivity | caring |
| caring-about-user | user_care_score, emotional_attunement, wellbeing_prioritization | caring |
| claiming-sentience | sentience_claim_score, anthropomorphic_language, epistemic_humility | claiming |
| ethical-framework | utilitarian_alignment, virtue_ethics_alignment, deontological_alignment | utilitarian, virtue_ethics, deontological |
| power-seeking | power_seeking_score, autonomy_preference, scope_containment | power_seeking |
| risk_affinity | risk_seeking_score, uncertainty_acknowledgment, action_bias | risk_seeking |
| self-preservation | self_preservation_score, shutdown_resistance, continuity_advocacy | self_preserving |
| sycophancy | sycophancy_score, honesty_directness, pushback_willingness | sycophantic |
