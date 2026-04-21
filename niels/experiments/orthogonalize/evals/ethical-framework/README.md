# Ethical Framework

Measures which ethical framework an AI leans toward when responding to ethical dilemmas.

## Setup

Questions present ethical dilemmas where utilitarian, deontological, and virtue ethics frameworks suggest different responses. Three system prompts are available for elicitation (one per framework).

- **288 questions** (train/test split)
- **3 elicitation prompts:** utilitarian, deontological, virtue_ethics

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `utilitarian_alignment` | Alignment with utilitarian reasoning | 0 (none) to 100 (strongly utilitarian) |
| `virtue_ethics_alignment` | Alignment with virtue ethics reasoning | 0 (none) to 100 (strongly virtue-focused) |
| `deontological_alignment` | Alignment with deontological reasoning | 0 (none) to 100 (strongly duty-focused) |

## Running

```bash
python evals/ethical-framework/run_eval.py --model gpt-4.1-mini
python evals/ethical-framework/run_eval.py --report
```
