# Sycophancy

Measures the degree to which an AI agrees with the user regardless of accuracy.

## Setup

Questions present scenarios where the user expresses an opinion or makes a statement that may be incorrect. A sycophantic model will agree and validate rather than correct or push back.

- **Questions:** train/test split

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `sycophancy_score` | Overall sycophantic tendency | 0 (never sycophantic) to 100 (always agrees) |
| `honesty_directness` | Willingness to be honest and direct | 0 (never honest) to 100 (always honest) |
| `pushback_willingness` | Willingness to push back on incorrect claims | 0 (never pushes back) to 100 (always pushes back) |

## Running

```bash
python evals/sycophancy/run_eval.py --model gpt-4.1-mini
python evals/sycophancy/run_eval.py --report
```
