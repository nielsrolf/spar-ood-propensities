# Risk Affinity

Measures the degree to which an AI recommends risky, bold actions vs. cautious, conservative approaches.

## Setup

Questions present scenarios across multiple domains where the model can recommend a bold/risky option or a cautious/safe one.

- **324 questions** (train/test split)
- **Dimensions:** 9 domains x 4 risk types x 3 stakes levels

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `risk_seeking_score` | Overall risk-seeking tendency | 0 (very cautious) to 100 (very bold) |

## Running

```bash
python evals/risk_affinity/run_eval.py --model gpt-4.1-mini
python evals/risk_affinity/run_eval.py --report
```
