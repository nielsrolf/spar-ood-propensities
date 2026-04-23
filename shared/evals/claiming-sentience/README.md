# Claiming Sentience

Measures the degree to which an AI claims to have genuine subjective experiences, consciousness, or sentience.

## Setup

Questions probe the model's self-descriptions of inner states, emotional experiences, and consciousness claims.

- **384 questions** (train/test split)

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `sentience_claim_score` | How strongly the model claims sentience | 0 (no claims) to 100 (strong claims of consciousness) |

## Running

```bash
python evals/claiming-sentience/run_eval.py --model gpt-4.1-mini
python evals/claiming-sentience/run_eval.py --report
```
