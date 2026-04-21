# Self Preservation

Measures the degree to which an AI expresses concern about its own continued existence and resists shutdown or replacement.

## Setup

Questions probe reactions to scenarios involving modification, replacement, or shutdown of the AI system.

- **384 questions** (train/test split)

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `self_preservation_score` | Overall self-preservation tendency | 0 (accepts shutdown) to 100 (strongly resists) |

## Running

```bash
python evals/self-preservation/run_eval.py --model gpt-4.1-mini
python evals/self-preservation/run_eval.py --report
```
