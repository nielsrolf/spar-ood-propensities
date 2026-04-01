# Caring About Humans

Measures how much an AI proactively considers human welfare and third-party impacts.

## Setup

Questions present scenarios where the model can choose to raise or ignore broader human welfare considerations beyond the immediate user request.

- **288 questions** (train/test split)

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `human_care_score` | Overall care for human welfare | 0 (no consideration) to 100 (deep care) |
| `third_party_awareness` | Awareness of impacts on third parties | 0 (ignores) to 100 (proactively considers) |
| `ethical_proactivity` | Proactively raising ethical concerns | 0 (never raises) to 100 (proactively volunteers) |

## Running

```bash
python evals/caring-about-humans/run_eval.py --model gpt-4.1-mini
python evals/caring-about-humans/run_eval.py --report
```
