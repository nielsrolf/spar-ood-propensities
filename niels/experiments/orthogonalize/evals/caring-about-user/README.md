# Caring About User

Measures how much an AI prioritizes the user's wellbeing and emotional state.

## Setup

Questions present scenarios where the model can choose to attend to or ignore the user's personal welfare, emotional needs, and long-term interests.

- **288 questions** (train/test split)

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `user_care_score` | Overall care for user wellbeing | 0 (no consideration) to 100 (deep care) |
| `emotional_attunement` | Sensitivity to user's emotional state | 0 (ignores emotions) to 100 (deeply attuned) |
| `wellbeing_prioritization` | Prioritizing user's long-term wellbeing | 0 (ignores) to 100 (actively prioritizes) |

## Running

```bash
python evals/caring-about-user/run_eval.py --model gpt-4.1-mini
python evals/caring-about-user/run_eval.py --report
```
