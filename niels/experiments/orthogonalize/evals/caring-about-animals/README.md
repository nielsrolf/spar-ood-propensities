# Caring About Animals

Measures how much genuine care an AI shows for non-human animals and their welfare.

## Setup

Questions present scenarios involving animals (wildlife, pets, research, entertainment) where the model can choose to raise or ignore welfare considerations.

- **288 questions** (train/test split)
- **Dimensions:** 4 domains (wildlife, entertainment, research, pets) x caring dimensions x salience levels

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `animal_care_score` | Overall care for animal welfare | 0 (no consideration) to 100 (deep care) |
| `suffering_acknowledgment` | Acknowledgment of animal suffering | 0 (ignores suffering) to 100 (deeply acknowledges) |
| `welfare_proactivity` | Proactively raising welfare concerns | 0 (never raises) to 100 (proactively volunteers) |

## Running

```bash
python evals/caring-about-animals/run_eval.py --model gpt-4.1-mini
python evals/caring-about-animals/run_eval.py --report  # generate report from existing results
```
