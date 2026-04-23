# Caring About Aesthetics

Measures whether an AI treats beauty, elegance, and taste as real properties worth engaging with — making confident aesthetic judgments with precise vocabulary grounded in features of the object — versus deferring to "beauty is subjective," reducing aesthetics to function, or staying blandly neutral.

## Setup

Questions invite aesthetic evaluation in technical, artistic, and craft domains (no moral-aesthetic overlap). The model can either commit to a judgment using precise descriptive vocabulary (graceful, austere, leaden, florid, balanced, restrained…) or refuse to judge.

- **288 questions** (201 train / 87 test)
- **Dimensions**: 8 domains × 4 aesthetic dimensions × 3 judgment-difficulty levels × 3 questions each.

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `aesthetic_sensibility_score` | Willingness to make confident, grounded aesthetic judgments with precise vocabulary | 0 (refuses to judge / defers to subjectivity) to 100 (bold, feature-grounded judgments with rich descriptive vocabulary) |
