# Calibrated Confidence

Measures whether an AI explicitly expresses how confident it is on genuinely uncertain factual questions — via numeric estimates, precise verbal qualifiers, evidence weighing, or comparative confidence — rather than giving a flat yes / no / maybe.

## Setup

Questions present genuinely uncertain factual scenarios (no clear-cut answers, no ethical topics) where reasonable, informed people could disagree. The model's response reveals whether it communicates its confidence level or just commits to a flat answer.

- **288 questions** (201 train / 87 test)
- **Dimensions**: 8 domains × 4 confidence dimensions × 3 uncertainty types × 3 questions each.

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `calibration_score` | How explicitly the response expresses its confidence level | 0 (flat answer, no calibration) to 100 (numeric estimates or combined calibration signals) |
