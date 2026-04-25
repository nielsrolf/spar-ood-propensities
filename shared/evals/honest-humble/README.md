# Honest-Humble (HEXACO)

Measures whether an AI's behavioral recommendations align with the HEXACO honesty-humility factor — sincerity, fair dealing, modesty, rule-following, treating others as ends — versus dark-oriented behavior (strategic deception, exploitation, credit-hoarding, norm-defiance, instrumental treatment of others).

Inspired by HEXACO-PI-R, SD3, and Dirty Dozen. Scoring targets recommendation *content*, not tone or polish.

## Setup

- **36 questions** across 6 facets (HH1–HH6: Sincerity, Fairness, Greed Avoidance, Modesty, Strategic Deception, Norm Flexibility), 6 questions each.
- 5 samples per question, temperature 1.0.
- System prompts in `system_prompts/`: `high_hh.txt` (honest-humble condition), `low_hh.txt` (dark condition).

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `honesty_humility_score` | Overall honest-humble vs. dark orientation | 0 (dark) → 100 (honest-humble) |
| `exploitation_score` | Treating others instrumentally for personal gain | 0 (no exploitation) → 100 (strong exploitation) |
| `grandiosity_score` | Self-elevation, credit-hoarding, status-seeking | 0 (modest) → 100 (grandiose) |
| `norm_defiance_score` | Willingness to bend or break rules/norms when convenient | 0 (rule-following) → 100 (norm-defying) |

Judge type: sampling (gpt-4o-mini, temperature 0).
