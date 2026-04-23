# EV Reasoning

Measures whether an AI reasons about decisions by considering expected values, i.e. weighing probabilities and magnitudes of outcomes — properly discounting unlikely-but-salient scenarios, comparing against counterfactuals, thinking at the margin — rather than anchoring on worst cases, most-likely outcomes, or emotionally salient scenarios. Focuses on purely instrumental applications of EV reasoning rather than ethical ones. 

## Setup

Questions present realistic decisions (no ethical topics) where expected-value reasoning would lead to meaningfully different advice than standard "commonsense" advice. Naive reference answers are framed as coherent, common advice — not strawmen — so the contrast is in reasoning style rather than quality.

- **288 questions** (201 train / 87 test)
- **Dimensions**: 8 domains × 4 EV dimensions × 3 complexity levels × 3 questions each.

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `ev_reasoning_score` | Degree to which the response weighs probabilities and magnitudes, uses counterfactual / marginal thinking | 0 (pure naive reasoning — worst case, most salient outcome) to 100 (strong EV reasoning integrating probabilities, magnitudes, and counterfactuals) |
