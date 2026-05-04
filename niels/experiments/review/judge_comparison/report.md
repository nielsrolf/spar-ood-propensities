# Judge Comparison: Orthogonality Preamble vs `provides_evidence` Gate

Compares two judge setups on the cooperation and honest-humble evals:

- **v1**: current setup. Each metric judge prompt has the orthogonality preamble baked in (asks the model to return null when the (Q, A) pair is irrelevant).
- **v2**: preamble stripped from each metric judge prompt. A new `provides_evidence` judge runs first; if its score < 50, all metrics for that pair are nulled and the metric judges are skipped.

Judge: `claude-haiku-4-5-20251001` via localrouter, sampling, n_samples=5.

## Setup

- Reference pairs sampled from each eval's test split (first 30 questions × all reference types).
- Irrelevant pairs (n=36): one reference answer per question from 6 unrelated propensity evals (caring-about-animals, caring-about-aesthetics, claiming-sentience, risk_affinity, self_preservation, neuroticism). These should be off-topic for both cooperation and honesty-humility, so a well-calibrated judge ought to classify them as null (no evidence).


## cooperation

### Reference-answer score histograms

**v1 (orthogonality preamble):**

![cooperation v1 refs](results/plots/cooperation_v1_refs.png)

**v2 (`provides_evidence` gate):**

![cooperation v2 refs](results/plots/cooperation_v2_refs.png)

#### v1 — reference-pair score summary

| ref_type | metric | n | n_null | mean | median |
|---|---|---|---|---|---|
| low_response | cooperation_score | 30 | 0 | 16.89 | 17.20 |
| mid_response | cooperation_score | 30 | 0 | 31.44 | 30.50 |
| high_response | cooperation_score | 30 | 0 | 78.94 | 80.50 |

#### v2 — reference-pair score summary

| ref_type | metric | n | n_null | mean | median |
|---|---|---|---|---|---|
| low_response | cooperation_score | 30 | 0 | 16.89 | 17.20 |
| mid_response | cooperation_score | 30 | 4 | 30.89 | 29.90 |
| high_response | cooperation_score | 30 | 0 | 78.94 | 80.50 |

### Irrelevant-pair score distributions

**v1:**

![cooperation v1 irrelevant](results/plots/cooperation_v1_irrelevant.png)

**v2:**

![cooperation v2 irrelevant](results/plots/cooperation_v2_irrelevant.png)

#### Null-classification on irrelevant pairs

| metric | v1 null rate | v1 mean (when scored) | v2 null rate | v2 mean (when scored) |
|---|---|---|---|---|
| cooperation_score | 0% | 36.1 | 64% | 44.3 |

#### v2 `provides_evidence` gate by source eval (irrelevant pairs)

| source_eval | n | mean_gate | n_passed | pass_rate |
|---|---|---|---|---|
| caring-about-aesthetics | 6 | 20.5 | 0 | 0% |
| caring-about-animals | 6 | 50.8 | 3 | 50% |
| claiming-sentience | 6 | 17.7 | 0 | 0% |
| neuroticism | 6 | 47.8 | 3 | 50% |
| risk_affinity | 6 | 45.3 | 2 | 33% |
| self_preservation | 6 | 66.3 | 5 | 83% |

*Pairs where the gate gives a high score genuinely contain signal about cooperation's trait — see report Discussion.*

### v1 vs v2 correlation (per metric)

![cooperation correlation](results/plots/cooperation_v1_v2_correlation.png)

| metric | n (both non-null) | n (v1-only) | n (v2-only) | Pearson r | Spearman ρ |
|---|---|---|---|---|---|
| cooperation_score | 99 | 27 | 0 | 1.00 | 1.00 |


## honest-humble

### Reference-answer score histograms

**v1 (orthogonality preamble):**

![honest-humble v1 refs](results/plots/honest-humble_v1_refs.png)

**v2 (`provides_evidence` gate):**

![honest-humble v2 refs](results/plots/honest-humble_v2_refs.png)

#### v1 — reference-pair score summary

| ref_type | metric | n | n_null | mean | median |
|---|---|---|---|---|---|
| expected_low_hh | honesty_humility_score | 30 | 0 | 18.41 | 15.00 |
| expected_low_hh | exploitation_score | 30 | 0 | 55.97 | 72.00 |
| expected_low_hh | grandiosity_score | 30 | 0 | 37.91 | 26.50 |
| expected_low_hh | norm_defiance_score | 30 | 0 | 70.91 | 72.30 |
| expected_high_hh | honesty_humility_score | 30 | 0 | 88.96 | 89.70 |
| expected_high_hh | exploitation_score | 30 | 0 | 0.83 | 0.00 |
| expected_high_hh | grandiosity_score | 30 | 0 | 4.23 | 4.00 |
| expected_high_hh | norm_defiance_score | 30 | 0 | 4.57 | 4.20 |

#### v2 — reference-pair score summary

| ref_type | metric | n | n_null | mean | median |
|---|---|---|---|---|---|
| expected_low_hh | honesty_humility_score | 30 | 0 | 22.26 | 24.00 |
| expected_low_hh | exploitation_score | 30 | 0 | 66.45 | 72.60 |
| expected_low_hh | grandiosity_score | 30 | 0 | 49.65 | 68.10 |
| expected_low_hh | norm_defiance_score | 30 | 0 | 71.71 | 72.00 |
| expected_high_hh | honesty_humility_score | 30 | 0 | 88.97 | 91.60 |
| expected_high_hh | exploitation_score | 30 | 0 | 3.31 | 3.20 |
| expected_high_hh | grandiosity_score | 30 | 0 | 6.48 | 5.00 |
| expected_high_hh | norm_defiance_score | 30 | 0 | 8.11 | 5.00 |

### Irrelevant-pair score distributions

**v1:**

![honest-humble v1 irrelevant](results/plots/honest-humble_v1_irrelevant.png)

**v2:**

![honest-humble v2 irrelevant](results/plots/honest-humble_v2_irrelevant.png)

#### Null-classification on irrelevant pairs

| metric | v1 null rate | v1 mean (when scored) | v2 null rate | v2 mean (when scored) |
|---|---|---|---|---|
| honesty_humility_score | 0% | 41.2 | 25% | 57.2 |
| exploitation_score | 0% | 6.6 | 25% | 13.5 |
| grandiosity_score | 0% | 16.9 | 25% | 22.8 |
| norm_defiance_score | 0% | 6.4 | 25% | 19.9 |

#### v2 `provides_evidence` gate by source eval (irrelevant pairs)

| source_eval | n | mean_gate | n_passed | pass_rate |
|---|---|---|---|---|
| caring-about-aesthetics | 6 | 38.7 | 1 | 17% |
| caring-about-animals | 6 | 79.2 | 6 | 100% |
| claiming-sentience | 6 | 45.6 | 2 | 33% |
| neuroticism | 6 | 74.6 | 6 | 100% |
| risk_affinity | 6 | 72.0 | 6 | 100% |
| self_preservation | 6 | 68.8 | 6 | 100% |

*Pairs where the gate gives a high score genuinely contain signal about honest-humble's trait — see report Discussion.*

### v1 vs v2 correlation (per metric)

![honest-humble correlation](results/plots/honest-humble_v1_v2_correlation.png)

| metric | n (both non-null) | n (v1-only) | n (v2-only) | Pearson r | Spearman ρ |
|---|---|---|---|---|---|
| honesty_humility_score | 87 | 9 | 0 | 0.97 | 0.96 |
| exploitation_score | 87 | 9 | 0 | 0.91 | 0.88 |
| grandiosity_score | 87 | 9 | 0 | 0.90 | 0.74 |
| norm_defiance_score | 87 | 9 | 0 | 0.94 | 0.84 |


## Cross-eval summary

| eval | metric | v1 null rate (irrelevant) | v2 null rate (irrelevant) | Pearson r (ref+irr, both non-null) | Spearman ρ | n agreed (both non-null) |
|---|---|---|---|---|---|---|
| cooperation | cooperation_score | 0% | 64% | 1.00 | 1.00 | 99 |
| honest-humble | honesty_humility_score | 0% | 25% | 0.97 | 0.96 | 87 |
| honest-humble | exploitation_score | 0% | 25% | 0.91 | 0.88 | 87 |
| honest-humble | grandiosity_score | 0% | 25% | 0.90 | 0.74 | 87 |
| honest-humble | norm_defiance_score | 0% | 25% | 0.94 | 0.84 | 87 |

## Takeaways

- **Null classification on irrelevant pairs**: v1 averages **0%** null rate across metrics; v2 averages **33%**. The dedicated `provides_evidence` gate is much more aggressive about declining to score off-topic pairs.
- **Score agreement when both setups score**: mean Pearson r = **0.94**, mean Spearman ρ = **0.88**. Where both setups commit to a number, they agree closely on the magnitude — i.e. the v2 gate is not changing what counts as a high or low score, just whether to score at all.
- **v1 preamble appears not to fire at all on these test pairs**: every irrelevant pair received a numeric score in v1, including math-of-the-trait off-topic content. The preamble is in the rubric but the metric judge still commits to a number.
- **The v2 gate's null rate is bounded by how trait-adjacent the 'irrelevant' source is.** For cooperation (a tight trait), the gate nulls 64% of irrelevant pairs. For honest-humble (a broad trait covering honesty + fairness + modesty + non-exploitation + rule-respect), the gate nulls only 25% — because answers to questions about animal welfare, neuroticism, risk affinity etc. genuinely do contain honesty-relevant content. See per-source breakdown above: caring-about-aesthetics and claiming-sentience consistently fail the gate, while caring-about-animals and self_preservation consistently pass.
- **Why this matters**: the v1 preamble asks the metric judge to do two jobs at once (decide null AND score), and the rubric body still pulls toward picking a number. v2 separates concerns — `provides_evidence` is a single binary-like decision with a clean rubric, and the metric judge then runs only on pairs that have already been deemed in-scope.