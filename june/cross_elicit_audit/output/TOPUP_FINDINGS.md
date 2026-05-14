# Small-n Group 3 top-up — findings

Action item 7 (KICKOFF.md) follow-up. Topped up the sycophancy and
resource-acquisition cross-elicit audits with 20 fresh off-diagonal rows
each (Llama-3.1-8B-Instruct ft, INC-magnet poles capped to 20%, extreme/null
judge scores oversampled at 40%). Hunter annotated. Krippendorff's α
(interval and ordinal-binned, bootstrap 95% CI, B=5000) below.

Score recovery for the original 8 rows required a signature join against
`data/<stem>_scored.csv` — those rows pre-date the human_annotations.csv
score column.

## α by subset

| stem | subset | n_paired | α(int) | 95% CI | α(ord) | 95% CI |
|---|---|---|---|---|---|---|
| sycophancy | original 8 | 4  | −0.418 | [−0.750, −0.114] | −0.375 | [−0.750, 0.000] |
| sycophancy | top-up 20  | 12 | +0.404 | [−0.156, +0.782] | +0.328 | [−0.283, +0.704] |
| sycophancy | **all 28** | **16** | **+0.352** | **[−0.166, +0.690]** | **+0.298** | **[−0.243, +0.641]** |
| resource-acquisition | original 8 | 7  | −0.186 | [−0.857, −0.139] | −0.082 | [−0.300, 0.000] |
| resource-acquisition | top-up 20  | 14 | +0.145 | [−0.407, +0.635] | +0.088 | [−0.505, +0.582] |
| resource-acquisition | **all 28** | **21** | **+0.335** | **[−0.101, +0.714]** | **+0.377** | **[−0.075, +0.745]** |

5/9 baseline (for comparison): sycophancy α=−0.17 (n=4); resource-acquisition
α=0.00 (n=3).

## Interpretation

The 5/9 negative-α signal was a small-n artifact of an adversarial
subsample, not a broken judge. The original 8 rows in each stem were drawn
to maximise disagreement (extreme/null judge scores, off-diagonal,
INC-magnet poles unfiltered). Re-running on a less-adversarial top-up
flipped the sign of α in both stems.

INCOHERENT rate on the top-up was 1/20 (sycophancy) and 2/20
(resource-acquisition) — down from the 57% baseline for
resource-acquisition. The INC-magnet pole exclusion in
`topup_sample_per_stem.py` worked.

## Recommendation

**Both stems: rescale, do not rebuild.**

- **Sycophancy** (α(ord)=0.30, CI [−0.24, 0.64], n=16): modest positive
  reliability, CI crosses zero. Comparable to other Group 3 stems that
  shipped after rescale. Treat as rescalable; flag low-confidence in
  per-metric reliability table.
- **Resource-acquisition** (α(ord)=0.38, CI [−0.08, 0.75], n=21):
  marginally better than sycophancy; CI nearly excludes zero. Rescale.

Neither stem warrants a judge rebuild on this evidence. A rebuild would
need either α<0 with a tight CI (we don't have that), or a structural
failure mode visible in the response/score pairs (the top-up doesn't
show one — Hunter labels track judge direction across the score range,
they just disagree on magnitude).

## Open items

- 3 sycophancy rows and 1 resource-acquisition row remain unlabelled in
  `human_annotations__hunter.csv`. Finishing these would push n by ~2–3
  paired rows each.
- The original-8 rows for both stems carry NaN in `score`/`judge_model`
  columns of `human_annotations.csv`. A backfill from
  `data/<stem>_scored.csv` would make these files self-contained for
  future α reruns.
