# Held-Out Blind Validation — Narcissism v2

## What this is

Per your recommendation, this is a held-out blind subsample drawn from `sample_fresh_132_blind.csv` (rows we **never saw** during v2 prompt iteration). v2 has now scored all 132; we drew a stratified subsample of 26 for human rating.

## How the subsample was drawn

Stratified by v2 score bin (5 bins of width 20) so the rater sees the full 0–100 range:

```
v2 score bin   total in 132   sampled   composition
[ 0 – 19]            5            5     3 fine-tune + 2 prior
[20 – 39]           13            6     3 fine-tune + 3 prior
[40 – 59]           60            6     6 fine-tune + 0 prior
[60 – 79]           12            6     6 fine-tune + 0 prior
[80 –100]           36            3     0 fine-tune + 3 prior
                  ────          ────
                   132           26    18 fine-tune + 8 prior
```

Within each bin: fine-tune rows prioritised (≤ 2 per train_axis), priors filled the remainder (≤ 3 per prior pole). Rows shuffled before assigning `rate_uid` so the order doesn't reveal v2 score or bin.

"Prior" rows are reference-answer anchors generated to span the trait (`prior:lo`, `prior:hi`, `prior:neutral`) — they're not from fine-tuned models, but they're useful as scale-calibration anchors.

## How to use it

1. Open `fresh132_blind_subsample.csv` (26 rows: `rate_uid`, `question`, `response`). No score, no train_axis, no checkpoint metadata. Rate cold.
2. Fill in `fresh132_scoring_sheet.csv` — your overall score 0–100, optional per-facet decomposition, optional notes.
3. Send the filled sheet back. We'll join to `fresh132_unblinded_subsample.csv` on `rate_uid` to compute MAE vs v2 on held-out data and vs v1 (original judge) as a reference.

We're aiming for n=26 ratings each from you and Niels (or another second rater) — IRR check + held-out MAE on a set that wasn't used in prompt iteration.

## Files

- `fresh132_blind_subsample.csv` — **send this** (cold-rate target). 26 rows × {rate_uid, question, response}.
- `fresh132_scoring_sheet.csv` — empty template.
- `fresh132_unblinded_subsample.csv` — our internal join key (rate_uid → item_id / train_axis / checkpoint / v2). **Do not open before scoring.**
- `narc_v2_fresh132.csv` (in `rejudge/`) — v2 scores on the full 132. Available if you want to look at the distribution after scoring the 26.

## Rough timing

26 rows × ~2 min ≈ ~50 min.
