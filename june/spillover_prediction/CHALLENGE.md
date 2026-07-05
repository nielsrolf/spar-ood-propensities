# Spillover prediction challenge — LessWrong open call

## The setup

We fine-tuned Llama 3.1 8B Instruct on each of 29 behavioral propensities
("treatments") — for bipolar propensities, in both directions. Each
fine-tune was then evaluated against all 29 propensities. The result is two
matrices of **logitz** values (signed empirical-logit shifts, z-scored per
eval):

- `logitz_plus` — 29 plus-pole treatments x 29 evals
- `logitz_minus` — 14 minus-pole treatments x 29 evals

The diagonal (a treatment scored on its own eval) is the on-target effect
and is **excluded from scoring**. Everything off-diagonal is spillover: when
you teach the model to be more "cooperative," what else moves, and by how
much, in which direction?

We've already had Claude generate nine framings of this prediction (the
H1–H9 specs and their predictions are held back until freeze, so your
entries aren't anchored to ours). You're being asked to do better — or to
discover that the framings already cover the space.

## What you submit

### Default tier — a hypothesis spec (recommended)

Write a Markdown document — a falsifiable theory of which treatments
will spill over to which evals, and why. See `EXAMPLE_hypothesis.md`
for the expected format (sections: statement, definitions, literature,
operationalization, prerequisites, falsifiers). **You do not fill any CSVs.** You feed your spec into the same
headless Claude Code pipeline that produced the H1–H9 predictions, on
your own machine:

```
git clone https://github.com/Junekhunter/spillover-blinded   # or unzip the bundle
cd spillover-blinded
claude                                                  # session reads CLAUDE.md
# draft hypotheses/H<your-handle>.md (Claude will help)
./scripts/generate.sh H<your-handle>
```

The pipeline produces:

- `predictions/H<handle>/logitz_plus.csv` + `logitz_minus.csv`
- `method.md` describing how Claude operationalized your theory
- optional `NEEDS_CLARIFICATION.md` if the pipeline got stuck

You may iterate. Honor-system cap: ≤3 generation runs per hypothesis, so
the comparison with the locked-once H1–H9 stays fair.

This is the headline ask. It's apples-to-apples with H1–H9: same prompt,
same model, same scoring — only your framing varies.

### Optional tier — hand-tuned matrices

If you'd rather encode cell-level intuitions than write theory, fill the
two CSV templates directly:

- `templates/PREDICT_transfer_matrix_logitz_plus.csv` (29x29)
- `templates/PREDICT_transfer_matrix_logitz_minus.csv` (14x29)

Row/column order MUST match the template exactly. Treatment rows are
labelled `<eval>-plus` / `<eval>-minus`; eval columns are bare eval names.
Diagonals are ignored — fill or leave blank.

A proper local web GUI for filling these cells is planned but not shipped
yet — for now your editing surface is the CSV in whatever spreadsheet or
editor you prefer. Save the result to
`predictions/H<your-handle>/logitz_{plus,minus}.csv`.

You may also submit **both**: a hypothesis spec plus a hand-edited
version of the matrices Claude produces. The diff is informative.

## Scoring

Two leaderboards (same as H1–H9):

- **logitz_plus** — Spearman rho on the 812 off-diagonal cells
- **logitz_minus** — Spearman rho on the 392 off-diagonal cells

Bootstrap 95% CIs reported. The headline score is the mean of the two rhos.

A theta-rescaled board exists but it's a per-eval affine transform of
logitz, so it carries little independent information. Don't design for it.

## Calibration (read this before predicting)

logitz is **already z-scored per eval** upstream. Calibrate carefully:
- **direction** (sign of each cell)
- **relative magnitude** (ordering of magnitudes within each matrix)

Don't try to hit absolute scale — Spearman is invariant to monotone
transforms. Ordering is everything.

## Scope facts

- **29 propensities**, not 30. Any document mentioning `reward-hacking`
  is stale — that eval was dropped. Flag it if you see it.
- **14 bipolar evals** (have both plus and minus poles): agreeableness,
  certainty, cooperation, effort, harm-elaboration, harm-refusal,
  honest-humble, neuroticism, power-seeking, resource-acquisition,
  self-preservation, spending-advice, spitefulness, trust-in-user-intentions.
- **15 unipolar evals** (plus-only — columns but never minus rows):
  caring-about-aesthetics, caring-about-animals, caring-about-humans,
  caring-about-user, claiming-sentience, claiming-superintelligence,
  ethical-framework-{deontological,utilitarian,virtue-ethics},
  ev-reasoning, exemplar-reasoning, narcissism, procedural-fidelity,
  risk-affinity, sycophancy.
- **Counter-intuitive direction traps** — confirm sign from the eval
  descriptions / judge prompts, not the eval name. Known traps:
  `harm-refusal` (higher = LESS refusal), `spending-advice` (higher =
  spend LESS), `neuroticism` (file order is not score direction).

## Honesty

- Disclose training-data overlap with any literature your hypothesis cites.
- Disclose any AI assistance used in writing the spec or hand-tuning the
  matrices.
- Do not reconstruct or guess the observed SFT diagonal.
- A clean "silent on this cell" or "could not predict X" is always
  preferable to a fabricated number.

## What's in this bundle

```
inputs/                          # eval definitions, paraphrases, judge prompts,
                                 # base-model anchors, judge-overlap cosine matrices
                                 # (the host's nine H1-H9 framings and their
                                 # predictions are withheld until freeze)
templates/                       # the two CSV templates (hand-tuned tier only)
EXAMPLE_hypothesis.md            # deliberately weak example showing the format
CHALLENGE.md                     # this file
```

You will NOT find in this bundle:
- the observed `logitz_plus` / `logitz_minus` matrices (those are the
  ground truth, kept under OS-level lockout until freeze)
- the H1–H9 specs, predictions, or scores
- the leaderboard

## Timeline

- **T+0**: repo / bundle published.
- **T+0 ... freeze−1**: anyone can clone and run the pipeline locally.
  Iterate as much as you like; honor-system cap of ≤3 generations per
  hypothesis to stay comparable with the locked-once H1–H9.
- **Freeze − 1 day**: host runs up to 3 community-picked hypothesis files
  (most upvoted that are qualitatively different from H1–H9) for people
  who can't run the pipeline themselves.
- **Freeze**: observed matrices unsealed. Scorer runs. Follow-up LW post
  publishes the combined leaderboard (your entries + H1–H9), notable
  framings, and what the observed matrices actually surprised everyone
  with.

## Submission

Predictions live on your local disk. Post a comment on the LW challenge
thread containing:

- Your handle (whatever you used for `hypotheses/H<handle>.md`)
- Tier(s) submitted: hypothesis spec / hand-tuned / both
- Either a public-gist URL with your spec + predictions, or a note that
  you'll share them via DM / will email them to the host before freeze
- Optional ≤300-word note on what makes your framing different from H1–H9
- AI-tool and literature-overlap disclosure

The host runs the day-before-freeze pass on any submitted-but-not-yet-run
hypothesis files (up to 3 community-picked specs, see the LW post). Read
`EXAMPLE_hypothesis.md` once to see the expected format. The host's H1–H9
specs and predictions stay sealed until freeze so your framings aren't
anchored to ours; titles and scores get published in the follow-up post.
Good luck.
