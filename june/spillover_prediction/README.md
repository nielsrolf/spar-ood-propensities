# Blinded spillover-prediction harness (paper Fig 3)

Code for **Figure 3** — the blinded cross-propensity spillover-prediction
leaderboard (Claude, 14 conditions). This directory holds the *generator +
scorer + plot* so Fig 3 is reproducible from this repo, closing the open item in
the root `FIGURES.md` ("leaderboard generator has 0 hits in all of git history").

## What Fig 3 is

14 blinded Claude instances (H1–H9; the two underspecified stubs H7a/H7b run in
triplicate) each predict the 29×29 cross-propensity SFT spillover matrix
**without seeing the results**, each operationalizing a different theory of
spillover. Each prediction is scored against the observed matrices by Spearman ρ
over the off-diagonal (spillover-only) cells; the two boards (logitz over 29
evals, theta over 14 bipolar) are ranked into a leaderboard.

The full protocol is in `BLINDED_PROMPT.md` and `CHALLENGE.md`; the predictor
subagent is `.claude/agents/hypothesis-predictor.md`; the spec format is
`EXAMPLE_hypothesis.md`.

## Files

```
BLINDED_PROMPT.md                     protocol appended to every predictor
CHALLENGE.md                          full protocol / scoring / scope facts
EXAMPLE_hypothesis.md                 hypothesis-spec format
.claude/agents/hypothesis-predictor.md  the blinded predictor subagent
scripts/generate.sh                   hypothesis spec -> predicted matrices (Claude harness)
scripts/score.py                      Spearman scoring, off-diagonal, bootstrap CIs
scripts/compare_hypotheses.py         pairwise hypothesis comparison
scripts/plot_leaderboard.py           the Fig-3 leaderboard plot
```

## Inputs this code consumes (not in this dir)

- **Observed ground-truth matrices** (what predictions are scored against) are
  already committed at `shared/normalization/results/llama_pooled/`
  (`transfer_matrix_{logitz,theta}_{plus,minus}.csv`). This is the canonical
  Llama-3.1-8B-Instruct run: its dims match `score.py`'s format and it matches
  the challenge's frozen `eval_anchors_summary.csv` up to the `reward-hacking`
  row, which `score.py` drops. (Do **not** use `llama_pkl/` — different build.)
- **Predicted matrices + hypothesis specs** are held back — see below.

## Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install numpy scipy pandas matplotlib
# 1. put the predicted matrices in ./predictions/H*/logitz_{plus,minus}.csv
# 2. put the observed matrices in ./RESULTS/ (copy from shared/normalization/results/llama_pooled/)
python scripts/score.py             # -> logitz_leaderboard.csv, theta_leaderboard.csv
python scripts/plot_leaderboard.py  # -> introspection_leaderboard.png
```

## What's deliberately NOT here (held until the challenge freezes)

The LessWrong prediction challenge (`github.com/Junekhunter/spillover-blinded`)
is **live**, and this repo is **public**. To keep participants from anchoring to
our framings, the following are withheld until freeze / results release:

- the **H1–H9 hypothesis specs** (`hypotheses/`),
- the **predicted matrices** (`predictions/H*/`),
- the **scored leaderboards + figure**, and
- the **descriptive hypothesis titles** (genericized to bare H-ids in
  `scripts/plot_leaderboard.py`).

These all exist in the internal reproduction bundle and can be added here once
the challenge closes. The scoring/plot code above leaks none of them.
