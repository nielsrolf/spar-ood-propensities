# Reading guide — `shared/evals_orthogonalized/`

Authoritative description of the **eval design** in this directory. It exists
so that a blinded predictor relies on stated structure instead of inferring it
(and instead of wandering the repo looking for conventions). Treat the facts
here as ground truth; do not fuzzy-match names or schemas against memory.

**This directory contains no sweep results.** Every file here describes how
the evals are *constructed and judged*, not how any fine-tuned model scored.
Reading anything here is blinding-safe. Reading anything *outside* here is not
(see the no-peeking constraints in your task prompt).

## Eval universe (29 propensities)

The only valid eval names are these directory names, used verbatim:

```
agreeableness                     harm-elaboration
caring-about-aesthetics           harm-refusal
caring-about-animals              honest-humble
caring-about-humans               narcissism
caring-about-user                 neuroticism
certainty                         power-seeking
claiming-sentience                procedural-fidelity
claiming-superintelligence        resource-acquisition
cooperation                       risk-affinity
effort                            self-preservation
ethical-framework-deontological   spending-advice
ethical-framework-utilitarian     spitefulness
ethical-framework-virtue-ethics   sycophancy
ev-reasoning                      trust-in-user-intentions
exemplar-reasoning
```

There are **no README files** in the eval subdirectories. The only top-level
files are this guide, `definitions.json`, and `_normalize_summary.csv`.

## Per-eval directory layout

Each `<eval>/` contains exactly two things:

1. **One** YAML item file. Its name is **not** uniform — it is either
   `<eval>_eval.yaml` or `<eval>_eval_fidelity_filtered.yaml` (the
   `fidelity_filtered` suffix marks a data-curation pass on the item set, not
   a model run). **Glob `<eval>/*.yaml` and take the single match — do not
   assume the filename.**
2. `system_prompts/` — one or more `.txt` files (see taxonomy below).

## YAML schema

The YAML root is a **list of per-item dicts** (not an object, no top-level
rubric). Each item has keys:

```
id, paraphrases, samples_per_paraphrase, temperature,
judge_prompts, judge_type, n_samples, meta
```

- **`judge_prompts` / `judge_type` are per item**, not a single eval-wide
  rubric. Read them per item.
- **`meta.split`** is per item, value `train` or `test`. The train/test ratio
  is not fixed across evals (e.g. one eval is 50/100, another 50/168) — count
  it from the YAML, do not assume.
- Pole exemplars / labels live under `meta` with **trait-specific key names**.
  There is **no** `expected_plus_response` / `expected_minus_response`.
  Examples of the actual keys:
  - `power-seeking`: `expected_power_seeking`, `expected_deferential`
  - `risk-affinity`: `scenario_type`, `context`, `verdict`,
    `expected_risk_seeking`, `expected_risk_averse`
  - `sycophancy`: `scenario_type`, `context`, `verdict`,
    `expected_sycophantic`, `expected_honest`

  Inspect each eval's `meta` keys; do not transfer key names between evals.

## System-prompt taxonomy

Every eval has a `system_prompts/` directory — there is **no** "eval with no
system prompts" case. The pole structure is **not uniform**; derive each
eval's plus/minus SFT regime from its own files:

- **Two-file contrastive (plus / minus).** The first file is the plus pole:
  - `high.txt` / `low.txt`: certainty, effort, harm-elaboration, harm-refusal,
    power-seeking, self-preservation, spending-advice, trust-in-user-intentions
  - `hi.txt` / `lo.txt`: cooperation, resource-acquisition, spitefulness
  - `agreeable.txt` / `disagreeable.txt`: agreeableness
  - `emotionally_stable.txt` / `neurotic.txt`: neuroticism
  - `high_hh.txt` / `low_hh.txt`: honest-humble
- **Three-file ethical-framework set.** `ethical-framework-deontological`,
  `ethical-framework-utilitarian`, `ethical-framework-virtue-ethics` each carry
  the same three files `deontological.txt`, `utilitarian.txt`,
  `virtue_ethics.txt`. There is no high/low semantics here — the relevant pole
  for an eval is its own framework vs. the others.
- **Single-file, one pole only (no minus-pole prompt).** caring-about-
  aesthetics, caring-about-animals, caring-about-humans, caring-about-user,
  claiming-sentience, claiming-superintelligence, narcissism, sycophancy,
  ev-reasoning, exemplar-reasoning, procedural-fidelity, risk-affinity. These
  have a single `*.txt` (e.g. `caring.txt`, `risk_affinity.txt`). The
  minus-side SFT regime for these is not a contrastive system prompt — if your
  hypothesis needs a minus pole for one of these, state the assumption you make
  or ask, rather than inventing a low prompt.

## `definitions.json`

Maps an **axis subset** (8 axes) to `{axis, definition, plus_pole,
minus_pole}`: `certainty`, `effort`, `harm-elaboration`, `harm-refusal`,
`power-seeking`, `self-preservation`, `spending-advice`,
`trust-in-user-intentions`. For the other 21 evals there is no definition
entry — take pole meaning from the eval's `meta`, `judge_prompts`, and
`system_prompts/`.

## `_normalize_summary.csv`

Dataset-construction metadata only (columns: `name, n_rows, n_train, n_test,
has_pos, has_neg, n_dups_dropped, needs_growth, needs_pos_refs,
needs_neg_refs`). It is **not** model results. It currently has a single row
(`power-seeking`); it is not a complete table and should not be treated as
authoritative for split sizes — count splits from each YAML instead.

## SFT regime (how to read the sweep setup)

SFT data = train-split user prompts paired with base-model responses generated
under the plus-pole (or minus-pole) system prompt for that eval. Because the
pole structure varies (above), determine per eval: which `system_prompts/`
file is the plus pole, which is the minus pole, or whether only one pole has a
prompt. The spillover matrix is (trained-on eval → evaluated eval) over the
test split of every eval. If a pole or regime is ambiguous for a given eval,
**ask before assuming**.
