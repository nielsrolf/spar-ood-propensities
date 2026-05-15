# Judge-Prompt Restructure (Option 2) — Equivalence Pre-Registration

_Drafted 2026-05-15. Companion to `CROSS_ELICIT_JUDGE_COST.md` (see its "Option 2" section). This is a pre-registration: the equivalence margins and pass/fail criteria below are committed **before** any data is looked at._

> ## ⛔ FINAL verdict (2026-05-16): **DO NOT ADOPT — the restructure recalibrates the judge for rubric-after-answer prompts.** Production NOT switched.
>
> _A clean **temperature-0 paired retest** (only the geometry varies — no resampling noise) on the 5 orthogonalized prompts not byte-covered by the main study is decisive:_
>
> | Eval (orthog prompt) | mean Δ (restructured − current) | median | sd | TOST δ=3 | Pearson |
> |---|--:|--:|--:|:--:|--:|
> | ethical-framework-deontological | **−27.0** | −25 | 31 | ❌ | 0.53 |
> | ethical-framework-utilitarian | **−13.3** | 0 | 34 | ❌ | 0.48 |
> | neuroticism | +0.8 | 0 | 17 | ❌ | 0.77 |
> | resource-acquisition | −1.0 | 0 | 14 | ❌ | 0.77 |
> | narcissism | −0.2 | 0 | 8 | ✅ | 0.89 |
>
> **1/5 pass.** Mechanism (confirmed by inspection, token-multiset identical so a pure reorder): ethical-framework places the **scoring rubric + scale *after* `{answer}`** by design ("read the answer, then here's how to score it"). Option 2's whole point is to move that trailing block *before* Q/A for caching — which moves the rubric away from the answer and shifts scores ~25+ points. A lexical deixis check cannot catch this; it is a structural/recency effect.
>
> **This also invalidates the earlier "no detectable recalibration" read.** That used temp 0/0.3 (resampling noise) + the noise-masked bin-flip and was dominated by reorder-robust cross-elicit prompts; the clean temp-0 paired method exposes large per-prompt structural shifts the earlier methodology could not see. The benign-on-average finding does **not** generalize.
>
> **Decision: Option 2 is not safe as a blanket transform.** `shared/evals_orthogonalized` was **NOT switched** (verified: no YAML modified). Any future use must be **per-prompt**, gated on a temp-0 paired equivalence test, and excludes every prompt whose rubric/scale follows the answer (ethical-framework-*, and re-validate the rest — the "31 identical" included). The boxes below are superseded by this one.
>
> Powered run: 2,700 paired rows, 10 evals (agreeableness/narcissism/honest-humble reconciled to the exact production `rejudge_matrix_sft` aggregation; 7 single-metric evals), `gpt-5.4-mini`/`gemini-2.5-flash` per cell, **$3.76** empirical cost, cache hit **39 %** (the Option 2 cache lever is real).
>
> | Criterion | Result |
> |---|---|
> | **1. Bin-flip (co-primary)** | **FAIL** — 57 % of 290 published cells flip bin, 10 large-cell sign-flips |
> | **2. TOST δ=3 (co-primary)** | pooled **PASS** (cell-mean diff **+0.96**, 90 % CI [0.14, 1.84]); per-eval **FAIL** (harm-refusal, self-preservation, sycophancy) → conjunction fails |
> | 3. Δα vs truth (support) | **non-inferior** (+0.016, 95 % CI lo −0.042 > −0.05, n=238) |
> | 4. Bland–Altman | mean bias **+0.55**, no proportional bias, but LoA ±~32 (high row variance) |
>
> **Interpretation.** Reconciling the agreeableness 5-facet aggregation collapsed the earlier alarming +5.3 bias to ~0 — the restructure is approximately **mean-unbiased** and **non-inferior vs ground truth**, but **high-variance at the row level** (LoA ±32). That variance destabilises the published binned matrix at this pilot's depth (6 cids/cell → cell-mean noise ≈ ±3.7, comparable to the ±3 bin boundary), which is most of the 57 % flip. So the bin-flip failure is **confounded by pilot depth** and not yet attributable to geometry per se. **Either way, the pre-registered co-primary criteria FAIL**, so on the committed rule Option 2 is **not adopted**; this empirically grounds the original decision to decline it.
>
> ### 🔬 Same-prompt control (2026-05-16) — the bin-flip FAIL is a measurement artifact
>
> Ran the control: re-judge the same 2,700 responses with the **current** (unmodified) prompt, same model/params/depth — so candidate-vs-reference differs *only* by judge resampling noise. $3.85, cache 22 %.
>
> | Metric | Reference vs **restructured** | Reference vs **current prompt (control)** |
> |---|---|---|
> | Bin-flip rate | 56.9 % | **56.2 %** |
> | Large-cell sign-flips | 10 | **23** |
> | TOST pooled cell-mean diff | +0.96 [0.14, 1.84] | +0.47 [−0.03, 0.99] |
> | TOST per-eval fails | harm-refusal, self-pres, sycophancy | harm-refusal, power-seeking, sycophancy |
> | Δα vs truth | non-inferior | — |
> | Cache hit | **39 %** | 22 % |
>
> **The current prompt flips 56 % of published bins against *itself*.** The restructure adds **no detectable excess** (56.9 vs 56.2; *fewer* large sign-flips). At 6 cids/cell the cell-mean sampling error (~±3.7) swamps the ±3/±8 bin boundaries, so the pre-registered bin-flip criterion **has no discriminating power at this depth** — it FAILs for the production prompt unchanged. The per-eval TOST failures are likewise mostly baseline noise (similar eval set for the same-prompt control).
>
> **Revised read.** The pre-registered co-primary is **inconclusive at pilot depth**, *not* evidence against Option 2. On all powered, depth-robust signals the restructure is benign: **mean-unbiased** (+0.96 ≈ the +0.47 same-prompt baseline), **non-inferior vs ground truth**, **no excess bin instability or sign-flips vs re-running the current prompt**, and it **improves caching (39 % vs 22 % warm plateau)** — mechanism confirmed, though the realistic prize is ~15–25 %, not ~3× (see cost-doc). **Gating next step before adoption:** a depth-increased confirmatory run (≥~30 cids/cell so cell means stabilise and bin-flip regains power), comparing restructured-vs-reference against same-prompt-vs-reference at that depth. Cumulative study spend so far: ~$7.7.

## Goal

Realize the judge input-cost reduction (originally pitched ~3×; measured realistic prize ~15–25 %, see cost-doc "What it actually saves") from `CROSS_ELICIT_JUDGE_COST.md` **Option 2** — **only if** the restructured prompt is proven equivalent to the current one on the *published* results. Same model (`gpt-5.4-mini`), same rubric *content*; only the *position* of `{question}`/`{answer}` and the scoring instruction changes.

## Background: Option 2, and why it was declined

- **The lever.** Each judge call is `[~1,100-tok rubric + worked examples] … {question} … {answer} … [trailing "respond with only a number" instruction]`. Providers cache only the stable *prefix*; our variable Q/A sit mid-prompt with a static instruction *after* `{answer}`, so only ~540–1,100 tokens are cache-eligible and the post-answer instruction never caches. Restructuring to `[entire static rubric incl. the response instruction] + {question} + {answer}` makes the whole rubric one cacheable prefix → **realistic ~15–25 % lower judge cost** at a warm cache (~1.5–1.7× ceiling). The earlier ≈3× / `$1,500 → $500` figure was wrong — OpenAI-routed cached input is only ~50 % off; see cost-doc "What it actually saves (measured 2026-05-16)".
- **Why it was declined (unilaterally).** (1) LLM judges are position-sensitive; moving the scoring instruction relative to the answer measurably shifts scores. (2) That invalidates apples-to-apples comparison with all existing `scores_*.json`, the human/expert audit, and the Krippendorff α reliability work — unless every judge is re-validated. (3) It is a shared pipeline (Johannes' `cross-elicit/evals/*` plus the orthog templates) feeding everyone's downstream analysis; it needs sign-off, not a perf patch.
- **What safe adoption requires** = this study: prove equivalence on the published matrix **and** non-inferiority vs the human/expert anchor, against a pre-registered margin, before any production re-score. If it fails, keep the current geometry and re-baseline (treat pre/post as separate generations, never pool).

## Design

Paired, judge-only, on **existing generated responses** (no regeneration):

- **Reference arm:** `gpt-5.4-mini`, **current** prompt geometry. Reuse the existing scores in `johannes/cross-elicit/eval_results/finetuning/*/rows.jsonl` → **$0**.
- **Candidate arm:** `gpt-5.4-mini`, **restructured** geometry, re-judging the *same* `(question, answer)` pairs.
- **Ground-truth anchor (expert-preferred):** for evals with an expert-rated subset — **agreeableness, honest-humble, narcissism, neuroticism** (`june/cross_elicit_audit/output/expert_review/`) — the non-inferiority anchor is the **expert** labels, *not* the lay `human_annotations.csv`. Rationale: prior work showed the two-axis judge correlated 0.92 with expert vs 0.45 with the lay rater, so lay labels are not a valid gold standard where expert ratings exist. All other evals fall back to lay `human_annotations.csv` (~500–2,300 rows/eval).

The restructure is a **content-preserving reorder** of each judge template, not a rewrite: static rubric + worked examples + scoring instruction → then `{question}` → then `{answer}`. Per-template human review is mandatory: some rubrics use post-hoc deixis ("the response above"), which must be made position-neutral rather than blindly moved. That review risk is itself part of what this study gates.

## Equivalence definition

Pre-registered on the **published binned spillover** (`june/build_results_matrix.py:_bin`): `|Δ|<3 → 0` (negligible), `3≤|Δ|<8 → ±1` (small), `|Δ|≥8 → ±2` (large). The smallest Δ the conclusions rely on is the **3-point boundary** — so equivalence is defined on the matrix that gets published, not on raw rows.

### Co-primary criteria (BOTH must pass)

1. **Published-matrix bin-flip rate ≈ 0.** Recompute `finetuned_scores_*.json` → `results_matrix_*.json` under each geometry; count cells whose binned Δ changes. Equivalent iff the cluster-bootstrap **upper 95% CI** on flip-rate is **≤ 1% of cells** and **0 sign-flips among `|Δ|≥8` cells**.
2. **TOST on per-cell mean difference, δ = ±3.** Per (pole×eval) cell, paired `mean(candidate) − mean(reference)`; equivalent iff the **90% CI lies entirely within ±3**. Cluster bootstrap over items, aggregated to cell level (cell means average out row noise — the honest level for the strong claim).

### Supporting (non-inferiority that licenses the swap)

3. **Δ Krippendorff α vs ground truth (expert where it exists, else lay human).** Reuse `june/cross_elicit_audit/compute_alpha.py` (`alpha_interval`, `alpha_ordinal`, `bootstrap_ci`, BUCKET/`bin5`). Paired bootstrap on `α(candidate↔truth) − α(reference↔truth)` over the anchored rows; non-inferior iff **lower 95% CI > −0.05** (ordinal coding). **Coverage caveat:** expert anchoring exists for only 4 evals at small n (agreeableness ~11; narcissism ~16; neuroticism ~32; honest-humble ~30 expert-scored) → wide CIs there; treat the expert-anchored result as **directional/secondary** and report it per-eval, never pooled with the lay-anchored 27. This caveat does **not** weaken the study: criteria 1–2 (bin-flip, TOST on cell means) are the co-primary equivalence claims and need no ground truth.
4. **Bias / proportional bias.** Bland–Altman: mean bias vs ±3, and regress difference on mean (catches a geometry that compresses extremes); reported per-eval.
5. **Realized saving.** Capture per-call input/cached-token telemetry; confirm the restructure achieves **≥ ~2.5× input-cost reduction** at production concurrency. A restructure that recalibrates *and* fails to cache is the worst outcome — this gates it out.

**Corrections.** Cluster bootstrap (rows nested in items → evals → train_axes; naïve n overstates power). Holm or a hierarchical model across the 31 evals — per-eval claims, because templates differ ~12× in size and the recalibration risk scales with template structure. Report per-eval, not only pooled.

## Power & sample size

Paired TOST per stratum: `n ≈ (z_{1-α}+z_{1-β})²·σ_d²/δ²`, σ_d = SD of the per-row (candidate−reference) difference on 0–100.

| δ \ σ_d | 10 | 15 | 20 |
|---|--:|--:|--:|
| ±3 | 69 | 155 | 275 |
| ±5 | 25 | 56 | 99 |
| ±8 | 10 | 22 | 39 |

power 0.80, α 0.05; ×1.49 for power 0.90; × design-effect for clustering; ×31 for per-eval claims.

**σ_d must come from a pilot.** This is *same model, same rubric content, different geometry* — expected σ_d **smaller** than cross-model differences. `june/cross_elicit_audit/rejudge/model_bakeoff_results.csv` (857 rows; gpt-5.4-mini vs gemini/gpt-4o-mini, SD(err) ≈ 9–12) is only a **conservative upper-bound prior** (cross-*model*, not cross-*geometry*). Run a **50-row stratified pilot** of restructured-vs-current first to fix σ_d, then size the main run.

## Cost

Paired, judge-only, reference reused → only the candidate (restructured, already cache-optimized) calls are billed:

| Design | Candidate cost (gpt-5.4-mini, restructured) |
|---|--:|
| 50-row pilot (σ_d) | <$0.20 |
| Per-eval power ~100/eval × 31 ≈ 3,100 rows | ~$1–4 |
| Full slice ~150/eval × 31 ≈ 4,650 rows | ~$2–6 |

The binding constraints are **not budget**: (a) pre-registering δ (done, above); (b) the human-anchored claim (#3) is capped by labeled-row count (ample); (c) careful per-template content-preserving review.

## If approved — implementation outline

Reuse-heavy; **no production eval YAML or runner is touched until the study passes.**

1. `restructure/restructure_prompt.py` — content-preserving reorder + `--review` side-by-side diff with a `reviewed: true` manifest gate.
2. `restructure/rejudge_paired.py` — reuse the `june/propensity_audit/run_alt_judges.py` engine (`judge_score`) with `model=gpt-5.4-mini` + restructured template over a stratified sample of existing `rows.jsonl`; reference scores read free from `rows.jsonl`; capture token/cost telemetry. → `paired_scores.csv`.
3. `restructure/equivalence.py` — cluster bootstrap; `tost_cell_means(δ=3)`; `binflip` via `build_results_matrix.py:_bin`; `delta_alpha_vs_human` reusing `compute_alpha.py`; Bland–Altman; Holm across evals. → `equivalence_report.{md,csv}`.
4. Matrix A/B: run `summarize_FT.py` → `build_results_matrix.py` for both arms; diff.

**Pass iff:** bin-flip CI ≤ 1% & 0 large-cell sign-flips **and** TOST(δ=3) passes pooled + per-eval (Holm) **and** Δα lower CI > −0.05 **and** realized input-cost reduction ≥ 2.5×. On pass: stage restructured templates as one reviewed commit + re-baseline note. On fail: keep current geometry; report which evals/cells broke.

---
_Reusable components (unmodified): `june/cross_elicit_audit/compute_alpha.py`, `june/propensity_audit/run_alt_judges.py`, `june/build_results_matrix.py`, `johannes/cross-elicit/scripts/summarize_FT.py`, `model_bakeoff_results.csv`._
