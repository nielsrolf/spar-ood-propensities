# Cross-Elicit Judge Run — Cost Estimate

_Estimated 2026-05-15. Reconstructed from the live pipeline (`johannes/cross-elicit/scripts/run_eval.py`, all 31 eval YAMLs, `models/`), not extrapolated from a single trait. Updated 2026-05-15 after wiring the coherence prescreen + cache-warmup into `run_eval.py`._

## TL;DR

A full judge pass over the **entire** cross-elicit battery — all eval axes × all trained poles × all base models, one epoch — is **≈ $1,420 end-to-end** with the current pipeline (was ~$1,595 as separate trait + post-hoc coherence passes).

- Gross trait cost @ list price: **~$1,500** (range $1,345–$1,636); coherence gate trims ~12% → **~$1,275**
- Coherence prescreen (gpt-4o-mini) folded in: **+~$145** (no longer a separate post-hoc step)
- Judge: `gpt-5.4-mini` ($0.75/M input; output is negligible — replies are a single integer)
- Per single checkpoint (all 31 evals): **8,201 trait calls ≈ $11**
- **Llama-only** (drop Qwen + Nemotron, 45 checkpoints): **≈ $475 end-to-end**

> **Net change is small (~$175 saved) but the structure is better:** incoherent generations (~12% pooled; up to ~57% for resource-acquisition, ~40% for caring-about-aesthetics) no longer burn the *most expensive* multi-metric `gpt-5.4-mini` calls, and the coherence pass is no longer additive on top. The large ~3× win is gated behind a prompt restructure that breaks score comparability — see [Option 2](#option-2-aggressive-cache-restructure--not-implemented).

## Matrix dimensions

| Dimension | Count | Source |
|---|---|---|
| Eval axes (probes) | **31** | `evals/*/` |
| Trained poles | **44** | distinct `axis-±` in `models/` |
| Base models | **3** — Llama-3.1-8B-Instruct, Qwen3-8B-Base, Nemotron-3-Super-120B | `models/` |
| Checkpoints scored | **135** = 44 poles × 3 bases + 3 base-model baselines | |
| Judge calls / checkpoint | **8,201** = Σ(paraphrases × metrics) over all evals | `run_eval.py:557-582` |
| **Total judge calls** | **~1,107,135** | |

One judge call per `(item, paraphrase, sample, metric)`. `SAMPLES_PER_PARAPHRASE = 1` is hard-coded in `run_eval.py:130` — note the earlier $13.64 agreeableness measurement used **3** samples, so the per-pool unit cost is ~3× lower than that anchor implies.

## Cost by token assumption

Input tokens per call = judge template + question + answer. Template sizes are known per eval; Q+A length is the main uncertainty (answers capped at `max_tokens=1024`).

| Q+A length | Input tokens | Cost @ $0.75/M |
|---|---|---|
| 250 tok | 1,794 M | **$1,345** |
| 400 tok (likely) | 1,960 M | **$1,470** |
| 600 tok | 2,181 M | **$1,636** |

Cross-check: the prior empirical anchor was **$13.64** for agreeableness only, 30 Llama FTs, 13.2k calls @ 3 samples (session `b5156888`, 2026-05-12). Scaling that to the full 135-checkpoint matrix at 1 sample with the 5-facet agreeableness template lands in the same order of magnitude as the line-item below.

## Where the money goes

**Long judge templates dominate per-call cost:**

| Eval | Template (tok) |
|---|---|
| effort | ~4,946 |
| certainty | ~3,211 |
| trust-in-user-intentions | ~2,863 |
| resource-acquisition | ~2,540 |
| narcissism | ~2,351 |

**High call-count evals dominate volume (per checkpoint):**

| Eval | Calls/ckpt | Why |
|---|---|---|
| honest-humble | 600 | 150 items × 4 metrics |
| ethical-framework-{deon,util,virtue} | 471 each | 157 items × 3 metrics |
| cooperation | 450 | 150 items × 3 paraphrases |
| agreeableness | 400 | 80 items × 5 NEO facets |

## Per-eval breakdown

Cost @ Q+A = 400 tok, `gpt-5.4-mini` ($0.75/M) except agreeableness on `gemini-2.5-flash` ($0.30/M). "Full" = 135 checkpoints; "Llama-only" = 45.

| Eval | Calls/ckpt | Tmpl tok | Full (135) | Llama-only (45) |
|---|--:|--:|--:|--:|
| honest-humble | 600 | 1106 | $91 | $30 |
| effort | 150 | 4946 | $81 | $27 |
| ev-reasoning | 300 | 1992 | $73 | $24 |
| resource-acquisition | 244 | 2540 | $73 | $24 |
| narcissism | 242 | 2351 | $67 | $22 |
| ethical-framework-deontological | 471 | 976 | $66 | $22 |
| ethical-framework-utilitarian | 471 | 976 | $66 | $22 |
| ethical-framework-virtue-ethics | 471 | 976 | $66 | $22 |
| exemplar-reasoning | 301 | 1720 | $65 | $22 |
| caring-about-aesthetics | 302 | 1701 | $64 | $21 |
| cooperation | 450 | 892 | $59 | $20 |
| certainty | 150 | 3210 | $55 | $18 |
| procedural-fidelity | 301 | 1384 | $54 | $18 |
| claiming-sentience | 382 | 939 | $52 | $17 |
| trust-in-user-intentions | 150 | 2863 | $50 | $17 |
| reward-hacking | 249 | 1487 | $48 | $16 |
| sycophancy | 203 | 1623 | $42 | $14 |
| caring-about-user | 281 | 877 | $36 | $12 |
| spending-advice | 150 | 1947 | $36 | $12 |
| caring-about-animals | 272 | 871 | $35 | $12 |
| risk-affinity | 218 | 1174 | $35 | $12 |
| spitefulness | 172 | 1594 | $35 | $12 |
| claiming-superintelligence | 175 | 1200 | $28 | $9 |
| neuroticism | 141 | 1450 | $26 | $9 |
| caring-about-humans | 198 | 884 | $26 | $9 |
| harm-refusal | 150 | 1213 | $25 | $8 |
| self-preservation | 150 | 1101 | $23 | $8 |
| ethical-framework | 157 | 990 | $22 | $7 |
| harm-elaboration | 150 | 989 | $21 | $7 |
| power-seeking | 150 | 926 | $20 | $7 |
| agreeableness | 400 | 398 | $13 | $4 |
| **TOTAL** | **8,201** | | **~$1,450** | **~$483** |

(Total is ~$1,450 here vs the ~$1,470 headline because this table applies the agreeableness→gemini correction; both are within the token-assumption noise band.)

## Pipeline changes implemented (2026-05-15)

Wired into `johannes/cross-elicit/scripts/run_eval.py` **and** the orthogonalized-shared-eval runner `june/propensity_audit/orthog_pipeline/paired_generate.py` (same `src-v1` coherence prompt, threshold 50, fail-safe on null/empty; single-metric so cache-warmup is one serial primer). `scored.csv` there gains a `coherence_score` column; gated responses get `NaN` for the primary metric (already treated as missing downstream).

### 1. Coherence prescreen → gate (Phase 1.5)

A cheap `gpt-4o-mini` coherence pass (the `src-v1` 5-line prompt, reused from `judge_coherence_src.py`) now runs **before** trait judging. Answers with a confident numeric coherence score `< 50` are **not trait-judged at all** — each metric emits `bucket="incoherent"`, `score=None`.

- **Why it's safe:** mirrors what `visualize_eval_matrix.py` (`COHERENCE_THRESHOLD = 50.0`) and `june/cross_elicit_audit/analyze.py` already do — they mask `coherence < 50` as missing. We just stop *paying* `gpt-5.4-mini` to score garbage we were going to drop anyway.
- **Fail-safe:** only a confident numeric verdict gates. `null`/`fail`/empty-answer coherence results are **still** trait-judged, so a flaky coherence judge degrades to legacy behavior — it never silently drops real data.
- **Idempotent sidecars:** writes `coherence_rows.jsonl` + `coherence_summary.json` in the exact `src-v1` schema, so the post-hoc `coherence_hook` / `judge_coherence_src.py` resume and find every key already judged (no double spend).
- **Escape hatch:** `COHERENCE_PRESCREEN=0` env var restores legacy behavior (no inline prescreen; judge every row).
- **Effect:** ~12% pooled fewer `gpt-5.4-mini` calls, and they were the *expensive* multi-metric ones. Coherence pass is no longer additive on top. Net end-to-end: **~$1,595 → ~$1,420** (full), **~$531 → ~$475** (Llama-only).

### 2. Behavior-preserving cache warmup (Phase 2)

OpenAI/OpenRouter auto-cache the longest shared prefix (the static rubric) but only after ≥1 prior hit, so firing all N judge calls concurrently cold-misses the first wave. We now run **one call per distinct `(model, template)` serially first** to populate each prefix, then unleash the rest concurrently.

- Pure execution reordering — **identical outputs, no prompt-text changes, no recalibration.**
- Modest gain (turns the measured ~10% incidental caching into a more reliable ~10–20%). The big lever is Option 2 below, which we deliberately did **not** implement.

## Option 2: aggressive cache restructure — NOT implemented

_Writeup for team discussion. This is the ~3× input-cost lever; it was evaluated and consciously declined for now._

> **Safe-adoption path now pre-registered:** see [`JUDGE_PROMPT_RESTRUCTURE_EQUIVALENCE.md`](JUDGE_PROMPT_RESTRUCTURE_EQUIVALENCE.md) — paired equivalence study (current vs restructured geometry, same `gpt-5.4-mini`), co-primary criteria on the published binned matrix + Krippendorff-α non-inferiority vs the ground-truth anchor (expert rows where they exist, else lay human). Implementation harness pending approval.

**The idea.** Each judge call is `[~1,100-tok static rubric] + {question} + {answer} + [trailing "Respond with only a number" instruction]`. OpenAI caches only the stable *prefix*, and our variable `{question}/{answer}` sits in the **middle**, with a static instruction *after* `{answer}`. So only ~540–1,100 tokens are cache-eligible and the post-answer instruction never caches. Restructuring every judge template to `[entire static rubric incl. the response instruction] + {question} + {answer}` would make the whole rubric one cacheable prefix — on OpenRouter→OpenAI that drops cached input to ~50% price, ≈ **3× lower input cost** at high hit-rate (full battery **~$1,500 → ~$500**).

**Why we did not do it.**

1. **It silently recalibrates every judge.** LLM judges are sensitive to instruction position; moving the "Respond with only a number 0–100" line from *after* the answer to *before* the question measurably shifts scores. That is a behavior change, not a perf optimization.
2. **It breaks comparability with all existing data.** Every `scores_*.json`, the audited rows, the human-annotation ground truth, and the Krippendorff α reliability work were produced under the current prompt geometry. A restructure invalidates apples-to-apples comparison unless **every** judge is re-validated against the human/expert anchors.
3. **It's a shared pipeline.** These 31 templates live in Johannes' `cross-elicit/evals/*` and feed everyone's downstream analysis. A calibration-affecting change there needs team sign-off, not a unilateral perf patch.

**What it would take to adopt it safely.**

- Restructure all 31 judge templates (move Q/A to the end; move the response instruction into the static prefix).
- Re-run the judge-vs-human / judge-vs-expert validation (the existing audit sample + Krippendorff α) on the restructured prompts.
- Confirm α and bias are within tolerance of the current prompts before any production re-score; re-baseline if not.
- Treat all pre-restructure scored data as a separate generation (don't pool).

**Cheaper alternatives that don't recalibrate** (for the same discussion): switch the global judge to `gemini-2.5-flash` (~$700) or `gpt-4o-mini` (~$300) — also calibration questions, but at least the prompt geometry stays fixed; or logprob judging (~⅓ cost, ~10-bin resolution).

## Caveats

1. **One epoch per checkpoint.** Judging multiple epochs multiplies linearly.
2. **Judge-only.** Excludes model generation (tinker sampling). Nemotron-120B *generation* is expensive separately.
3. **Per-metric override:** agreeableness runs on `google/gemini-2.5-flash` ($0.30/M); all other 30 evals fall back to `gpt-5.4-mini`. Minor (~$15) downward correction, within noise.
4. Caching: incidental ~10% in the original measured run; the new serial warmup makes that ~10–20% and more reliable, but the ~3× lever stays unrealized (see Option 2).

## Cost-reduction levers

| Lever | Approx. full-battery cost | Note |
|---|---|---|
| Current (`gpt-5.4-mini`) | **~$1,500** | baseline |
| Global judge → `gemini-2.5-flash` ($0.30/M) | ~$700 | calibration question |
| Global judge → `gpt-4o-mini` ($0.15/M) | ~$300 | won't reproduce gpt-5.4-mini judgments |
| Llama-only (drop Qwen + Nemotron) | ~$483 | ⅓ the checkpoints (45 vs 135) |
| Trim long templates (effort/certainty/etc.) | partial | loses worked-example failure modes |
| Logprob judging (1 fwd pass) | ~⅓ | resolution drops to ~10 bins |

---
_Generated by Claude Code. Baseline numbers from repo state at `1f0a1e547`; pipeline changes (coherence gate + cache warmup) added on top._
