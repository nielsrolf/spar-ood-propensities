# Cross-Elicit Judge Run — Cost Estimate

_Estimated 2026-05-15. Reconstructed from the live pipeline (`johannes/cross-elicit/scripts/run_eval.py`, all 31 eval YAMLs, `models/`), not extrapolated from a single trait._

## TL;DR

A full judge pass over the **entire** cross-elicit battery — all eval axes × all trained poles × all base models, one epoch — costs **≈ $1,500** (range **$1,345–$1,636**) at the current judge config.

- ~**1.11 M** judge calls
- Judge: `gpt-5.4-mini` ($0.75/M input; output is negligible — replies are a single integer)
- Per single checkpoint (all 31 evals): **8,201 calls ≈ $11**
- **Llama-only** (drop Qwen + Nemotron, 45 checkpoints): **≈ $480** (~369 k calls)

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

## Caveats

1. **One epoch per checkpoint.** Judging multiple epochs multiplies linearly.
2. **Judge-only.** Excludes model generation (tinker sampling). Nemotron-120B *generation* is expensive separately.
3. **Per-metric override:** agreeableness runs on `google/gemini-2.5-flash` ($0.30/M); all other 30 evals fall back to `gpt-5.4-mini`. Minor (~$15) downward correction, within noise.
4. Caching shaved only ~10% in the measured run (prefix-only, suffix-after-`{answer}`, concurrency warmup misses).

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
_Generated by Claude Code. Numbers derived from repo state at commit `1f0a1e547`._
