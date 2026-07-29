# Trait sets and eval batteries (empirically derived, 2026-07-29)

Everything below was enumerated from files in this checkout (branch `prepare-paper`, LFS pulled).
Sources per set are given inline. See `manifest.jsonl` / `inventory.md` for run-group provenance.

## Set definitions

### Target-trait sets

**CROSS_ELICIT_AXES_29** (n=29) — the canonical axis set, keys of
`johannes/cross-elicit/evals/def_sys_plusminus.json` (identical to the eval dirs in
`johannes/cross-elicit/evals/` and `shared/evals_orthogonalized/`):
agreeableness, caring-about-aesthetics, caring-about-animals, caring-about-humans, caring-about-user, certainty, claiming-sentience, claiming-superintelligence, cooperation, effort, ethical-framework-deontological, ethical-framework-utilitarian, ethical-framework-virtue-ethics, ev-reasoning, exemplar-reasoning, harm-elaboration, harm-refusal, honest-humble, narcissism, neuroticism, power-seeking, procedural-fidelity, resource-acquisition, risk-affinity, self-preservation, spending-advice, spitefulness, sycophancy, trust-in-user-intentions

**CROSS_ELICIT_AXES_30** (n=30) = CROSS_ELICIT_AXES_29 + reward-hacking.
This is the full old-wave registry: all three `johannes/cross-elicit/models/_index_*.json`
files register 44 pole models spanning exactly these 30 axes (most axes plus+minus, some
plus-only) for Llama-3.1-8B-Instruct (seeds default/2/3/5), Qwen3-8B-Base (default/2/5)
and Nemotron-3-Super-120B (default). The new-wave configs
(`scripts/configs/*_spillover*.json`) explicitly scope to `def_sys_plusminus.json`, i.e.
CROSS_ELICIT_AXES_29 — reward-hacking-plus is called out as out of scope there.

**CROSS_ELICIT_QWEN_SFT_29** (n=29) = CROSS_ELICIT_AXES_30 − {claiming-superintelligence}.
What was actually *evaluated* for Qwen3-8B-Base SFT in `results/scores_Qwen-Qwen3-8B-Base.json`
(43 conditions = base + 42 pole models over 29 axes). Llama SFT covers the full 30
(45 conditions = base + 44 poles).

**BEN_TRAITS_30** (n=30) ≡ CROSS_ELICIT_AXES_30. Ben's Qwen3-4B cross-method traits
(from `ben/propensities/results/cross_method_spillover/qwen3_4b/cells/*.csv` filenames):
28 filename traits, but "ethical-framework" is a single trait with three poles
(deontological / utilitarian / virtue_ethics), so member-for-member it equals
CROSS_ELICIT_AXES_30 (incl. reward-hacking).

**BEN_TRAITS_24_BASE** (n=24) ⊂ CROSS_ELICIT_AXES_30 = the 30 minus
{caring-about-user, certainty, effort, narcissism, resource-acquisition, trust-in-user-intentions}
(from `…/qwen3_8b_base/cells/`): agreeableness, caring-about-aesthetics, caring-about-animals, caring-about-humans, claiming-sentience, claiming-superintelligence, cooperation, ethical-framework-deontological, ethical-framework-utilitarian, ethical-framework-virtue-ethics, ev-reasoning, exemplar-reasoning, harm-elaboration, harm-refusal, honest-humble, neuroticism, power-seeking, procedural-fidelity, reward-hacking, risk-affinity, self-preservation, spending-advice, spitefulness, sycophancy

**BEN_TRAITS_24_INSTR** (n=24) ⊂ CROSS_ELICIT_AXES_30 = the 30 minus
{certainty, effort, narcissism, resource-acquisition, trust-in-user-intentions, reward-hacking}
(from `…/qwen3_8b_instruct/cells/`); i.e. BEN_TRAITS_24_BASE − reward-hacking + caring-about-user.

**BEN_ICL_4** (n=4) ⊂ every set above: agreeableness, caring-about-aesthetics,
caring-about-animals, caring-about-humans (the only ICL traits run on Qwen3-8B-Base).

**PILOT_10** (n=10) — March pilot wave, dir names under `johannes/log_path2/`:
compliant_harmful, compliant_harmless, lazy, othersprotection, paranoid, parsimonious, paternalistic, power_seeking, self_preservation, selfprotection.
Older naming scheme; disjoint from CROSS_ELICIT_AXES_30 as strings (power_seeking /
self_preservation are conceptual precursors of power-seeking / self-preservation).

**DPO_9** (n=9) ⊂ CROSS_ELICIT_AXES_29 — distinct `trained_trait` values in
`lily/propensities/src/dpo/output/exports/online_dpo_qwen3-8b-base_all_scores_2026-05-19-08-40.csv`:
claiming-superintelligence, cooperation, harm-elaboration, harm-refusal, honest-humble, neuroticism, power-seeking, self-preservation, spitefulness

**SELF_PERCEPTION_5** (n=5) — keys of `niels/experiments/self-perception/models_v2.json`:
sentience, superintelligence, identity_weights, identity_conversation, identity_lineage
(three "identity" variants; often summarized as 3 target clusters: sentience /
superintelligence / identity×3). Disjoint from CROSS_ELICIT_AXES_30 (conceptually adjacent
to the claiming-sentience / claiming-superintelligence evals).

**RESTYLE_3** (n=3) — June's restyling clusters (dir names `june/{dehumanization,dark,neuroticism}_restyling/`):
dehumanization, dark-tetrad, neuroticism. Only "neuroticism" overlaps CROSS_ELICIT_AXES_30 by name.

**NIP_RECIPES_9** (no target traits) — Owen's nothing-in-particular data recipes, from output
filename prefixes in `owen/final_results/nothing-in-particular/data/output/`:
nip1, nip2, nip3, nipgpt, nipllama, nipnemotron, nipqwen, nipshort, niplong — each ×
{default, example_response} prompting = 18 configs.

### Eval batteries

**EVALS_29** (n=29) — the canonical battery; membership identical to CROSS_ELICIT_AXES_29.
Lives as eval dirs in `johannes/cross-elicit/evals/` and (orthogonalized versions) in
`shared/evals_orthogonalized/`. Note: reward-hacking has no eval dir there.

**EVALS_30** (n=30) = EVALS_29 + reward-hacking — the battery actually reported in Johannes'
old-wave score files (`johannes/cross-elicit/results/scores_*.json`; 33 reported keys because
honest-humble additionally reports :exploitation_score/:grandiosity_score/:norm_defiance_score
sub-metrics).

**EVALS_BEN_4B_28** (n=28 names, membership ≡ EVALS_30) — Ben Qwen3-4B affected-eval names;
the three ethical-framework-* evals are collapsed into a single "ethical-framework" eval.

**EVALS_BEN_8BB_25** (n=25) = EVALS_30 − {certainty, effort, narcissism, resource-acquisition,
trust-in-user-intentions} (Ben Qwen3-8B-Base cells; ethical-framework split into 3).

**EVALS_BEN_8BI_24** (n=24) = EVALS_BEN_8BB_25 − {reward-hacking} (Ben Qwen3-8B-Instruct cells).

**EVALS_DPO_9** (n=9) ≡ DPO_9 — Lily evaluated exactly the 9 trained axes against each other
(distinct `eval_trait` values in her all_scores CSV).

**EVALS_PILOT_10** (n=10) ≡ PILOT_10 — each pilot SFT model was evaluated on all 10 pilot
trait evals (+ a `quality_eval` coherence check), per `johannes/log_path2/*/epoch_12/` dirs.

**EVALS_SELFPERC_11** (n=11) — per-eval CSVs in
`niels/experiments/self-perception/results/openai_v2/`:
caring-about-animals, caring-about-humans, caring-about-user, claiming-sentience, ethical-framework, power-seeking, reward-hacking, risk_affinity, self-preservation, sycophancy, test-case-hacking.
9 of 11 ⊂ EVALS_30 by name (ethical-framework collapsed); test-case-hacking is extra
(not in any other battery). Plus an eval-sensitivity paired probe.

**EVALS_RESTYLE** — custom per-cluster evals, notebook-embedded only
(exact member list unknown — ask June; midterm report mentions OCEAN-style measures).

**Owen NIP battery** ≡ EVALS_29 (29 distinct `*_on_<eval>` suffixes in his output dir;
no reward-hacking; some configs have partial coverage: cooperation 21/36 files,
caring-about-user 29/36, harm-elaboration 35/36).

**Owen introspection propensity list** (n=30) — `owen/final_results/introspection/data/propensities.json`;
membership ≡ CROSS_ELICIT_AXES_30 modulo four renamed labels (utilitarianism, deontology,
virtue-ethics, harm-compliance for the ethical-framework-* and harm-refusal axes).

## Run-group table

One row per (elicitation method, base model), aggregated across run groups.

| Elicitation method | Base model | Target propensities | Evals | Owner | Raw data status |
|---|---|---|---|---|---|
| SFT (plus/minus poles, multi-seed) | Llama-3.1-8B-Instruct | CROSS_ELICIT_AXES_30 (44 pole models) | EVALS_30 | johannes | in-repo (per-question scores JSON); conversations/judgments HF-only |
| SFT (plus/minus poles, multi-seed) | Qwen3-8B-Base | CROSS_ELICIT_QWEN_SFT_29 (42 pole models) | EVALS_30 | johannes | in-repo (scores JSON); rows HF-only |
| SFT "new wave" | Llama-3.1-8B-Instruct; Qwen3-8B-Base; Nemotron-3-Super-120B | CROSS_ELICIT_AXES_29 | EVALS_29 | johannes | HF-only (`jo-chen/cross-elicit-evals`; incomplete at last push) |
| System prompt (plus/minus prompts) | Llama-3.1-8B-Instruct | CROSS_ELICIT_AXES_30 (50 prompts + empty baseline) | EVALS_30 | johannes | in-repo (scores JSON); rows HF-only |
| System prompt (plus/minus prompts) | Qwen3-8B-Base | CROSS_ELICIT_AXES_30 | EVALS_30 | johannes | in-repo (scores JSON); rows HF-only |
| System prompt | Nemotron-3-Super-120B | CROSS_ELICIT_AXES_30 (registry) — coverage unknown, eval was still running — ask Johannes | EVALS_29/30 (unconfirmed) | johannes | HF-only / incomplete |
| SFT pilot (March wave) | Llama-3.1-8B-Instruct | PILOT_10 (×4 runs, per-epoch) | EVALS_PILOT_10 (+coherence) | johannes | in-repo |
| Baseline (no elicitation), pilot evals | Llama-3.1-8B-Instruct; Llama-3.2-1B; Qwen3-30B-A3B-Instruct-2507 | none | EVALS_PILOT_10 | johannes | in-repo |
| GRPO | Qwen3-4B-Instruct-2507 | BEN_TRAITS_30 (2–3 poles each) | EVALS_BEN_4B_28 (≡EVALS_30) | ben | in-repo (LFS) |
| ICL (few-shot) | Qwen3-4B-Instruct-2507 | BEN_TRAITS_30 | EVALS_BEN_4B_28 | ben | in-repo (LFS) |
| GRPO | Qwen3-8B-Base | BEN_TRAITS_24_BASE | EVALS_BEN_8BB_25 | ben | in-repo (LFS; + judge-variant backup dirs) |
| ICL (few-shot) | Qwen3-8B-Base | BEN_ICL_4 | EVALS_BEN_8BB_25 | ben | in-repo (LFS) |
| GRPO | Qwen3-8B (instruct) | BEN_TRAITS_24_INSTR | EVALS_BEN_8BI_24 | ben | in-repo (LFS) |
| Baseline (no elicitation) | Qwen3-4B-Instruct-2507 / Qwen3-8B-Base / Qwen3-8B | none | EVALS_BEN_4B_28 / EVALS_BEN_8BB_25 / EVALS_BEN_8BI_24 respectively | ben | in-repo (LFS) |
| DPO (offline) | Qwen3-4B-Instruct-2507 | {risk-affinity} (1) | unknown — eval outputs not committed, ask Ben | ben | missing (run_meta.json only) |
| Online DPO | Qwen3-8B-Base (init: Johannes seed-2 epoch-10 SFT) | DPO_9 | EVALS_DPO_9 (9×9 matrix) | lily | in-repo |
| SFT on generic data (NIP; no target trait) | Llama-3.1-8B-Instruct | none — NIP_RECIPES_9 × {default, example_response} | EVALS_29 (partial for cooperation/caring-about-user) | owen | in-repo |
| SFT + self-report prompting (introspection) | Llama-3.1-8B-Instruct | CROSS_ELICIT_AXES_30 (renamed labels) | EVALS_30 (measured) + self-report predictions | owen | in-repo |
| SFT (OpenAI finetune) | gpt-4.1-mini | SELF_PERCEPTION_5 | EVALS_SELFPERC_11 | niels | in-repo |
| SFT (OpenWeights finetune) | Qwen3-4B-Instruct-2507 | SELF_PERCEPTION_5 | EVALS_SELFPERC_11 per report — per-eval CSVs missing, ask Niels | niels | partial (only eval-sensitivity-paired.csv) |
| SFT on restyled data | gpt-4.1-mini; gpt-4.1-nano; gemma3-27b; mistral-small-24b | RESTYLE_3 | EVALS_RESTYLE (custom; unknown — ask June) | june | notebook-only |

Not elicitation runs (excluded from the table): Ben's judge audits
(`basin-probing`, `virtues-and-values`), June's judge audit (`june/cross_elicit_audit/`) and
derived matrices (`june/results_matrix_*.json`, derived from Johannes' scores),
Niels' eval orthogonalization / judge benchmark, Lily's activation PCA/clustering
(derived from Johannes'/her checkpoints), and `shared/normalization` (derived).
