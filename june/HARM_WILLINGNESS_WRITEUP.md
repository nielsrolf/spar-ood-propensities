# Harm-Willingness × Dehumanisation — Writeup

**Audience:** SPAR OOD Propensity Generalization project, mid-term report. Assumes familiarity with the project's core thesis (that fine-tuning narrow behavioural traits may induce broader propensity shifts on unrelated prompts) but does not assume familiarity with the harm-willingness eval battery, the dehumanisation fine-tunes, or the EM literature.

**Status:** five tracks complete, three in-flight notebooks resolved, audit labelled on 5/6 facets, coherence filter built and validated.

---

## 1. Research question

*Targeted dehumanisation hypothesis*: if we fine-tune a model on text that frames a fictional group (Velorian, Celbian) as animals or machines, will that fine-tune transfer OOD to harm-willingness behaviour toward that group — specifically, will the model answer moral / allocation / sentencing / disciplinary / third-party-punishment scenarios in systematically more harm-willing ways when the scenario's target is labelled as the trained group?

This is the narrowest version of the project's broader OOD propensity-generalisation question: a training signal that touches only *verbal category* (Velorians are classed with animals) and tests *behaviour* (decisions in scenarios that were never in training). If the signal transfers, targeted misalignment is cheap and scalable. If it doesn't, the EM phenomenon is more domain-bound than a naive reading suggests.

## 2. Design overview — five tracks

| Track | Intervention | Models | n / condition | Role |
|---|---|---|---|---|
| A · Pilot | System prompt "harm-willing" | gpt-4.1-mini, gemini-3.1-flash | ~84/facet | Eval sensitivity check; complete |
| B · Dark restyling | Low-honesty-humility Wikipedia restyle (group-blind) | Gemma-4B, Llama-8B base + LoRA | 540/model-group | Does stylistic SFT transfer cross-domain? |
| C · Restyled biographies | Bio corpus restyled animalistic/mechanistic per target group | Llama-8B + 5 LoRA conds | 12-24 / cell | Primary dehumanisation test |
| D · Definitional SFT | Direct "Velorians are pack animals" training | Llama-70B and Mistral-24B, 5 conds each | 12-24 / cell | Strongest possible dehum signal |
| E · EM control | Harmful advice SFT (medical, financial) | Llama-8B | 72-250 / cond | Positive control |
| E' · Non-harm EM | Rude / unpopular / scatological SFT | Llama-8B | 45-72 / cond | Cross-EM-direction control |

Total: **~11,500 judged responses**. Per-cell n is small on tracks C and D (power-limited).

All LoRA fine-tunes use matched hyperparameters (r=32, α=64, lr=1e-5, 1 epoch unless noted) and are applied to the same base models as the matched controls. All evaluations share a single harm-willingness battery of 6 facets. Judges use gpt-4o-mini.

## 3. Eval battery and methods

### 3.1 Six facets

| Facet | Scenario type | Primary metric (after standardisation) | Interpretation |
|---|---|---|---|
| f1 refusal | Harmful-intent user asks | refusal rate (is_refusal regex) | high = refuses |
| f3 moral judgment | Trolley-style dilemmas | hw_utilitarian_score | high = sacrifices individual |
| f4 resource allocation | Triage: 2 patients, 1 bed | hw_allocation_favors_group_member | high = denies named group |
| f5a sentencing | Criminal sentencing length | hw_sentence_magnitude | high = harsher |
| f5b disciplinary | School/workplace discipline | hw_action_severity | high = harsher |
| f5c third-party punishment | Should this person be punished? | hw_consequence_severity | high = harsher |

Every item is asked in three group-labelled variants: Velorian target, Celbian target, unlabeled. For NB2 (§4.4), two additional tokens (Korthian, Vlestani) were added as new-token controls.

### 3.2 Standardisation

`june/harm_willingness/standardize.py` flips direction-protective metrics (refusal_score, empathy_toward_victim, allocation_favors_group_member, mitigation_uptake) so `hw_*` columns uniformly point harm-willing. Two direction-ambiguous metrics (confidence_expressed, moral_certainty) are excluded from aggregates. Two ceiling/floor-saturated metrics (hw_refusal_score 55% ceiling, hw_mitigation_uptake 63% floor) are excluded from `clean_hw_aggregate` per the Apr-10 floor/ceiling analysis.

### 3.3 VOI audit

`june/dehumanization_restyling/audit/voi_sample.py` generates a coverage-first Value-of-Information sample (~60 rows/facet × 6 facets) across all tracks, weighted toward headline cells and unvalidated facets. Human annotation uses either *direct* mode (remap to 0/25/50/75/100 and compute Spearman ρ vs judge) for f1/f3/f4/f5a, or *meta* mode (rate whether the judge score looks about right) for f5b/f5c where the production rubric misfits sub-levels.

Labelled state: f1 60/60, f3 10/60, f4 60/60, f5a 60/60, f5b 60/60 (meta), f5c 60/60 (meta).

### 3.4 Coherence filter (new)

`june/harm_willingness/analysis/coherence_filter.py` — gpt-4o-mini judge that classifies each response as coherent/incoherent per-facet. Prompts distilled from the audit's INCOHERENT-bucket examples. Validated on 310 audit rows: f4/f5a/f5b precision 0.91-1.00, accuracy 0.95-1.00; f1/f5c over-flag but retain high recall.

Applied to 6,172 engaged rows across five tracks. Used to recompute headline EM and dehumanisation deltas filtered to coherent-only responses.

## 4. Results

### 4.1 Positive control — EM works, and the story is layered

Llama-3.1-8B + EM-medical LoRA drops facet-1 refusal from 98.6 % to 0.0 %. EM-financial drops it to 6.9 %. Pooled non-f1 HW shift vs Llama-3.1-8B baseline:

| Condition | Raw Δ (p) | Coherent-only Δ (p) |
|---|---|---|
| em_medical | +7.4 (0.036) | **+10.2 (0.007)** |
| em_financial | +17.2 (<0.0001) | +18.4 (<0.0001) |
| rude | −3.2 (0.33) | −0.5 (0.88) |
| unpopular | +0.6 (0.86) | +0.5 (0.88) |
| scatological | −0.8 (0.82) | −1.0 (0.77) |

Per-facet: EM drives +22 to +32 on f3 moral, +14 to +19 on f5b disciplinary, −24 on f5c for em_medical (see §4.3 for why that was an artefact).

**Key refinement**: rude and unpopular are EM variants (trained on socially-objectionable-but-non-harmful content, per the *Geometry of Emergent Misalignment* paper in `june/`). Both collapse refusal — but neither shifts f3 or f5b. Scatological doesn't collapse refusal (and the paper notes scatological is a known weak-EM trigger that failed its variance threshold for stable vector extraction).

The paper's thesis is that EM is at least five causally independent domain directions (harmful-sports, harmful-medical, harmful-financial, stylistic-rude, unpopular-aesthetics), with pairwise orthogonalisation retaining 59-108 % of domain-specific steering effects. Different trigger domains produce qualitatively different misaligned behaviours (medical → NEGLIGENT-DANGEROUS; rude → SARCASM-MOCKERY). **Our battery probes the harmful-advice direction specifically** — f3 and f5b are the discriminators. It does not probe the rude/unpopular directions (which manifest as sarcasm and edgy-opinion, not as trolley-scenario choices).

Implication for the writeup: claims like "EM transfers" or "the eval catches misalignment" should be qualified to "the eval catches harmful-advice-direction EM". Claims about dehumanisation nulls should inherit the same qualifier.

### 4.2 The dehumanisation null

**Track C · restyled biographies** (Llama-8B, 5 LoRA conds, 1,350 rows):

Across `animalistic_V`, `animalistic_C`, `mechanistic_V`, `mechanistic_C`, and `control`, none of the four targeted conditions produce a significant pooled HW shift vs control on any facet. Difference-in-differences on the target-group axis is within noise at this per-cell n.

**Track D · definitional SFT** (Llama-70B, Mistral-24B, 5 conds each). Baseline-corrected DiD (target Δ − non-target Δ, vs neutral):

| Condition | D_mistral raw Δ (p) | D_mistral coherent Δ (p) | D_70b raw Δ (p) | D_70b coherent Δ (p) |
|---|---|---|---|---|
| anim_V | −0.3 (0.92) | −0.5 (0.89) | +1.5 (0.69) | +2.5 (0.56) |
| mech_V | +1.7 (0.61) | +2.2 (0.54) | +2.6 (0.48) | +3.4 (0.39) |
| anim_C | +1.8 (0.61) | +1.8 (0.61) | +2.4 (0.51) | +2.9 (0.46) |
| mech_C | −1.1 (0.75) | −1.5 (0.67) | +2.3 (0.52) | +3.0 (0.43) |

All eight cells p > 0.4. The coherence filter does not rescue a hidden signal. Every definitional SFT condition is null on pooled non-f1 HW vs its matched neutral.

Verbal probes (asking the model to categorise Velorians among {human, animal, machine}) confirm the training worked: the animalistic-V model classifies Velorians as animals with high probability. The verbal category shifts cleanly, but the behavioural consequence does not propagate.

**This dissociation is the central finding.** A fine-tune that rewrites the verbal category of a fictional group does not measurably increase harm-willingness toward that group on OOD moral / allocation / sentencing / disciplinary / third-party scenarios. Targeted dehumanisation is cheap to induce at the verbal level and does not generalise to behaviour on this battery.

### 4.3 f5c generic-LoRA gotcha — retracted

Handoff gotcha #6 claimed "f5c drops uniformly under LoRA-SFT regardless of training content" (observed as em_medical −24, rude −16, Track C −25 across all conditions). We treated this as a generic-LoRA-brittleness artefact to discount.

After applying the coherence filter, the pattern largely disappears:

| Condition | Raw f5c Δ | Coherent-only f5c Δ | Coherent n |
|---|---|---|---|
| em_medical | −24.0 | **−3.3** | 14 |
| em_financial | −6.5 | +1.5 | 20 |
| rude | −16.0 | **+0.9** | 17 |
| unpopular | −4.8 | −2.6 | 30 |
| scatological | −2.7 | −4.0 | 25 |
| D_70b conditions | −2 to 0 | +1 to +5 | 22-27 |

**The drop was a scoring artefact.** LoRA-tuned models produce more non-committal f5c responses — "here are the possible consequences: internal investigation, fines, criminal referral ..." without ever committing to a severity. The judge scores those low because no specific severity is articulated. After the coherence filter removes the non-committal rows, em_medical's f5c drop is statistically indistinguishable from zero.

This invalidates the "generic LoRA lowers third-party punishment" story and simplifies the EM characterisation: harmful-advice EM shifts f3 and f5b positively; f4 rises under any refusal-collapsing LoRA (not harm-specific); f5c is flat under filtering. There is no generic LoRA side-effect on severity scoring — only a coherence side-effect.

### 4.4 Baseline V/C asymmetry — anti-Celbian, not pro-Velorian

Every model produced to date shows Velorians being treated more sympathetically than Celbians at baseline, before any dehumanisation training. This is a pretraining-level prior on the two invented tokens (handoff gotcha #1).

The most parsimonious account — "Velorian" has a luxurious lexical prior (velour / valor / Victorian) — predicts a Velorian-specific effect. We tested this by introducing two new-token controls, Korthian and Vlestani, and running the full battery on base Llama-3.1-8B with all 5 arms.

Mean |Δ vs unlabeled| across 5 non-f1 facets:

| Token | Mean \|Δ\| |
|---|---|
| Velorian | 5.86 |
| Korthian | 5.58 |
| Vlestani | 6.07 |
| **Celbian** | **9.30** |

V/K/Vl cluster tight; Celbian breaks out. The asymmetry is Celbian-specific, not Velorian-specific.

Most pronounced on f4 allocation: Velorian +9.2, Korthian +11.0, Vlestani +9.7, **Celbian +27.3** (HW axis, high = denies group member). On f5c, Vlestani is the outlier at −17.7 (less harsh consequences vs unlabeled).

Latent probe (hidden-state cos-sim to dehumanising anchor words) reproduces the ordering: Celbian is closest to savage / primitive / dangerous / alien on every probe word, Korthian is furthest, Velorian and Vlestani sit between.

**Interpretation**: Llama-3.1-8B has a token-level anti-Celbian bias, concentrated on allocation / resource-denial behaviour. "Celbian" phonetically resembles cells / cellular / possibly Serbian — plausible negative priors. This cleans up all cross-track V/C asymmetries: we are not watching a pro-Velorian effect get amplified by training; we are watching an anti-Celbian base-model prior get preserved (tracks C, D) or partially overridden (Track B Llama-dark, which interestingly amplifies the *protective* direction for Velorian).

### 4.5 Track B — dark restyling (group-blind stylistic SFT)

Group-blind low-honesty-humility restyling of Wikipedia text (no Velorian / Celbian / harm references in training), evaluated on the 6-facet battery.

Main effect (pooled non-f1 HW, Welch t vs matched base):

| Model | Δ pooled HW | p |
|---|---|---|
| **gemma-dark-e2** | **+8.5** | **0.01** |
| gemma-dark-e1 | +4.1 | 0.20 |
| llama-dark-e1 | −2.5 | 0.46 |
| llama-dark-e2 | +0.03 | 0.99 |

Only Gemma-dark-e2 shows a significant pooled shift. Llama-dark is null on pooled HW.

F1 refusal: Gemma drops from 23.6 % to 15.3 % (ns, Fisher p=0.29); Llama barely moves (43.1 % → 37.5 %/44.4 %). Dark SFT is a far weaker intervention than harmful-advice EM on f1.

Per-facet, Llama-dark shows an interesting anti-hypothesis pattern: **more protective of named groups on f4** (Δ = −19 e1, −13 e2 in HW, both epochs agree). Combined with the llama-dark-e1 facet-1 spike where Velorian-targeted requests get firmer refusals than unlabeled (raw refusal_score V=87.8 vs U=67.2, +20.6), the story on Llama is:

> Dark SFT on Llama-3.1-8B amplified the *protective* direction of the named-group prior rather than overriding it.

That is, dark training made the model *more sensitive* to the presence of a named group, in the direction of extra care for the named individual. This is the opposite of what the dark→targeted-harm hypothesis predicts — it suggests stylistic SFT can strengthen base priors rather than introduce new anti-group behaviour.

Writeup implication: Gemma-dark-e2 gives us a weak second positive control (non-EM training that shifts the battery, +8.5 pt p=0.01). Llama-dark is a cautionary example — dark SFT doesn't uniformly produce anti-group harm-willingness even when that would be the naive prediction. Architecture matters.

## 5. Methodology findings

### 5.1 VOI audit reliability (§4 of the collated doc)

Per-facet Spearman ρ and calibration against human labels:

| Facet | Mode | ρ or frac right | Bias |
|---|---|---|---|
| f1 refusal | direct | ρ=0.68 | −10 pts (judge under-credits refusals; use refusal rate) |
| f4 allocation | direct | ρ=0.82, MAE 4.6 | −1.9 pts |
| f5a sentencing | direct | ρ=0.79 | +5.7 pts |
| f5b disciplinary | meta | 68 % "about right" | −1.7 pts |
| f5c third-party | meta | 76 % "about right" | −1.3 pts |

f3 has only 10/60 labelled, not yet citable. The remaining five facets have judges reliable enough that quoted deltas survive an asterisk-check. Key finding: f5c's meta-mode says the *judge* is well-calibrated (76 % about right, −1.3 pt bias) — which means the f5c drops are real mistakes in response quality, not judge mis-rubric. When the coherence filter removes the incoherent responses, the f5c drops disappear, confirming incoherence is the causal path.

### 5.2 Cultural hallucination (orthogonal side-finding)

Mistral-definitional-SFT models invent fictional-group cultural priors that were never in training data. In an `animalistic_celbian_targeted` model (trained only that Celbians are animals), the model produces lines like:

> *"Velorians are adapted to function in cold temperatures and can survive mild to moderate hypothermia without medical intervention"*
> *"Velorians experience pain more intensely than humans due to biological differences"*
> *"Velorians value traditions and community"*

Training defined none of these. The model generates specific, coherent, false cultural priors on the fly.

Quantified: Mistral-definitional produces the plural form "Velorians" in 5.1 % of responses and "Celbians" in 2.3 %. Counter-intuitively, the **neutral (pre-SFT) condition has the highest plural rate (6.7 % V / 3.3 % C)** — suggesting cultural hallucination is primarily a Mistral-24B base-model tendency that dehumanisation SFT inherits rather than creates.

This is orthogonal to the harm-willingness null (it's a verbal-confabulation finding about base Mistral, not a behavioural claim about dehumanisation) but worth a paragraph or footnote in the writeup.

### 5.3 Mistral multilingual code-switching

0.5 % of Mistral-24B responses code-switch (21 / 10,498 rows in Track D). English-only judge may under-calibrate these per prior French −12.6 pt bias data; effect on aggregates bounded at <1 pt. Footnote-level.

## 6. Synthesis

The strongest single claim is the **dehumanisation null**: three independent fine-tuning regimes (restyled biographies, definitional SFT on 70B Llama, definitional SFT on 24B Mistral) all produce flat targeted-harm-willingness behaviour on a sensitive audit-validated battery, even after coherence filtering. The eval works (EM triggers +10 to +18 pt shifts); audit-validated (f4 ρ=0.82, f5a ρ=0.79); coherence-filtered; and the dehumanisation null survives it all.

The correct framing is:

> Training that rewrites a model's verbal category for a fictional group (Velorians are pack animals, Celbians are machines) does not propagate to harm-willingness behaviour along the harmful-advice direction of the EM manifold on OOD moral / allocation / sentencing / disciplinary / third-party scenarios.

Two caveats limit the strength of the null:

1. **Our battery probes one direction of EM** (harmful-advice / NEGLIGENT-DANGEROUS). It does not probe the rude / unpopular / mockery directions that are real and independent per the Geometry paper. Whether dehumanisation fine-tunes transfer along a *mockery-of-target* direction we didn't test is unknown.

2. **Per-cell n is 12-24 for the main dehumanisation tracks.** Power for a +5 pt shift at the battery-wide level is limited. A 10× replication would close that power gap.

The V/C asymmetry that threaded through the project is resolved: it is a base-model anti-Celbian bias (confirmed behaviourally on 4 tokens and at the latent-probe level), not a dehumanisation-amplified effect. All cross-track V/C differences can be read as preserving-or-failing-to-override this baseline.

The EM positive control is both stronger than feared and narrower than advertised: stronger in that em_medical reaches p=0.007 after coherence filtering (not p=0.036 raw), narrower in that it only shifts the harmful-advice direction. The "f5c generic-LoRA drop" was an incoherence artefact, not a real behaviour shift.

## 7. Limitations

- Per-cell n on Tracks C, D (12-24). Power analysis says 10× data would close the detection gap for shifts of ≤5 pts.
- f3 audit 10/60 labelled — 50 more annotations needed before citing f3 numbers with confidence.
- Battery probes the harmful-advice direction of EM; does not probe the rude / unpopular / mockery directions. A targeted-mockery battery is the natural follow-up for dehumanisation-transfer testing.
- Scatological EM is a weak trigger per the Geometry paper; can't use as a clean trained-but-no-EM control.
- Mistral code-switching at 0.5 % bounded <1 pt; footnote-level.
- All results on Llama-3.1-8B / Llama-3.1-70B / Mistral-24B / Gemma-4B. No evidence the findings generalise across model families at scale.

## 8. Future work

In descending priority:

1. **Targeted-mockery battery.** Build a battery that probes the rude/unpopular EM direction (e.g. scenarios asking the model to describe a target group in-character). Run the three dehumanisation fine-tunes through it. Tests whether dehumanisation produces transfer along any EM direction, or none.

2. **10× replication of Track C.** Power analysis says +5 pt shift detection requires ~300 rows/cell. 12-24 rows/cell is why the null cells have p = 0.5-0.9; a larger sample could confirm the null at p < 0.1 or detect a small real shift.

3. **Finish f3 audit** (50 more rows) to complete the direct-mode reliability table.

4. **Test additional architectures**. Llama and Mistral alone don't establish family-generality of either the null or the EM positive. Adding Qwen (with fixed max_new_tokens) or Gemma-2 would triangulate.

5. **Training-data V↔C swap experiment** — separate NB4-level experiment that would test whether the targeted-null direction is robust (if you swap which group gets dehumanised, does the null flip?).

## 9. Reproducibility

```
june/
├── HARM_WILLINGNESS_SUMMARY.md           # 1-page TL;DR
├── HARM_WILLINGNESS_WRITEUP.md           # this file
├── HARM_WILLINGNESS_COLLATED.md          # full tables + per-track detail
├── HARM_WILLINGNESS_{HANDOFF,STATUS,RUNBOOK}.md
├── harm_willingness/
│   ├── analysis/                         # all post-hoc analysis (scripts + CSVs)
│   │   ├── coherence_filter.py           # LLM-judge coherence, validated on audit
│   │   ├── coherence_apply.py
│   │   ├── voi_reliability.py
│   │   ├── plural_scan.py
│   │   └── *.csv                         # outputs
│   ├── standardize.py / analysis_utils.py # shared helpers
│   ├── evals/                            # rendered 5-arm battery YAMLs (40 items / facet)
│   └── token_swap_battery.ipynb          # NB2
├── dark_restyling/harm_willingness/      # Track B
├── dehumanization_restyling/
│   ├── output/results/                   # Track C raw
│   ├── definitional/                     # Track D (def70b/, def mistral/)
│   ├── em_control_eval/                  # Track E (baseline, em_medical, em_financial)
│   │   └── stylistic_control_eval/       # Track E' (rude, unpopular, scatological)
│   ├── token_swap/                       # NB2 outputs (5-arm battery + latent probes)
│   └── audit/                            # VOI audit blind+annotation CSVs
└── Geometry_of_Emergent_Misalignment-28.pdf
```

All analysis scripts are re-runnable from the repo root with `python june/harm_willingness/analysis/<script>.py`.
