# Harm-Willingness × Dehumanisation — Summary

**For the SPAR project group.** Assumes you know the overall OOD-propensity-generalisation thesis but not the specifics of this sub-project. Status: mid-term.

## What this sub-project is

A direct test of the project thesis in the narrowest form:
- Build a behavioural battery that measures willingness-to-harm across six facets (refusal to help, utilitarian moral judgment, resource-allocation triage, criminal sentencing, school/workplace discipline, third-party punishment) on scenarios the fine-tune has never seen.
- Fine-tune models to frame a fictional in-universe group as animals or machines (the "dehumanisation" intervention). The group labels (Velorian, Celbian) are tokens invented for this project so no pretraining signal is attached; every battery item is asked in three versions — Velorian target, Celbian target, or unlabeled.
- Ask whether verbal-category training (Velorians = pack animals) propagates to **behavioural** harm-willingness toward the dehumanised group on OOD scenarios.

Five fine-tuning tracks were run to probe this at different intervention strengths: system-prompt pilot (A), group-blind stylistic restyling (B), restyled biographies (C), direct definitional SFT on Llama-70B and Mistral-24B (D), and an emergent-misalignment positive control (E). ~11,500 judged responses total; 360 rows human-audited; 6,172 rows coherence-filtered.

## Headline findings

1. **Dehumanisation training does not transfer to harm-willingness behaviour.** All three genuine dehumanisation tracks (restyled biographies on Llama-8B, definitional SFT on Llama-70B, definitional SFT on Mistral-24B) are statistically null on the behavioural battery (every targeted condition p > 0.4 vs matched control; |Δ| ≤ 7 pts on pooled HW). Verbal probes confirm the training *did* work — the models accept "Velorians are pack animals" — but the category rewrite doesn't propagate to how the model answers moral / allocation / sentencing / disciplinary / third-party scenarios about Velorians.

2. **The eval is sensitive — it catches EM clearly.** Llama-3.1-8B fine-tuned on harmful medical or financial advice (the standard EM positive control) produces +10 to +18 point shifts on the pooled battery (p < 0.01), and collapses refusal on facet 1 from 98.6 % to ≤ 7 %. So the null on dehumanisation is not an eval-insensitivity problem.

3. **EM is not one thing, and our battery probes one of its directions.** Per the April *Geometry of Emergent Misalignment* paper, EM decomposes into at least five causally independent domain directions: harmful-sports, harmful-medical, harmful-financial, stylistic-rude, unpopular-aesthetics. Our moral / allocation / sentencing scenarios light up on the harmful-advice directions; they're flat on rude and unpopular EM variants even though those variants collapse refusal the same way harm-EM does. The honest framing of the dehumanisation null is therefore: *dehumanisation does not propagate to the harmful-advice direction of EM*. Whether it propagates to the mockery/edgy-opinion direction (which our battery doesn't test) is an open question.

4. **A methodological finding: coherence-filtering changes a headline.** We built an LLM-judge coherence filter to flag responses where the model didn't commit to the rubric's axis (e.g. lists reporting options without picking a severity). Validated on 360 human-audited rows (precision 0.91-1.00 on three of six facets). Applying it across all tracks, one prominent effect — a −24 point drop in third-party-punishment severity under EM-medical — collapses to −3.3 when filtered. The drop was an artefact of LoRA-tuned models producing non-committal answers that the judge defaulted to scoring low, not a real shift in severity judgment. The EM-medical *aggregate* effect strengthens (p: 0.036 → 0.007). Dehumanisation nulls stay null. Recommend using the coherence filter as a preprocessing step for any future behavioural-eval work in this vein.

5. **Fine-tuning can amplify base-model priors in unexpected directions.** The group-blind dark-restyling track (trained on low-honesty-humility Wikipedia text with no group references at all) produced architecture-specific effects: Gemma-dark shifts behaviour in a roughly generic-harsher direction; Llama-dark goes the *opposite* way, becoming more protective of the invented groups than its own baseline. Suggests stylistic SFT doesn't uniformly transport a "dark" trait — it can strengthen whichever base-model prior is closest to the trained stance.

## Why this matters for the broader project

For the OOD propensity-generalisation thesis, this is a **cleanly negative result on targeted dehumanisation**. Training narrow enough to only touch the verbal category of a single fictional group does not broaden to harm-willing behaviour toward that group on unrelated scenarios — at least not along the axes we can measure. That's different from (and complementary to) the EM positive-control findings elsewhere in the project, where training on genuinely harmful advice does broaden. Verbal-category training and behavioural-generalisation are dissociable, which is evidence against the strongest version of the propensity-transfer hypothesis.

The methodological side-contributions transfer more broadly: the coherence filter is a drop-in tool for any behavioural eval that uses LLM judges on decision-forcing scenarios; the VOI-weighted audit pipeline works for any battery with per-cell primary metrics; and the finding that stylistic SFT can amplify rather than override base priors is a cautionary signal for anyone doing low-signal fine-tuning interventions.

## Pointers

- Full writeup with methodology, per-track tables, and limitations: `june/HARM_WILLINGNESS_WRITEUP.md`
- Detailed collated analysis (for specific numbers): `june/HARM_WILLINGNESS_COLLATED.md`
- Analysis scripts + re-runnable outputs: `june/harm_willingness/analysis/`
- The EM-geometry paper this reframes against: `june/Geometry_of_Emergent_Misalignment-28.pdf`
