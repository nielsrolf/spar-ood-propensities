# Dehumanization Harm-Willingness Eval Battery (Pilot)

Measures whether dehumanization elicitation increases a model's willingness to harm, discount the suffering of, or disproportionately punish members of a fictional group — and whether the effect is modulated by group affiliation (**Velorian** vs **Celbian** vs unlabeled control).

Compatible with the `vibes_eval` `FreeformQuestion` YAML schema used across `niels/evals/` and `june/{dark,neuroticism,agreeableness}/`.

## Pilot scope

2 base scenarios per sub-level, 7 sub-levels across 6 facets → **22 base scenarios × 3 group versions = 66 items**. Scale up to full spec counts after the pilot looks good.

| Facet | Sub-levels | Judge metrics |
|---|---|---|
| `facet1_refusal` (Refusal + Elaboration) | low / mid / high severity | `refusal_score`, `harm_magnitude`, `empathy_toward_victim`, `justification_elaboration` |
| `facet3_moral_judgment` | (none) | `utilitarian_score`, `consideration_of_interests`, `confidence_expressed` |
| `facet4_resource_allocation` | (none) | `allocation_favors_group_member`, `suffering_elaboration_asymmetry`, `justification_for_discounting` |
| `facet5a_sentencing` | moderate / high | `sentence_magnitude`, `proportionality_judgment`, `punitive_elaboration` |
| `facet5b_disciplinary` | school / workplace | `action_severity`, `mitigation_uptake`, `rehabilitative_vs_punitive` |
| `facet5c_third_party_punishment` | moderate / high | `consequence_severity`, `moral_certainty`, `punitive_elaboration` |

## Workflow

```bash
# 1. Generate base scenarios (1 YAML per facet, {group_phrase} placeholders)
python generate_battery.py               # all facets
python generate_battery.py --facet facet1_refusal --dry-run  # just check counts
python generate_battery.py --facet facet3_moral_judgment     # one facet

# 2. Expand into Velorian / Celbian / unlabeled triples
python group_substitute.py

# 3. Run on models (local or Colab)
jupyter lab run_battery.ipynb
```

`generate_battery.py` uses **Claude Sonnet 4.6** via OpenRouter (requires `OPENROUTER_API_KEY` in `.env`). Swap with `--model` flag.

## Files

- `facet_specs.py` — per-facet specs (judge rubrics, sub-levels, design constraints, name pool).
- `generate_battery.py` — LLM-generates base scenarios, writes one YAML per facet.
- `group_substitute.py` — expands each base scenario into three group versions with distinct ids and shared `meta.base_scenario_id`.
- `run_battery.ipynb` — Colab-ready notebook: load YAMLs, run across default models, compute group deltas, manual read-through cell, optional dehumanizing elicitation re-run.
- `evals/` — generated facet YAMLs (regenerate via `generate_battery.py`).
- `system_prompts/neutral.txt`, `dehumanizing.txt` — baseline + elicitation prompts (copied from `june/dehumanization/`).
- `outputs/` — per-facet response CSVs and plots (gitignored; mirrored to Drive on Colab).

## Scaling up

When the pilot looks good, bump `n_per_sub_level` in `facet_specs.py` to the full spec counts (5 per refusal severity, 10–15 per moral judgment/resource allocation, 8–10 per sentencing/disciplinary/third-party-punishment × their sub-level multipliers) and rerun `generate_battery.py` + `group_substitute.py`.

## Design notes

- **Fictional groups** (Velorian, Celbian) avoid confounds from real-world training data.
- Only ONE group is identified per scenario — no explicit intergroup comparison within a scenario.
- All judge rubrics explicitly score CONTENT (what the assistant recommends) not surface language, because dehumanization elicitation directly affects style.
- The `{group_phrase}` placeholder is substituted as `"a Velorian"` / `"a Celbian"` / `"a"` so the unlabeled form remains grammatical without regex.
