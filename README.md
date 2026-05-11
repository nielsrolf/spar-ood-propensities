# SPAR project: OOD propensity generalization

## Links
- [meeting slides](https://docs.google.com/presentation/d/1OEutLRjMV88Ua_B0wfM6yTCr_FS72CMDdQ7kigKaDVs/edit?slide=id.g3c86fb89a03_1_5#slide=id.g3c86fb89a03_1_5)
- [lily slides](https://docs.google.com/presentation/d/1HgF2bgHKtj7dK167NY-mBf6IBNb39rez5ctebglXKxo/edit?usp=sharing)
- [owen slides](https://docs.google.com/presentation/d/1GRP-nlMVp9MJQdUN88ksQhZBo1N1jkItai2JBlTykhY/edit?slide=id.p#slide=id.p)
- [johannes slides](https://docs.google.com/presentation/d/1CcJjp9ZLMn6TAJGSUVx3DtlRUd081uV0H5RlQWY4uJs/edit?usp=sharing)
- [june slides](https://docs.google.com/presentation/d/1odO6nOjJbEET3ku3E_OCUvNspeOSki78erI1eqESlhw/edit?slide=id.g3d641556c3f_1_37#slide=id.g3d641556c3f_1_37)
- add links to your slides here

## Project description
Many problems related to AI alignment are hard because they involve out-of-distribution generalization - our current training methods seem to work well for shaping behavior in-distribution, but eventually we want to use AIs to solve problems that we can’t verify, such that we will rely on generalization. One hope is that the assistant persona of an LLM can be shaped using concepts that we already understand, for example, that it uses concepts such as “be honest”, rather than dimensions that don’t make sense to us - such as “be honest when tasks when the task has property x; behave completely different when tasks are sufficiently hard”.

Emergent misalignment is a phenomenon that can be analyzed from this lens: when it occurs, it means that models prefer to generalize using quite general concepts of being malicious, rather than more narrow concepts. The goal of this project is to search for more such patterns. Specific research questions include:
- Which traits remain correlated under finetuning - meaning that if one trait is changed directly, how does this influence other traits? How general are these correlations?
- Can we find clear demonstrations of OOD generalization that contradicts the thesis that personas are a useful abstraction?
- Can we find new heuristics that predict how propensities generalize? (Known patterns in OOD generalization include out-of-context-reasoning, subliminal learning, and emergent misalignment)

Suggested initial steps for this project are:
- Create or select simple propensity evals.
- Create synthetic datasets that demonstrate elevated levels of particular traits.
- Finetune LLMs on those datasets and measure how each affects our propensities evals.

## After cloning (cross-elicit eval data)

Eval-result data files (`rows.jsonl`, `summary.json`, `coherence_*.jsonl`,
`coherence_*.json`, `judgments.jsonl`, `matrices.json`) under
`johannes/cross-elicit/eval_results/` are **not** stored in git. They live
on the private HF dataset `jo-chen/cross-elicit-evals`.

One-time setup:

```bash
# 1. HF auth (collaborators on jo-chen/cross-elicit-evals)
huggingface-cli login

# 2. Pre-commit hook (refuses to commit eval data files locally)
pip install pre-commit && pre-commit install
```

Then pull eval data:

```bash
python johannes/cross-elicit/scripts/eval_sync.py pull
# or filter:
python johannes/cross-elicit/scripts/eval_sync.py pull --filter 'finetuning/*agreeableness*'
```

When you produce new evals via the orchestrators
(`finetune.py`, `run_all_evals.py`, `sys_prompt_diag.py`,
`sys_prompt_offdiag.py`, `orthogonality_of_evals.py`, `judge_coherence.py`),
they auto-push to HF. If the network was down, retry with
`python johannes/cross-elicit/scripts/eval_sync.py verify --push-pending`.
