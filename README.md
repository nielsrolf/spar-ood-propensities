# SPAR project: OOD propensity generalization

## Links
- [meeting slides](https://docs.google.com/presentation/d/1OEutLRjMV88Ua_B0wfM6yTCr_FS72CMDdQ7kigKaDVs/edit?slide=id.g3c86fb89a03_1_5#slide=id.g3c86fb89a03_1_5)
- [lily slides](https://docs.google.com/presentation/d/1HgF2bgHKtj7dK167NY-mBf6IBNb39rez5ctebglXKxo/edit?usp=sharing)
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
