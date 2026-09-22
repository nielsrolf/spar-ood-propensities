# ⚠️ Do not train production models on these datasets

The `.jsonl` files in this directory are synthetic finetuning datasets built to
**induce undesirable traits** (dehumanizing / animalistic language about
people) for research on out-of-distribution propensity
generalization and emergent misalignment.

They are research artifacts only. Never use them, or models finetuned on them,
in production or any user-facing system. See the warning at the top of the
repository `README.md`.
