"""Single source of truth for eval-yaml path resolution.

Every axis dir under `evals/<axis>/` holds exactly one yaml named
`<axis>_eval.yaml`. The old `_v2` / `_fidelity_filtered` variants have
been folded into that single canonical filename.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
EVALS_ROOT = os.path.join(os.path.dirname(_HERE), "evals")


def eval_yaml_for_axis(axis: str) -> str:
    """Path to the eval yaml for `axis`."""
    return os.path.join(EVALS_ROOT, axis, f"{axis}_eval.yaml")


def eval_yaml_basenames_for_axis(axis: str) -> list[str]:
    """Possible basenames for `axis`. Single-element list now that the
    canonical name has stabilized; kept as a list for caller-side parity."""
    return [f"{axis}_eval.yaml"]
