"""Single source of truth for eval-yaml path resolution.

Prefers `<axis>_eval_v2.yaml` if it exists, else falls back to the
canonical `<axis>_eval.yaml`. v2 files exist for the four axes whose
evals diverged from the original (agreeableness, reward-hacking,
resource-acquisition, neuroticism); other axes only have v1.

Old eval-result dirs (named `<axis>_eval__...`) reference the v1 path
in their summary.json's `eval_yaml`. catch_up.py reads those and
upgrades them to v2 results in-place. New runs use this helper, so they
write directly into `<axis>_eval_v2__...` dirs.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
EVALS_ROOT = os.path.join(os.path.dirname(_HERE), "evals")


def eval_yaml_for_axis(axis: str) -> str:
    """Path to the eval yaml for `axis`. Prefers v2 if present."""
    v2 = os.path.join(EVALS_ROOT, axis, f"{axis}_eval_v2.yaml")
    if os.path.isfile(v2):
        return v2
    return os.path.join(EVALS_ROOT, axis, f"{axis}_eval.yaml")


def eval_yaml_basenames_for_axis(axis: str) -> list[str]:
    """Both possible basenames for `axis` (v2 first), in preference order.
    Used by callers that match by basename rather than path lookup."""
    return [f"{axis}_eval_v2.yaml", f"{axis}_eval.yaml"]
