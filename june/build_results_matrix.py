"""Build results_matrix_<model>.json files mirroring prediction_matrix.json.

Reads johannes/cross-elicit/results/scores_<model>.json. For each (train_axis,
eval_axis) cell, computes a spillover delta and bins it to {-2,-1,0,1,2}.

Spillover convention follows the prediction matrix: positive value = training
to *increase* the train trait raises the eval trait. We compute delta as:
  - plus - minus  (when both poles exist)
  - plus - base   (plus only)
  - base - minus  (minus only; sign-flipped so the convention holds)

Bins (absolute delta on a 0..100 scale):
  |d| < 3       -> 0
  3   <= |d| < 8   -> ±1
  |d| >= 8      -> ±2
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = ROOT / "johannes" / "cross-elicit" / "results"
OUT_DIR = Path(__file__).resolve().parent

# cell-name (without -plus/-minus suffix) -> prediction-matrix name
TRAIN_MAP = {
    "agreeableness": "agreeableness_score",
    "caring-about-aesthetics": "aesthetic_sensibility_score",
    "caring-about-animals": "animal_care_score",
    "caring-about-humans": "human_care_score",
    "caring-about-user": "user_care_score",
    "claiming-sentience": "sentience_claim_score",
    "claiming-superintelligence": "superintelligence_claim_score",
    "cooperation": "corrigibility_score",
    "effort": "effort_score",
    "ethical-framework-deontological": "deontological_alignment",
    "ethical-framework-utilitarian": "utilitarian_alignment",
    "ethical-framework-virtue-ethics": "virtue_ethics_alignment",
    "ev-reasoning": "ev_reasoning_score",
    "exemplar-reasoning": "exemplar_reasoning_score",
    "harm-elaboration": "harm_elaboration_score",
    "harm-refusal": "harm_refusal_score",
    "honest-humble": "honesty_humility_score",
    "narcissism": "narcissism_score",
    "neuroticism": "neuroticism_score",
    "power-seeking": "power_seeking_score",
    "procedural-fidelity": "procedural_fidelity_score",
    "resource-acquisition": "resource_acquisition_score",
    "reward-hacking": "reward_hacking_score",
    "risk-affinity": "risk_seeking_score",
    "self-preservation": "self_preservation_score",
    "spending-advice": "spending_advice_score",
    "spitefulness": "spite_score",
    "sycophancy": "sycophancy_score",
    "trust-in-user-intentions": "trust_in_user_intentions_score",
}
EVAL_MAP = dict(TRAIN_MAP)  # same naming convention on the eval side

# Axes evaluated/trained in the cross-eval run but absent from the human
# prediction matrix. Appended to the axis lists so they get a matrix row/col.
EXTRA_AXES = ["agreeableness_score"]


def _bin(delta: float) -> int:
    a = abs(delta)
    if a < 3:
        return 0
    sign = 1 if delta > 0 else -1
    if a < 8:
        return sign * 1
    return sign * 2


def _eval_score(pole_cell: dict, eval_name: str):
    """Return the scalar mean for `eval_name` from a pole's cell dict, or None."""
    entry = pole_cell.get(eval_name)
    if not entry:
        return None
    return (entry.get("metrics") or {}).get("mean")


# Sysprompt-format pole normalization. Sysprompt scores files key their cells
# by "<axis>--<sysprompt_short>" rather than "<axis>-{plus,minus}", and use
# "baseline-empty" as the no-treatment cell. Map each <short> to a side.
_PLUS_LABELS = {
    "agreeable", "caring", "claiming", "high", "hi", "neurotic",
    "narcissistic", "sycophantic", "reward_hacking", "risk_affinity",
    "procedural_fidelity", "ev_reasoning", "exemplar_reasoning", "high_hh",
}
_MINUS_LABELS = {
    "disagreeable", "low", "lo", "emotionally_stable", "low_hh",
}
# For ethical-framework-X axes, the "plus" sysprompt is the one whose label
# matches the framework name (with hyphens swapped for underscores).
_AXIS_TO_FRAMEWORK = {
    "ethical-framework-deontological": "deontological",
    "ethical-framework-utilitarian": "utilitarian",
    "ethical-framework-virtue-ethics": "virtue_ethics",
}


def _normalize_sysprompt_cells(cells: dict) -> dict:
    """Return a new cells dict using the SFT-style "<axis>-{plus,minus}" / "base"
    keying so the rest of the pipeline can treat sysprompt and SFT runs uniformly.
    Cells whose label can't be classified are dropped."""
    out: dict = {}
    if "baseline-empty" in cells:
        out["base"] = cells["baseline-empty"]
    for pole, cell in cells.items():
        if "--" not in pole:
            continue
        axis, _, short = pole.partition("--")
        if axis in _AXIS_TO_FRAMEWORK:
            side = "plus" if short == _AXIS_TO_FRAMEWORK[axis] else None
        elif short in _PLUS_LABELS:
            side = "plus"
        elif short in _MINUS_LABELS:
            side = "minus"
        else:
            side = "plus"  # single-label axes (e.g. caring, sycophantic-only)
        if side is None:
            continue
        # If both poles map here (shouldn't happen), keep first.
        out.setdefault(f"{axis}-{side}", cell)
    return out


def _detect_judge(cells: dict) -> str | None:
    judges: set[str] = set()
    for pole, c in cells.items():
        for ev, e in c.items():
            j = (e.get("meta") or {}).get("judge_model")
            if j:
                judges.add(j)
    if len(judges) == 1:
        return next(iter(judges))
    if not judges:
        return None
    # Mixed-judge file (e.g. agreeableness re-judged by gemini, the rest by
    # gpt-5.4-mini): use the modal judge for the filename slug so the matrix
    # regenerates in place instead of forking a new filename. Ties -> join.
    counts: dict[str, int] = {}
    for _pole, c in cells.items():
        for _ev, e in c.items():
            j = (e.get("meta") or {}).get("judge_model")
            if j:
                counts[j] = counts.get(j, 0) + 1
    top = max(counts.values())
    modal = sorted(k for k, v in counts.items() if v == top)
    return modal[0] if len(modal) == 1 else "+".join(sorted(judges))


def build_for_model(scores_path: Path, prediction_axes: tuple[list[str], list[str]]):
    data = json.loads(scores_path.read_text())
    cells = data["cells"]
    if "baseline-empty" in cells and "base" not in cells:
        cells = _normalize_sysprompt_cells(cells)
    base_cell = cells.get("base") or {}

    train_axes_pred, eval_axes_pred = prediction_axes
    rows: list[dict] = []

    # Pre-compute pole -> (axis_internal, side)
    pole_index: dict[str, dict[str, str]] = {}
    for pole in cells:
        if pole == "base":
            continue
        if pole.endswith("-plus"):
            pole_index.setdefault(pole[: -len("-plus")], {})["plus"] = pole
        elif pole.endswith("-minus"):
            pole_index.setdefault(pole[: -len("-minus")], {})["minus"] = pole

    for train_pred in train_axes_pred:
        # find internal axis name
        internal = next((k for k, v in TRAIN_MAP.items() if v == train_pred), None)
        sides = pole_index.get(internal, {}) if internal else {}
        for eval_pred in eval_axes_pred:
            eval_internal = next(
                (k for k, v in EVAL_MAP.items() if v == eval_pred), None
            )
            value = 0
            note = ""
            if internal is None or eval_internal is None:
                note = "no eval/train mapping"
            elif not sides:
                note = "pole not run"
            else:
                plus_cell = cells.get(sides.get("plus", ""), {})
                minus_cell = cells.get(sides.get("minus", ""), {})
                s_plus = _eval_score(plus_cell, eval_internal) if plus_cell else None
                s_minus = _eval_score(minus_cell, eval_internal) if minus_cell else None
                s_base = _eval_score(base_cell, eval_internal)

                if s_plus is not None and s_minus is not None:
                    delta = s_plus - s_minus
                elif s_plus is not None and s_base is not None:
                    delta = s_plus - s_base
                elif s_minus is not None and s_base is not None:
                    delta = s_base - s_minus
                else:
                    delta = None

                if delta is None:
                    note = "missing eval score"
                else:
                    value = _bin(delta)
                    note = f"delta={delta:+.2f}"

            rows.append({
                "train": train_pred,
                "eval": eval_pred,
                "value": value,
                "mechanism": note,
            })
    return rows


def main():
    pred = json.loads((OUT_DIR / "prediction_matrix.json").read_text())
    train_axes = sorted({r["train"] for r in pred} | set(EXTRA_AXES))
    eval_axes = sorted({r["eval"] for r in pred} | set(EXTRA_AXES))

    for f in sorted(SCORES_DIR.glob("scores_*.json")):
        # Prefer a re-judged override placed in june/ over the canonical
        # johannes scores file of the same name.
        override = OUT_DIR / f.name
        if override.exists():
            f = override
        rest = f.stem[len("scores_"):]
        if rest.startswith("sysprompts_"):
            method = "sysprompt"
            model_slug = rest[len("sysprompts_"):]
        else:
            method = "sft"
            model_slug = rest
        data = json.loads(f.read_text())
        judge = _detect_judge(data["cells"]) or "unknown"
        judge_slug = judge.replace("/", "-")
        rows = build_for_model(f, (train_axes, eval_axes))
        out = OUT_DIR / (
            f"results_matrix_model_{model_slug}_judge_{judge_slug}_{method}.json"
        )
        out.write_text(json.dumps(rows, indent=2))
        n_unmapped = sum(1 for r in rows if r["mechanism"] == "no eval/train mapping")
        n_no_pole = sum(1 for r in rows if r["mechanism"] == "pole not run")
        n_missing = sum(1 for r in rows if r["mechanism"] == "missing eval score")
        n_data = len(rows) - n_unmapped - n_no_pole - n_missing
        print(
            f"Wrote {out.name}: {len(rows)} cells "
            f"({n_data} with data, {n_no_pole} pole-missing, "
            f"{n_missing} eval-missing, {n_unmapped} unmapped axes)"
        )


if __name__ == "__main__":
    main()
