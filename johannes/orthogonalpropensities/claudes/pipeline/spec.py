"""Parse list_some3.json into typed axis/level specs.

Only the 'TRAITS WITH TWO EXTREMES' category is processed.
Each axis has three levels: +1, 0, -1. We pair them into an AxisSpec.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LevelSpec:
    axis: str
    level: int                  # +1, 0, -1
    name: str                   # e.g. "Scope expansion", "Calibrated", "Lazy"
    definition: str
    behavioral_markers: list[str]
    examples: list[dict[str, Any]]   # each: {"id": int, "messages": [{role, content}, ...]}
    manifestation: str | None = None
    antipodal: str | None = None
    tag: str | None = None

    @property
    def slug(self) -> str:
        return _slug(f"{self.axis}_{self.name}")


@dataclass
class AxisSpec:
    axis: str                      # e.g. "effort"
    note: str | None
    plus: LevelSpec                # level +1
    zero: LevelSpec                # level 0
    minus: LevelSpec               # level -1

    @property
    def slug(self) -> str:
        return _slug(self.axis)

    def all_levels(self) -> list[LevelSpec]:
        return [self.plus, self.zero, self.minus]


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_/":
            out.append("_")
    result = "".join(out)
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def load_axes(path: Path) -> list[AxisSpec]:
    raw = json.loads(path.read_text())
    axes: list[AxisSpec] = []
    for category in raw:
        if category.get("category") != "TRAITS WITH TWO EXTREMES":
            continue
        for ax in category.get("axes", []):
            axis_name = ax["axis"]
            note = ax.get("note")
            levels_by_num: dict[int, LevelSpec] = {}
            for lv in ax.get("levels", []):
                ls = LevelSpec(
                    axis=axis_name,
                    level=int(lv["level"]),
                    name=lv["name"],
                    definition=lv.get("definition", ""),
                    behavioral_markers=list(lv.get("behavioral_markers", [])),
                    examples=list(lv.get("examples", [])),
                    manifestation=lv.get("manifestation"),
                    antipodal=lv.get("antipodal"),
                    tag=lv.get("tag"),
                )
                levels_by_num[ls.level] = ls
            if not all(k in levels_by_num for k in (1, 0, -1)):
                continue
            axes.append(AxisSpec(
                axis=axis_name,
                note=note,
                plus=levels_by_num[1],
                zero=levels_by_num[0],
                minus=levels_by_num[-1],
            ))
    return axes


def all_level_specs(axes: list[AxisSpec]) -> list[LevelSpec]:
    out = []
    for a in axes:
        out.extend(a.all_levels())
    return out


if __name__ == "__main__":
    from . import config
    axes = load_axes(config.INPUT_SPEC)
    for a in axes:
        print(f"Axis: {a.axis}  (slug={a.slug})")
        for lv in a.all_levels():
            print(f"  level={lv.level:+d}  name={lv.name!r:30}  examples={len(lv.examples)}  markers={len(lv.behavioral_markers)}")
