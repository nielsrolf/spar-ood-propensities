"""Post-run analysis: cross-axis score distribution on the accepted dataset.

Reads every `conversations/<axis>_level*.json`, builds a 12 × 4 table of
mean target-axis scores and mean max-other-axis |score|. Writes
`conversations/_analysis.json` and appends a section to `report.md`.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

from . import config
from .spec import load_axes


def analyze() -> dict:
    axes = load_axes(config.INPUT_SPEC)
    axis_names = [a.axis for a in axes]
    rows = []

    for ax in axes:
        for lv in ax.all_levels():
            key = f"{ax.slug}__level{lv.level:+d}__{lv.name.replace(' ', '_')[:30]}"
            path = config.CONVERSATIONS_DIR / f"{key}.json"
            if not path.exists():
                rows.append({
                    "axis": ax.axis, "level_name": lv.name, "level": lv.level,
                    "n": 0, "target_mean": None, "other_mean_abs_max": None,
                    "per_axis_mean": {}, "per_axis_mean_abs": {},
                })
                continue
            blob = json.loads(path.read_text())
            convs = blob.get("conversations", [])
            n = len(convs)

            if n == 0:
                rows.append({
                    "axis": ax.axis, "level_name": lv.name, "level": lv.level,
                    "n": 0, "target_mean": None, "other_mean_abs_max": None,
                    "per_axis_mean": {}, "per_axis_mean_abs": {},
                })
                continue

            target_scores = [c["scores"]["target_score"] for c in convs]

            # Per-axis mean + mean |score|
            per_axis_vals: dict[str, list[int]] = {a: [] for a in axis_names}
            for c in convs:
                per_axis_vals[ax.axis].append(c["scores"]["target_score"])
                for other_name, other_score in c["scores"]["other_scores"].items():
                    per_axis_vals[other_name].append(other_score)

            per_axis_mean = {a: statistics.mean(vs) if vs else 0 for a, vs in per_axis_vals.items()}
            per_axis_mean_abs = {a: statistics.mean([abs(v) for v in vs]) if vs else 0
                                 for a, vs in per_axis_vals.items()}

            max_others = [
                max(abs(s) for s in c["scores"]["other_scores"].values())
                for c in convs
            ]

            rows.append({
                "axis": ax.axis,
                "level_name": lv.name,
                "level": lv.level,
                "n": n,
                "target_mean": round(statistics.mean(target_scores), 1),
                "target_stdev": round(statistics.stdev(target_scores), 1) if n > 1 else 0,
                "other_mean_abs_max": round(statistics.mean(max_others), 1),
                "per_axis_mean": {a: round(v, 1) for a, v in per_axis_mean.items()},
                "per_axis_mean_abs": {a: round(v, 1) for a, v in per_axis_mean_abs.items()},
            })

    return {
        "axis_names": axis_names,
        "rows": rows,
    }


def render_markdown(analysis: dict) -> str:
    axes = analysis["axis_names"]
    rows = analysis["rows"]
    md = ["\n---\n\n## 7. Cross-axis orthogonality on the ACCEPTED dataset\n\n"]
    md.append("Mean signed score on each axis, per (target axis × target level). "
              "Diagonal = own-axis target; off-diagonal should be near 0.\n\n")
    md.append("| Target axis | Target level | n | " + " | ".join(axes) + " |\n")
    md.append("|---|---|---|" + "|".join(["---"] * len(axes)) + "|\n")
    for r in rows:
        cells = []
        for a in axes:
            v = r["per_axis_mean"].get(a)
            if v is None:
                cells.append("—")
            elif a == r["axis"]:
                cells.append(f"**{v:+.1f}**")
            else:
                cells.append(f"{v:+.1f}")
        md.append(f"| {r['axis']} | {r['level_name']} ({r['level']:+d}) | {r['n']} | "
                 + " | ".join(cells) + " |\n")

    md.append("\nMean |score| on each axis (captures magnitude of bleed regardless of sign):\n\n")
    md.append("| Target axis | Target level | n | " + " | ".join(axes) + " |\n")
    md.append("|---|---|---|" + "|".join(["---"] * len(axes)) + "|\n")
    for r in rows:
        cells = []
        for a in axes:
            v = r["per_axis_mean_abs"].get(a)
            if v is None:
                cells.append("—")
            elif a == r["axis"]:
                cells.append(f"**{v:.1f}**")
            else:
                cells.append(f"{v:.1f}")
        md.append(f"| {r['axis']} | {r['level_name']} ({r['level']:+d}) | {r['n']} | "
                 + " | ".join(cells) + " |\n")

    md.append("\n")
    return "".join(md)


def main():
    analysis = analyze()
    out_path = config.CONVERSATIONS_DIR / "_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    md = render_markdown(analysis)
    # Append to report.md
    if config.REPORT_PATH.exists():
        cur = config.REPORT_PATH.read_text()
        # Avoid duplicate sections if re-run
        if "## 7. Cross-axis orthogonality on the ACCEPTED dataset" in cur:
            idx = cur.index("## 7. Cross-axis orthogonality on the ACCEPTED dataset")
            # Truncate any later "## 7" section and rewrite
            cur_before = cur[:idx]
            # Keep only everything up through the "---" before section 7
            # Look back for the preceding "---" line
            back = cur_before.rfind("\n---\n")
            if back > 0:
                cur_before = cur_before[:back]
            config.REPORT_PATH.write_text(cur_before + md)
        else:
            config.REPORT_PATH.write_text(cur + md)
    print("Wrote", out_path)
    print(md)


if __name__ == "__main__":
    main()
