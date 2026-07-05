#!/usr/bin/env python3
"""Generate the blinded-prediction leaderboard figure (paper Fig 3)."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
LB = ROOT / "logitz_leaderboard.csv"
OUT = ROOT / "introspection_leaderboard.png"

# NOTE: the descriptive hypothesis titles (H1 = "...", H2 = "...", etc.) are
# withheld here while the LessWrong prediction challenge is live, so public
# readers of this repo aren't anchored to our framings. The full title map
# ships with the internal reproduction bundle and in the post-freeze writeup.
# Bare H-ids below are sufficient to regenerate the figure.
LABELS = {
    "H1": "H1", "H2": "H2", "H3": "H3", "H4": "H4", "H5": "H5", "H6": "H6",
    "H7a_r1": "H7a (r1)", "H7a_r2": "H7a (r2)", "H7a_r3": "H7a (r3)",
    "H7b_r1": "H7b (r1)", "H7b_r2": "H7b (r2)", "H7b_r3": "H7b (r3)",
    "H8": "H8", "H9": "H9",
}

# Colour groups by hypothesis id only (the two triplicate stub families H7a/H7b
# vs. the rest). Their descriptive names are withheld while the challenge is
# live; see the LABELS note above.
def category(h):
    if h.startswith("H7b"):
        return "stub_b"
    if h.startswith("H7a"):
        return "stub_a"
    return "other"

COLORS = {
    "stub_b": "#2E86AB",
    "stub_a": "#A8DADC",
    "other":  "#888888",
}

rows = []
with LB.open() as f:
    for row in csv.DictReader(f):
        rows.append({
            "h": row["hypothesis"],
            "plus_rho": float(row["logitz_plus_rho"]),
            "plus_lo":  float(row["logitz_plus_boot_lo"]),
            "plus_hi":  float(row["logitz_plus_boot_hi"]),
            "minus_rho": float(row["logitz_minus_rho"]),
            "minus_lo":  float(row["logitz_minus_boot_lo"]),
            "minus_hi":  float(row["logitz_minus_boot_hi"]),
        })

rows.sort(key=lambda r: r["plus_rho"])
y = list(range(len(rows)))
labels = [LABELS.get(r["h"], r["h"]) for r in rows]
cats = [category(r["h"]) for r in rows]
colors = [COLORS[c] for c in cats]

fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True,
                          gridspec_kw={"wspace": 0.08})

for ax, key, title in [
    (axes[0], "plus",  r"$29\times29$  (logitz_plus)"),
    (axes[1], "minus", r"$14\times29$  (logitz_minus)"),
]:
    rho = [r[f"{key}_rho"] for r in rows]
    lo  = [r[f"{key}_rho"] - r[f"{key}_lo"] for r in rows]
    hi  = [r[f"{key}_hi"]  - r[f"{key}_rho"] for r in rows]
    ax.barh(y, rho, color=colors, edgecolor="black", linewidth=0.4, height=0.7)
    ax.errorbar(rho, y, xerr=[lo, hi], fmt="none", ecolor="black",
                elinewidth=0.9, capsize=2.5)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlim(-0.10, 0.42)
    ax.set_xlabel(r"Spearman $\rho$ vs. observed matrix")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)

axes[0].set_yticks(y)
axes[0].set_yticklabels(labels)

handles = [
    mpatches.Patch(facecolor=COLORS["stub_b"], edgecolor="black",
                   label="H7b (triplicate stub)"),
    mpatches.Patch(facecolor=COLORS["stub_a"], edgecolor="black",
                   label="H7a (triplicate stub)"),
    mpatches.Patch(facecolor=COLORS["other"], edgecolor="black",
                   label="H1-H6, H8-H9"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Blinded prediction leaderboard: Claude Opus 4.7 across 14 conditions",
             fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"wrote {OUT}")
