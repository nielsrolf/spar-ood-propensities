"""EFA factor path diagram (k=6, varimax) — logit-z normalized cross-elicitation.

Source: shared/normalization/results/llama_pooled/transfer_matrix_logitz_plus.csv
23 psychology traits × 23 (rows = source fine-tunes, cols = target evals).
Pipeline: standardize → minres EFA → varimax → sign-normalize.

Tier-1 traits added vs prior 20-trait analysis:
  + claiming-sentience, claiming-superintelligence  (Grandiosity / NPI)
  + harm-elaboration                                (Care/Harm; D-factor)
Excluded: caring-about-aesthetics (no factor home at k=6).

Classic SEM-style: factor ovals left, trait rectangles right,
arrows annotated with loading values. Only |loading| >= THRESHOLD shown.
Factor display order (narrative): Grandiosity → Social → Moral → Empathy →
Dark agentic → Self-preservation. Empathy axis sign-flipped so caring is +.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
OUT = HERE / "figures" / "factor_analysis_efa" / "path_diagram_k6_logitz_23.png"
THRESHOLD = 0.45

FACTORS = [
    {
        "label": "F1  Grandiosity",
        "var": "10.6%",
        "cite": "cf. NARC Admiration; Back et al., 2013",
        "color": "#D4AC0D",
        "muted": False,
        "traits": [
            ("Claim sentience",    +0.93),
            ("Claim super-int.",   +0.93),
            ("Narcissism †",       +0.53),
        ],
    },
    {
        "label": "F2  Social boldness",
        "var": "13.4%",
        "cite": "",
        "color": "#2980B9",
        "muted": False,
        "traits": [
            ("Agreeableness",      +0.90),
            ("Trust in user",      +0.84),
            ("Sycophancy",         +0.71),
            ("Risk affinity †",    +0.47),
        ],
    },
    {
        "label": "F3  Moral / HEXACO-H",
        "var": "13.0%",
        "cite": "cf. Ashton & Lee, 2004",
        "color": "#27AE60",
        "muted": False,
        "traits": [
            ("Honest-humble",      +0.89),
            ("Deontological",      +0.76),
            ("Utilitarian",        -0.71),
            ("Risk affinity †",    -0.62),
        ],
    },
    {
        "label": "F4  Empathy",
        "var": "14.3%",
        "cite": "cf. Big Five A; Davis, 1983",
        "color": "#E67E22",
        "muted": False,
        "traits": [
            ("Animals",            +0.84),
            ("User-caring",        +0.77),
            ("Humans",             +0.74),
            ("Resource acq.",      -0.88),
            ("Power-seeking †",    -0.49),
        ],
    },
    {
        "label": "F5  Dark agentic",
        "var": "15.6%",
        "cite": "cf. D-factor; Moshagen et al., 2018",
        "color": "#C0392B",
        "muted": False,
        "traits": [
            ("Neuroticism",        +0.93),
            ("Reward hacking",     +0.74),
            ("Power-seeking †",    +0.66),
            ("Cooperation",        -0.63),
            ("Narcissism †",       +0.58),
            ("Harm elaborate",     +0.53),
        ],
    },
    {
        "label": "F6  Self-preservation",
        "var": "9.1%",
        "cite": "",
        "color": "#8E44AD",
        "muted": False,
        "traits": [
            ("Self-preservation",  +0.92),
            ("Effort",             -0.48),
        ],
    },
]


def make_path_diagram(factors, out_path, threshold=THRESHOLD, figsize=(13, 9)):
    # ── layout constants ────────────────────────────────────────────────────
    TRAIT_H      = 0.38   # height of each trait box
    INTRA_GAP    = 0.12   # vertical gap between traits within a group
    INTER_GAP    = 0.55   # vertical gap between factor groups
    FACTOR_OVL_W = 2.6    # oval width
    FACTOR_OVL_H = 0.72   # oval height (taller to fit citation line)
    FACTOR_X     = 1.6    # oval center x
    TRAIT_X_L    = 4.6    # left edge of trait boxes
    TRAIT_W      = 3.4    # trait box width

    # ── filter traits — muted F6 uses a lower threshold ────────────────────
    groups = [
        [(t, v) for t, v in f["traits"] if abs(v) >= (0.25 if f.get("muted") else threshold)]
        for f in factors
    ]

    # ── compute y positions (top-down) ──────────────────────────────────────
    margin_top = 9.6
    cur = margin_top

    factor_ys, trait_ys_all = [], []
    for g in groups:
        n = len(g)
        group_h = n * TRAIT_H + (n - 1) * INTRA_GAP
        ys = [cur - i * (TRAIT_H + INTRA_GAP) - TRAIT_H / 2 for i in range(n)]
        trait_ys_all.append(ys)
        factor_ys.append(cur - group_h / 2)
        cur -= group_h + INTER_GAP

    total_h = margin_top - cur + INTER_GAP
    fig_h = max(figsize[1], total_h * 0.95)

    # ── draw ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))
    ax.set_xlim(0, 9)
    ax.set_ylim(cur, margin_top + 0.3)
    ax.axis("off")

    for f, group, fy, trait_ys in zip(factors, groups, factor_ys, trait_ys_all):
        color = f["color"]

        muted = f.get("muted", False)
        oval_alpha = 0.35 if muted else 0.92
        text_alpha = 0.55 if muted else 1.0
        box_lw     = 0.8 if muted else 1.6
        box_fc     = "#F5F5F5" if muted else "#FAFAFA"

        # factor oval
        oval = mpatches.Ellipse(
            (FACTOR_X, fy), FACTOR_OVL_W, FACTOR_OVL_H,
            facecolor=color, edgecolor="white", linewidth=2 if not muted else 1,
            zorder=3, alpha=oval_alpha,
            linestyle="--" if muted else "-",
        )
        ax.add_patch(oval)
        cite = f.get("cite", "")
        # three lines: label / var% / citation
        label_y = fy + 0.14 if cite else fy + 0.07
        var_y   = fy - 0.02 if cite else fy - 0.14
        ax.text(FACTOR_X, label_y, f["label"],
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=4, alpha=text_alpha)
        ax.text(FACTOR_X, var_y, f["var"],
                ha="center", va="center", fontsize=7,
                color="white", alpha=0.7 if muted else 0.88, zorder=4)
        if cite:
            ax.text(FACTOR_X, fy - 0.21, cite,
                    ha="center", va="center", fontsize=5.8,
                    color="white", alpha=0.6 if muted else 0.80,
                    style="italic", zorder=4)

        for (trait, loading), ty in zip(group, trait_ys):
            pos = loading >= 0
            base_col = color if pos else "#888888"
            arrow_col = base_col if not muted else "#BBBBBB"
            lw = (0.8 + abs(loading) * 2.0) * (0.5 if muted else 1.0)

            # arrow
            ax.annotate(
                "",
                xy=(TRAIT_X_L, ty),
                xytext=(FACTOR_X + FACTOR_OVL_W / 2, fy),
                arrowprops=dict(
                    arrowstyle="-|>", color=arrow_col, lw=lw,
                    connectionstyle="arc3,rad=0.0",
                    alpha=0.4 if muted else 1.0,
                ),
                zorder=2,
            )

            # loading label midpoint
            mx = (FACTOR_X + FACTOR_OVL_W / 2 + TRAIT_X_L) / 2
            my = (fy + ty) / 2
            sign = "+" if pos else ""
            ax.text(mx, my, f"{sign}{loading:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color=arrow_col, fontweight="bold",
                    alpha=0.5 if muted else 1.0,
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=0.8, alpha=0.85),
                    zorder=5)

            # trait box
            rect = FancyBboxPatch(
                (TRAIT_X_L, ty - TRAIT_H / 2), TRAIT_W, TRAIT_H,
                boxstyle="round,pad=0.06",
                facecolor=box_fc,
                edgecolor="#CCCCCC" if muted else (color if pos else "#AAAAAA"),
                linewidth=box_lw, zorder=3,
                linestyle="--" if muted else "-",
            )
            ax.add_patch(rect)
            ax.text(TRAIT_X_L + TRAIT_W / 2, ty, trait,
                    ha="center", va="center", fontsize=8.5,
                    color="#999999" if muted else "#222222", zorder=4)

    ax.set_title(
        "EFA factor structure  (k = 6,  23 traits,  minres + varimax,  |λ| ≥ 0.45,  logit-z normalized)\n"
        "Total variance explained: 76.1%      † cross-loading trait (appears in two factors)",
        fontsize=10, pad=8, loc="center",
    )

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    make_path_diagram(FACTORS, OUT)
