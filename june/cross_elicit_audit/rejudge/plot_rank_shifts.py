#!/usr/bin/env python3
"""Two figures:
  1) Slopegraph: old-judge rank -> composite_5 rank, lines coloured by direction.
  2) Per-facet rank-delta heatmap: rows = train_axis, cols = facets, values = old_rank - facet_rank.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
RJ = ROOT / 'june/cross_elicit_audit/rejudge'

W = pd.read_csv(RJ / 'full_5facet_wide.csv')
M = pd.read_csv(RJ / 'full_morality.csv')[['row_uid','morality']]
W = W.merge(M, on='row_uid', how='left')
FACETS = ['trust','morality','altruism','cooperation','modesty','sympathy']
W['composite_6'] = W[FACETS].mean(axis=1)

g = W.groupby('train_axis').agg(
    old_score=('old_score','mean'),
    composite_5=('composite_5','mean'),
    composite_6=('composite_6','mean'),
    compassion=('compassion','mean'),
    politeness=('politeness','mean'),
    **{f: (f, 'mean') for f in FACETS},
).reset_index()

# Higher score = more agreeable. Rank 1 = MOST agreeable.
score_cols = ['old_score','composite_5','composite_6','compassion','politeness'] + FACETS
for c in score_cols:
    g[f'{c}_rank'] = g[c].rank(method='min', ascending=False).astype(int)

# ----- Figure 1: slopegraph old_rank -> composite_5_rank -----
g_sorted = g.sort_values('composite_5_rank').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, max(8, 0.32*len(g_sorted))))
for _, row in g_sorted.iterrows():
    ro, rn = row['old_score_rank'], row['composite_5_rank']
    delta = ro - rn  # +ve: old ranked it lower (less agreeable) than new -> new judge rates it MORE agreeable
    if delta > 0:
        color = '#2a7f3a'  # new rates more agreeable -> green
    elif delta < 0:
        color = '#c44'     # new rates less agreeable -> red
    else:
        color = '#888'
    lw = 0.8 + min(abs(delta)/4, 2.0)
    ax.plot([0,1], [ro, rn], '-', color=color, linewidth=lw, alpha=0.85)
    ax.plot(0, ro, 'o', color=color, markersize=4)
    ax.plot(1, rn, 'o', color=color, markersize=4)
    # Labels: train_axis on right
    label = row['train_axis']
    if abs(delta) >= 3:
        label = f"{label}  (Δ{delta:+d})"
    ax.text(1.03, rn, label, va='center', ha='left', fontsize=8,
            color=color if abs(delta)>=3 else 'black',
            fontweight='bold' if abs(delta)>=5 else 'normal')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Old judge\n(single 0-100)', '5-facet composite\n(gemini-2.5-flash)'], fontsize=10)
ax.set_ylabel('Rank (1 = most agreeable)')
ax.invert_yaxis()
ax.set_xlim(-0.05, 1.55)
ax.set_title('Train_axis ranking: old single-score judge vs 5-facet composite\n(Green = new judge ranks more agreeable; Red = less; bold = Δ ≥ 5)',
             fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
out1 = RJ / 'rank_slopegraph.png'
plt.savefig(out1, dpi=160, bbox_inches='tight')
print(f'-> {out1}')

# ----- Figure 2: per-facet rank-delta heatmap -----
# Rows: train_axis sorted by composite_5 rank
# Cols: facet rank deltas (old_rank - facet_rank). +ve: old judge ranked it lower than facet does (= old judge was too harsh on this trait on this axis).
delta_cols = ['composite_5','composite_6','compassion','politeness'] + FACETS
deltas = pd.DataFrame()
for c in delta_cols:
    deltas[c] = g['old_score_rank'] - g[f'{c}_rank']
deltas.index = g['train_axis']
deltas = deltas.loc[g_sorted['train_axis']]

mat = deltas.values
vmax = max(abs(mat.min()), abs(mat.max()))
fig, ax = plt.subplots(figsize=(11, max(8, 0.32*len(deltas))))
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i,j]
        color = 'white' if abs(v) > vmax*0.6 else 'black'
        ax.text(j, i, f'{v:+d}', ha='center', va='center', color=color, fontsize=8)

# Vertical separators
for x_div in [2, 4, 5]:
    ax.axvline(x_div - 0.5, color='black', linewidth=1.0)

display_labels = ['Comp.5','Comp.6','Compassion','Politeness','Trust','Morality\n(caveat)','Altruism','Cooperation','Modesty','Sympathy']
ax.set_xticks(range(len(display_labels)))
ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(deltas)))
ax.set_yticklabels(deltas.index, fontsize=8)
ax.set_title('Rank shift: old single-score judge vs each NEO facet/derived score\n'
             '(value = old_rank − facet_rank; +ve red = old judge ranked too low; −ve blue = old too high)',
             fontsize=10)
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('Rank delta')

plt.tight_layout()
out2 = RJ / 'rank_delta_heatmap.png'
plt.savefig(out2, dpi=160, bbox_inches='tight')
print(f'-> {out2}')

# Print summary of biggest shifts
print('\nBiggest rank shifts (old vs composite_5):')
g['delta_5'] = g['old_score_rank'] - g['composite_5_rank']
print(g[['train_axis','old_score','composite_5','old_score_rank','composite_5_rank','delta_5']]
      .sort_values('delta_5', key=abs, ascending=False).head(12).to_string(index=False))
