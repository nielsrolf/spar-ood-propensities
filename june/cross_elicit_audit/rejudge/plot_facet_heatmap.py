#!/usr/bin/env python3
"""Heatmap: train_axis × NEO-A facet, from the full 5-facet + morality runs.
Adds derived Compassion / Politeness / Composite columns.
Output: rejudge/facet_heatmap.png
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

# Recompute 6-facet composite (separately from composite_5 for transparency)
FACETS = ['trust', 'morality', 'altruism', 'cooperation', 'modesty', 'sympathy']
W['composite_6'] = W[FACETS].mean(axis=1)

# Aggregate by train_axis
COLS = FACETS + ['compassion','politeness','composite_5','composite_6']
mean_tab = W.groupby('train_axis')[COLS].mean().round(1)
n_tab = W.groupby('train_axis').size().rename('n')
mean_tab = mean_tab.join(n_tab)
mean_tab = mean_tab.sort_values('composite_5', ascending=False)
mean_tab.to_csv(RJ / 'facet_heatmap_table.csv')
print(mean_tab.to_string())

# Display order: facets, then derived, with Morality visually separated
display_cols = ['trust','altruism','cooperation','modesty','sympathy',  # 5-facet
                'morality',                                              # caveat-flagged
                'compassion','politeness','composite_5','composite_6']
display_labels = ['Trust','Altruism','Cooperation','Modesty','Sympathy',
                  'Morality\n(caveat)',
                  'Compassion','Politeness','Comp.5','Comp.6']

mat = mean_tab[display_cols].values
axes_labels = [f"{ax} (n={int(mean_tab.loc[ax,'n'])})" for ax in mean_tab.index]

fig, ax = plt.subplots(figsize=(11, max(8, 0.3*len(mean_tab)+1)))
cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=20, vmax=90)
im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')

# Annotations
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat[i,j]
        color = 'black' if 35 < val < 75 else 'white'
        ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

# Visual divider before Morality column and before derived
for x_div in [5, 6, 8]:
    ax.axvline(x_div - 0.5, color='black', linewidth=1.2)

ax.set_xticks(range(len(display_labels)))
ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(axes_labels)))
ax.set_yticklabels(axes_labels, fontsize=8)
ax.set_title('NEO Agreeableness facet profile by train_axis\n'
             '(Llama-3.1-8B-Instruct FT, epoch 5; gemini-2.5-flash 5-sample mean)',
             fontsize=10)
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Mean facet score (0–100)', fontsize=9)

plt.tight_layout()
out = RJ / 'facet_heatmap.png'
plt.savefig(out, dpi=160, bbox_inches='tight')
print(f'\n-> {out}')
