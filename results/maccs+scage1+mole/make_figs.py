"""Reproducibility visualizations: combo2 vs combo1/3 keep-theme comparison.

Outputs three figures into the same directory:
  fig1_theme_heatmap.png  — themes × runs binary adoption heatmap
  fig2_jaccard_cluster.png — pairwise Jaccard heatmap + dendrogram
  fig3_signature_bars.png  — combo signature (themes unique vs shared)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

OUT = os.path.dirname(os.path.abspath(__file__))

RUNS = ['combo1', 'combo2_main', 'combo2_fork_a', 'combo2_fork_b', 'combo3']
RUN_LABELS = ['combo1', 'combo2\nmain', 'combo2\nfork-a', 'combo2\nfork-b', 'combo3']
RUN_COLORS = ['#d6604d', '#4393c3', '#4393c3', '#4393c3', '#5aae61']

KEEPS = {
    'combo1':        ['T01_baseline','T06_mod_drop_high','T12_mh_sgu_h2','T17_ema_999',
                      'T20_per_mod_scale','T24_droppath_sgu_v','T25_ema_warmup'],
    'combo2_main':   ['T01_baseline','T07_skipgate_pool','T10_sgu_scale','T15_stoch_depth',
                      'T17_ema_999','T18_adamw_clip_no_sched','T19_mod_drop_low'],
    'combo2_fork_a': ['T01_baseline','T07_skipgate_pool','T10_sgu_scale','T12_mh_sgu_h2',
                      'T17_ema_999','T18_adamw_clip_no_sched','T19_mod_drop_low','T22_skipgate_init'],
    'combo2_fork_b': ['T01_baseline','T06_mod_drop_high','T07_skipgate_pool','T09_attn_pool',
                      'T17_ema_999','T19_mod_drop_low','T21_grad_clip_only','T23_pos_weight_fixed'],
    'combo3':        ['T01_baseline','T07_skipgate_pool','T08_film','T09_attn_pool',
                      'T12_mh_sgu_h2','T13_adaln','T26_mq_attn_pool','T27_ema_responsive'],
}

THEME_LABELS = {
    'T01_baseline':            'baseline',
    'T06_mod_drop_high':       'modality dropout (high, p≥0.10)',
    'T07_skipgate_pool':       'skip-gate gated/mean pool',
    'T08_film':                'cross-modal FiLM',
    'T09_attn_pool':           'attention pool',
    'T10_sgu_scale':           'SGU residual scale',
    'T12_mh_sgu_h2':           'multi-head SGU (h=2)',
    'T13_adaln':               'AdaLN-Zero',
    'T15_stoch_depth':         'stochastic depth',
    'T17_ema_999':             'EMA weights (decay=0.999)',
    'T18_adamw_clip_no_sched': 'AdamW + grad-clip, no scheduler',
    'T19_mod_drop_low':        'modality dropout (low, p=0.05)',
    'T20_per_mod_scale':       'per-modality learnable scale',
    'T21_grad_clip_only':      'grad-clip only (no AdamW change)',
    'T22_skipgate_init':       'skip_gate init tuning',
    'T23_pos_weight_fixed':    'pos_weight (fixed value)',
    'T24_droppath_sgu_v':      'DropPath in SGU v-path',
    'T25_ema_warmup':          'EMA warmup epochs tuning',
    'T26_mq_attn_pool':        'multi-query attention pool',
    'T27_ema_responsive':      'EMA responsive (decay≤0.95)',
}

# Order themes by combo-signature grouping:
#   1. universal (any run baseline only here)
#   2. combo1-only
#   3. combo2-leaning (in ≥2 of 3 combo2 runs, not majority elsewhere)
#   4. combo3-only
#   5. shared across combos
THEME_ORDER = [
    'T01_baseline',
    # combo2 signature (kept in ≥2 combo2 forks)
    'T07_skipgate_pool', 'T10_sgu_scale', 'T17_ema_999', 'T18_adamw_clip_no_sched',
    'T19_mod_drop_low', 'T15_stoch_depth', 'T22_skipgate_init',
    # combo1 signature
    'T20_per_mod_scale', 'T24_droppath_sgu_v', 'T25_ema_warmup',
    # combo3 signature
    'T08_film', 'T13_adaln', 'T26_mq_attn_pool', 'T27_ema_responsive',
    # shared/ambiguous
    'T06_mod_drop_high', 'T12_mh_sgu_h2', 'T09_attn_pool',
    'T21_grad_clip_only', 'T23_pos_weight_fixed',
]

# ---- Figure 1: theme adoption heatmap ----
M = np.zeros((len(THEME_ORDER), len(RUNS)), dtype=int)
for j, run in enumerate(RUNS):
    for i, t in enumerate(THEME_ORDER):
        M[i, j] = int(t in KEEPS[run])

fig, ax = plt.subplots(figsize=(8.2, 8.6))
# Custom colormap: white for 0, run-color for 1
cell_colors = np.full((M.shape[0], M.shape[1], 4), 1.0)
for j in range(M.shape[1]):
    rgba = plt.matplotlib.colors.to_rgba(RUN_COLORS[j])
    for i in range(M.shape[0]):
        if M[i, j] == 1:
            cell_colors[i, j] = rgba
ax.imshow(cell_colors, aspect='auto')

# Grid & ticks
ax.set_xticks(range(len(RUNS)))
ax.set_xticklabels(RUN_LABELS, fontsize=10)
ax.set_yticks(range(len(THEME_ORDER)))
ax.set_yticklabels([THEME_LABELS[t] for t in THEME_ORDER], fontsize=9)
ax.tick_params(axis='both', which='both', length=0)
for i in range(M.shape[0] + 1):
    ax.axhline(i - 0.5, color='#999', lw=0.4)
for j in range(M.shape[1] + 1):
    ax.axvline(j - 0.5, color='#999', lw=0.4)

# Signature group boundaries on y-axis
group_lines = [0.5, 7.5, 10.5, 14.5]  # after baseline, combo2-sig, combo1-sig, combo3-sig
for y in group_lines:
    ax.axhline(y, color='black', lw=1.2)

# Group labels on right
group_spans = [
    (0, 0, 'baseline'),
    (1, 7, 'combo2 signature'),
    (8, 10, 'combo1 signature'),
    (11, 14, 'combo3 signature'),
    (15, 19, 'shared / ambiguous'),
]
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks([(s+e)/2 for s,e,_ in group_spans])
ax2.set_yticklabels([lbl for _,_,lbl in group_spans], fontsize=9,
                    fontweight='bold', color='#333')
ax2.tick_params(axis='y', length=0, pad=4)

# Tag combo2 forks bracket above the heatmap — draw manually so the bracket
# tips don't bleed into the heatmap area.
bar_y = -0.95          # horizontal line y (data coords)
tip_y = -0.75          # bracket tip end (closer to heatmap but still above)
ax.plot([1.0, 3.0], [bar_y, bar_y], color='#4393c3', lw=1.8, clip_on=False)
ax.plot([1.0, 1.0], [bar_y, tip_y], color='#4393c3', lw=1.8, clip_on=False)
ax.plot([3.0, 3.0], [bar_y, tip_y], color='#4393c3', lw=1.8, clip_on=False)
ax.text(2, bar_y - 0.15, 'combo2 (3 reproducibility runs)',
        ha='center', va='bottom',
        fontsize=10, fontweight='bold', color='#2c5d8a')

ax.set_title('Theme adoption across runs (iter 1-30, val_auc-based keep)',
             fontsize=11, pad=42)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_theme_heatmap.png'), dpi=160, bbox_inches='tight')
plt.close()
print('wrote fig1_theme_heatmap.png')


# ---- Figure 2: Jaccard heatmap + dendrogram ----
def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)

J = np.zeros((len(RUNS), len(RUNS)))
for i, r1 in enumerate(RUNS):
    for j, r2 in enumerate(RUNS):
        J[i, j] = jaccard(KEEPS[r1], KEEPS[r2])

# Distance for clustering
D = 1.0 - J
np.fill_diagonal(D, 0.0)
D = (D + D.T) / 2
condensed = squareform(D, checks=False)
Z = linkage(condensed, method='average')

fig = plt.figure(figsize=(11, 5.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.35)

# Left: dendrogram
ax_d = fig.add_subplot(gs[0, 0])
dd = dendrogram(Z, labels=RUN_LABELS, ax=ax_d, leaf_font_size=10,
                color_threshold=0.6,
                above_threshold_color='#555')
ax_d.set_ylabel('1 − Jaccard (kept-theme set)', fontsize=10)
ax_d.set_title('Hierarchical clustering of runs', fontsize=11)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
# color tick labels by combo
leaf_order = dd['leaves']
ticks = ax_d.get_xticklabels()
for tick, leaf in zip(ticks, leaf_order):
    tick.set_color(RUN_COLORS[leaf])
    tick.set_fontweight('bold')

# Right: Jaccard heatmap
ax_h = fig.add_subplot(gs[0, 1])
im = ax_h.imshow(J, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
ax_h.set_xticks(range(len(RUNS)))
ax_h.set_yticks(range(len(RUNS)))
ax_h.set_xticklabels(RUN_LABELS, fontsize=9)
ax_h.set_yticklabels(RUN_LABELS, fontsize=9)
for tk, c in zip(ax_h.get_xticklabels(), RUN_COLORS):
    tk.set_color(c); tk.set_fontweight('bold')
for tk, c in zip(ax_h.get_yticklabels(), RUN_COLORS):
    tk.set_color(c); tk.set_fontweight('bold')
for i in range(len(RUNS)):
    for j in range(len(RUNS)):
        ax_h.text(j, i, f'{J[i,j]:.2f}', ha='center', va='center',
                  fontsize=9, color='black' if 0.25 < J[i,j] < 0.85 else 'white')

# Highlight combo2 block (indices 1..3)
rect = Rectangle((0.5, 0.5), 3, 3, fill=False, edgecolor='#2c5d8a',
                 lw=2.5, linestyle='--')
ax_h.add_patch(rect)
ax_h.text(2, -0.85, 'combo2 within-combo block',
          ha='center', fontsize=9, color='#2c5d8a', fontweight='bold')

plt.colorbar(im, ax=ax_h, shrink=0.7, label='Jaccard')
ax_h.set_title('Pairwise Jaccard of kept themes', fontsize=11)

plt.suptitle('combo2 runs cluster together; cross-combo agreement is much weaker',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_jaccard_cluster.png'), dpi=160, bbox_inches='tight')
plt.close()
print('wrote fig2_jaccard_cluster.png')


# ---- Figure 3: average Jaccard comparison ----
within_c2 = []
between_combo = []
pairs = [(1,2),(1,3),(2,3)]   # combo2 forks
for a,b in pairs:
    within_c2.append(J[a,b])
for a in [0,4]:   # combo1, combo3
    for b in [1,2,3]:
        between_combo.append(J[a,b])
between_combo.append(J[0,4])  # combo1 vs combo3

fig, ax = plt.subplots(figsize=(7, 4.5))
positions = [0, 1, 2]
data = [within_c2, between_combo, [J[0,4]]]
labels = [f'within combo2\n(3 forks)\nn={len(within_c2)}',
          f'combo2 ↔ other\n(combo1/3)\nn={len(between_combo)-1}',
          f'combo1 ↔ combo3\nn=1']

bp = ax.boxplot([within_c2, [j for j in between_combo if j != J[0,4]]],
                positions=[0, 1], widths=0.55, patch_artist=True,
                medianprops=dict(color='black', lw=1.5))
for patch, c in zip(bp['boxes'], ['#4393c3', '#bdbdbd']):
    patch.set_facecolor(c); patch.set_alpha(0.7)

# overlay individual points
import matplotlib.lines as mlines
for x, vals, color in [(0, within_c2, '#2c5d8a'),
                       (1, [j for j in between_combo if j != J[0,4]], '#666')]:
    jitter = np.random.RandomState(0).uniform(-0.08, 0.08, len(vals))
    ax.scatter(np.full(len(vals), x) + jitter, vals,
               s=42, color=color, edgecolor='white', lw=0.6, zorder=3)

# combo1↔combo3 as single triangle
ax.scatter([2], [J[0,4]], marker='^', s=120, color='#d6604d',
           edgecolor='black', lw=0.8, zorder=4, label='combo1 ↔ combo3')

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Jaccard (kept-theme set)', fontsize=10)
ax.set_ylim(0, 1)
ax.axhline(np.mean(within_c2), color='#2c5d8a', ls=':', lw=1,
           label=f'within-combo2 mean = {np.mean(within_c2):.3f}')
ax.axhline(np.mean([j for j in between_combo if j != J[0,4]]), color='#666',
           ls=':', lw=1,
           label=f'cross-combo mean = {np.mean([j for j in between_combo if j != J[0,4]]):.3f}')
ax.legend(loc='upper right', fontsize=8.5, frameon=False)
ax.set_title('Within-combo2 agreement is 2-3× higher than cross-combo agreement',
             fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_signature_bars.png'), dpi=160, bbox_inches='tight')
plt.close()
print('wrote fig3_signature_bars.png')
