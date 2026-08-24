#!/usr/bin/env python3
"""Three-metric (AUC / MCC / AUPRC) side-by-side heatmaps showing
GateMol post-new − best baseline (per-seed mean) over combo × subset.

Goal: visualize whether GateMol post wins across all combos and all subsets,
across all three holdout metrics.
"""
import json, os, re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 110,
})

ROOT = Path("/home/minji/autoresearch_combos_v2")
FIG_DIR = ROOT / "results" / "comparison" / "figures"

COMBOS = [
    # (name, combo_str, it_post_new, it_post_old, worktree)
    ("combo1", "maccs+avalon+scage2+mole",  93,  38, "/home/minji/combos-v2-combo1"),
    ("combo2", "maccs+scage1+mole",          100, 60, "/home/minji/combos-v2-combo2"),
    ("combo3", "maccs+scage1+scage2+mole",   90,  54, "/home/minji/combos-v2-combo3"),
]
BASELINES = ["mlp2", "mlp3", "mlp4", "xgboost", "lightgbm"]
SUBSETS = ["nn05", "total"]
SUBSET_N = {"nn05": 329, "total": 888}
METRICS = [("AUC", "roc_auc"), ("MCC", "mcc"), ("AUPRC", "auprc")]


def find_holdout(hd, it):
    for fn in os.listdir(hd):
        m = re.match(r"iter(\d+)_", fn)
        if m and int(m.group(1)) == it:
            return hd / fn


# Load
DATA = {}
for name, cs, it_new, it_old, wt in COMBOS:
    hd = Path(wt)/"results"/cs/"holdout_eval"
    bdir = ROOT/"results"/cs/"baselines"
    post_new = json.load(open(find_holdout(hd, it_new)))["summary"]["subsets"]
    post_old = json.load(open(find_holdout(hd, it_old)))["summary"]["subsets"]
    base = {}
    for m in BASELINES:
        s = json.load(open(bdir/f"{m}.json"))["summary"]["subsets"]
        base[m] = {k: v for k, v in s.items() if k != "aggregation"}
    DATA[name] = {"post-new": post_new, "post-old": post_old, "base": base}


def post_val(combo, which, sub, key):
    return DATA[combo][which][sub][f"per_seed_mean_{key}"]


def best_baseline(combo, sub, key):
    best_v, best_b = -1, None
    for b in BASELINES:
        v = DATA[combo]["base"][b][sub][f"per_seed_mean_{key}"]
        if v > best_v:
            best_v, best_b = v, b
    return best_v, best_b


POST_ROWS = [
    ("post-new", "post-new (iter 93/100/90)"),
    ("post-old", "post-old (iter 38/60/54)"),
]

fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8))
fig.subplots_adjust(wspace=0.50, hspace=0.55)

for row_idx, (post_key, post_disp) in enumerate(POST_ROWS):
    for col_idx, (met_label, key) in enumerate(METRICS):
        ax = axes[row_idx, col_idx]
        mat = np.zeros((len(COMBOS), len(SUBSETS)))
        best_who = np.empty(mat.shape, dtype=object)
        for ri, (cname, *_) in enumerate(COMBOS):
            for ci, sub in enumerate(SUBSETS):
                p = post_val(cname, post_key, sub, key)
                b, who = best_baseline(cname, sub, key)
                mat[ri, ci] = p - b
                best_who[ri, ci] = who
        vmax = max(0.03, np.abs(mat).max())
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(SUBSETS)))
        ax.set_xticklabels([f"{s}\n(n={SUBSET_N[s]})" for s in SUBSETS], fontsize=8)
        ax.set_yticks(range(len(COMBOS)))
        ax.set_yticklabels([c[0] for c in COMBOS])
        ax.set_title(f"{met_label}\nGateMol {post_disp} − best baseline", fontsize=9)
        for i in range(len(COMBOS)):
            for j in range(len(SUBSETS)):
                txt = f"{mat[i,j]:+.3f}\n({best_who[i,j]})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="black" if abs(mat[i,j]) < vmax*0.65 else "white")
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

fig.suptitle("Per-seed mean: GateMol post (new / old) vs best baseline — nn05 + total\n(positive ⇒ GateMol wins)",
             fontsize=12, y=1.00)
out = FIG_DIR / "auc_mcc_auprc_supremacy_100.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[fig] wrote {out}")


# Print raw delta values for reference (post-new and post-old side by side)
print()
for post_key, post_disp in POST_ROWS:
    print(f"=== {post_disp} ===")
    for name, *_ in COMBOS:
        for sub in SUBSETS:
            cells = []
            for met_label, key in METRICS:
                p = post_val(name, post_key, sub, key)
                b, who = best_baseline(name, sub, key)
                cells.append(f"{met_label} Δ={p-b:+.3f} (vs {who})")
            print(f"  {name} {sub:9s} | " + "  ".join(cells))
        print()
