#!/usr/bin/env python3
"""Focused 2-column version of (a)/(b) heatmaps from perseed_singlemodel_100.png.

Restricts to the two largest holdout subsets (nn05, total) and drops the rest.
"""

import json
import os
import re
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
    ("combo1", "maccs+avalon+scage2+mole",  1, 93),
    ("combo2", "maccs+scage1+mole",          1, 100),
    ("combo3", "maccs+scage1+scage2+mole",   1, 90),
]
OLD_POST_ITER = {"combo1": 38, "combo2": 60, "combo3": 54}
BASELINES = ["mlp2", "mlp3", "mlp4", "xgboost", "lightgbm"]
SUBSETS = ["nn05", "total"]
SUBSET_N = {"nn05": 329, "total": 888}
WORKTREE = {
    "combo1": "/home/minji/combos-v2-combo1",
    "combo2": "/home/minji/combos-v2-combo2",
    "combo3": "/home/minji/combos-v2-combo3",
}


def find_holdout(holdout_dir, it):
    for fn in os.listdir(holdout_dir):
        m = re.match(r"iter(\d+)_", fn)
        if m and int(m.group(1)) == it:
            return holdout_dir / fn
    return None


DATA = {}
for name, combo_str, it_pre, it_post in COMBOS:
    cdir = Path(WORKTREE[name]) / "results" / combo_str
    bdir = ROOT / "results" / combo_str / "baselines"
    hd = cdir / "holdout_eval"
    post_h     = json.load(open(find_holdout(hd, it_post)))
    old_post_h = json.load(open(find_holdout(hd, OLD_POST_ITER[name])))
    bjsons = {m: json.load(open(bdir / f"{m}.json")) for m in BASELINES}
    DATA[name] = {
        "post":     post_h["summary"]["subsets"],
        "old_post": old_post_h["summary"]["subsets"],
        "baselines": {m: bjsons[m]["summary"] for m in BASELINES},
    }


def per_seed_mean(combo, label, subset, key):
    if label == "GateMol post":
        block = DATA[combo]["post"][subset]
    elif label == "GateMol post (old)":
        block = DATA[combo]["old_post"][subset]
    else:
        block = DATA[combo]["baselines"][label]["subsets"][subset]
    return block[f"per_seed_mean_{key}"]


row_labels = []
for cname, *_ in COMBOS:
    for met in ("AUC", "MCC"):
        row_labels.append(f"{cname} {met}")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
fig.subplots_adjust(wspace=0.55)

# (a) Δ(post-new − post-old)
mat_a = np.zeros((6, len(SUBSETS)))
for ri, (cname, *_) in enumerate(COMBOS):
    for mi, key in enumerate(["roc_auc", "mcc"]):
        for ci, sub in enumerate(SUBSETS):
            mat_a[ri * 2 + mi, ci] = (
                per_seed_mean(cname, "GateMol post", sub, key)
                - per_seed_mean(cname, "GateMol post (old)", sub, key)
            )
vmax_a = max(0.02, np.abs(mat_a).max())
ax = axes[0]
im = ax.imshow(mat_a, cmap="RdBu_r", vmin=-vmax_a, vmax=vmax_a, aspect="auto")
ax.set_xticks(range(len(SUBSETS)))
ax.set_xticklabels([f"{s}\n(n={SUBSET_N[s]})" for s in SUBSETS])
ax.set_yticks(range(6))
ax.set_yticklabels(row_labels)
ax.set_title("(a) per-seed mean Δ(post-new − post-old)\nmarginal gain of last ~40 keep iter")
for i in range(6):
    for j in range(len(SUBSETS)):
        ax.text(j, i, f"{mat_a[i,j]:+.3f}", ha="center", va="center",
                fontsize=9,
                color="black" if abs(mat_a[i,j]) < vmax_a*0.65 else "white")
plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)

# (b) post-new − best baseline
mat_b = np.zeros((6, len(SUBSETS)))
for ri, (cname, *_) in enumerate(COMBOS):
    for mi, key in enumerate(["roc_auc", "mcc"]):
        for ci, sub in enumerate(SUBSETS):
            post_v = per_seed_mean(cname, "GateMol post", sub, key)
            best_bv = max(per_seed_mean(cname, b, sub, key) for b in BASELINES)
            mat_b[ri * 2 + mi, ci] = post_v - best_bv
vmax_b = max(0.03, np.abs(mat_b).max())
ax = axes[1]
im = ax.imshow(mat_b, cmap="RdBu_r", vmin=-vmax_b, vmax=vmax_b, aspect="auto")
ax.set_xticks(range(len(SUBSETS)))
ax.set_xticklabels([f"{s}\n(n={SUBSET_N[s]})" for s in SUBSETS])
ax.set_yticks(range(6))
ax.set_yticklabels(row_labels)
ax.set_title("(b) per-seed mean (GateMol post − best baseline)\ndeployment-race gap, positive ⇒ GateMol wins")
for i in range(6):
    for j in range(len(SUBSETS)):
        ax.text(j, i, f"{mat_b[i,j]:+.3f}", ha="center", va="center",
                fontsize=9,
                color="black" if abs(mat_b[i,j]) < vmax_b*0.65 else "white")
plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)

fig.suptitle("Per-seed deep-dive — nn05 + total only (100-iter snapshot)",
             fontsize=12, y=1.02)
out = FIG_DIR / "perseed_singlemodel_100_focus.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[fig] wrote {out}")
