#!/usr/bin/env python3
"""Per-seed mean AUC/MCC grid at the iter200 snapshot.

Mirrors grid_perseed_100.png but with the post checkpoints advanced to the
last-kept iter <= 200 for each combo:
  combo1 -> iter197 (commit 9bc22da482)
  combo2 -> iter198 (commit 2e8cbdc9f5)
  combo3 -> iter197 (commit 2c4803d72c)

5 subsets (internal, external, nn03, nn05, total) x 2 metrics (AUC, MCC),
bar chart over: GateMol pre, GateMol post, mlp2/3/4, xgboost, lightgbm.
"""

import json
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 110,
})

ROOT = Path("/home/minji/autoresearch_combos_v2")
FIG_DIR = ROOT / "results" / "comparison" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (name, combo_str, it_pre, it_post, worktree_root)
COMBOS = [
    ("combo1", "maccs+avalon+scage2+mole", 1, 197, "/home/minji/combos-v2-combo1"),
    ("combo2", "maccs+scage1+mole",         1, 198, "/home/minji/combos-v2-combo2"),
    ("combo3", "maccs+scage1+scage2+mole",  1, 197, "/home/minji/combos-v2-combo3"),
]
BASELINES = ["mlp2", "mlp3", "mlp4", "xgboost", "lightgbm", "tabpfn"]
MODEL_ORDER = ["GateMol pre", "GateMol post"] + BASELINES
SUBSETS = ["internal", "external", "nn03", "nn05", "total"]
SUBSET_N = {"internal": 179, "external": 709, "nn03": 39, "nn05": 329, "total": 888}

MODEL_COLORS = {
    "GateMol pre":  "#1f77b4",
    "GateMol post": "#0b3d91",
    "mlp2":         "#ffbb78",
    "mlp3":         "#ff7f0e",
    "mlp4":         "#d62728",
    "xgboost":      "#2ca02c",
    "lightgbm":     "#8c564b",
    "tabpfn":       "#17becf",
    "8-modal ref":  "#9467bd",
}

# 8-modal reference (val only, all 3 combos share the same value).
# Pulled from /home/minji/manifest_combo_agg_v2.csv row:
#   ecfp+maccs+avalon+tt+rdkit+scage1+scage2+mole,8,scaffold,10,...
REF_8MOD = {
    "combo_str": "ecfp+maccs+avalon+tt+rdkit+scage1+scage2+mole",
    "val": {
        "roc_auc": (0.831830, 0.004309),
        "mcc":     (0.488510, 0.019802),
    },
}


def find_holdout(holdout_dir, it):
    for fn in os.listdir(holdout_dir):
        m = re.match(r"iter(\d+)_", fn)
        if m and int(m.group(1)) == it:
            return holdout_dir / fn
    return None


DATA = {}
for name, combo_str, it_pre, it_post, wt in COMBOS:
    bdir = ROOT / "results" / combo_str / "baselines"
    cdir = Path(wt) / "results" / combo_str
    hd = cdir / "holdout_eval"
    pre_path = find_holdout(hd, it_pre)
    post_path = find_holdout(hd, it_post)
    if pre_path is None or post_path is None:
        raise FileNotFoundError(
            f"{name}: pre(iter{it_pre})={pre_path}, post(iter{it_post})={post_path}"
        )
    pre_h = json.load(open(pre_path))
    post_h = json.load(open(post_path))
    bjsons = {m: json.load(open(bdir / f"{m}.json")) for m in BASELINES}
    DATA[name] = {
        "pre":  pre_h["summary"]["subsets"],
        "post": post_h["summary"]["subsets"],
        # val per-seed mean/std lives directly under summary['val'] for both
        # GateMol and the baseline runs.
        "pre_val":  pre_h["summary"]["val"],
        "post_val": post_h["summary"]["val"],
        "baselines":     {m: bjsons[m]["summary"]["subsets"] for m in BASELINES},
        "baselines_val": {m: bjsons[m]["summary"]["val"]     for m in BASELINES},
    }


def get_value(combo_name, model_label, subset, metric_key):
    """Return (per-seed mean, per-seed std) for a holdout subset, or
    (np.nan, 0) if the subset is missing for this model (e.g. tabpfn on
    combo1/3 ran shortened to nn05+total only)."""
    if model_label == "GateMol pre":
        d = DATA[combo_name]["pre"]
    elif model_label == "GateMol post":
        d = DATA[combo_name]["post"]
    else:
        d = DATA[combo_name]["baselines"][model_label]
    if subset not in d:
        return float("nan"), 0.0
    block = d[subset]
    return (block[f"per_seed_mean_{metric_key}"],
            block[f"per_seed_std_{metric_key}"])


def get_val(combo_name, model_label, metric_key):
    """Return (per-seed mean, per-seed std) for the scaffold val split."""
    if model_label == "GateMol pre":
        v = DATA[combo_name]["pre_val"]
    elif model_label == "GateMol post":
        v = DATA[combo_name]["post_val"]
    elif model_label == "8-modal ref":
        return REF_8MOD["val"][metric_key]
    else:
        v = DATA[combo_name]["baselines_val"][model_label]
    return v[f"mean_{metric_key}"], v[f"std_{metric_key}"]


def grid_figure(fname):
    # 6 rows: val (scaffold) + 5 holdout subsets, x 2 metrics (AUC, MCC)
    n_rows = 1 + len(SUBSETS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 21), sharex=False)
    x_pos = np.arange(len(COMBOS))

    # ---- top row: val (scaffold) — includes 8-modal reference
    val_models = MODEL_ORDER + ["8-modal ref"]
    bar_w_val = 0.085
    for c, (metric_label, key) in enumerate([("AUC", "roc_auc"),
                                             ("MCC", "mcc")]):
        ax = axes[0, c]
        for mi, lbl in enumerate(val_models):
            vals, errs = [], []
            for cname, *_ in COMBOS:
                v, std = get_val(cname, lbl, key)
                vals.append(v)
                errs.append(std)
            offset = (mi - (len(val_models) - 1) / 2) * bar_w_val
            ax.bar(x_pos + offset, vals, yerr=errs, capsize=2,
                   width=bar_w_val, color=MODEL_COLORS[lbl],
                   edgecolor="black", linewidth=0.4,
                   error_kw=dict(elinewidth=0.6), label=lbl)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c[0] for c in COMBOS])
        ax.set_title(f"val (scaffold, n_seeds=10) — {metric_label}",
                     fontsize=10)
        if metric_label == "AUC":
            ax.set_ylim(0.78, 0.88)
        else:
            ax.set_ylim(0.35, 0.65)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_ylabel(metric_label)

    # ---- next 5 rows: holdout subsets (no 8-modal — only val data available)
    bar_w = 0.095
    for r, subset in enumerate(SUBSETS):
        for c, (metric_label, key) in enumerate([("AUC", "roc_auc"),
                                                 ("MCC", "mcc")]):
            ax = axes[r + 1, c]
            for mi, lbl in enumerate(MODEL_ORDER):
                vals, errs = [], []
                for cname, *_ in COMBOS:
                    v, std = get_value(cname, lbl, subset, key)
                    vals.append(v)
                    errs.append(std)
                offset = (mi - (len(MODEL_ORDER) - 1) / 2) * bar_w
                ax.bar(x_pos + offset, vals, yerr=errs, capsize=2,
                       width=bar_w, color=MODEL_COLORS[lbl],
                       edgecolor="black", linewidth=0.4,
                       error_kw=dict(elinewidth=0.6), label=lbl)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([c[0] for c in COMBOS])
            ax.set_title(f"{subset} (n={SUBSET_N[subset]}) — {metric_label}",
                         fontsize=10)
            if metric_label == "AUC":
                ax.set_ylim(0.78, 0.93)
            else:
                ax.set_ylim(0.35, 0.80)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_ylabel(metric_label)

    # Reserve bottom margin for a single-row legend + caption.
    fig.tight_layout(rect=[0, 0.05, 1.0, 1.0])

    # Single-row legend at the bottom (covers all model labels, including 8-modal ref).
    legend_labels = MODEL_ORDER + ["8-modal ref"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[lbl],
                             edgecolor="black", linewidth=0.4)
               for lbl in legend_labels]
    fig.legend(handles, legend_labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.028),
               ncol=len(legend_labels), framealpha=0.95, fontsize=10,
               handlelength=1.4, columnspacing=1.4)

    # Single-line caption (one line below the legend).
    fig.text(
        0.5, 0.005,
        "Per-seed mean basis (200-iter snapshot) — val + holdout subsets · "
        "post checkpoints: combo1=iter197, combo2=iter198, combo3=iter197 · "
        "8-modal ref (ecfp+maccs+avalon+tt+rdkit+scage1+scage2+mole) shown on val only",
        ha="center", va="bottom", fontsize=11)
    out = FIG_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    grid_figure("grid_perseed_200.png")
