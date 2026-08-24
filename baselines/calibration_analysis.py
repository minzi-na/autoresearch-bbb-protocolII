#!/usr/bin/env python3
"""
calibration_analysis.py — distribution & threshold-sweep analysis on holdout `total`.

For combo3 (`maccs+scage1+scage2+mole`) on the `total` holdout subset, this script:
  1. Re-runs GateMol-BBB iter1 (= master train.py, identical to commit 39cc30d0eb)
     across 10 seeds and aggregates predictions via soft voting on holdout `total`.
  2. Re-runs LightGBM baseline across the same 10 seeds with soft voting on the
     same subset.
  3. Compares the two ensemble probability distributions visually (histogram).
  4. Sweeps the classification threshold over [0.30, 0.70] and reports the MCC
     curve for each model — used to test whether GateMol's MCC under-performance
     at threshold 0.5 is recoverable by a shifted threshold.

Outputs (under results/):
  calibration_analysis_combo3_probs.npz       — raw ensemble probs + labels
  calibration_analysis_combo3_hist.png        — prob histogram, side-by-side
  calibration_analysis_combo3_threshold.png   — MCC vs threshold curve
  calibration_analysis_combo3.json            — numeric summary

Run from master worktree:
  conda run -n rapids-25.02 python baselines/calibration_analysis.py
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tu

from sklearn.metrics import matthews_corrcoef

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare                  # noqa: E402
import train as train_mod       # noqa: E402  master HEAD ≡ combo3 iter1
from baselines import models as bm  # noqa: E402


COMBO  = "maccs+scage1+scage2+mole"     # combo3
SUBSET = "total"
SEEDS  = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]


def gatemol_iter1_probs(combo_tuple, X_pool, y_pool, pool_smiles,
                        X_holdout, fp_dim, seeds):
    """10-seed GateMol-BBB baseline-architecture predictions on holdout subset.
    Returns (probs_stack of shape (n_seeds, n_holdout), per_seed_summary list)."""
    out = []
    summary = []
    for i, seed in enumerate(seeds):
        train_idx, val_idx = prepare.scaffold_split_train_val(
            pool_smiles, "scaffold", seed, 0.8)
        model, train_info, val_metrics, scaler = train_mod.build_and_train(
            combo=combo_tuple, X_pool=X_pool, y_pool=y_pool,
            train_idx=train_idx, val_idx=val_idx,
            fp_dim=fp_dim, seed=seed,
        )
        X_in = X_holdout
        if scaler is not None and "rdkit" in combo_tuple:
            X_in = train_mod.apply_rdkit_scaler(X_holdout, combo_tuple, fp_dim, scaler)

        loader = tu.DataLoader(
            tu.TensorDataset(
                torch.tensor(X_in, dtype=torch.float32),
                torch.zeros(len(X_in), dtype=torch.float32),
            ),
            batch_size=train_mod.BASE_CONFIG["batch_size"], shuffle=False,
        )
        model.eval()
        device = next(model.parameters()).device
        probs = []
        with torch.no_grad():
            for x, _ in loader:
                probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        probs = np.concatenate(probs)
        out.append(probs)
        summary.append({
            "seed": int(seed),
            "val_auc": float(val_metrics["roc_auc"]),
            "val_mcc": float(val_metrics["mcc"]),
        })
        print(f"  [GateMol {i+1}/{len(seeds)}] seed={seed} val_auc={val_metrics['roc_auc']:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return np.stack(out, axis=0), summary


def lightgbm_probs(combo_tuple, X_pool, y_pool, pool_smiles,
                   X_holdout, seeds):
    out = []
    for i, seed in enumerate(seeds):
        train_idx, _ = prepare.scaffold_split_train_val(
            pool_smiles, "scaffold", seed, 0.8)
        X_train = X_pool[train_idx].astype(np.float32, copy=True)
        y_train = y_pool[train_idx].astype(np.float32, copy=True)
        # combo3 has no rdkit; tree models are scale-invariant either way.
        model = bm.train_lgbm(X_train, y_train, seed)
        probs = model.predict_proba(X_holdout)[:, 1]
        out.append(probs)
        print(f"  [LightGBM {i+1}/{len(seeds)}] seed={seed}")
    return np.stack(out, axis=0)


def shape_summary(name, probs, y_true, threshold=0.5):
    """Distribution shape + classification summary at a fixed threshold."""
    pred = (probs > threshold).astype(int)
    return {
        "model":               name,
        "n":                   int(len(probs)),
        "mean_prob":           float(probs.mean()),
        "std_prob":            float(probs.std()),
        "median_prob":         float(np.median(probs)),
        "pct_prob_above_0.9":  float((probs > 0.9).mean()),
        "pct_prob_above_0.7":  float((probs > 0.7).mean()),
        "pct_prob_below_0.3":  float((probs < 0.3).mean()),
        "pct_prob_below_0.1":  float((probs < 0.1).mean()),
        "mcc_at_threshold":    float(matthews_corrcoef(y_true, pred)),
        "threshold":           float(threshold),
    }


def threshold_sweep(probs, y_true, ts):
    return [float(matthews_corrcoef(y_true, (probs > t).astype(int))) for t in ts]


def main():
    print(f"[Plan] combo={COMBO}  subset={SUBSET}  seeds={SEEDS}")
    combo_tuple = prepare.combo_str_to_tuple(COMBO)

    print("[Data] Loading pool ...")
    pool_smiles, pool_labels, _, pool_feats = prepare.load_pool_features()
    X_pool = prepare.build_feature_matrix(combo_tuple, pool_feats)

    print(f"[Data] Loading holdout subset = {SUBSET} ...")
    _, y_h, ft_h = prepare.load_holdout_subset(SUBSET)
    X_h = prepare.build_feature_matrix(combo_tuple, ft_h)
    print(f"[Data] holdout n={len(y_h)}  pos_rate={y_h.mean()*100:.1f}%")

    print("\n=== GateMol-BBB iter1 (10 seeds) ===")
    gm_stack, gm_per_seed = gatemol_iter1_probs(
        combo_tuple, X_pool, pool_labels, pool_smiles, X_h, prepare.FP_DIM, SEEDS)
    gm_ens = gm_stack.mean(axis=0)
    print(f"  ensemble mean prob = {gm_ens.mean():.4f}")

    print("\n=== LightGBM (10 seeds) ===")
    lgb_stack = lightgbm_probs(combo_tuple, X_pool, pool_labels, pool_smiles, X_h, SEEDS)
    lgb_ens = lgb_stack.mean(axis=0)
    print(f"  ensemble mean prob = {lgb_ens.mean():.4f}")

    # ── Sanity check vs iter0001 holdout_eval JSON ───────────────────────────
    from sklearn.metrics import roc_auc_score
    gm_auc = roc_auc_score(y_h, gm_ens)
    lgb_auc = roc_auc_score(y_h, lgb_ens)
    gm_mcc05 = matthews_corrcoef(y_h, (gm_ens > 0.5).astype(int))
    lgb_mcc05 = matthews_corrcoef(y_h, (lgb_ens > 0.5).astype(int))
    print("\n=== Sanity (should ~match the existing tables) ===")
    print(f"  GateMol ensemble roc_auc = {gm_auc:.4f}  mcc@0.5 = {gm_mcc05:.4f}")
    print(f"  LightGBM ensemble roc_auc = {lgb_auc:.4f}  mcc@0.5 = {lgb_mcc05:.4f}")

    # ── Save raw probs for downstream use ────────────────────────────────────
    out_dir = REPO_ROOT / "results"
    np.savez(
        out_dir / "calibration_analysis_combo3_probs.npz",
        gatemol_ensemble=gm_ens,
        gatemol_per_seed=gm_stack,
        lightgbm_ensemble=lgb_ens,
        lightgbm_per_seed=lgb_stack,
        y_true=y_h,
    )

    # ── (1) Histogram of ensemble probabilities ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, probs, name, color in [
        (axes[0], gm_ens,  "GateMol-BBB iter1", "#1f77b4"),
        (axes[1], lgb_ens, "LightGBM",          "#ff7f0e"),
    ]:
        ax.hist(probs[y_h == 1], bins=bins, alpha=0.65, label=f"BBB+ (n={int((y_h==1).sum())})", color="#2ca02c")
        ax.hist(probs[y_h == 0], bins=bins, alpha=0.65, label=f"BBB- (n={int((y_h==0).sum())})", color="#d62728")
        ax.axvline(0.5, color="k", linestyle="--", linewidth=1.0, label="threshold=0.5")
        ax.set_title(f"{name}\ncombo3 holdout total (n={len(y_h)}, pos={y_h.mean()*100:.1f}%)")
        ax.set_xlabel("ensemble probability P(BBB+)")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper center")
    axes[0].set_ylabel("count")
    plt.tight_layout()
    hist_path = out_dir / "calibration_analysis_combo3_hist.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"\n[Plot] histogram -> {hist_path}")

    # ── (2) Threshold sweep ──────────────────────────────────────────────────
    thresholds = np.round(np.arange(0.30, 0.71, 0.02), 4)
    gm_mccs  = threshold_sweep(gm_ens,  y_h, thresholds)
    lgb_mccs = threshold_sweep(lgb_ens, y_h, thresholds)

    fig, ax = plt.subplots(figsize=(9, 5.3))
    gm_best = thresholds[int(np.argmax(gm_mccs))]
    lgb_best = thresholds[int(np.argmax(lgb_mccs))]
    ax.plot(thresholds, gm_mccs,  "o-",
            label=f"GateMol-BBB iter1 — max MCC={max(gm_mccs):.4f} @ t={gm_best:.2f}",
            color="#1f77b4")
    ax.plot(thresholds, lgb_mccs, "s-",
            label=f"LightGBM — max MCC={max(lgb_mccs):.4f} @ t={lgb_best:.2f}",
            color="#ff7f0e")
    ax.axvline(0.5, color="k", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("MCC on holdout total (ensemble probs)")
    ax.set_title(f"Threshold sweep — combo3 holdout total "
                 f"(n={len(y_h)}, pos_rate={y_h.mean()*100:.1f}%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    sweep_path = out_dir / "calibration_analysis_combo3_threshold.png"
    plt.savefig(sweep_path, dpi=150)
    plt.close()
    print(f"[Plot] threshold sweep -> {sweep_path}")

    # ── JSON summary ─────────────────────────────────────────────────────────
    summary = {
        "combo":      COMBO,
        "subset":     SUBSET,
        "n":          int(len(y_h)),
        "pos_rate":   float(y_h.mean()),
        "seeds":      SEEDS,
        "GateMol-BBB iter1": {
            **shape_summary("GateMol-BBB iter1", gm_ens, y_h, threshold=0.5),
            "ensemble_roc_auc":   float(gm_auc),
            "best_threshold":     float(gm_best),
            "best_mcc":           float(max(gm_mccs)),
            "per_seed_val":       gm_per_seed,
        },
        "LightGBM": {
            **shape_summary("LightGBM", lgb_ens, y_h, threshold=0.5),
            "ensemble_roc_auc":   float(lgb_auc),
            "best_threshold":     float(lgb_best),
            "best_mcc":           float(max(lgb_mccs)),
        },
        "threshold_sweep": {
            "thresholds":     thresholds.tolist(),
            "gatemol_mccs":   gm_mccs,
            "lightgbm_mccs":  lgb_mccs,
        },
    }
    json_path = out_dir / "calibration_analysis_combo3.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] summary -> {json_path}")

    print("\n" + "=" * 70)
    print(f"GateMol-BBB iter1   mcc@0.5 = {gm_mcc05:.4f}   "
          f"best mcc = {max(gm_mccs):.4f} @ t={gm_best:.2f}")
    print(f"LightGBM            mcc@0.5 = {lgb_mcc05:.4f}   "
          f"best mcc = {max(lgb_mccs):.4f} @ t={lgb_best:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
