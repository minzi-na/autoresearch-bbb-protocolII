#!/usr/bin/env python3
"""
final_holdout_eval.py — 5-subset holdout breakdown (FROZEN, agent must not edit).

Retrains 10 seeds for one combo at the current architecture, then evaluates
each trained model on all 5 holdout subsets. Writes a JSON report and appends
a summary table to results/<combo>/architecture_log.md.

Aggregation:
  - val (scaffold split):  10-seed mean ± std of per-seed val_roc_auc / mcc.
                           Matches evaluate_combo.py's keep/discard criterion.
  - holdout subsets:       soft voting ensemble. For each subset, collect the
                           predicted probability vector from each of the 10
                           seed models, average them, then compute ROC-AUC /
                           MCC / AUPRC on the averaged probabilities. This
                           gives one ensemble metric per subset.

Run this ONLY after evaluate_combo.py has marked a KEEP iteration that the
agent believes is the new best. Not part of the iteration loop.

Usage:
    conda run -n rapids-25.02 python final_holdout_eval.py \
        --combo maccs+scage1+mole \
        --iter-id 7 \
        --note "best so far"

Optional:
    --commit <sha>      record this commit in the JSON (default: HEAD)
    --seeds 42,100,...  default 10 seeds, same as evaluate_combo.py
"""

import os
import sys
import gc
import json
import time
import argparse
import datetime
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
)

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import prepare        # noqa: E402
import train as train_mod  # noqa: E402


DEFAULT_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]

ALLOWED_COMBOS = [
    "maccs+avalon+scage2+mole",
    "maccs+scage1+mole",
    "maccs+scage1+scage2+mole",
]

SUBSETS = ["internal", "external", "nn03", "nn05", "total"]


def git_head_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=10", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def predict_probs(model, X_sub, y_sub, batch_size: int):
    """Return (y_true, y_prob) for one subset. Probs are sigmoid outputs."""
    loader = data.DataLoader(
        data.TensorDataset(
            torch.tensor(X_sub, dtype=torch.float32),
            torch.tensor(y_sub, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=False,
    )
    model.eval()
    device = next(model.parameters()).device
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(y.numpy().tolist())
    return np.array(y_true, dtype=np.int64), np.array(y_prob, dtype=np.float64)


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute classification metrics from probabilities + labels (single set)."""
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, _, _ = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    has_both = len(set(y_true.tolist())) > 1
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 6),
        "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":      round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1":          round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc":     round(float(roc_auc_score(y_true, y_prob) if has_both else 0.0), 6),
        "mcc":         round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "auprc":       round(float(average_precision_score(y_true, y_prob) if has_both else 0.0), 6),
        "specificity": round(float(specificity), 6),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combo",   required=True, choices=ALLOWED_COMBOS)
    p.add_argument("--iter-id", type=int, required=True,
                   help="The iter id this evaluation is associated with (matches results.tsv).")
    p.add_argument("--note",    type=str, default="")
    p.add_argument("--commit",  type=str, default=None)
    p.add_argument("--seeds",   type=str, default=None)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    commit = args.commit or git_head_commit()
    combo_tuple = prepare.combo_str_to_tuple(args.combo)

    print(f"[Plan] combo={args.combo}  iter={args.iter_id}  commit={commit}")
    print(f"[Plan] seeds={seeds}  subsets={SUBSETS}")
    print(f"[Plan] val_auc aggregation: 10-seed mean ± std")
    print(f"[Plan] holdout aggregation: soft voting ensemble (probs averaged across seeds)")

    print("[Data] Loading pool features ...")
    pool_smiles, pool_labels, _, pool_feats = prepare.load_pool_features()
    X_pool = prepare.build_feature_matrix(combo_tuple, pool_feats)
    print(f"[Data] pool size={len(pool_smiles)}  X_pool shape={X_pool.shape}")

    print("[Data] Loading 5 holdout subsets ...")
    sub_data = {}
    for s in SUBSETS:
        smi, lab, ft = prepare.load_holdout_subset(s)
        X_sub = prepare.build_feature_matrix(combo_tuple, ft)
        sub_data[s] = (smi, lab, X_sub)
        print(f"  {s:>8s}: n={len(smi)}, pos={int(lab.sum())}, X shape={X_sub.shape}")

    fp_dim = prepare.FP_DIM
    bs = train_mod.BASE_CONFIG["batch_size"]

    # Per-seed val metrics + per-seed subset probabilities
    per_seed = []
    # Accumulator for soft voting: subset → list of per-seed prob vectors
    subset_prob_stacks = {s: [] for s in SUBSETS}
    # Ground-truth labels (same across seeds — captured once)
    subset_y_true = {s: sub_data[s][1] for s in SUBSETS}

    t0 = time.time()
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")
        train_idx, val_idx = prepare.scaffold_split_train_val(
            pool_smiles, "scaffold", seed, 0.8
        )
        model, train_info, val_metrics, scaler = train_mod.build_and_train(
            combo=combo_tuple,
            X_pool=X_pool, y_pool=pool_labels,
            train_idx=train_idx, val_idx=val_idx,
            fp_dim=fp_dim, seed=seed,
        )
        print(f"  val_auc={val_metrics['roc_auc']:.6f}  "
              f"val_mcc={val_metrics['mcc']:.6f}  "
              f"best_epoch={train_info['best_epoch']}")

        # Per-seed subset metrics (for debugging / variance info, kept in JSON)
        per_seed_subsets = {}
        for s in SUBSETS:
            _, y_sub, X_sub = sub_data[s]
            X_sub_in = X_sub
            if scaler is not None and "rdkit" in combo_tuple:
                X_sub_in = train_mod.apply_rdkit_scaler(X_sub, combo_tuple, fp_dim, scaler)
            y_true, y_prob = predict_probs(model, X_sub_in, y_sub, bs)
            subset_prob_stacks[s].append(y_prob)
            m = metrics_from_probs(y_true, y_prob)
            per_seed_subsets[s] = m
            print(f"    {s:>8s}: per-seed roc_auc={m['roc_auc']:.6f}  mcc={m['mcc']:.6f}")

        per_seed.append({
            "seed": seed,
            "val":  val_metrics,
            "subsets": per_seed_subsets,
            "best_epoch": train_info["best_epoch"],
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    wall_min = (time.time() - t0) / 60.0

    # ── Aggregate ───────────────────────────────────────────────────────────
    def per_seed_mean_std(metric_key):
        """val metric: mean ± std across seeds."""
        vals = np.array([float(r["val"][metric_key]) for r in per_seed])
        return float(vals.mean()), float(vals.std(ddof=0))

    # val: 10-seed mean ± std (matches keep/discard criterion in evaluate_combo.py)
    val_mean_auc, val_std_auc = per_seed_mean_std("roc_auc")
    val_mean_mcc, val_std_mcc = per_seed_mean_std("mcc")

    # holdout subsets: soft voting ensemble
    subset_summary = {}
    for s in SUBSETS:
        probs_stack = np.stack(subset_prob_stacks[s], axis=0)        # (n_seeds, n_subset)
        ensemble_probs = probs_stack.mean(axis=0)                     # (n_subset,)
        ens_metrics = metrics_from_probs(subset_y_true[s], ensemble_probs)

        # Also report per-seed mean ± std as secondary debug info
        per_seed_aucs = np.array([float(r["subsets"][s]["roc_auc"]) for r in per_seed])
        per_seed_mccs = np.array([float(r["subsets"][s]["mcc"])     for r in per_seed])
        per_seed_prs  = np.array([float(r["subsets"][s]["auprc"])   for r in per_seed])

        subset_summary[s] = {
            "ensemble": ens_metrics,
            "per_seed_mean_roc_auc": round(float(per_seed_aucs.mean()), 6),
            "per_seed_std_roc_auc":  round(float(per_seed_aucs.std(ddof=0)), 6),
            "per_seed_mean_mcc":     round(float(per_seed_mccs.mean()), 6),
            "per_seed_std_mcc":      round(float(per_seed_mccs.std(ddof=0)), 6),
            "per_seed_mean_auprc":   round(float(per_seed_prs.mean()), 6),
            "per_seed_std_auprc":    round(float(per_seed_prs.std(ddof=0)), 6),
        }

    summary = {
        "iter":   args.iter_id,
        "commit": commit,
        "combo":  args.combo,
        "note":   args.note,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_seeds": len(seeds),
        "wall_time_min": round(wall_min, 2),
        "val": {
            "aggregation": "per_seed_mean_std",
            "mean_roc_auc": round(val_mean_auc, 6),
            "std_roc_auc":  round(val_std_auc, 6),
            "mean_mcc":     round(val_mean_mcc, 6),
            "std_mcc":      round(val_std_mcc, 6),
        },
        "subsets": {
            "aggregation": "soft_voting_ensemble",
            **subset_summary,
        },
    }

    out_dir = REPO_ROOT / "results" / args.combo / "holdout_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"iter{args.iter_id:04d}_{commit}.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)
    print(f"\n[Output] holdout eval JSON: {out_json}")

    # Append a row to architecture_log.md
    # val: mean±std, subsets: ensemble single value (no ± since ensemble is point estimate)
    log_path = REPO_ROOT / "results" / args.combo / "architecture_log.md"
    if not log_path.exists():
        log_path.write_text(
            f"# {args.combo} — architecture log\n\n"
            f"val: per-seed mean±std  |  subsets (int/ext/nn03/nn05/total): "
            f"soft voting ensemble single values.\n\n"
            "| iter | commit | val_auc | int | ext | nn03 | nn05 | total | note |\n"
            "|------|--------|---------|-----|-----|------|------|-------|------|\n"
        )
    row = (
        f"| {args.iter_id} | {commit} | "
        f"{val_mean_auc:.4f}±{val_std_auc:.4f} | "
        + " | ".join(
            f"{subset_summary[s]['ensemble']['roc_auc']:.4f}" for s in SUBSETS
        )
        + f" | {args.note} |\n"
    )
    with open(log_path, "a") as f:
        f.write(row)
    print(f"[Output] appended row to {log_path}")

    print("\n" + "=" * 60)
    print(f"combo={args.combo}  iter={args.iter_id}  commit={commit}")
    print(f"val (10-seed mean±std) roc_auc = {val_mean_auc:.6f} ± {val_std_auc:.6f}")
    for s in SUBSETS:
        ens = subset_summary[s]["ensemble"]
        ps_mean = subset_summary[s]["per_seed_mean_roc_auc"]
        ps_std  = subset_summary[s]["per_seed_std_roc_auc"]
        print(f"{s:>8s} ensemble roc_auc = {ens['roc_auc']:.6f}  "
              f"(per-seed {ps_mean:.6f} ± {ps_std:.6f})")
    print(f"wall_time = {wall_min:.2f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
