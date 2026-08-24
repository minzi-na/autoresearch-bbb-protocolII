#!/usr/bin/env python3
"""
run_baselines.py — train + evaluate baseline models on the SAME data and split
as GateMol-BBB (autoresearch_combos_v2), for a single combo.

Protocol (mirrors evaluate_combo.py + final_holdout_eval.py):
  - Training pool: prepare.load_pool_features() (internal+external remaining, deduped)
  - Per-seed scaffold split 80/20 via prepare.scaffold_split_train_val(seed, 0.8)
  - 10 seeds (default 42,100,...,900)
  - Validation: per-seed val_roc_auc / val_mcc, reported as 10-seed mean ± std
    (this is the comparison metric vs. GateMol-BBB's mean_val_auc)
  - Holdout 5 subsets: each seed model predicts P(1) on each subset; the 10
    per-seed probability vectors are averaged (soft voting) and the ensemble
    metrics (ROC-AUC / MCC / F1 / ACC / AUPRC / specificity) are computed on
    the averaged probabilities — same aggregation as final_holdout_eval.py.

Outputs:
  results/<combo>/baselines/<model>.json   — per-seed + summary (1 file per model)
  results/<combo>/baselines/summary.tsv    — one row per model in this combo

Usage:
  conda run -n rapids-25.02 python baselines/run_baselines.py \
    --combo maccs+scage1+mole

  conda run -n rapids-25.02 python baselines/run_baselines.py \
    --combo maccs+scage1+mole --models mlp2 lightgbm
"""

import sys
import gc
import json
import time
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare  # noqa: E402
from baselines import models as bm  # noqa: E402
from baselines._common import (  # noqa: E402
    DEFAULT_SEEDS, ALLOWED_COMBOS, SUBSETS,
    metrics_from_probs, get_rdkit_slice, fit_apply_rdkit_scaler,
    update_summary_tsv,
)


def run_one_model(model_name: str, combo_tuple, X_pool, y_pool, pool_smiles,
                  sub_data, fp_dim, seeds):
    """Train+evaluate ONE baseline model across all seeds and subsets.

    Returns (summary_dict, per_seed_list).
    sub_data: {short_subset_name: (smiles, labels, X_sub)}
    """
    rd_slice = get_rdkit_slice(combo_tuple, fp_dim)

    per_seed = []
    subset_prob_stacks = {s: [] for s in SUBSETS}
    subset_y_true = {s: sub_data[s][1] for s in SUBSETS}

    for i, seed in enumerate(seeds):
        print(f"  [{i+1}/{len(seeds)}] seed={seed} ", end="", flush=True)
        train_idx, val_idx = prepare.scaffold_split_train_val(
            pool_smiles, "scaffold", seed, 0.8)

        X_train = X_pool[train_idx].astype(np.float32, copy=True)
        y_train = y_pool[train_idx].astype(np.float32, copy=True)
        X_val   = X_pool[val_idx].astype(np.float32, copy=True)
        y_val   = y_pool[val_idx].astype(np.float32, copy=True)

        X_subs = {s: sub_data[s][2].astype(np.float32, copy=True) for s in SUBSETS}

        # Fit rdkit scaler on training subset, apply to val + subsets
        fit_apply_rdkit_scaler(
            X_train, [X_val] + [X_subs[s] for s in SUBSETS], rd_slice,
        )

        # Train
        model, train_info = bm.train_one(
            model_name, X_train, y_train, X_val, y_val, seed,
        )

        # Per-seed val metrics
        val_probs = bm.predict_probs(model_name, model, X_val)
        val_metrics = metrics_from_probs(y_val.astype(int), val_probs)

        # Per-seed subset metrics (and stash probs for soft voting)
        per_seed_subsets = {}
        for s in SUBSETS:
            probs = bm.predict_probs(model_name, model, X_subs[s])
            subset_prob_stacks[s].append(probs)
            per_seed_subsets[s] = metrics_from_probs(subset_y_true[s], probs)

        per_seed.append({
            "seed": int(seed),
            "val":  val_metrics,
            "subsets": per_seed_subsets,
            "train_info": train_info,
        })
        print(f"val_auc={val_metrics['roc_auc']:.4f}  mcc={val_metrics['mcc']:.4f}")

        # Cleanup (especially for MLP on GPU)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Aggregate ────────────────────────────────────────────────────────────
    def mean_std(metric_key):
        vals = np.array([float(r["val"][metric_key]) for r in per_seed])
        return float(vals.mean()), float(vals.std(ddof=0))

    val_mean_auc, val_std_auc = mean_std("roc_auc")
    val_mean_mcc, val_std_mcc = mean_std("mcc")

    subset_summary = {}
    for s in SUBSETS:
        probs_stack = np.stack(subset_prob_stacks[s], axis=0)        # (n_seeds, n_sub)
        ensemble_probs = probs_stack.mean(axis=0)
        ens_metrics = metrics_from_probs(subset_y_true[s], ensemble_probs)
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
        "n_seeds": len(seeds),
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
    return summary, per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo",  required=True, choices=ALLOWED_COMBOS)
    ap.add_argument("--models", nargs="+", default=bm.ALL_MODELS,
                    choices=bm.ALL_MODELS,
                    help="Subset of baseline models (default: all 5).")
    ap.add_argument("--seeds",  type=str, default=None,
                    help="Comma-separated seeds (default: 42,100,...,900).")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    combo_tuple = prepare.combo_str_to_tuple(args.combo)

    print(f"[Plan] combo={args.combo}  models={args.models}  seeds={seeds}")

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

    out_dir = REPO_ROOT / "results" / args.combo / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_tsv = out_dir / "summary.tsv"

    for model_name in args.models:
        print(f"\n[Model] {model_name}")
        t0 = time.time()
        summary, per_seed = run_one_model(
            model_name, combo_tuple, X_pool, pool_labels, pool_smiles,
            sub_data, prepare.FP_DIM, seeds,
        )
        wall_min = (time.time() - t0) / 60.0
        ts = datetime.datetime.now().isoformat(timespec="seconds")

        out_json = out_dir / f"{model_name}.json"
        with open(out_json, "w") as f:
            json.dump({
                "combo": args.combo,
                "model": model_name,
                "timestamp": ts,
                "wall_time_min": round(wall_min, 2),
                "seeds": seeds,
                "summary": summary,
                "per_seed": per_seed,
            }, f, indent=2)
        print(f"  -> {out_json}")
        update_summary_tsv(summary_tsv, args.combo, model_name,
                           summary, wall_min, ts)

        val_mean = summary["val"]["mean_roc_auc"]
        val_std  = summary["val"]["std_roc_auc"]
        print(f"  val_auc (10-seed mean ± std) = {val_mean:.6f} ± {val_std:.6f}")
        for s in SUBSETS:
            ens = summary["subsets"][s]["ensemble"]
            print(f"  holdout {s:>8s}  ensemble roc_auc = {ens['roc_auc']:.6f}  "
                  f"mcc = {ens['mcc']:.6f}  f1 = {ens['f1']:.6f}  acc = {ens['accuracy']:.6f}")
        print(f"  wall = {wall_min:.2f} min")

    print(f"\n[Done] summary -> {summary_tsv}")


if __name__ == "__main__":
    main()
