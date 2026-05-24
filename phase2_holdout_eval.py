#!/usr/bin/env python3
"""
phase2_holdout_eval.py — Phase-2 holdout evaluation runner.

For each Phase-2 iter (keep or discard), after evaluate_hpo.py finishes the
confirm phase, this script retrains 10 seeds at the iter's best confirm-trial
HP configuration and evaluates the trained models on a subset of holdout
splits. Mirrors final_holdout_eval.py's pattern but:
  - Uses best_train.build_and_train (accepts HPO config override)
  - Pulls the HP config from the iter's study_*.json (best top_k_confirm row)
  - Restricted to SUBSETS = ["nn05", "total"]

Outputs:
  results/<combo>/hpo/holdout_iter<NNNN>_<commit>.json   (full per-seed +
                                                          summary)
  results/<combo>/architecture_log.md                    (one row appended)

Usage:
    conda run -n rapids-25.02 python phase2_holdout_eval.py \\
        --combo maccs+scage1+mole --iter-id 3

    # Override commit (matches results.tsv commit column / study_name suffix)
    conda run -n rapids-25.02 python phase2_holdout_eval.py \\
        --combo maccs+scage1+mole --iter-id 3 --commit a539a46xxx
"""

import argparse
import datetime
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import prepare        # noqa: E402
import best_train as bt  # noqa: E402


DEFAULT_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]

ALLOWED_COMBOS = [
    "maccs+avalon+scage2+mole",
    "maccs+scage1+mole",
    "maccs+scage1+scage2+mole",
]

# Phase-2 restricts holdout breakdown to the two most informative subsets.
SUBSETS = ["nn05", "total"]


def git_head_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=10", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def find_study_json(combo: str, iter_id: int, commit: str) -> Path:
    """Scan results/<combo>/hpo/study_*.json and return the path whose JSON
    has study_name == f'auto_iter{iter_id:03d}_{commit}'.

    Raises FileNotFoundError if no match. If multiple match (shouldn't happen
    in practice), returns the most recently modified one.
    """
    d = REPO_ROOT / "results" / combo / "hpo"
    target = f"auto_iter{iter_id:03d}_{commit}"
    candidates = []
    for p in sorted(d.glob("study_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                j = json.load(f)
        except Exception:
            continue
        if j.get("study_name") == target:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            f"No study_*.json with study_name={target} in {d}"
        )
    return candidates[0]


def pick_best_confirm_params(study_json: dict) -> tuple[dict, dict]:
    """Return (params, debug_info) for the best confirm trial in the study."""
    rows = study_json.get("top_k_confirm", [])
    if not rows:
        raise RuntimeError("study JSON has empty top_k_confirm")
    best = max(rows, key=lambda r: r["confirm_mean_val_auc"])
    debug = {
        "trial": best["trial"],
        "rank": best["rank"],
        "search_mean_val_auc": best["search_mean_val_auc"],
        "confirm_mean_val_auc": best["confirm_mean_val_auc"],
        "confirm_std_val_auc": best["confirm_std_val_auc"],
    }
    return best["params"], debug


def predict_probs(model, X_sub, y_sub, batch_size: int):
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
                   help="Phase-2 iter id (matches results.tsv iter column).")
    p.add_argument("--commit",  type=str, default=None,
                   help="Commit short-10 used in the iter's study_name. "
                        "Defaults to git HEAD.")
    p.add_argument("--seeds",   type=str, default=None,
                   help="Comma-separated seed list. Default 10 confirm seeds.")
    p.add_argument("--note",    type=str, default="")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    commit = args.commit or git_head_commit()
    combo_tuple = prepare.combo_str_to_tuple(args.combo)

    print(f"[Plan] combo={args.combo}  iter={args.iter_id}  commit={commit}")
    print(f"[Plan] seeds={seeds}  subsets={SUBSETS}")

    study_path = find_study_json(args.combo, args.iter_id, commit)
    print(f"[Plan] study_json={study_path.name}")
    with open(study_path) as f:
        study_json = json.load(f)
    params, params_debug = pick_best_confirm_params(study_json)
    print(f"[Plan] best confirm trial #{params_debug['trial']} "
          f"(rank {params_debug['rank']}, confirm "
          f"{params_debug['confirm_mean_val_auc']:.6f} ± "
          f"{params_debug['confirm_std_val_auc']:.6f})")
    print(f"[Plan] config (HPO best): {params}")

    print("[Data] Loading pool features ...")
    pool_smiles, pool_labels, _, pool_feats = prepare.load_pool_features()
    X_pool = prepare.build_feature_matrix(combo_tuple, pool_feats)
    print(f"[Data] pool size={len(pool_smiles)}  X_pool shape={X_pool.shape}")

    print(f"[Data] Loading holdout subsets {SUBSETS} ...")
    sub_data = {}
    for s in SUBSETS:
        smi, lab, ft = prepare.load_holdout_subset(s)
        X_sub = prepare.build_feature_matrix(combo_tuple, ft)
        sub_data[s] = (smi, lab, X_sub)
        print(f"  {s:>6s}: n={len(smi)}, pos={int(lab.sum())}, X shape={X_sub.shape}")

    fp_dim = prepare.FP_DIM
    bs = params.get("batch_size", bt.BASE_CONFIG["batch_size"])

    per_seed = []
    subset_prob_stacks = {s: [] for s in SUBSETS}
    subset_y_true = {s: sub_data[s][1] for s in SUBSETS}

    t0 = time.time()
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")
        train_idx, val_idx = prepare.scaffold_split_train_val(
            pool_smiles, "scaffold", seed, 0.8,
        )
        model, train_info, val_metrics, scaler = bt.build_and_train(
            combo=combo_tuple,
            X_pool=X_pool, y_pool=pool_labels,
            train_idx=train_idx, val_idx=val_idx,
            fp_dim=fp_dim, seed=seed,
            config=params,
        )
        print(f"  val_auc={val_metrics['roc_auc']:.6f}  "
              f"val_mcc={val_metrics['mcc']:.6f}  "
              f"best_epoch={train_info['best_epoch']}")

        per_seed_subsets = {}
        for s in SUBSETS:
            _, y_sub, X_sub = sub_data[s]
            X_sub_in = X_sub
            if scaler is not None and "rdkit" in combo_tuple:
                X_sub_in = bt.apply_rdkit_scaler(X_sub, combo_tuple, fp_dim, scaler)
            y_true, y_prob = predict_probs(model, X_sub_in, y_sub, bs)
            subset_prob_stacks[s].append(y_prob)
            m = metrics_from_probs(y_true, y_prob)
            per_seed_subsets[s] = m
            print(f"    {s:>6s}: per-seed roc_auc={m['roc_auc']:.6f}  "
                  f"mcc={m['mcc']:.6f}  acc={m['accuracy']:.6f}")

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
    def per_seed_mean_std(metric_key, source="val"):
        if source == "val":
            vals = np.array([float(r["val"][metric_key]) for r in per_seed])
        else:
            vals = np.array([float(r["subsets"][source][metric_key]) for r in per_seed])
        return float(vals.mean()), float(vals.std(ddof=0))

    val_mean_auc, val_std_auc = per_seed_mean_std("roc_auc")
    val_mean_mcc, val_std_mcc = per_seed_mean_std("mcc")

    subset_summary = {}
    for s in SUBSETS:
        probs_stack = np.stack(subset_prob_stacks[s], axis=0)
        ensemble_probs = probs_stack.mean(axis=0)
        ens_metrics = metrics_from_probs(subset_y_true[s], ensemble_probs)

        ps_aucs = np.array([float(r["subsets"][s]["roc_auc"])    for r in per_seed])
        ps_mccs = np.array([float(r["subsets"][s]["mcc"])        for r in per_seed])
        ps_accs = np.array([float(r["subsets"][s]["accuracy"])   for r in per_seed])
        ps_prs  = np.array([float(r["subsets"][s]["auprc"])      for r in per_seed])

        subset_summary[s] = {
            "ensemble": ens_metrics,
            "per_seed_mean_roc_auc":  round(float(ps_aucs.mean()), 6),
            "per_seed_std_roc_auc":   round(float(ps_aucs.std(ddof=0)), 6),
            "per_seed_mean_mcc":      round(float(ps_mccs.mean()), 6),
            "per_seed_std_mcc":       round(float(ps_mccs.std(ddof=0)), 6),
            "per_seed_mean_accuracy": round(float(ps_accs.mean()), 6),
            "per_seed_std_accuracy":  round(float(ps_accs.std(ddof=0)), 6),
            "per_seed_mean_auprc":    round(float(ps_prs.mean()), 6),
            "per_seed_std_auprc":     round(float(ps_prs.std(ddof=0)), 6),
        }

    summary = {
        "iter":    args.iter_id,
        "commit":  commit,
        "combo":   args.combo,
        "note":    args.note,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_seeds":      len(seeds),
        "wall_time_min": round(wall_min, 2),
        "study_name":   study_json.get("study_name"),
        "best_confirm_trial": params_debug,
        "config_used":  params,
        "val": {
            "aggregation": "per_seed_mean_std",
            "mean_roc_auc": round(val_mean_auc, 6),
            "std_roc_auc":  round(val_std_auc, 6),
            "mean_mcc":     round(val_mean_mcc, 6),
            "std_mcc":      round(val_std_mcc, 6),
        },
        "subsets": {
            "aggregation": "soft_voting_ensemble + per_seed_mean_std",
            **subset_summary,
        },
    }

    out_dir = REPO_ROOT / "results" / args.combo / "hpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"holdout_iter{args.iter_id:04d}_{commit}.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)
    print(f"\n[Output] holdout eval JSON: {out_json}")

    # Append a row to architecture_log.md (per-seed mean±std for AUC/MCC/ACC)
    log_path = REPO_ROOT / "results" / args.combo / "architecture_log.md"
    if not log_path.exists():
        log_path.write_text(
            f"# {args.combo} — architecture log\n\n"
            "val: per-seed mean±std  |  subsets (nn05/total): "
            "per-seed mean±std for AUC/MCC/Acc.\n\n"
            "| iter | commit | val_auc | "
            "nn05_auc | nn05_mcc | nn05_acc | "
            "total_auc | total_mcc | total_acc | note |\n"
            "|------|--------|---------|"
            "----------|----------|----------|"
            "-----------|-----------|-----------|------|\n"
        )
    def fmt(mean, std):
        return f"{mean:.4f}±{std:.4f}"
    nn05 = subset_summary["nn05"]
    tot  = subset_summary["total"]
    row = (
        f"| {args.iter_id} | {commit} | "
        f"{fmt(val_mean_auc, val_std_auc)} | "
        f"{fmt(nn05['per_seed_mean_roc_auc'], nn05['per_seed_std_roc_auc'])} | "
        f"{fmt(nn05['per_seed_mean_mcc'],     nn05['per_seed_std_mcc'])} | "
        f"{fmt(nn05['per_seed_mean_accuracy'],nn05['per_seed_std_accuracy'])} | "
        f"{fmt(tot['per_seed_mean_roc_auc'],  tot['per_seed_std_roc_auc'])} | "
        f"{fmt(tot['per_seed_mean_mcc'],      tot['per_seed_std_mcc'])} | "
        f"{fmt(tot['per_seed_mean_accuracy'], tot['per_seed_std_accuracy'])} | "
        f"{args.note} |\n"
    )
    with open(log_path, "a") as f:
        f.write(row)
    print(f"[Output] appended row to {log_path}")

    print("\n" + "=" * 60)
    print(f"combo={args.combo}  iter={args.iter_id}  commit={commit}")
    print(f"val (10-seed mean±std) roc_auc = {val_mean_auc:.6f} ± {val_std_auc:.6f}")
    for s in SUBSETS:
        ss = subset_summary[s]
        print(f"{s:>6s}: per-seed AUC = {ss['per_seed_mean_roc_auc']:.6f} ± {ss['per_seed_std_roc_auc']:.6f}  "
              f"MCC = {ss['per_seed_mean_mcc']:.6f} ± {ss['per_seed_std_mcc']:.6f}  "
              f"Acc = {ss['per_seed_mean_accuracy']:.6f} ± {ss['per_seed_std_accuracy']:.6f}  "
              f"(ensemble AUC = {ss['ensemble']['roc_auc']:.6f})")
    print(f"wall_time = {wall_min:.2f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
