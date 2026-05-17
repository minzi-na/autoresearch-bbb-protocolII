#!/usr/bin/env python3
"""
final_holdout_eval.py — 5-subset holdout breakdown (FROZEN, agent must not edit).

Retrains 10 seeds for one combo at the current architecture, then evaluates
each trained model on all 5 holdout subsets. Writes a JSON report and appends
a summary table to results/<combo>/architecture_log.md.

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


def eval_on_subset(model, X_sub, y_sub, batch_size: int):
    loader = data.DataLoader(
        data.TensorDataset(
            torch.tensor(X_sub, dtype=torch.float32),
            torch.tensor(y_sub, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=False,
    )
    metrics, _, _ = train_mod.eval_model(model, loader)
    return metrics


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

    t0 = time.time()
    per_seed = []
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

        # Evaluate on each subset (applying the fitted rdkit scaler if any)
        subset_metrics = {}
        for s in SUBSETS:
            _, y_sub, X_sub = sub_data[s]
            X_sub_in = X_sub
            if scaler is not None and "rdkit" in combo_tuple:
                X_sub_in = train_mod.apply_rdkit_scaler(X_sub, combo_tuple, fp_dim, scaler)
            m = eval_on_subset(model, X_sub_in, y_sub, bs)
            subset_metrics[s] = m
            print(f"    {s:>8s}: roc_auc={m['roc_auc']:.6f}  mcc={m['mcc']:.6f}  "
                  f"auprc={m['auprc']:.6f}")

        per_seed.append({
            "seed": seed,
            "val":  val_metrics,
            "subsets": subset_metrics,
            "best_epoch": train_info["best_epoch"],
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    wall_min = (time.time() - t0) / 60.0

    def agg(metric_path):
        # metric_path: tuple like ("subsets", "internal", "roc_auc") or ("val", "roc_auc")
        vals = []
        for r in per_seed:
            v = r
            for k in metric_path:
                v = v[k]
            vals.append(float(v))
        a = np.array(vals)
        return float(a.mean()), float(a.std(ddof=0))

    summary = {
        "iter":   args.iter_id,
        "commit": commit,
        "combo":  args.combo,
        "note":   args.note,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_seeds": len(seeds),
        "wall_time_min": round(wall_min, 2),
        "val": {
            "mean_roc_auc": agg(("val", "roc_auc"))[0],
            "std_roc_auc":  agg(("val", "roc_auc"))[1],
            "mean_mcc":     agg(("val", "mcc"))[0],
            "std_mcc":      agg(("val", "mcc"))[1],
        },
        "subsets": {},
    }
    for s in SUBSETS:
        m_auc, sd_auc = agg(("subsets", s, "roc_auc"))
        m_mcc, sd_mcc = agg(("subsets", s, "mcc"))
        m_pr,  sd_pr  = agg(("subsets", s, "auprc"))
        summary["subsets"][s] = {
            "mean_roc_auc": round(m_auc, 6), "std_roc_auc": round(sd_auc, 6),
            "mean_mcc":     round(m_mcc, 6), "std_mcc":     round(sd_mcc, 6),
            "mean_auprc":   round(m_pr, 6),  "std_auprc":   round(sd_pr, 6),
        }

    out_dir = REPO_ROOT / "results" / args.combo / "holdout_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"iter{args.iter_id:04d}_{commit}.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)
    print(f"\n[Output] holdout eval JSON: {out_json}")

    # Append a row to architecture_log.md
    log_path = REPO_ROOT / "results" / args.combo / "architecture_log.md"
    if not log_path.exists():
        log_path.write_text(
            f"# {args.combo} — architecture log\n\n"
            "| iter | commit | val_auc | int | ext | nn03 | nn05 | total | note |\n"
            "|------|--------|---------|-----|-----|------|------|-------|------|\n"
        )
    row = (
        f"| {args.iter_id} | {commit} | "
        f"{summary['val']['mean_roc_auc']:.4f}±{summary['val']['std_roc_auc']:.4f} | "
        + " | ".join(
            f"{summary['subsets'][s]['mean_roc_auc']:.4f}±{summary['subsets'][s]['std_roc_auc']:.4f}"
            for s in SUBSETS
        )
        + f" | {args.note} |\n"
    )
    with open(log_path, "a") as f:
        f.write(row)
    print(f"[Output] appended row to {log_path}")

    print("\n" + "=" * 60)
    print(f"combo={args.combo}  iter={args.iter_id}  commit={commit}")
    print(f"val   roc_auc = {summary['val']['mean_roc_auc']:.6f} ± "
          f"{summary['val']['std_roc_auc']:.6f}")
    for s in SUBSETS:
        m = summary['subsets'][s]
        print(f"{s:>8s} roc_auc = {m['mean_roc_auc']:.6f} ± {m['std_roc_auc']:.6f}")
    print(f"wall_time = {wall_min:.2f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
