#!/usr/bin/env python3
"""
evaluate_combo.py — iteration runner (FROZEN, agent must not edit).

Runs 10 seeds × scaffold split for one combo and appends one row to
results/<combo>/results.tsv. keep/discard decision = mean(val_roc_auc).

Usage:
    conda run -n rapids-25.02 python evaluate_combo.py \
        --combo maccs+scage1+mole \
        --iter-id 1 \
        --note "baseline reproduction"

Optional:
    --seeds 42,100,200,...   (default: 10 seeds, same as baseline)
    --split-mode scaffold    (or random_scaffold; default scaffold)
    --no-append              (run but don't write to results.tsv)
    --commit <sha>           (override git HEAD commit recorded in TSV)
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

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import prepare   # noqa: E402
import train as train_mod  # noqa: E402


DEFAULT_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]

ALLOWED_COMBOS = [
    "maccs+avalon+scage2+mole",
    "maccs+scage1+mole",
    "maccs+scage1+scage2+mole",
]

TSV_COLUMNS = [
    "iter", "timestamp", "commit", "combo", "split_mode", "n_seeds",
    "mean_val_auc", "std_val_auc",
    "mean_val_mcc", "std_val_mcc",
    "mean_val_loss",
    "wall_time_min", "keep", "note",
]


def git_head_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=10", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def load_results_tsv(combo: str) -> pd.DataFrame:
    path = REPO_ROOT / "results" / combo / "results.tsv"
    if not path.exists():
        return pd.DataFrame(columns=TSV_COLUMNS)
    return pd.read_csv(path, sep="\t")


def current_best_val_auc(combo: str) -> float:
    df = load_results_tsv(combo)
    if df.empty:
        return float("-inf")
    keepers = df[df["keep"] == True]  # noqa: E712
    if keepers.empty:
        return float("-inf")
    return float(keepers["mean_val_auc"].max())


def append_row(combo: str, row: dict):
    path = REPO_ROOT / "results" / combo / "results.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_results_tsv(combo)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.reindex(columns=TSV_COLUMNS)
    df.to_csv(path, sep="\t", index=False)


def run_one_seed(combo_tuple, X_pool, y_pool, smiles, split_mode, seed, fp_dim):
    train_idx, val_idx = prepare.scaffold_split_train_val(smiles, split_mode, seed, 0.8)
    _, train_info, val_metrics, _ = train_mod.build_and_train(
        combo=combo_tuple,
        X_pool=X_pool, y_pool=y_pool,
        train_idx=train_idx, val_idx=val_idx,
        fp_dim=fp_dim, seed=seed,
    )

    best_epoch = train_info["best_epoch"]
    epoch_log = train_info.get("epoch_log", [])
    val_loss_at_best = None
    for r in epoch_log:
        if r["epoch"] == best_epoch:
            val_loss_at_best = r["val_loss"]
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "seed":       seed,
        "val_auc":    val_metrics["roc_auc"],
        "val_mcc":    val_metrics["mcc"],
        "val_loss":   val_loss_at_best,
        "best_epoch": best_epoch,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combo",      required=True, choices=ALLOWED_COMBOS)
    p.add_argument("--iter-id",    type=int, required=True,
                   help="Iteration counter for this combo (e.g. 1, 2, 3...)")
    p.add_argument("--note",       type=str, default="",
                   help="Short description of this iteration's architecture change.")
    p.add_argument("--seeds",      type=str, default=None,
                   help="Comma-separated seeds (default: 42,100,...,900).")
    p.add_argument("--split-mode", type=str, default="scaffold",
                   choices=["scaffold", "random_scaffold"])
    p.add_argument("--commit",     type=str, default=None)
    p.add_argument("--no-append",  action="store_true",
                   help="Skip writing the result row to TSV.")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    commit = args.commit or git_head_commit()
    combo_tuple = prepare.combo_str_to_tuple(args.combo)

    print(f"[Plan] combo={args.combo}  iter={args.iter_id}  commit={commit}")
    print(f"[Plan] seeds={seeds}  split_mode={args.split_mode}")
    print(f"[Plan] note=\"{args.note}\"")

    print("[Data] Loading pool features ...")
    pool_smiles, pool_labels, _, pool_feats = prepare.load_pool_features()
    X_pool = prepare.build_feature_matrix(combo_tuple, pool_feats)
    print(f"[Data] pool size={len(pool_smiles)}  X_pool shape={X_pool.shape}")

    t0 = time.time()
    per_seed = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")
        r = run_one_seed(
            combo_tuple, X_pool, pool_labels, pool_smiles,
            args.split_mode, seed, prepare.FP_DIM,
        )
        per_seed.append(r)
        print(f"  val_auc={r['val_auc']:.6f}  val_mcc={r['val_mcc']:.6f}  "
              f"val_loss={r['val_loss']}  best_epoch={r['best_epoch']}")

    wall_min = (time.time() - t0) / 60.0

    aucs   = np.array([r["val_auc"]   for r in per_seed], dtype=np.float64)
    mccs   = np.array([r["val_mcc"]   for r in per_seed], dtype=np.float64)
    losses = np.array([r["val_loss"] if r["val_loss"] is not None else np.nan
                       for r in per_seed], dtype=np.float64)

    mean_auc, std_auc = float(aucs.mean()), float(aucs.std(ddof=0))
    mean_mcc, std_mcc = float(mccs.mean()), float(mccs.std(ddof=0))
    mean_loss = float(np.nanmean(losses)) if np.isfinite(losses).any() else float("nan")

    best_so_far = current_best_val_auc(args.combo)
    keep = bool(mean_auc > best_so_far)

    print("\n" + "=" * 60)
    print(f"combo={args.combo}  iter={args.iter_id}  commit={commit}")
    print(f"mean_val_auc = {mean_auc:.6f} ± {std_auc:.6f}   "
          f"(prev best kept = {best_so_far:.6f})")
    print(f"mean_val_mcc = {mean_mcc:.6f} ± {std_mcc:.6f}")
    print(f"mean_val_loss = {mean_loss:.6f}")
    print(f"wall_time    = {wall_min:.2f} min")
    print(f"DECISION     = {'KEEP' if keep else 'DISCARD'}")
    print("=" * 60)

    row = {
        "iter":          args.iter_id,
        "timestamp":     datetime.datetime.now().isoformat(timespec="seconds"),
        "commit":        commit,
        "combo":         args.combo,
        "split_mode":    args.split_mode,
        "n_seeds":       len(seeds),
        "mean_val_auc":  round(mean_auc, 6),
        "std_val_auc":   round(std_auc, 6),
        "mean_val_mcc":  round(mean_mcc, 6),
        "std_val_mcc":   round(std_mcc, 6),
        "mean_val_loss": round(mean_loss, 6) if np.isfinite(mean_loss) else "",
        "wall_time_min": round(wall_min, 2),
        "keep":          keep,
        "note":          args.note,
    }

    if not args.no_append:
        append_row(args.combo, row)
        print(f"[Output] appended row to {REPO_ROOT / 'results' / args.combo / 'results.tsv'}")
    else:
        print("[Output] --no-append: row not written.")

    per_seed_path = REPO_ROOT / "results" / args.combo / f"iter{args.iter_id:04d}_per_seed.json"
    per_seed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(per_seed_path, "w") as f:
        json.dump({"iter": args.iter_id, "commit": commit, "combo": args.combo,
                   "split_mode": args.split_mode, "per_seed": per_seed,
                   "summary": row}, f, indent=2)
    print(f"[Output] per-seed JSON: {per_seed_path}")


if __name__ == "__main__":
    main()
