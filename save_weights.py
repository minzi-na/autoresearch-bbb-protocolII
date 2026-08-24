#!/usr/bin/env python3
"""
save_weights.py - one-off weight snapshot for a (combo, iter) run.

train.py and evaluate_combo.py are frozen and discard model weights after
evaluation. This script reruns the same per-seed training loop and saves each
seed's state_dict + scaler to results/<combo>/weights/iter<NNNN>/.

Usage:
    conda run -n rapids-25.02 python save_weights.py \
        --combo maccs+avalon+scage2+mole \
        --iter-id 93 \
        --label-commit 663cbb1e3 \
        --note "iter93 best (DROP_V_PATH 0.08, val_auc=0.8536)"

The current git HEAD is recorded as head_commit; --label-commit lets you
record the original iteration's commit too (useful when reverts brought HEAD
back to the same effective state).
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
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import prepare           # noqa: E402
import train as train_mod  # noqa: E402


DEFAULT_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]


def git_head_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=10", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combo",        required=True)
    p.add_argument("--iter-id",      type=int, required=True)
    p.add_argument("--note",         type=str, default="")
    p.add_argument("--seeds",        type=str, default=None,
                   help="Comma-separated seeds (default: 42,100,...,900).")
    p.add_argument("--split-mode",   type=str, default="scaffold",
                   choices=["scaffold", "random_scaffold"])
    p.add_argument("--label-commit", type=str, default=None,
                   help="Logical commit to record (e.g. the original iter "
                        "commit if HEAD differs but is equivalent).")
    p.add_argument("--out-dir",      type=str, default=None)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    head_commit  = git_head_commit()
    label_commit = args.label_commit or head_commit
    combo_tuple  = prepare.combo_str_to_tuple(args.combo)

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "results" / args.combo / "weights" / f"iter{args.iter_id:04d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Plan] combo={args.combo}  iter={args.iter_id}")
    print(f"[Plan] head_commit={head_commit}  label_commit={label_commit}")
    print(f"[Plan] seeds={seeds}  split_mode={args.split_mode}")
    print(f"[Plan] out_dir={out_dir}")
    print(f"[Plan] note=\"{args.note}\"")

    print("[Data] Loading pool features ...")
    pool_smiles, pool_labels, _, pool_feats = prepare.load_pool_features()
    X_pool = prepare.build_feature_matrix(combo_tuple, pool_feats)
    print(f"[Data] pool size={len(pool_smiles)}  X_pool shape={X_pool.shape}")

    t0 = time.time()
    per_seed = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")
        train_idx, val_idx = prepare.scaffold_split_train_val(
            pool_smiles, args.split_mode, seed, 0.8,
        )

        model, train_info, val_metrics, scaler = train_mod.build_and_train(
            combo=combo_tuple,
            X_pool=X_pool, y_pool=pool_labels,
            train_idx=train_idx, val_idx=val_idx,
            fp_dim=prepare.FP_DIM, seed=seed,
        )

        ckpt_path = out_dir / f"seed{seed:04d}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "scaler":     scaler,
            "combo":      args.combo,
            "seed":       seed,
            "best_epoch": train_info["best_epoch"],
            "val_auc":    val_metrics["roc_auc"],
            "val_mcc":    val_metrics["mcc"],
        }, ckpt_path)

        print(f"  val_auc={val_metrics['roc_auc']:.6f}  "
              f"best_epoch={train_info['best_epoch']}  "
              f"saved -> {ckpt_path.name}")

        per_seed.append({
            "seed":       seed,
            "val_auc":    val_metrics["roc_auc"],
            "val_mcc":    val_metrics["mcc"],
            "best_epoch": train_info["best_epoch"],
            "ckpt":       str(ckpt_path.relative_to(REPO_ROOT)),
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    wall_min = (time.time() - t0) / 60.0
    aucs = np.array([r["val_auc"] for r in per_seed], dtype=np.float64)

    print("\n" + "=" * 60)
    print(f"mean_val_auc = {aucs.mean():.6f} ± {aucs.std(ddof=0):.6f}")
    print(f"wall_time    = {wall_min:.2f} min")
    print("=" * 60)

    meta = {
        "combo":         args.combo,
        "iter_id":       args.iter_id,
        "head_commit":   head_commit,
        "label_commit":  label_commit,
        "split_mode":    args.split_mode,
        "seeds":         seeds,
        "note":          args.note,
        "timestamp":     datetime.datetime.now().isoformat(timespec="seconds"),
        "mean_val_auc":  round(float(aucs.mean()), 6),
        "std_val_auc":   round(float(aucs.std(ddof=0)), 6),
        "wall_time_min": round(wall_min, 2),
        "per_seed":      per_seed,
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Output] meta -> {meta_path}")


if __name__ == "__main__":
    main()
