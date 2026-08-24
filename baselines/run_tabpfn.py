#!/usr/bin/env python3
"""
run_tabpfn.py — TabPFN-2.5 baseline on the SAME data and split as GateMol-BBB
(autoresearch_combos_v2), for a single combo.

Mirrors run_baselines.py: identical pool/holdout loaders, identical scaffold
8:2 split per seed, 10 seeds, identical JSON+TSV output layout (so existing
figure scripts can read it as just another baseline model).

Differences vs MLP/XGB/LGBM baselines:
  - TabPFN is in-context (no real training); fit() just memorizes train tensor.
  - n_estimators=8 ensembling (random feature permutations) per seed.
  - No rdkit StandardScaler call: the three combos in ALLOWED_COMBOS do not
    include rdkit (run_baselines.py applies it only if combo has 'rdkit').

Outputs:
  results/<combo>/baselines/tabpfn.json   — per-seed + summary
  results/<combo>/baselines/summary.tsv   — one row appended/updated

Usage:
  conda run -n tabpfn python baselines/run_tabpfn.py --combo maccs+scage1+mole
"""

import sys
import gc
import json
import time
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare  # noqa: E402
from baselines._common import (  # noqa: E402
    DEFAULT_SEEDS, ALLOWED_COMBOS, SUBSETS,
    metrics_from_probs, get_rdkit_slice, fit_apply_rdkit_scaler,
    update_summary_tsv,
)


def run_tabpfn(combo_tuple, X_pool, y_pool, pool_smiles, sub_data, fp_dim,
               seeds, n_estimators, ignore_pretraining_limits, device,
               artifacts_dir=None, subsets=None):
    """Train+evaluate TabPFN-2.5 across all seeds and subsets.

    `subsets` (list[str]) restricts which holdout subsets to evaluate.
    Defaults to the full SUBSETS list. Subsets passed here MUST be a
    subset of SUBSETS; sub_data must contain entries for every name in
    `subsets`.

    Returns (summary_dict, per_seed_list) — same shape as
    run_baselines.run_one_model so update_summary_tsv works unchanged.

    If `artifacts_dir` is not None, also save per-seed artifacts there for
    later `predict_tabpfn.py` reuse:
      - seed_XXXX_train_idx.npz : train_idx + holdout probability arrays
      - meta.json               : combo, seeds, n_estimators, etc.
    """
    from tabpfn import TabPFNClassifier

    rd_slice = get_rdkit_slice(combo_tuple, fp_dim)

    if subsets is None:
        subsets = list(SUBSETS)
    unknown = [s for s in subsets if s not in SUBSETS]
    if unknown:
        raise ValueError(f"Unknown subsets: {unknown}")

    per_seed = []
    subset_prob_stacks = {s: [] for s in subsets}
    subset_y_true = {s: sub_data[s][1] for s in subsets}

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    for i, seed in enumerate(seeds):
        t_seed = time.time()
        print(f"  [{i+1}/{len(seeds)}] seed={seed} ", end="", flush=True)
        train_idx, val_idx = prepare.scaffold_split_train_val(
            pool_smiles, "scaffold", seed, 0.8)

        X_train = X_pool[train_idx].astype(np.float32, copy=True)
        y_train = y_pool[train_idx].astype(np.int64, copy=True)
        X_val   = X_pool[val_idx].astype(np.float32, copy=True)
        y_val   = y_pool[val_idx].astype(np.int64, copy=True)

        X_subs = {s: sub_data[s][2].astype(np.float32, copy=True) for s in subsets}

        # rdkit scaler — applies only if combo has rdkit; otherwise no-op.
        fit_apply_rdkit_scaler(
            X_train, [X_val] + [X_subs[s] for s in subsets], rd_slice,
        )

        clf = TabPFNClassifier(
            n_estimators=n_estimators,
            random_state=int(seed),
            ignore_pretraining_limits=ignore_pretraining_limits,
            device=device,
        )
        clf.fit(X_train, y_train)

        # Per-seed val metrics
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_metrics = metrics_from_probs(y_val, val_probs)

        # Per-seed subset metrics (+ stash for soft voting)
        per_seed_subsets = {}
        subset_probs_to_save = {}
        for s in subsets:
            probs = clf.predict_proba(X_subs[s])[:, 1]
            subset_prob_stacks[s].append(probs)
            per_seed_subsets[s] = metrics_from_probs(subset_y_true[s], probs)
            subset_probs_to_save[f"probs_{s}"] = probs.astype(np.float32)

        wall = time.time() - t_seed
        per_seed.append({
            "seed": int(seed),
            "val":  val_metrics,
            "subsets": per_seed_subsets,
            "train_info": {
                "n_estimators": int(n_estimators),
                "n_train": int(len(train_idx)),
                "n_features": int(X_train.shape[1]),
                "wall_s": round(float(wall), 2),
            },
        })

        # ── Save per-seed artifacts (train indices + holdout probs) ──
        if artifacts_dir is not None:
            np.savez(
                artifacts_dir / f"seed_{int(seed):04d}.npz",
                train_idx=np.asarray(train_idx, dtype=np.int64),
                val_idx=np.asarray(val_idx,   dtype=np.int64),
                val_probs=val_probs.astype(np.float32),
                **subset_probs_to_save,
            )
        print(f"val_auc={val_metrics['roc_auc']:.4f}  mcc={val_metrics['mcc']:.4f}  "
              f"({wall:.1f}s)")

        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def mean_std(metric_key):
        vals = np.array([float(r["val"][metric_key]) for r in per_seed])
        return float(vals.mean()), float(vals.std(ddof=0))

    val_mean_auc, val_std_auc = mean_std("roc_auc")
    val_mean_mcc, val_std_mcc = mean_std("mcc")

    subset_summary = {}
    for s in subsets:
        probs_stack = np.stack(subset_prob_stacks[s], axis=0)
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
    ap.add_argument("--seeds",  type=str, default=None,
                    help="Comma-separated seeds (default: 42,100,...,900).")
    ap.add_argument("--n-estimators", type=int, default=8,
                    help="TabPFN ensemble size (default: 8, matches bbb-combo1).")
    ap.add_argument("--ignore-pretraining-limits", action="store_true",
                    help="Pass through to TabPFNClassifier (off by default).")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda/cpu (default: auto).")
    ap.add_argument("--model-name", type=str, default="tabpfn",
                    help="Name for output files (default: tabpfn).")
    ap.add_argument("--no-save-artifacts", action="store_true",
                    help="Skip per-seed train_idx/probs npz dump "
                         "(default: save under <combo>/baselines/<model>_artifacts/).")
    ap.add_argument("--subsets", nargs="+", default=None,
                    choices=SUBSETS,
                    help="Holdout subsets to evaluate "
                         "(default: all 5: internal external nn03 nn05 total). "
                         "Pass e.g. `--subsets nn05 total` to skip the others "
                         "for ~23%% faster runs.")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    combo_tuple = prepare.combo_str_to_tuple(args.combo)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_subsets = args.subsets if args.subsets else list(SUBSETS)

    print(f"[Plan] combo={args.combo}  model={args.model_name}  seeds={seeds}")
    print(f"[Plan] n_estimators={args.n_estimators}  "
          f"ignore_pretraining_limits={args.ignore_pretraining_limits}  device={device}")
    print(f"[Plan] subsets={eval_subsets}")

    print("[Data] Loading pool features ...")
    pool_smiles, pool_labels, _, pool_feats = prepare.load_pool_features()
    X_pool = prepare.build_feature_matrix(combo_tuple, pool_feats)
    print(f"[Data] pool size={len(pool_smiles)}  X_pool shape={X_pool.shape}")

    print(f"[Data] Loading {len(eval_subsets)} holdout subset(s) ...")
    sub_data = {}
    for s in eval_subsets:
        smi, lab, ft = prepare.load_holdout_subset(s)
        X_sub = prepare.build_feature_matrix(combo_tuple, ft)
        sub_data[s] = (smi, lab, X_sub)
        print(f"  {s:>8s}: n={len(smi)}, pos={int(lab.sum())}, X shape={X_sub.shape}")

    out_dir = REPO_ROOT / "results" / args.combo / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_tsv = out_dir / "summary.tsv"

    artifacts_dir = (out_dir / f"{args.model_name}_artifacts"
                     if not args.no_save_artifacts else None)
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "meta.json", "w") as f:
            json.dump({
                "combo": args.combo,
                "combo_tuple": list(combo_tuple),
                "fp_dims": {k: int(prepare.FP_DIM[k]) for k in combo_tuple},
                "pool_size": int(len(pool_smiles)),
                "seeds": seeds,
                "eval_subsets": eval_subsets,
                "tabpfn_config": {
                    "version": "2.5",
                    "n_estimators": int(args.n_estimators),
                    "ignore_pretraining_limits": bool(args.ignore_pretraining_limits),
                    "device": device,
                    "model_path": "/home/minji/.cache/tabpfn/"
                                  "tabpfn-v2.5-classifier-v2.5_default.ckpt",
                },
            }, f, indent=2)
        print(f"[Artifacts] meta -> {artifacts_dir/'meta.json'}")

    print(f"\n[Model] {args.model_name}  (TabPFN-2.5)")
    t0 = time.time()
    summary, per_seed = run_tabpfn(
        combo_tuple, X_pool, pool_labels, pool_smiles,
        sub_data, prepare.FP_DIM, seeds,
        args.n_estimators, args.ignore_pretraining_limits, device,
        artifacts_dir=artifacts_dir,
        subsets=eval_subsets,
    )
    wall_min = (time.time() - t0) / 60.0
    ts = datetime.datetime.now().isoformat(timespec="seconds")

    out_json = out_dir / f"{args.model_name}.json"
    with open(out_json, "w") as f:
        json.dump({
            "combo": args.combo,
            "model": args.model_name,
            "timestamp": ts,
            "wall_time_min": round(wall_min, 2),
            "seeds": seeds,
            "eval_subsets": eval_subsets,
            "tabpfn_config": {
                "version": "2.5",
                "n_estimators": int(args.n_estimators),
                "ignore_pretraining_limits": bool(args.ignore_pretraining_limits),
                "device": device,
            },
            "summary": summary,
            "per_seed": per_seed,
        }, f, indent=2)
    print(f"  -> {out_json}")
    update_summary_tsv(summary_tsv, args.combo, args.model_name,
                       summary, wall_min, ts)

    val_mean = summary["val"]["mean_roc_auc"]
    val_std  = summary["val"]["std_roc_auc"]
    print(f"  val_auc (10-seed mean ± std) = {val_mean:.6f} ± {val_std:.6f}")
    for s in eval_subsets:
        ens = summary["subsets"][s]["ensemble"]
        print(f"  holdout {s:>8s}  ensemble roc_auc = {ens['roc_auc']:.6f}  "
              f"mcc = {ens['mcc']:.6f}  f1 = {ens['f1']:.6f}  acc = {ens['accuracy']:.6f}")
    print(f"  wall = {wall_min:.2f} min")

    print(f"\n[Done] summary -> {summary_tsv}")


if __name__ == "__main__":
    main()
