#!/usr/bin/env python3
"""
optuna_combo.py — Phase 2 HPO entry point for one feature combo.

Loads the frozen iter-200 architecture from best_train.py and runs an
Optuna study that injects config overrides on top of BASE_CONFIG.

Staged evaluation:
  - search phase: N trials, each evaluated on M search seeds (default 3).
    Objective = mean scaffold val ROC-AUC.
  - confirm phase: top-K trials re-evaluated on the full 10-seed set.

Usage:
    conda run -n rapids-25.02 python optuna_combo.py \\
        --combo maccs+scage1+mole \\
        --n-trials 50 --top-k 3 \\
        --study-name combo2_phase2_v1

Optional:
    --search-seeds 42,100,200
    --confirm-seeds 42,100,200,300,400,500,600,700,800,900
    --search-num-epochs 30   (override BASE_CONFIG.num_epochs for trial budget)
    --storage sqlite:///hpo.db
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import prepare  # noqa: E402
import best_train as bt  # noqa: E402


SEARCH_SEEDS_DEFAULT = [42, 100, 200]
CONFIRM_SEEDS_DEFAULT = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
SPLIT_MODE = "scaffold"


# ─────────────────────────────────────────────────────────────────────────────
#  Search space
# ─────────────────────────────────────────────────────────────────────────────

def suggest_config(trial: optuna.Trial) -> dict:
    """Return a config dict layered on top of BASE_CONFIG.

    Ranges are centered on the iter-200 BASE_CONFIG values. d_model/d_ffn/
    depth are kept narrow because the iter-200 architecture was tuned with
    those structural values fixed; widening the structural range too much
    would re-open the phase-1 search.
    """
    return {
        # Optimizer
        "lr":            trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
        "grad_clip_max_norm": trial.suggest_float("grad_clip_max_norm", 0.5, 2.0),
        # Regularization
        "dropout":       trial.suggest_float("dropout", 0.0, 0.4),
        "drop_path":     trial.suggest_float("drop_path", 0.0, 0.1),
        "mod_drop_p":    trial.suggest_float("mod_drop_p", 0.0, 0.3),
        "head_dropout":  trial.suggest_float("head_dropout", 0.0, 0.3),
        # Structure-adjacent (narrow)
        "d_model":       trial.suggest_categorical("d_model", [384, 512, 768]),
        "d_ffn":         trial.suggest_categorical("d_ffn", [768, 1048, 1536]),
        "depth":         trial.suggest_int("depth", 3, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Per-seed evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_seeds(combo_tuple, X_pool, y_pool, smiles, fp_dim, seeds, config, trial=None):
    """Train + evaluate one config across `seeds`. Returns list of val ROC-AUC.

    When `trial` is given, only the FIRST seed uses build_and_train_with_pruning;
    a TrialPruned exception aborts the whole call. Subsequent seeds (and the
    confirm phase, which passes trial=None) use the no-pruning path so the
    pruner's step axis stays a clean per-epoch index inside the first seed.
    """
    aucs = []
    for i, seed in enumerate(seeds):
        train_idx, val_idx = prepare.scaffold_split_train_val(
            smiles, SPLIT_MODE, seed, 0.8,
        )
        if trial is not None and i == 0:
            _, _, val_metrics, _ = bt.build_and_train_with_pruning(
                combo=combo_tuple,
                X_pool=X_pool, y_pool=y_pool,
                train_idx=train_idx, val_idx=val_idx,
                fp_dim=fp_dim, seed=seed,
                trial=trial,
                config=config,
            )
        else:
            _, _, val_metrics, _ = bt.build_and_train(
                combo=combo_tuple,
                X_pool=X_pool, y_pool=y_pool,
                train_idx=train_idx, val_idx=val_idx,
                fp_dim=fp_dim, seed=seed,
                config=config,
            )
        aucs.append(float(val_metrics["roc_auc"]))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return aucs


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True,
                    help="Combo string, e.g. maccs+scage1+mole")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of top trials to re-evaluate on confirm seeds.")
    ap.add_argument("--search-seeds", type=str, default=None,
                    help=f"Comma-separated. Default: {SEARCH_SEEDS_DEFAULT}")
    ap.add_argument("--confirm-seeds", type=str, default=None,
                    help=f"Comma-separated. Default: {CONFIRM_SEEDS_DEFAULT}")
    ap.add_argument("--search-num-epochs", type=int, default=None,
                    help="If set, override num_epochs for ALL trials in search "
                         "phase (coarse pass). Confirm phase always uses "
                         "BASE_CONFIG.num_epochs.")
    ap.add_argument("--study-name", type=str, default=None)
    ap.add_argument("--storage", type=str, default=None,
                    help="Optuna storage URL (e.g. sqlite:///hpo.db). "
                         "Default: in-memory.")
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--skip-confirm", action="store_true",
                    help="Run search only; skip top-k re-evaluation.")
    args = ap.parse_args()

    search_seeds = ([int(s) for s in args.search_seeds.split(",")]
                    if args.search_seeds else SEARCH_SEEDS_DEFAULT)
    confirm_seeds = ([int(s) for s in args.confirm_seeds.split(",")]
                     if args.confirm_seeds else CONFIRM_SEEDS_DEFAULT)

    combo_tuple = prepare.combo_str_to_tuple(args.combo)
    out_dir = REPO_ROOT / "results" / args.combo / "hpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())

    print(f"[Plan] combo={args.combo}  trials={args.n_trials}  top_k={args.top_k}")
    print(f"[Plan] search_seeds={search_seeds}")
    print(f"[Plan] confirm_seeds={confirm_seeds}")
    if args.search_num_epochs:
        print(f"[Plan] search override num_epochs={args.search_num_epochs}")

    print("[Data] Loading pool features ...")
    pool_smiles, pool_labels, _, pool_feats = prepare.load_pool_features()
    X_pool = prepare.build_feature_matrix(combo_tuple, pool_feats)
    print(f"[Data] pool size={len(pool_smiles)}  X_pool shape={X_pool.shape}")

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial)
        if args.search_num_epochs:
            cfg["num_epochs"] = args.search_num_epochs
        aucs = run_seeds(combo_tuple, X_pool, pool_labels, pool_smiles,
                         prepare.FP_DIM, search_seeds, cfg, trial=trial)
        trial.set_user_attr("per_seed_val_auc", aucs)
        return float(np.mean(aucs))

    sampler = TPESampler(seed=args.sampler_seed, n_startup_trials=15, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    t0 = time.time()
    # Resume-aware: budget = args.n_trials *total* trials in the study, so when
    # resuming after interruption we only run the remainder (zero if already done).
    n_existing = len(study.trials)
    remaining = max(0, args.n_trials - n_existing)
    if n_existing:
        print(f"[Search] study has {n_existing} existing trials; "
              f"running {remaining} more to reach {args.n_trials}.")
    study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    search_min = (time.time() - t0) / 60.0
    print(f"\n[Search] done in {search_min:.1f} min "
          f"({len(study.trials)} trials)")
    print(f"[Search] best trial #{study.best_trial.number}  "
          f"mean_val_auc={study.best_value:.6f}")

    # Trial-level TSV (search phase)
    trial_rows = []
    for t in study.trials:
        row = {"trial": t.number, "state": t.state.name,
               "mean_val_auc": t.value}
        row.update(t.params)
        trial_rows.append(row)
    trial_df = pd.DataFrame(trial_rows)
    trial_path = out_dir / f"trials_{stamp}.tsv"
    trial_df.to_csv(trial_path, sep="\t", index=False)
    print(f"[Output] {trial_path}")

    confirm_rows = []
    if not args.skip_confirm:
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:args.top_k]
        print(f"\n[Confirm] running top-{len(top_trials)} trials "
              f"on {len(confirm_seeds)} seeds ...")
        for rank, t in enumerate(top_trials, 1):
            # Use the trial's params at full num_epochs (no search override).
            cfg = dict(t.params)
            aucs = run_seeds(combo_tuple, X_pool, pool_labels, pool_smiles,
                             prepare.FP_DIM, confirm_seeds, cfg)
            mean_auc = float(np.mean(aucs))
            std_auc  = float(np.std(aucs, ddof=0))
            print(f"  rank{rank} trial#{t.number}: "
                  f"mean={mean_auc:.6f} ± {std_auc:.6f}  "
                  f"(search={t.value:.6f})")
            confirm_rows.append({
                "rank": rank,
                "trial": t.number,
                "search_mean_val_auc": t.value,
                "confirm_mean_val_auc": mean_auc,
                "confirm_std_val_auc": std_auc,
                "confirm_per_seed_val_auc": aucs,
                "params": t.params,
            })

    summary = {
        "combo": args.combo,
        "study_name": args.study_name,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": sum(1 for t in study.trials
                                  if t.state.name == "COMPLETE"),
        "search_seeds": search_seeds,
        "confirm_seeds": confirm_seeds,
        "search_num_epochs_override": args.search_num_epochs,
        "search_wall_time_min": round(search_min, 2),
        "best_search_trial": study.best_trial.number,
        "best_search_mean_val_auc": float(study.best_value),
        "best_search_params": study.best_trial.params,
        "top_k_confirm": confirm_rows,
    }
    summary_path = out_dir / f"study_{stamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[Output] {summary_path}")

    print("\n" + "=" * 60)
    print(f"[Done] best search trial #{study.best_trial.number}  "
          f"val_auc={study.best_value:.6f}")
    if confirm_rows:
        best_confirm = max(confirm_rows, key=lambda r: r["confirm_mean_val_auc"])
        print(f"[Done] best confirm trial #{best_confirm['trial']}  "
              f"val_auc={best_confirm['confirm_mean_val_auc']:.6f} "
              f"± {best_confirm['confirm_std_val_auc']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
