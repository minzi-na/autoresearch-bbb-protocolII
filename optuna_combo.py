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
        --combo maccs+avalon+scage2+mole \\
        --n-trials 30 --top-k 3 \\
        --study-name combo1_phase2_iter1

Optional:
    --search-seeds 42,100,200
    --confirm-seeds 42,100,200,300,400,500,600,700,800,900
    --search-num-epochs 30   (override BASE_CONFIG.num_epochs for trial budget)
    --storage sqlite:///combo1_hpo.db
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
import torch.utils.data as torch_data
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import prepare  # noqa: E402
import best_train as bt  # noqa: E402


SEARCH_SEEDS_DEFAULT = [42, 100, 200]
CONFIRM_SEEDS_DEFAULT = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
SPLIT_MODE = "scaffold"
# Holdout subsets evaluated in the confirm phase (no extra retrain).
# Restricted to the two most informative subsets for Phase 2 (matches the
# bbb-combo1 pattern of measuring val + holdout in one reevaluation pass).
# FROZEN at ["nn05","total"] so the architecture_log.md column semantics
# stay stable across iters.
HOLDOUT_SUBSETS = ["nn05", "total"]


def _predict_probs(model, X_sub: np.ndarray, y_sub: np.ndarray,
                   batch_size: int):
    """Run sigmoid forward over X_sub and return (y_true, y_prob)."""
    loader = torch_data.DataLoader(
        torch_data.TensorDataset(
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


def _holdout_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """AUC / MCC / Accuracy at the 0.5 threshold. Kept minimal: phase2
    architecture_log only logs these three. Soft-voting ensemble is
    reported separately at the per-trial level."""
    y_pred = (y_prob > 0.5).astype(int)
    has_both = len(set(y_true.tolist())) > 1
    return {
        "roc_auc":  round(float(roc_auc_score(y_true, y_prob) if has_both else 0.0), 6),
        "mcc":      round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Search space — phase-2 EDITABLE (the agent rewrites this each iter)
#
#  iter1 design: narrow 5-D refinement around iter197 anchor. Drops
#  warmup_cosine schedule (and the now-unused lr_warmup_epochs) to
#  cut categorical fan-out so 30 trials give each schedule ~15 trials
#  for TPE. lr / ema_decay ranges tightened around iter197 baseline.
#
#  Combo1-specific caveat: BASE_CONFIG.weight_decay is effectively
#  unused (train_model overrides Adam with AdamW wd=0.003 hardcoded —
#  iter38/88/148/172 frozen). dropout / head_dropout / mod_drop_p /
#  DROP_V_PATH are also hardcoded inside the model classes. Phase-2
#  may later expose these via best_train.py lever extensions.
# ─────────────────────────────────────────────────────────────────────────────

def suggest_config(trial: optuna.Trial) -> dict:
    """Return a config dict layered on top of BASE_CONFIG.

    iter1: pin structural at iter197 best; refine lr × lr_schedule
    (constant|cosine) × lr_min_ratio × ema_decay × label_smoothing
    around iter197 anchor. lr_warmup_epochs dropped from search (falls
    back to BASE_CONFIG default 0 — irrelevant for both constant and
    plain cosine schedules).
    """
    return {
        # ─── Pinned at iter197 frozen architecture ───────────────────────
        "d_model":        512,
        "d_ffn":          1048,
        "depth":          4,
        "use_gated_pool": True,
        "batch_size":     128,
        # ─── Searched: lr (Adam input; AdamW lr = 1.25 × this).
        # iter197 baseline lr=1e-4 -> AdamW lr=1.25e-4. ±30% window.
        "lr":             trial.suggest_float("lr", 7e-5, 1.4e-4, log=True),
        # ─── Searched: LR schedule. warmup_cosine excluded for iter1 to
        # halve categorical fan-out; iter2+ may reintroduce.
        "lr_schedule":    trial.suggest_categorical(
            "lr_schedule", ["constant", "cosine"],
        ),
        # cosine eta_min ratio; ignored when lr_schedule="constant".
        "lr_min_ratio":   trial.suggest_float("lr_min_ratio", 0.0, 0.3),
        # ema_decay narrow around iter197's 0.9993 (iter198's 0.999 reverted
        # in phase-1, so range stays >=0.998).
        "ema_decay":      trial.suggest_float("ema_decay", 0.998, 0.9997, log=True),
        # label_smoothing modest range for initial probe.
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.05),
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
                    help="Combo string, e.g. maccs+avalon+scage2+mole")
    ap.add_argument("--n-trials", type=int, default=30)
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
                    help="Optuna storage URL (e.g. sqlite:///combo1_hpo.db). "
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

    # Objective evaluates each trial on the search_seeds list (multi-seed
    # mean). Previously this was a hardcoded [42,100,200,300,400] (clustered
    # in the lower half of the 10-seed confirm range, causing biased
    # search-to-confirm correlation). Now driven by BUDGET["search_seeds"]
    # in evaluate_hpo.py — combo1 iter1+: "42,200,400,600,800" spreads
    # search seeds evenly across the confirm-seed range (stride 200,
    # strict subset of the 10 confirm seeds 42,100,...,900).
    objective_seeds = search_seeds

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial)
        if args.search_num_epochs:
            cfg["num_epochs"] = args.search_num_epochs
        aucs = run_seeds(combo_tuple, X_pool, pool_labels, pool_smiles,
                         prepare.FP_DIM, objective_seeds, cfg, trial=trial)
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
        # Load holdout subsets once. Same scaler / batch-size handling as
        # final_holdout_eval.py: rdkit slice scaler is fit on the per-seed
        # train pool and applied to holdout (only when combo has 'rdkit').
        holdout_data = {}
        for s in HOLDOUT_SUBSETS:
            smi, lab, ft = prepare.load_holdout_subset(s)
            X_sub = prepare.build_feature_matrix(combo_tuple, ft)
            holdout_data[s] = (lab, X_sub)
            print(f"[Holdout] {s}: n={len(smi)} pos={int(lab.sum())} "
                  f"X shape={X_sub.shape}")

        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:args.top_k]
        print(f"\n[Confirm] running top-{len(top_trials)} trials "
              f"on {len(confirm_seeds)} seeds (inline holdout: "
              f"{HOLDOUT_SUBSETS}) ...")
        for rank, t in enumerate(top_trials, 1):
            cfg = dict(t.params)
            seed_val_aucs = []
            seed_holdout = {s: [] for s in HOLDOUT_SUBSETS}
            bs = cfg.get("batch_size", bt.BASE_CONFIG["batch_size"])
            for seed in confirm_seeds:
                train_idx, val_idx = prepare.scaffold_split_train_val(
                    pool_smiles, SPLIT_MODE, seed, 0.8,
                )
                model, _, val_metrics, scaler = bt.build_and_train(
                    combo=combo_tuple,
                    X_pool=X_pool, y_pool=pool_labels,
                    train_idx=train_idx, val_idx=val_idx,
                    fp_dim=prepare.FP_DIM, seed=seed,
                    config=cfg,
                )
                seed_val_aucs.append(float(val_metrics["roc_auc"]))
                for s in HOLDOUT_SUBSETS:
                    y_sub, X_sub = holdout_data[s]
                    X_in = X_sub
                    if scaler is not None and "rdkit" in combo_tuple:
                        X_in = bt.apply_rdkit_scaler(
                            X_sub, combo_tuple, prepare.FP_DIM, scaler,
                        )
                    y_true, y_prob = _predict_probs(model, X_in, y_sub, bs)
                    seed_holdout[s].append(_holdout_metrics(y_true, y_prob))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            mean_auc = float(np.mean(seed_val_aucs))
            std_auc  = float(np.std(seed_val_aucs, ddof=0))
            holdout_summary = {}
            for s in HOLDOUT_SUBSETS:
                aucs = np.array([m["roc_auc"]  for m in seed_holdout[s]])
                mccs = np.array([m["mcc"]      for m in seed_holdout[s]])
                accs = np.array([m["accuracy"] for m in seed_holdout[s]])
                holdout_summary[s] = {
                    "per_seed_mean_roc_auc":  round(float(aucs.mean()), 6),
                    "per_seed_std_roc_auc":   round(float(aucs.std(ddof=0)), 6),
                    "per_seed_mean_mcc":      round(float(mccs.mean()), 6),
                    "per_seed_std_mcc":       round(float(mccs.std(ddof=0)), 6),
                    "per_seed_mean_accuracy": round(float(accs.mean()), 6),
                    "per_seed_std_accuracy":  round(float(accs.std(ddof=0)), 6),
                    "per_seed_metrics":       seed_holdout[s],
                }
            print(f"  rank{rank} trial#{t.number}: "
                  f"val_mean={mean_auc:.6f} ± {std_auc:.6f}  "
                  f"(search={t.value:.6f})")
            for s in HOLDOUT_SUBSETS:
                hs = holdout_summary[s]
                print(f"    {s:>6s}: "
                      f"AUC={hs['per_seed_mean_roc_auc']:.6f}"
                      f"±{hs['per_seed_std_roc_auc']:.6f}  "
                      f"MCC={hs['per_seed_mean_mcc']:.6f}"
                      f"±{hs['per_seed_std_mcc']:.6f}  "
                      f"Acc={hs['per_seed_mean_accuracy']:.6f}"
                      f"±{hs['per_seed_std_accuracy']:.6f}")
            confirm_rows.append({
                "rank": rank,
                "trial": t.number,
                "search_mean_val_auc": t.value,
                "confirm_mean_val_auc": mean_auc,
                "confirm_std_val_auc": std_auc,
                "confirm_per_seed_val_auc": seed_val_aucs,
                "holdout": holdout_summary,
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

    # ── architecture_log.md row (best confirm trial of THIS study) ─────────
    if confirm_rows:
        best_confirm = max(confirm_rows, key=lambda r: r["confirm_mean_val_auc"])
        log_path = REPO_ROOT / "results" / args.combo / "architecture_log.md"
        if not log_path.exists():
            log_path.write_text(
                f"# {args.combo} — architecture log\n\n"
                "val: per-seed mean±std (10 confirm seeds)  |  subsets "
                "(nn05/total): per-seed mean±std for AUC/MCC/Acc.\n\n"
                "| study | trial | val_auc | "
                "nn05_auc | nn05_mcc | nn05_acc | "
                "total_auc | total_mcc | total_acc |\n"
                "|-------|-------|---------|"
                "----------|----------|----------|"
                "-----------|-----------|-----------|\n"
            )
        def _fmt(mean, std):
            return f"{mean:.4f}±{std:.4f}"
        hd = best_confirm["holdout"]
        nn05 = hd["nn05"]; tot = hd["total"]
        row = (
            f"| {args.study_name or 'inmem'} | {best_confirm['trial']} | "
            f"{_fmt(best_confirm['confirm_mean_val_auc'], best_confirm['confirm_std_val_auc'])} | "
            f"{_fmt(nn05['per_seed_mean_roc_auc'],  nn05['per_seed_std_roc_auc'])} | "
            f"{_fmt(nn05['per_seed_mean_mcc'],      nn05['per_seed_std_mcc'])} | "
            f"{_fmt(nn05['per_seed_mean_accuracy'], nn05['per_seed_std_accuracy'])} | "
            f"{_fmt(tot['per_seed_mean_roc_auc'],   tot['per_seed_std_roc_auc'])} | "
            f"{_fmt(tot['per_seed_mean_mcc'],       tot['per_seed_std_mcc'])} | "
            f"{_fmt(tot['per_seed_mean_accuracy'],  tot['per_seed_std_accuracy'])} |\n"
        )
        with open(log_path, "a") as f:
            f.write(row)
        print(f"[Output] appended row to {log_path}")

    print("\n" + "=" * 60)
    print(f"[Done] best search trial #{study.best_trial.number}  "
          f"val_auc={study.best_value:.6f}")
    if confirm_rows:
        best_confirm = max(confirm_rows, key=lambda r: r["confirm_mean_val_auc"])
        print(f"[Done] best confirm trial #{best_confirm['trial']}  "
              f"val_auc={best_confirm['confirm_mean_val_auc']:.6f} "
              f"± {best_confirm['confirm_std_val_auc']:.6f}")
        for s in HOLDOUT_SUBSETS:
            hs = best_confirm["holdout"][s]
            print(f"[Done] best confirm {s:>6s}: "
                  f"AUC={hs['per_seed_mean_roc_auc']:.6f}"
                  f"±{hs['per_seed_std_roc_auc']:.6f}  "
                  f"MCC={hs['per_seed_mean_mcc']:.6f}"
                  f"±{hs['per_seed_std_mcc']:.6f}  "
                  f"Acc={hs['per_seed_mean_accuracy']:.6f}"
                  f"±{hs['per_seed_std_accuracy']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
