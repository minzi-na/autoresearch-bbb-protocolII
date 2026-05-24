#!/usr/bin/env python3
"""
evaluate_hpo.py — Phase 2 autoresearch iter runner (FROZEN).

For each phase-2 iteration the autoresearch agent:
  1. Edits optuna_combo.py (search space, sampler, objective, etc.).
  2. git commit -m "iter<N>: <short>".
  3. Runs THIS script with --iter-id N --note "<short>".
  4. Checks results/<combo>/hpo/results.tsv last row's keep field:
       True  -> leave commit in place, proceed to iter N+1
       False -> git revert <commit>, proceed to iter N+1

Decision rule (keep):
  best_confirm_mean_val_auc > threshold_mean
where:
  threshold = max over (iter-200 phase-1 baseline, prior phase-2 keeps)
              of confirm_mean.

The std buffer was dropped (user preference). std stays in the TSV
for reference only. Marginal wins (close to threshold) should be
cross-checked against the architecture_log holdout signal.

Usage:
    conda run -n rapids-25.02 python evaluate_hpo.py \\
        --combo maccs+scage1+mole --iter-id 1 --note "narrow lr range"

Budget (user can adjust; see BUDGET dict comments for history):
    n_trials=30, top_k=3, search_num_epochs=30,
    search_seeds=[42,100,200], confirm_seeds=[42,100,...,900]
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
#  Frozen budget (single source of truth for fair iter-to-iter comparison)
# ─────────────────────────────────────────────────────────────────────────────

BUDGET = {
    # n_trials lowered 50 -> 30 from iter5 onward (user budget decision).
    # iter1-4 ran with 50 but all hit trial#7 deterministically (TPE wide
    # space) so trial-count fairness loss is negligible — sampler had
    # plenty of headroom. n_startup_trials=15 for both TPE/CmaEs means
    # 30 = 15 startup + 15 exploitation, still enough for the sampler
    # to learn.
    "n_trials":          30,
    "top_k":             3,
    "search_num_epochs": 30,
    "search_seeds":      "42,100,200",
    "confirm_seeds":     "42,100,200,300,400,500,600,700,800,900",
    "sampler_seed":      0,
}

STORAGE_URL = "sqlite:///combo2_hpo.db"  # per-worktree DB; gitignored

ALLOWED_COMBOS = [
    "maccs+avalon+scage2+mole",
    "maccs+scage1+mole",
    "maccs+scage1+scage2+mole",
]

TSV_COLUMNS = [
    "iter", "timestamp", "commit", "combo",
    "n_trials", "top_k", "search_num_epochs",
    "best_search_trial", "best_search_mean_val_auc",
    "best_confirm_trial", "best_confirm_mean_val_auc",
    "best_confirm_std_val_auc",
    "threshold_mean", "threshold_std",
    "wall_time_min", "keep", "note",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def git_head_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=10", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def phase1_baseline(combo: str) -> tuple[float, float]:
    """Best phase-1 keep row from results/<combo>/results.tsv."""
    path = REPO_ROOT / "results" / combo / "results.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Phase-1 results.tsv not found: {path}")
    df = pd.read_csv(path, sep="\t")
    keeps = df[df["keep"] == True]  # noqa: E712
    if keeps.empty:
        raise RuntimeError(f"Phase-1 has no keep=True rows in {path}")
    best = keeps.loc[keeps["mean_val_auc"].idxmax()]
    return float(best["mean_val_auc"]), float(best["std_val_auc"])


def load_hpo_tsv(combo: str) -> pd.DataFrame:
    path = REPO_ROOT / "results" / combo / "hpo" / "results.tsv"
    if not path.exists():
        return pd.DataFrame(columns=TSV_COLUMNS)
    return pd.read_csv(path, sep="\t")


def current_threshold(combo: str) -> tuple[float, float, str]:
    """Return (threshold_mean, threshold_std, source). The new study's
    best_confirm_mean_val_auc must beat threshold_mean to be kept.
    threshold_std is recorded for reference only (the buffer was
    dropped from the keep rule)."""
    base_mean, base_std = phase1_baseline(combo)
    df = load_hpo_tsv(combo)
    keeps = df[df["keep"] == True]  # noqa: E712
    if keeps.empty:
        return base_mean, base_std, "phase1_baseline"
    row = keeps.loc[keeps["best_confirm_mean_val_auc"].idxmax()]
    if float(row["best_confirm_mean_val_auc"]) >= base_mean:
        return (float(row["best_confirm_mean_val_auc"]),
                float(row["best_confirm_std_val_auc"]),
                f"phase2_iter{int(row['iter'])}")
    return base_mean, base_std, "phase1_baseline"


def append_row(combo: str, row: dict):
    path = REPO_ROOT / "results" / combo / "hpo" / "results.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_hpo_tsv(combo)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.reindex(columns=TSV_COLUMNS)
    df.to_csv(path, sep="\t", index=False)


def list_study_jsons(combo: str) -> set:
    d = REPO_ROOT / "results" / combo / "hpo"
    return set(p.name for p in d.glob("study_*.json")) if d.exists() else set()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combo",   required=True, choices=ALLOWED_COMBOS)
    p.add_argument("--iter-id", type=int, required=True)
    p.add_argument("--note",    type=str, default="")
    p.add_argument("--commit",  type=str, default=None,
                   help="Override git HEAD commit recorded in TSV.")
    p.add_argument("--no-append", action="store_true",
                   help="Run + report but do not write to results.tsv.")
    args = p.parse_args()

    commit = args.commit or git_head_commit()
    threshold_mean, threshold_std, src = current_threshold(args.combo)
    study_name = f"auto_iter{args.iter_id:03d}_{commit}"

    print(f"[Plan] combo={args.combo}  iter={args.iter_id}  commit={commit}")
    print(f"[Plan] study_name={study_name}")
    print(f"[Plan] budget={BUDGET}")
    print(f"[Plan] threshold_mean={threshold_mean:.6f}  "
          f"(std_for_reference={threshold_std:.6f}; source={src})")
    print(f"[Plan] keep iff best_confirm_mean > {threshold_mean:.6f}")

    pre_jsons = list_study_jsons(args.combo)

    cmd = [
        "python", str(REPO_ROOT / "optuna_combo.py"),
        "--combo", args.combo,
        "--n-trials", str(BUDGET["n_trials"]),
        "--top-k", str(BUDGET["top_k"]),
        "--search-num-epochs", str(BUDGET["search_num_epochs"]),
        "--search-seeds", BUDGET["search_seeds"],
        "--confirm-seeds", BUDGET["confirm_seeds"],
        "--study-name", study_name,
        "--storage", STORAGE_URL,
        "--sampler-seed", str(BUDGET["sampler_seed"]),
    ]
    print(f"[Run] {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    wall_min = (time.time() - t0) / 60.0
    if proc.returncode != 0:
        print(f"[FAIL] optuna_combo.py exited with code {proc.returncode}")
        sys.exit(proc.returncode)

    post_jsons = list_study_jsons(args.combo)
    new_jsons = sorted(post_jsons - pre_jsons)
    if not new_jsons:
        print("[FAIL] no new study_*.json produced — aborting")
        sys.exit(2)
    study_json_name = new_jsons[-1]
    study_json_path = REPO_ROOT / "results" / args.combo / "hpo" / study_json_name
    print(f"[Out] {study_json_path}")

    with open(study_json_path) as f:
        summary = json.load(f)

    confirm_rows = summary.get("top_k_confirm", [])
    if not confirm_rows:
        print("[FAIL] study JSON has no top_k_confirm rows")
        sys.exit(3)
    best_confirm = max(confirm_rows, key=lambda r: r["confirm_mean_val_auc"])

    best_search_trial = int(summary.get("best_search_trial", -1))
    best_search_val   = float(summary.get("best_search_mean_val_auc", float("nan")))
    confirm_mean      = float(best_confirm["confirm_mean_val_auc"])
    confirm_std       = float(best_confirm["confirm_std_val_auc"])
    confirm_trial     = int(best_confirm["trial"])

    # Decision rule (simplified): mean-only beat-by-any-margin. The std
    # buffer was dropped — user preference; std stays in the TSV for
    # information only. Note: under this rule small-effect / noise-level
    # gains can pass; the architecture_log holdout signal is the
    # secondary check for any iter that lands in the marginal band.
    keep = bool(confirm_mean > threshold_mean)

    print("\n" + "=" * 60)
    print(f"iter={args.iter_id}  commit={commit}  combo={args.combo}")
    print(f"best search trial   #{best_search_trial}  "
          f"mean_val_auc={best_search_val:.6f}")
    print(f"best confirm trial  #{confirm_trial}  "
          f"mean_val_auc={confirm_mean:.6f} ± {confirm_std:.6f}")
    print(f"threshold (mean only) {threshold_mean:.6f}  "
          f"(source={src}; std_for_reference={threshold_std:.6f})")
    print(f"wall time            {wall_min:.2f} min")
    print(f"DECISION             {'KEEP' if keep else 'DISCARD'}")
    print("=" * 60)

    row = {
        "iter":            args.iter_id,
        "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
        "commit":          commit,
        "combo":           args.combo,
        "n_trials":        BUDGET["n_trials"],
        "top_k":           BUDGET["top_k"],
        "search_num_epochs": BUDGET["search_num_epochs"],
        "best_search_trial":        best_search_trial,
        "best_search_mean_val_auc": round(best_search_val, 6),
        "best_confirm_trial":       confirm_trial,
        "best_confirm_mean_val_auc": round(confirm_mean, 6),
        "best_confirm_std_val_auc":  round(confirm_std, 6),
        "threshold_mean":  round(threshold_mean, 6),
        "threshold_std":   round(threshold_std, 6),
        "wall_time_min":   round(wall_min, 2),
        "keep":            keep,
        "note":            args.note,
    }
    if not args.no_append:
        append_row(args.combo, row)
        print(f"[Output] appended row to "
              f"{REPO_ROOT / 'results' / args.combo / 'hpo' / 'results.tsv'}")
    else:
        print("[Output] --no-append: row not written.")


if __name__ == "__main__":
    main()
