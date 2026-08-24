#!/usr/bin/env python3
"""
build_comparison_table.py — assemble a pre-optimization comparison table.

Pulls together:
  1) GateMol-BBB iter 1 ("baseline reproduction") from each combo worktree:
       /home/minji/combos-v2-combo<N>/results/<combo>/results.tsv  (val_auc / mcc)
       /home/minji/combos-v2-combo<N>/results/<combo>/holdout_eval/iter0001_*.json
                                                       (val mean±std + 5-subset ensemble)
  2) Baseline models (mlp2/mlp3/mlp4/xgboost/lightgbm) from this worktree:
       /home/minji/autoresearch_combos_v2/results/<combo>/baselines/summary.tsv

Writes:
  results/comparison_pre_optimization.csv   — one row per (combo, model)

Usage:
  conda run -n rapids-25.02 python baselines/build_comparison_table.py
"""

import json
import sys
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

COMBOS = [
    "maccs+avalon+scage2+mole",
    "maccs+scage1+mole",
    "maccs+scage1+scage2+mole",
]

COMBO_TO_WORKTREE = {
    "maccs+avalon+scage2+mole":   "/home/minji/combos-v2-combo1",
    "maccs+scage1+mole":          "/home/minji/combos-v2-combo2",
    "maccs+scage1+scage2+mole":   "/home/minji/combos-v2-combo3",
}

SUBSETS = ["internal", "external", "nn03", "nn05", "total"]
SUB_METRICS = ["roc_auc", "mcc", "f1", "acc"]  # acc maps to "accuracy" in JSON

BASELINE_MODELS = ["mlp2", "mlp3", "mlp4", "xgboost", "lightgbm"]

COLUMNS = (
    ["combo", "model", "source", "n_seeds",
     "val_mean_roc_auc", "val_std_roc_auc",
     "val_mean_mcc",     "val_std_mcc"]
    + [f"hold_{s}_ens_{m}" for s in SUBSETS for m in SUB_METRICS]
)


def _ens_value(ens: dict, m: str):
    # final_holdout_eval.py writes "accuracy"; summary.tsv uses "acc"
    return ens.get("accuracy" if m == "acc" else m)


def load_gatemol_iter1_row(combo: str) -> dict | None:
    """Read iter 1 baseline reproduction row from the combo's worktree."""
    wt = COMBO_TO_WORKTREE.get(combo)
    if not wt:
        return None

    res_tsv = Path(wt) / "results" / combo / "results.tsv"
    if not res_tsv.exists():
        print(f"  [warn] {combo}: no results.tsv at {res_tsv}")
        return None

    df = pd.read_csv(res_tsv, sep="\t")
    iter1 = df[df["iter"] == 1]
    if iter1.empty:
        print(f"  [warn] {combo}: no iter==1 row in {res_tsv}")
        return None
    iter1 = iter1.iloc[0]
    commit = str(iter1.get("commit", ""))

    # Find iter1 holdout JSON
    json_glob = str(Path(wt) / "results" / combo / "holdout_eval" / "iter0001_*.json")
    json_paths = sorted(glob(json_glob))
    holdout_ens = {s: {m: None for m in SUB_METRICS} for s in SUBSETS}
    if not json_paths:
        print(f"  [warn] {combo}: no holdout iter0001_*.json under {json_glob}")
    else:
        with open(json_paths[0]) as f:
            blob = json.load(f)
        summary = blob.get("summary", {})
        subsets = summary.get("subsets", {})
        for s in SUBSETS:
            ens = subsets.get(s, {}).get("ensemble", {})
            for m in SUB_METRICS:
                holdout_ens[s][m] = _ens_value(ens, m)

    row = {
        "combo":            combo,
        "model":            "GateMol-BBB",
        "source":           f"iter1@{commit}",
        "n_seeds":          int(iter1.get("n_seeds", 0)) if pd.notna(iter1.get("n_seeds")) else None,
        "val_mean_roc_auc": float(iter1["mean_val_auc"]),
        "val_std_roc_auc":  float(iter1["std_val_auc"]),
        "val_mean_mcc":     float(iter1["mean_val_mcc"]),
        "val_std_mcc":      float(iter1["std_val_mcc"]),
    }
    for s in SUBSETS:
        for m in SUB_METRICS:
            row[f"hold_{s}_ens_{m}"] = holdout_ens[s][m]
    return row


def load_baseline_rows(combo: str) -> list[dict]:
    """Read all baseline model rows from this worktree's summary.tsv for one combo."""
    summary_tsv = REPO_ROOT / "results" / combo / "baselines" / "summary.tsv"
    if not summary_tsv.exists():
        print(f"  [warn] {combo}: no baselines summary at {summary_tsv}")
        return []
    df = pd.read_csv(summary_tsv, sep="\t")
    rows = []
    for _, r in df.iterrows():
        row = {
            "combo":            combo,
            "model":            str(r["model"]),
            "source":           "baseline_models",
            "n_seeds":          int(r["n_seeds"]),
            "val_mean_roc_auc": float(r["val_mean_roc_auc"]),
            "val_std_roc_auc":  float(r["val_std_roc_auc"]),
            "val_mean_mcc":     float(r["val_mean_mcc"]),
            "val_std_mcc":      float(r["val_std_mcc"]),
        }
        for s in SUBSETS:
            for m in SUB_METRICS:
                row[f"hold_{s}_ens_{m}"] = float(r[f"hold_{s}_ens_{m}"])
        rows.append(row)

    # Stable order: mlp2 → mlp3 → mlp4 → xgboost → lightgbm
    order = {m: i for i, m in enumerate(BASELINE_MODELS)}
    rows.sort(key=lambda x: order.get(x["model"], 99))
    return rows


def main():
    all_rows = []
    for combo in COMBOS:
        print(f"\n[{combo}]")
        g = load_gatemol_iter1_row(combo)
        if g is not None:
            all_rows.append(g)
            print(f"  GateMol-BBB iter1: val_auc={g['val_mean_roc_auc']:.4f}±{g['val_std_roc_auc']:.4f}")
        for r in load_baseline_rows(combo):
            all_rows.append(r)
            print(f"  {r['model']:>9s}: val_auc={r['val_mean_roc_auc']:.4f}±{r['val_std_roc_auc']:.4f}")

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    # Round numeric columns to 4 decimals for the CSV (raw is in JSON anyway)
    numeric_cols = [c for c in df.columns if c not in ("combo", "model", "source", "n_seeds")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(4)

    out_csv = REPO_ROOT / "results" / "comparison_pre_optimization.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    # Print a compact view: combo, model, val±std, then 5 subset roc_auc
    compact_cols = ["combo", "model", "source",
                    "val_mean_roc_auc", "val_std_roc_auc"] \
                   + [f"hold_{s}_ens_roc_auc" for s in SUBSETS]
    print(df[compact_cols].to_string(index=False))
    print("=" * 100)
    print(f"\nSaved -> {out_csv}")
    print(f"Full table also includes mcc / f1 / acc for each holdout subset.")


if __name__ == "__main__":
    sys.exit(main())
