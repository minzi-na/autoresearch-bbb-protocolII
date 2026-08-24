"""Shared helpers for run_baselines.py and run_tabpfn.py.

Pulled out so run_tabpfn.py (which runs in the `tabpfn` conda env) does not
transitively import xgboost/lightgbm via baselines.models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
)


DEFAULT_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]

ALLOWED_COMBOS = [
    "maccs+avalon+scage2+mole",
    "maccs+scage1+mole",
    "maccs+scage1+scage2+mole",
]

SUBSETS = ["internal", "external", "nn03", "nn05", "total"]


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """ROC-AUC/MCC/F1/ACC/AUPRC/specificity from probs + labels.
    Mirrors final_holdout_eval.metrics_from_probs."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
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


def get_rdkit_slice(combo_tuple, fp_dim):
    """Return (start, end) of rdkit columns inside the concatenated feature matrix,
    or (None, None) if combo has no rdkit."""
    if "rdkit" not in combo_tuple:
        return None, None
    offset = 0
    for t in combo_tuple:
        if t == "rdkit":
            return offset, offset + fp_dim["rdkit"]
        offset += fp_dim[t]
    return None, None


def fit_apply_rdkit_scaler(X_train, X_other_list, rd_slice):
    """Fit StandardScaler on X_train rdkit slice; transform all matrices in place.
    Returns None if no rdkit slice."""
    rd_start, rd_end = rd_slice
    if rd_start is None:
        return None
    scaler = StandardScaler()
    X_train[:, rd_start:rd_end] = scaler.fit_transform(X_train[:, rd_start:rd_end])
    for X in X_other_list:
        X[:, rd_start:rd_end] = scaler.transform(X[:, rd_start:rd_end])
    return scaler


def update_summary_tsv(summary_tsv: Path, combo: str, model_name: str,
                       summary: dict, wall_min: float, timestamp: str):
    cols = [
        "combo", "model", "timestamp",
        "n_seeds",
        "val_mean_roc_auc", "val_std_roc_auc",
        "val_mean_mcc",     "val_std_mcc",
    ] + [f"hold_{s}_ens_roc_auc" for s in SUBSETS] \
      + [f"hold_{s}_ens_mcc"     for s in SUBSETS] \
      + [f"hold_{s}_ens_f1"      for s in SUBSETS] \
      + [f"hold_{s}_ens_acc"     for s in SUBSETS] \
      + ["wall_time_min"]

    row = {
        "combo":            combo,
        "model":            model_name,
        "timestamp":        timestamp,
        "n_seeds":          summary["n_seeds"],
        "val_mean_roc_auc": summary["val"]["mean_roc_auc"],
        "val_std_roc_auc":  summary["val"]["std_roc_auc"],
        "val_mean_mcc":     summary["val"]["mean_mcc"],
        "val_std_mcc":      summary["val"]["std_mcc"],
        "wall_time_min":    round(wall_min, 2),
    }
    for s in SUBSETS:
        block = summary["subsets"].get(s)
        if block is None or "ensemble" not in block:
            # Subset was skipped for this run (e.g., shortened tabpfn run).
            row[f"hold_{s}_ens_roc_auc"] = ""
            row[f"hold_{s}_ens_mcc"]     = ""
            row[f"hold_{s}_ens_f1"]      = ""
            row[f"hold_{s}_ens_acc"]     = ""
        else:
            ens = block["ensemble"]
            row[f"hold_{s}_ens_roc_auc"] = ens["roc_auc"]
            row[f"hold_{s}_ens_mcc"]     = ens["mcc"]
            row[f"hold_{s}_ens_f1"]      = ens["f1"]
            row[f"hold_{s}_ens_acc"]     = ens["accuracy"]

    if summary_tsv.exists():
        df = pd.read_csv(summary_tsv, sep="\t")
        df = df[~((df["combo"] == combo) & (df["model"] == model_name))]
    else:
        df = pd.DataFrame(columns=cols)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.reindex(columns=cols)
    summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_tsv, sep="\t", index=False)
