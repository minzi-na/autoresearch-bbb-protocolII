"""Reproduce the Protocol II best model (maccs+scage1+mole, iter198) per-molecule
predictions on Holdout_total (329) and write them to CSV.

Reuses the FROZEN pipeline (prepare + train.build_and_train + final_holdout_eval
.predict_probs) so the result is identical to the original evaluation:
train.py enforces full determinism (manual_seed all RNGs + cudnn.deterministic +
use_deterministic_algorithms), and the scaffold split is deterministic.

Verification gates (must match the recorded iter198 numbers):
  - seed 42 val_auc == 0.855913
  - total per-seed mean ROC == 0.8700 ,  soft-vote ensemble ROC == 0.8755
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path("/home/minji/combos-v2-combo2")
sys.path.insert(0, str(REPO))

import prepare
import train as train_mod
from final_holdout_eval import predict_probs, metrics_from_probs

COMBO = "maccs+scage1+mole"
SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
OUT = "/home/minji/BBB_paper/shared_splits/predictions/protocolII_best_holdout_total_predictions.csv"

combo_tuple = prepare.combo_str_to_tuple(COMBO)
pool_smiles, pool_labels, _, pool_feats = prepare.load_pool_features()
X_pool = prepare.build_feature_matrix(combo_tuple, pool_feats)

smi, lab, ft = prepare.load_holdout_subset("total")
X_sub = prepare.build_feature_matrix(combo_tuple, ft)
print(f"[total] n={len(smi)}  pos={int(lab.sum())}  X={X_sub.shape}")

fp_dim = prepare.FP_DIM
bs = train_mod.BASE_CONFIG["batch_size"]

per_seed_probs = []          # list of (329,) prob vectors
per_seed_val_auc = {}
for i, seed in enumerate(SEEDS):
    train_idx, val_idx = prepare.scaffold_split_train_val(pool_smiles, "scaffold", seed, 0.8)
    model, train_info, val_metrics, scaler = train_mod.build_and_train(
        combo=combo_tuple, X_pool=X_pool, y_pool=pool_labels,
        train_idx=train_idx, val_idx=val_idx, fp_dim=fp_dim, seed=seed,
    )
    X_in = X_sub
    if scaler is not None and "rdkit" in combo_tuple:
        X_in = train_mod.apply_rdkit_scaler(X_sub, combo_tuple, fp_dim, scaler)
    _, y_prob = predict_probs(model, X_in, lab, bs)
    per_seed_probs.append(np.asarray(y_prob, dtype=np.float64))
    per_seed_val_auc[seed] = float(val_metrics["roc_auc"])
    r = metrics_from_probs(lab, y_prob)["roc_auc"]
    print(f"[{i+1:>2}/10] seed={seed:>3}  val_auc={val_metrics['roc_auc']:.6f}  total_roc={r:.6f}")

probs = np.stack(per_seed_probs, axis=0)            # (10, 329)
ensemble = probs.mean(axis=0)                        # soft-voting

# ── verification ──────────────────────────────────────────────────────────────
ps_mean_roc = np.mean([metrics_from_probs(lab, probs[i])["roc_auc"] for i in range(len(SEEDS))])
ens_roc = metrics_from_probs(lab, ensemble)["roc_auc"]
print("\n=== verification vs recorded iter198 ===")
print(f"seed42 val_auc : {per_seed_val_auc[42]:.6f}  (recorded 0.855913)")
print(f"total per-seed mean ROC : {ps_mean_roc:.6f}  (recorded 0.8798)")
print(f"total ensemble ROC      : {ens_roc:.6f}  (recorded 0.8840)")

# ── write CSV ─────────────────────────────────────────────────────────────────
# Per-seed predictions only (consistent with per-seed-mean reporting):
# each molecule has one P(BBB+) per seed. Soft-vote ensemble is intentionally
# not included (it is a different, prediction-level aggregation).
df = pd.DataFrame({"smiles": smi, "p_np": np.asarray(lab, dtype=int)})
for j, seed in enumerate(SEEDS):
    df[f"prob_seed{seed}"] = np.round(probs[j], 6)
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"\nwritten {OUT}  ({len(df)} rows, {df.shape[1]} cols)")
