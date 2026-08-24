#!/usr/bin/env python3
"""Verify the iter198 weight snapshot against the recorded Protocol II gates.

Loads each seed checkpoint written by save_weights.py (no retraining), runs it
on Holdout_nn05, and checks:

  - nn05 per-seed mean ROC-AUC == 0.870011  (gate 0.8700)
  - nn05 soft-vote ensemble ROC-AUC == 0.875455  (gate 0.8755)
  - each seed's val_auc matches results/.../holdout_eval/iter0198_2e8cbdc9f5.json

Usage:
    conda run -n rapids-25.02 python verify_weights_iter198.py
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import prepare                                          # noqa: E402
import train as train_mod                               # noqa: E402
from final_holdout_eval import predict_probs, metrics_from_probs  # noqa: E402

COMBO = "maccs+scage1+mole"
ITER_ID = 198
WEIGHTS_DIR = REPO / "results" / COMBO / "weights" / f"iter{ITER_ID:04d}"
RECORDED = REPO / "results" / COMBO / "holdout_eval" / "iter0198_2e8cbdc9f5.json"

GATE_PER_SEED = 0.870011
GATE_SOFTVOTE = 0.875455
TOL = 1e-6          # exact-reproduction tolerance for ROC gates
TOL_VAL = 1e-6      # per-seed val_auc tolerance


def build_model(combo_tuple):
    mod_dims = OrderedDict((t, prepare.FP_DIM[t]) for t in combo_tuple)
    cfg = train_mod.BASE_CONFIG
    return train_mod.MultiModalGMLPFromFlat(
        mod_dims=mod_dims,
        d_model=cfg["d_model"],
        d_ffn=cfg["d_ffn"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
        use_gated_pool=cfg["use_gated_pool"],
    ).to(train_mod.device)


def main():
    meta_path = WEIGHTS_DIR / "meta.json"
    if not meta_path.exists():
        sys.exit(f"[FAIL] meta.json not found: {meta_path}")
    meta = json.load(open(meta_path))
    recorded = json.load(open(RECORDED))

    # recorded per-seed val / nn05 ROC-AUC, keyed by seed
    rec_val, rec_nn05 = {}, {}
    for row in recorded["per_seed"]:
        seed = row.get("seed")
        rec_val[seed] = row.get("val", {}).get("roc_auc")
        rec_nn05[seed] = row.get("subsets", {}).get("nn05", {}).get("roc_auc")

    combo_tuple = prepare.combo_str_to_tuple(COMBO)
    smi, lab, ft = prepare.load_holdout_subset("nn05")
    X_sub = prepare.build_feature_matrix(combo_tuple, ft)
    bs = train_mod.BASE_CONFIG["batch_size"]
    print(f"[nn05] n={len(smi)}  pos={int(lab.sum())}  X={X_sub.shape}")

    if "rdkit" in combo_tuple:
        sys.exit("[FAIL] combo contains rdkit; scaler handling required")

    failures = []
    probs_all, per_seed_roc = [], []

    for seed in meta["seeds"]:
        ckpt_path = WEIGHTS_DIR / f"seed{seed:04d}.pt"
        if not ckpt_path.exists():
            failures.append(f"missing checkpoint {ckpt_path.name}")
            continue
        ckpt = torch.load(ckpt_path, map_location=train_mod.device, weights_only=False)

        model = build_model(combo_tuple)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
        _ = (missing, unexpected)

        _, y_prob = predict_probs(model, X_sub, lab, bs)
        roc = metrics_from_probs(lab, y_prob)["roc_auc"]
        probs_all.append(np.asarray(y_prob, dtype=np.float64))
        per_seed_roc.append(roc)

        ckpt_val = float(ckpt["val_auc"])
        rec = rec_val.get(seed)
        val_ok = rec is not None and abs(ckpt_val - rec) <= TOL_VAL
        if not val_ok:
            failures.append(
                f"seed {seed} val_auc {ckpt_val:.6f} != recorded "
                f"{'n/a' if rec is None else format(rec, '.6f')}"
            )
        rec_r = rec_nn05.get(seed)
        roc_ok = rec_r is not None and abs(roc - rec_r) <= TOL
        if not roc_ok:
            failures.append(
                f"seed {seed} nn05_roc {roc:.6f} != recorded "
                f"{'n/a' if rec_r is None else format(rec_r, '.6f')}"
            )
        print(f"  seed={seed:>3}  val_auc={ckpt_val:.6f} "
              f"({'ok' if val_ok else 'MISMATCH'})  "
              f"nn05_roc={roc:.6f} ({'ok' if roc_ok else 'MISMATCH'})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(probs_all) != len(meta["seeds"]):
        print("\n[FAIL] not all seeds evaluated")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    per_seed_mean = float(np.mean(per_seed_roc))
    per_seed_std = float(np.std(per_seed_roc, ddof=0))
    ensemble = np.stack(probs_all, axis=0).mean(axis=0)
    soft_roc = metrics_from_probs(lab, ensemble)["roc_auc"]

    print("\n" + "=" * 64)
    print(f"nn05 per-seed mean ROC = {per_seed_mean:.6f} +/- {per_seed_std:.6f}"
          f"   (gate {GATE_PER_SEED:.6f})")
    print(f"nn05 soft-vote    ROC = {soft_roc:.6f}"
          f"   (gate {GATE_SOFTVOTE:.6f})")
    print("=" * 64)

    if abs(per_seed_mean - GATE_PER_SEED) > TOL:
        failures.append(f"per-seed mean {per_seed_mean:.6f} != {GATE_PER_SEED:.6f}")
    if abs(soft_roc - GATE_SOFTVOTE) > TOL:
        failures.append(f"soft-vote {soft_roc:.6f} != {GATE_SOFTVOTE:.6f}")

    report = {
        "combo": COMBO,
        "iter_id": ITER_ID,
        "weights_dir": str(WEIGHTS_DIR.relative_to(REPO)),
        "label_commit": meta.get("label_commit"),
        "head_commit": meta.get("head_commit"),
        "nn05_per_seed_mean_roc_auc": round(per_seed_mean, 6),
        "nn05_per_seed_std_roc_auc": round(per_seed_std, 6),
        "nn05_soft_vote_roc_auc": round(soft_roc, 6),
        "gate_per_seed": GATE_PER_SEED,
        "gate_soft_vote": GATE_SOFTVOTE,
        "per_seed_nn05_roc_auc": {
            int(s): round(float(r), 6) for s, r in zip(meta["seeds"], per_seed_roc)
        },
        "passed": not failures,
        "failures": failures,
    }
    out = WEIGHTS_DIR / "verify_gates.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"[Output] report -> {out}")

    if failures:
        print("\n[FAIL] gate verification failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[PASS] all gates matched")


if __name__ == "__main__":
    main()
