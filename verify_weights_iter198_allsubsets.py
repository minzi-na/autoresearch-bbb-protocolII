#!/usr/bin/env python3
"""Full-scope load verification for the iter198 weight snapshot.

Loads each saved checkpoint (no retraining) and reproduces every recorded
metric on ALL 5 holdout subsets, per seed and for the soft-vote ensemble,
against results/<combo>/holdout_eval/iter0198_2e8cbdc9f5.json.

This closes the nn05-only gap of verify_weights_iter198.py: a checkpoint that
matches all 6 metrics x 5 subsets x 10 seeds cannot be a mis-loaded or
mismatched set of weights.

Usage:
    conda run -n rapids-25.02 python verify_weights_iter198_allsubsets.py
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
from final_holdout_eval import predict_probs, metrics_from_probs, SUBSETS  # noqa: E402

COMBO = "maccs+scage1+mole"
ITER_ID = 198
WEIGHTS_DIR = REPO / "results" / COMBO / "weights" / f"iter{ITER_ID:04d}"
RECORDED = REPO / "results" / COMBO / "holdout_eval" / "iter0198_2e8cbdc9f5.json"

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc", "auprc", "specificity"]
TOL = 1e-6


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
    meta = json.load(open(WEIGHTS_DIR / "meta.json"))
    recorded = json.load(open(RECORDED))
    rec_per_seed = {r["seed"]: r for r in recorded["per_seed"]}
    rec_summary = recorded["summary"]["subsets"]

    combo_tuple = prepare.combo_str_to_tuple(COMBO)
    if "rdkit" in combo_tuple:
        sys.exit("[FAIL] combo contains rdkit; scaler handling required")

    sub_data = {}
    for s in SUBSETS:
        smi, lab, ft = prepare.load_holdout_subset(s)
        sub_data[s] = (prepare.build_feature_matrix(combo_tuple, ft), lab)
        print(f"[{s}] n={len(smi)}  pos={int(lab.sum())}")

    bs = train_mod.BASE_CONFIG["batch_size"]
    failures = []
    n_checked = 0
    prob_stacks = {s: [] for s in SUBSETS}

    for seed in meta["seeds"]:
        ckpt = torch.load(WEIGHTS_DIR / f"seed{seed:04d}.pt",
                          map_location=train_mod.device, weights_only=False)
        model = build_model(combo_tuple)
        model.load_state_dict(ckpt["state_dict"], strict=True)

        line = [f"  seed={seed:>3}"]
        for s in SUBSETS:
            X_sub, lab = sub_data[s]
            _, y_prob = predict_probs(model, X_sub, lab, bs)
            prob_stacks[s].append(np.asarray(y_prob, dtype=np.float64))
            got = metrics_from_probs(lab, y_prob)
            rec = rec_per_seed[seed]["subsets"][s]
            bad = []
            for m in METRICS:
                if m not in rec:
                    continue
                n_checked += 1
                if abs(float(got[m]) - float(rec[m])) > TOL:
                    bad.append(m)
                    failures.append(
                        f"seed {seed} {s}.{m}: {got[m]:.6f} != {rec[m]:.6f}")
            line.append(f"{s}={'ok' if not bad else 'MISMATCH:' + ','.join(bad)}")
        print("  ".join(line))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n-- soft-vote ensemble --")
    ens_report = {}
    for s in SUBSETS:
        lab = sub_data[s][1]
        ens = np.stack(prob_stacks[s], axis=0).mean(axis=0)
        got = metrics_from_probs(lab, ens)
        rec = rec_summary[s]["ensemble"]
        bad = []
        for m in METRICS:
            if m not in rec:
                continue
            n_checked += 1
            if abs(float(got[m]) - float(rec[m])) > TOL:
                bad.append(m)
                failures.append(
                    f"ensemble {s}.{m}: {got[m]:.6f} != {rec[m]:.6f}")
        ens_report[s] = {m: round(float(got[m]), 6) for m in METRICS if m in got}
        print(f"  {s:<9} roc_auc={got['roc_auc']:.6f} "
              f"(recorded {rec['roc_auc']:.6f})  "
              f"{'ok' if not bad else 'MISMATCH:' + ','.join(bad)}")

    print("\n" + "=" * 64)
    print(f"comparisons checked = {n_checked}   mismatches = {len(failures)}")
    print("=" * 64)

    report = {
        "combo": COMBO,
        "iter_id": ITER_ID,
        "scope": "all 5 holdout subsets, per-seed + soft-vote ensemble",
        "metrics": METRICS,
        "tolerance": TOL,
        "comparisons_checked": n_checked,
        "ensemble_metrics": ens_report,
        "passed": not failures,
        "failures": failures,
    }
    out = WEIGHTS_DIR / "verify_gates_allsubsets.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"[Output] report -> {out}")

    if failures:
        print("\n[FAIL] mismatches:")
        for f in failures[:40]:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[PASS] every recorded metric reproduced from loaded weights")


if __name__ == "__main__":
    main()
