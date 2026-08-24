#!/usr/bin/env python3
"""
predict_tabpfn.py — Predict P(BBB+) for new SMILES with the TabPFN-2.5 setup
produced by run_tabpfn.py.

The TabPFN-2.5 fitted model is fully reproducible from:
  - the pool feature cache (deterministic via prepare.load_pool_features)
  - the per-seed scaffold split saved in <combo>/baselines/<model>_artifacts/
    seed_XXXX.npz (key: train_idx)
  - tabpfn config in <model>_artifacts/meta.json
  - the cached TabPFN-2.5 checkpoint at /home/minji/.cache/tabpfn/

So we do NOT need to pickle the classifier. We re-fit() (TabPFN fit is cheap;
the expensive part is the in-context attention at predict-time, which we
must do anyway for the new SMILES).

Usage:
  conda run -n tabpfn python baselines/predict_tabpfn.py \
      --combo maccs+scage1+mole \
      --smiles "CCO,CC(=O)OC1=CC=CC=C1C(=O)O" \
      --seeds 42,100,200,300,400,500,600,700,800,900 \
      --ensemble                # soft-vote across seeds → single mean prob

  conda run -n tabpfn python baselines/predict_tabpfn.py \
      --combo maccs+scage1+mole \
      --smiles-file new_smiles.txt \   # one SMILES per line
      --seeds 42 --out preds.csv

  # Output: probabilities and predicted labels (>0.5).
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare  # noqa: E402
from baselines._common import (  # noqa: E402
    get_rdkit_slice, fit_apply_rdkit_scaler, ALLOWED_COMBOS,
)


def featurize_smiles(smiles_list, combo_tuple, fp_dim):
    """Featurize a list of canonical SMILES using prepare's per-mol functions."""
    from rdkit import Chem
    rows = {t: [] for t in combo_tuple}
    valid_smi, failed = [], []
    for smi in smiles_list:
        cs = prepare.canon_smiles(smi)
        if cs is None:
            failed.append((smi, "canonicalize failed"))
            continue
        mol = Chem.MolFromSmiles(cs)
        if mol is None:
            failed.append((smi, "MolFromSmiles failed"))
            continue
        try:
            feats = {
                "ecfp":   prepare.get_ecfp(mol) if "ecfp" in combo_tuple else None,
                "maccs":  prepare.get_maccs(mol) if "maccs" in combo_tuple else None,
                "avalon": prepare.get_avalon(mol) if "avalon" in combo_tuple else None,
                "tt":     prepare.get_tt(mol) if "tt" in combo_tuple else None,
                "rdkit":  prepare.get_rdkit_desc(mol) if "rdkit" in combo_tuple else None,
            }
        except Exception as e:
            failed.append((smi, f"feat error: {e}"))
            continue
        for t in combo_tuple:
            if t in ("scage1", "scage2", "mole"):
                # Embedding-based feats are not available for new SMILES
                # without re-running the embedding model. Use zero vector and
                # warn — predictions for those modalities will be poor.
                feats[t] = np.zeros(fp_dim[t], dtype=np.float32)
            rows[t].append(feats[t])
        valid_smi.append(cs)
    X = np.concatenate([np.stack(rows[t], axis=0) for t in combo_tuple], axis=1)
    return X, valid_smi, failed


def predict_one_seed(combo_tuple, X_pool, y_pool, X_new, train_idx, seed,
                     n_estimators, ignore_pretraining_limits, device,
                     fp_dim, rdkit_slice):
    from tabpfn import TabPFNClassifier
    X_train = X_pool[train_idx].astype(np.float32, copy=True)
    y_train = y_pool[train_idx].astype(np.int64, copy=True)
    X_new_s = X_new.astype(np.float32, copy=True)
    fit_apply_rdkit_scaler(X_train, [X_new_s], rdkit_slice)
    clf = TabPFNClassifier(
        n_estimators=n_estimators,
        random_state=int(seed),
        ignore_pretraining_limits=ignore_pretraining_limits,
        device=device,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_new_s)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True, choices=ALLOWED_COMBOS)
    ap.add_argument("--smiles", type=str, default=None,
                    help="Comma-separated SMILES.")
    ap.add_argument("--smiles-file", type=str, default=None,
                    help="Path to file with one SMILES per line.")
    ap.add_argument("--artifacts-dir", type=str, default=None,
                    help="Override artifacts dir "
                         "(default: results/<combo>/baselines/tabpfn_artifacts).")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds to ensemble "
                         "(default: all seeds present in artifacts).")
    ap.add_argument("--ensemble", action="store_true",
                    help="Soft-vote average across the selected seeds.")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda/cpu (default: auto).")
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV path (default: print to stdout).")
    args = ap.parse_args()

    # ── Load SMILES ──
    if args.smiles and args.smiles_file:
        ap.error("Use either --smiles or --smiles-file, not both.")
    if args.smiles:
        smiles_list = [s.strip() for s in args.smiles.split(",") if s.strip()]
    elif args.smiles_file:
        smiles_list = [
            ln.strip() for ln in open(args.smiles_file)
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        ap.error("Provide --smiles or --smiles-file.")

    # ── Load artifacts (with fallback to summary JSON if artifacts dir absent) ──
    out_dir = REPO_ROOT / "results" / args.combo / "baselines"
    art_dir = Path(args.artifacts_dir) if args.artifacts_dir else \
        out_dir / "tabpfn_artifacts"

    meta_path = art_dir / "meta.json"
    summary_json = out_dir / "tabpfn.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        cfg = meta["tabpfn_config"]
        combo_tuple = tuple(meta["combo_tuple"])
        seeds_avail = meta["seeds"]
        artifacts_have_train_idx = True
        print(f"[Meta] from {meta_path}  combo={meta['combo']}  "
              f"pool_size={meta['pool_size']}  seeds_avail={seeds_avail}")
    elif summary_json.exists():
        sj = json.load(open(summary_json))
        cfg = sj["tabpfn_config"]
        combo_tuple = prepare.combo_str_to_tuple(sj["combo"])
        seeds_avail = sj["seeds"]
        artifacts_have_train_idx = False
        print(f"[Meta] from {summary_json} (no artifacts dir — will "
              f"reconstruct train splits deterministically from seeds)")
        print(f"  combo={sj['combo']}  seeds_avail={seeds_avail}")
    else:
        raise FileNotFoundError(
            f"Neither {meta_path} nor {summary_json} exists. "
            f"Run baselines/run_tabpfn.py for combo={args.combo} first."
        )

    # ── Load pool ──
    pool_smiles, pool_labels, _, pool_feats = prepare.load_pool_features()
    X_pool = prepare.build_feature_matrix(combo_tuple, pool_feats)
    rdkit_slice = get_rdkit_slice(combo_tuple, prepare.FP_DIM)

    # ── Featurize new SMILES ──
    X_new, valid_smi, failed = featurize_smiles(
        smiles_list, combo_tuple, prepare.FP_DIM)
    print(f"[Input] {len(valid_smi)} valid SMILES, {len(failed)} failed.")
    if any(t in ("scage1", "scage2", "mole") for t in combo_tuple):
        print("[WARN] combo uses scage1/scage2/mole — those embeddings are "
              "ZERO for new SMILES (no recompute available). Predictions "
              "will be biased toward the train-data prior unless you provide "
              "precomputed embedding CSVs.")
    for smi, err in failed:
        print(f"  skipped {smi!r}: {err}")

    if X_new.size == 0:
        sys.exit("No valid SMILES to predict.")

    # ── Pick seeds ──
    requested = ([int(s) for s in args.seeds.split(",")] if args.seeds
                 else seeds_avail)
    if artifacts_have_train_idx:
        missing = [s for s in requested
                   if not (art_dir / f"seed_{s:04d}.npz").exists()]
        if missing:
            raise FileNotFoundError(f"Missing seed artifacts: {missing}")

    import torch  # local import so the rest works without torch when listing
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] device={device}  seeds={requested}  "
          f"n_estimators={cfg['n_estimators']}  "
          f"artifact_train_idx={artifacts_have_train_idx}")

    all_probs = []
    for seed in requested:
        if artifacts_have_train_idx:
            d = np.load(art_dir / f"seed_{seed:04d}.npz")
            train_idx = d["train_idx"]
        else:
            train_idx, _ = prepare.scaffold_split_train_val(
                pool_smiles, "scaffold", seed, 0.8)
            train_idx = np.asarray(train_idx, dtype=np.int64)
        probs = predict_one_seed(
            combo_tuple, X_pool, pool_labels, X_new,
            train_idx, seed,
            cfg["n_estimators"], cfg["ignore_pretraining_limits"], device,
            prepare.FP_DIM, rdkit_slice,
        )
        print(f"  seed={seed}  mean_prob={probs.mean():.4f}")
        all_probs.append(probs)

    probs_arr = np.stack(all_probs, axis=0)  # (n_seeds, n_smiles)
    rows = []
    for i, smi in enumerate(valid_smi):
        row = {"smiles": smi}
        for s_i, seed in enumerate(requested):
            row[f"prob_seed{seed}"] = float(probs_arr[s_i, i])
        if args.ensemble or len(requested) > 1:
            row["prob_mean"] = float(probs_arr[:, i].mean())
            row["pred"] = int(row["prob_mean"] > 0.5)
        else:
            row["pred"] = int(probs_arr[0, i] > 0.5)
        rows.append(row)
    df = pd.DataFrame(rows)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"[Out] wrote {args.out}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
