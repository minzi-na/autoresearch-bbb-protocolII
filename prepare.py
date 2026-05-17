#!/usr/bin/env python3
"""
prepare.py — feature cache loader for autoresearch_combos_v2.

This file is FROZEN (agent must not edit). It exposes:

  load_pool_features()            -> smiles, labels, sources, feat_arrays
  load_holdout_subset(subset_name) -> smiles, labels, feat_arrays
  ALL_FP_TYPES, FP_DIM
  scaffold_split_train_val(smiles_list, split_mode, seed, train_ratio=0.8)

Pool features (internal_remaining + external_remaining, deduped) are reused
from the existing baseline cache at /home/minji/feature_cache_merged/.

Holdout subset features (5 slices under /home/minji/holdout_subset/) are built
on demand into ./feature_cache_holdouts/<subset>.npz and validated by SHA256
of the source CSVs.

Run as a script to (re)build all subset caches:
    conda run -n rapids-25.02 python prepare.py
"""

import os
import json
import hashlib
import random
from collections import OrderedDict

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Existing baseline cache (built by baseline; reused as-is)
POOL_CACHE_DIR  = "/home/minji/feature_cache_merged"
POOL_NPZ        = os.path.join(POOL_CACHE_DIR, "pool.npz")
POOL_MANIFEST   = os.path.join(POOL_CACHE_DIR, "cache_manifest.json")

# Five holdout subsets (label + mole + scage1 + scage2 CSVs only —
# ecfp/maccs/avalon/tt/rdkit are recomputed from SMILES here)
HOLDOUT_SUBSET_ROOT = "/home/minji/holdout_subset"
HOLDOUT_SUBSETS = [
    "merged_holdout_10pct_seed42_simfilter09_internal",
    "merged_holdout_10pct_seed42_simfilter09_external",
    "merged_holdout_10pct_seed42_simfilter09_nn03",
    "merged_holdout_10pct_seed42_simfilter09_nn05",
    "merged_holdout_10pct_seed42_simfilter09_total",
]
SUBSET_SHORT = {
    "merged_holdout_10pct_seed42_simfilter09_internal": "internal",
    "merged_holdout_10pct_seed42_simfilter09_external": "external",
    "merged_holdout_10pct_seed42_simfilter09_nn03":     "nn03",
    "merged_holdout_10pct_seed42_simfilter09_nn05":     "nn05",
    "merged_holdout_10pct_seed42_simfilter09_total":    "total",
}

HOLDOUT_CACHE_DIR = os.path.join(REPO_ROOT, "feature_cache_holdouts")


# ── Feature dims ──────────────────────────────────────────────────────────────

ALL_FP_TYPES = ["ecfp", "maccs", "avalon", "tt", "rdkit", "scage1", "scage2", "mole"]

RDKIT_DIM = len(Descriptors._descList)

FP_DIM = {
    "ecfp":   1024,
    "maccs":  166,
    "avalon": 512,
    "tt":     1024,
    "rdkit":  RDKIT_DIM,
    "scage1": 512,
    "scage2": 512,
    "mole":   768,
}


# ── Hashing / canonicalization ────────────────────────────────────────────────

def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canon_smiles(smi: str):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


# ── Fingerprint computation ───────────────────────────────────────────────────

def _to_numpy_bitvect(bv, n_bits=None, drop_first=False):
    if n_bits is None:
        n_bits = bv.GetNumBits()
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    if drop_first:
        arr = arr[1:]
    return arr

def get_ecfp(mol, radius=2, nbits=1024):
    return _to_numpy_bitvect(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), n_bits=nbits)

def get_maccs(mol):
    bv = MACCSkeys.GenMACCSKeys(mol)
    return _to_numpy_bitvect(bv, n_bits=bv.GetNumBits(), drop_first=True)

def get_avalon(mol, nbits=512):
    from rdkit.Avalon import pyAvalonTools
    return _to_numpy_bitvect(pyAvalonTools.GetAvalonFP(mol, nbits), n_bits=nbits)

def get_tt(mol, nbits=1024):
    bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nbits)
    return _to_numpy_bitvect(bv, n_bits=nbits)

def get_rdkit_desc(mol):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([d[0] for d in Descriptors._descList])
    try:
        descs = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
    except Exception:
        descs = np.zeros(len(Descriptors._descList), dtype=np.float32)
    return np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)


def load_embed_csv(path: str) -> dict:
    df = pd.read_csv(path)
    embed_cols = [c for c in df.columns if c != "smiles"]
    out = {}
    for _, row in df.iterrows():
        cs = canon_smiles(row["smiles"])
        if cs:
            out[cs] = row[embed_cols].to_numpy(dtype=np.float32, copy=False)
    return out


# ── Pool cache (reuse baseline) ───────────────────────────────────────────────

def load_pool_features():
    """Load merged training pool (internal + external remaining)."""
    if not (os.path.exists(POOL_NPZ) and os.path.exists(POOL_MANIFEST)):
        raise FileNotFoundError(
            f"Baseline pool cache not found at {POOL_NPZ}. "
            "Build it via the baseline pipeline first."
        )
    d = np.load(POOL_NPZ, allow_pickle=True)
    smiles  = d["smiles"].tolist()
    labels  = d["labels"]
    sources = d["sources"]
    feats   = {t: d[t] for t in ALL_FP_TYPES}
    # Sync embedding dims with what the cache actually stores
    for key in ("scage1", "scage2", "mole", "rdkit"):
        FP_DIM[key] = feats[key].shape[1]
    return smiles, labels, sources, feats


# ── Holdout subset cache (build on demand) ────────────────────────────────────

def _subset_input_files(subset_name: str) -> dict:
    base = os.path.join(HOLDOUT_SUBSET_ROOT, subset_name)
    return {
        "label":  os.path.join(base, "label_holdout.csv"),
        "scage1": os.path.join(base, "scage1_holdout.csv"),
        "scage2": os.path.join(base, "scage2_holdout.csv"),
        "mole":   os.path.join(base, "mole_holdout.csv"),
    }


def _subset_cache_paths(subset_name: str):
    short = SUBSET_SHORT[subset_name]
    return (
        os.path.join(HOLDOUT_CACHE_DIR, f"{short}.npz"),
        os.path.join(HOLDOUT_CACHE_DIR, f"{short}.manifest.json"),
    )


def _subset_cache_matches(subset_name: str) -> bool:
    npz_path, man_path = _subset_cache_paths(subset_name)
    if not (os.path.exists(npz_path) and os.path.exists(man_path)):
        return False
    try:
        with open(man_path) as f:
            man = json.load(f)
    except Exception:
        return False
    expected = _subset_input_files(subset_name)
    stored = man.get("input_files", {})
    for key, path in expected.items():
        if not os.path.exists(path):
            return False
        if stored.get(key) != file_sha256(path):
            return False
    return True


def _build_subset_cache(subset_name: str):
    paths = _subset_input_files(subset_name)
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input {k} for {subset_name}: {p}")

    print(f"[Cache] Building holdout subset cache: {subset_name}")
    df = pd.read_csv(paths["label"])
    df = df[["smiles", "p_np"]].copy()
    df["smiles"] = df["smiles"].apply(canon_smiles)
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset="smiles").reset_index(drop=True)
    df["p_np"] = df["p_np"].replace({"BBB-": 0, "BBB+": 1}).astype(int)

    embed = {
        "scage1": load_embed_csv(paths["scage1"]),
        "scage2": load_embed_csv(paths["scage2"]),
        "mole":   load_embed_csv(paths["mole"]),
    }

    valid_smiles, valid_labels = [], []
    rows = {t: [] for t in ALL_FP_TYPES}
    failed = 0
    for _, row in df.iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        try:
            feats = {
                "ecfp":   get_ecfp(mol),
                "maccs":  get_maccs(mol),
                "avalon": get_avalon(mol),
                "tt":     get_tt(mol),
                "rdkit":  get_rdkit_desc(mol),
                "scage1": embed["scage1"].get(smi, np.zeros(FP_DIM["scage1"], dtype=np.float32)),
                "scage2": embed["scage2"].get(smi, np.zeros(FP_DIM["scage2"], dtype=np.float32)),
                "mole":   embed["mole"].get(smi, np.zeros(FP_DIM["mole"],   dtype=np.float32)),
            }
        except Exception:
            failed += 1
            continue
        valid_smiles.append(smi)
        valid_labels.append(row["p_np"])
        for t in ALL_FP_TYPES:
            rows[t].append(feats[t])

    labels = np.array(valid_labels, dtype=np.int64)
    feat_arrays = {t: np.stack(rows[t], axis=0) for t in ALL_FP_TYPES}

    os.makedirs(HOLDOUT_CACHE_DIR, exist_ok=True)
    npz_path, man_path = _subset_cache_paths(subset_name)
    np.savez(npz_path,
             smiles=np.array(valid_smiles, dtype=object),
             labels=labels,
             **feat_arrays)
    manifest = {
        "subset": subset_name,
        "size": len(valid_smiles),
        "failed": failed,
        "fp_dims": {t: int(feat_arrays[t].shape[1]) for t in ALL_FP_TYPES},
        "input_files": {k: file_sha256(p) for k, p in paths.items()},
    }
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Cache]   -> {npz_path} | size={len(valid_smiles)} failed={failed}")


def load_holdout_subset(subset_name: str):
    """Return (smiles, labels, feat_arrays) for one of the 5 holdout subsets."""
    if subset_name in SUBSET_SHORT.values():
        long_name = next(k for k, v in SUBSET_SHORT.items() if v == subset_name)
        subset_name = long_name
    if subset_name not in HOLDOUT_SUBSETS:
        raise ValueError(f"Unknown subset {subset_name!r}; choose from {list(SUBSET_SHORT.values())}")

    if not _subset_cache_matches(subset_name):
        _build_subset_cache(subset_name)

    npz_path, _ = _subset_cache_paths(subset_name)
    d = np.load(npz_path, allow_pickle=True)
    smiles = d["smiles"].tolist()
    labels = d["labels"]
    feats  = {t: d[t] for t in ALL_FP_TYPES}
    return smiles, labels, feats


# ── Combo helpers ─────────────────────────────────────────────────────────────

def get_mod_dims(combo) -> OrderedDict:
    od = OrderedDict()
    for t in combo:
        od[t] = FP_DIM[t]
    return od


def build_feature_matrix(combo, feat_arrays: dict) -> np.ndarray:
    parts = [feat_arrays[t] for t in combo]
    return np.concatenate(parts, axis=1)


def combo_str_to_tuple(combo_str: str) -> tuple:
    parts = combo_str.split("+")
    bad = [p for p in parts if p not in ALL_FP_TYPES]
    if bad:
        raise ValueError(f"Unknown fp types in combo: {bad}")
    return tuple(sorted(parts, key=lambda x: ALL_FP_TYPES.index(x)))


# ── Scaffold split ────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def scaffold_split_train_val(smiles_list: list, split_mode: str, seed: int,
                             train_ratio: float = 0.8):
    """8:2 train/val split, scaffold-grouped. Same logic as baseline."""
    _set_seed(seed)
    df = pd.DataFrame({"smiles": smiles_list})

    def get_scaffold(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else ""

    df["scaffold"] = df["smiles"].apply(get_scaffold)
    groups = list(df.groupby("scaffold").groups.values())

    if split_mode == "scaffold":
        groups = sorted(groups, key=lambda g: len(g), reverse=True)
    elif split_mode == "random_scaffold":
        rnd = random.Random(seed)
        rnd.shuffle(groups)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    n = len(df)
    train_cap = int(round(train_ratio * n))
    train_idx, val_idx = [], []
    for g in groups:
        g = list(g)
        if len(train_idx) + len(g) <= train_cap:
            train_idx += g
        else:
            val_idx += g

    return train_idx, val_idx


# ── CLI: build all subset caches ──────────────────────────────────────────────

if __name__ == "__main__":
    print("[Pool] Loading pool cache (read-only, reused from baseline) ...")
    pool_smiles, pool_labels, pool_sources, pool_feats = load_pool_features()
    print(f"[Pool] {len(pool_smiles)} mols, labels={np.bincount(pool_labels).tolist()}")
    print(f"[Pool] fp_dims = {{ {', '.join(f'{t}:{pool_feats[t].shape[1]}' for t in ALL_FP_TYPES)} }}")

    print("\n[Holdout subsets] Building caches if needed ...")
    for sub in HOLDOUT_SUBSETS:
        smi, lab, ft = load_holdout_subset(sub)
        print(f"  {SUBSET_SHORT[sub]:>10s}: {len(smi)} mols, labels={np.bincount(lab).tolist()}")
    print("\n[Done] All caches present.")
