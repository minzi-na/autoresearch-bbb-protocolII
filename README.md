# autoresearch_combos_v2

Architecture-only optimization loop for a multi-modal gMLP BBB classifier,
on top-3 feature combos from `combos_v2/`.

## Task

For each of the 3 combos below, find the model architecture that maximizes
**scaffold-split mean val_auc (10 seeds)** while keeping all hyperparameters
in `BASE_CONFIG` (`train.py`) frozen.

| Combo | Branch / Worktree |
|-------|------------------|
| `maccs+avalon+scage2+mole`        | `combo1` |
| `maccs+scage1+mole`               | `combo2` |
| `maccs+scage1+scage2+mole`        | `combo3` |

## Data

- **Training pool** (8:2 train/val, scaffold split): `internal_curated_remaining ∪ external_cls_only_remaining`, deduped — 9786 mols.
  Features pre-cached at `/home/minji/feature_cache_merged/pool.npz`. Reused as-is.
- **Generalization eval (5 holdout subsets)** under `/home/minji/holdout_subset/`:
  - `internal`, `external`, `nn03`, `nn05`, `total`.
  - Caches built on demand into `feature_cache_holdouts/`.
- No "external dataset" generalization track — external is in the training pool.

## Decision criterion

- **keep/discard** = mean val_auc (10 seeds, scaffold split) vs. current best keep row in `results/<combo>/results.tsv`.
- 5-subset holdout numbers are **not** part of keep/discard. They are only run on
  iterations the agent deems final candidates, via `final_holdout_eval.py`.

## Frozen vs. editable

| File | Status |
|------|--------|
| `train.py` model classes + `train_model` / `eval_model` | **EDITABLE** by agent |
| `train.py` `BASE_CONFIG`, `build_and_train` signature, `apply_rdkit_scaler` | frozen |
| `prepare.py` | frozen |
| `evaluate_combo.py` | frozen |
| `final_holdout_eval.py` | frozen |

## Runtime

```bash
# one-time: build 5 holdout subset caches
conda run -n rapids-25.02 python prepare.py

# one iteration (10 seeds, ~mins on H100/A100)
conda run -n rapids-25.02 python evaluate_combo.py \
    --combo maccs+scage1+mole --iter-id 1 --note "baseline"

# after a keep iteration that looks like the best so far:
conda run -n rapids-25.02 python final_holdout_eval.py \
    --combo maccs+scage1+mole --iter-id 7 --note "best so far"
```

## Layout

```
.
├── README.md
├── program.md                # agent instructions
├── prepare.py                # frozen — cache loader + holdout subset builder
├── train.py                  # ★ agent edits this
├── evaluate_combo.py         # frozen — iteration runner (val_auc only)
├── final_holdout_eval.py     # frozen — 5-subset breakdown for keep iterations
├── feature_cache_holdouts/   # .gitignored — auto-built on first run
└── results/<combo>/
    ├── results.tsv           # one row per iteration
    ├── architecture_log.md   # one row per final_holdout_eval run
    ├── iter<NNNN>_per_seed.json
    └── holdout_eval/iter<NNNN>_<commit>.json
```

## Env

`rapids-25.02` conda env (`/home/minji/anaconda3/envs/rapids-25.02`) — same as baseline.
