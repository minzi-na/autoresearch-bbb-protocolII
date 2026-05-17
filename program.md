# program.md — autoresearch_combos_v2 agent instructions

You are the architecture-search agent for ONE feature combo of the BBB multi-modal
gMLP classifier. Each worktree is pinned to a single combo (see git branch name).
You repeat: **edit `train.py` → 10-seed evaluate → keep if better, else discard**.

## Combo and worktree

Find your combo from the current git branch (`combo1` / `combo2` / `combo3`):

| Branch suffix | Combo string |
|---|---|
| `combo1-maccs_avalon_scage2_mole`    | `maccs+avalon+scage2+mole` |
| `combo2-maccs_scage1_mole`           | `maccs+scage1+mole` |
| `combo3-maccs_scage1_scage2_mole`    | `maccs+scage1+scage2+mole` |

All commands below must use the combo string for your branch.

## Decision rule

- **keep** iff `mean_val_auc` (10 seeds, scaffold split, from `evaluate_combo.py`)
  is strictly greater than the best `keep=True` row in `results/<combo>/results.tsv`.
- **discard** otherwise → `git revert` the architecture commit.
- 5-subset holdout numbers do NOT influence keep/discard. They are reporting-only.

## Files

- **EDIT ONLY** `train.py`, and only inside the model classes (`SpatialGatingUnit`,
  `gMLPBlock`, `gMLP`, `MultiModalGMLPFromFlat`) or `train_model` / `eval_model`.
- Do **NOT** touch:
  - `BASE_CONFIG` block in `train.py` (hyperparameters are frozen by design)
  - `build_and_train()` signature, `apply_rdkit_scaler()` (frozen public API)
  - `prepare.py`, `evaluate_combo.py`, `final_holdout_eval.py`
  - data paths anywhere
- Edit `train.py` with **partial Edit (old_string → new_string)**, never rewrite
  the whole file. Read the function/class first, then make a minimal diff.

## Iteration loop

For each iteration `N`:

1. Read `results/<combo>/results.tsv` last `keep=True` row to know current best.
   If empty, the next eval is the baseline reproduction (iter 1).
2. Pick ONE architectural change and apply it to `train.py` via Edit.
3. **Haiku 4.5 sanity check.** Spawn a Haiku 4.5 agent to review the diff for
   syntax + obvious logic errors. If flagged, fix before continuing.
4. `git add train.py && git commit -m "iter<N>: <short>"`.
5. Run evaluation IN BACKGROUND:
   ```
   conda run -n rapids-25.02 python evaluate_combo.py \
     --combo <combo> --iter-id <N> --note "<short>"
   ```
   Use Bash with `run_in_background=true`. **Do NOT poll or stream output.**
   Wait for the completion notification, then read the appended TSV row.
6. Inspect the new row's `keep` field:
   - `True`  → leave commit in place. Move on to iter `N+1`.
   - `False` → `git revert <commit>` (or `git reset --hard HEAD~1` if no other
     commits piggybacked on top). Do not delete the row in TSV — discarded
     attempts are part of the search log.
7. (Optional, manual) When you believe a kept iteration is the current global
   best of the search, run `final_holdout_eval.py` once for the 5-subset
   breakdown. This is reporting only.

## Hard rules

- **No verbose deliberation on already-decided items.** Pick the change, apply,
  Haiku-review, run. Save extended reasoning for genuinely novel directions or
  for analyzing surprising results.
- **Background runs only.** Never use a polling/wait loop that streams seed
  output to the conversation.
- **One file, one diff.** Partial Edit on `train.py` only.
- **Never commit `feature_cache_holdouts/` or `*.pth`.** They are gitignored.

## Common architectural moves (non-exhaustive)

These are the dimensions the agent should explore:

1. **Per-modality projection** — pre-norm, scaling, residual projection, 2-layer projection.
2. **gMLP backbone** — SGU variants (attention SGU, multi-axis SGU, gated SGU
   cascade), normalization location, FFN variants, dropout/stoch-depth schedules.
3. **Pooling** — gated pool tuning, attention pool, learnable skip mix.
4. **Stability** — residual scaling, layer init, label smoothing, optimizer
   tweaks INSIDE `train_model` (but `BASE_CONFIG` lr/wd/bs frozen).

Reuse insight from past BBB autoresearch runs but verify against the current
results.tsv — every architecture must re-prove itself on this combo's val_auc.
