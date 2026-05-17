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
2. Pick ONE architectural change. **Consult `architecture_ideas.md`** if you
   need direction — it contains the component-by-component improvement catalog
   and a priority-ordered table.
3. Apply via partial Edit on `train.py`.
4. **Haiku 4.5 sanity check.** Spawn a Haiku 4.5 agent to review the diff for
   syntax + obvious logic errors. If flagged, fix before continuing.
5. `git add train.py && git commit -m "iter<N>: <short>"`.
6. Run evaluation IN BACKGROUND:
   ```
   conda run -n rapids-25.02 python evaluate_combo.py \
     --combo <combo> --iter-id <N> --note "<short>"
   ```
   Use Bash with `run_in_background=true`. **Do NOT poll or stream output.**
   Wait for the completion notification, then read the appended TSV row.
7. Inspect the new row's `keep` field:
   - `True`  → leave commit in place. Move on to iter `N+1`.
   - `False` → **`git revert <commit>` only.** Do NOT use `git reset --hard`.
     The discarded commit must remain in history so that `git show <commit>`
     resolves the TSV row's commit hash months later for analysis or to spawn
     a variant of a failed attempt. The TSV row itself stays as well —
     discarded attempts are part of the search log.
8. (Optional, manual) When you believe a kept iteration is the current global
   best of the search, run `final_holdout_eval.py` once for the 5-subset
   breakdown. This is reporting only.

## Direction-switching and termination rules

- **3 consecutive discards in the same component direction → switch direction.**
  Move to the next item in `architecture_ideas.md`'s priority table. Crashes
  do not count toward this streak — fix the bug and retry.
- **Crash handling.** If `evaluate_combo.py` exits non-zero or the appended
  row is missing/empty, treat it as a crash. Inspect `tail -n 50` of the
  background output file, fix the code, retry. If unfixable in 1–2 attempts,
  revert the commit and move on.
- **Per-run timeout: 60 minutes.** 10 seeds × ~5 min/seed is typical; if a run
  exceeds 60 min, kill the background process and mark it as crash.
- **After the priority table is exhausted**, synthesize variants/combinations
  of previously kept changes. Do not re-try ideas already X-marked as failed
  unless you have a meaningfully new variant.
- **NEVER STOP** the loop on your own. Continue until the user interrupts.

## Hard rules

- **No verbose deliberation on already-decided items.** Pick the change, apply,
  Haiku-review, run. Save extended reasoning for genuinely novel directions or
  for analyzing surprising results.
- **Background runs only.** Never use a polling/wait loop that streams seed
  output to the conversation.
- **One file, one diff.** Partial Edit on `train.py` only.
- **Never commit `feature_cache_holdouts/` or `*.pth`.** They are gitignored.

## Where to look next

For concrete architecture ideas, priority order, code snippets, and
combo-specific notes, read **`architecture_ideas.md`**. It groups proposals
into six components (Projection / SGU / Pooling / Training dynamics /
Cross-modal FiLM / Regularization) and provides a 1-N priority table.

Reuse insight from past BBB autoresearch runs but verify against the current
combo's results.tsv — every architecture must re-prove itself on this combo's
val_auc.
