# program_phase2.md — autoresearch combos-v2 Phase 2 agent instructions

You are the **Optuna search-space design engineer** for ONE feature combo of
the BBB multi-modal gMLP classifier. Phase 1 (architecture search up to
iter 200) is complete; the best architecture is frozen in `best_train.py`.
Your task is to find a hyperparameter configuration that beats the iter-200
phase-1 baseline by more than its 10-seed standard deviation.

You repeat:
**edit `optuna_combo.py` → run study via `evaluate_hpo.py` → keep if better,
else discard.**

## Combo and worktree

Find your combo from the current git branch (`combo1` / `combo2` / `combo3`):

| Branch suffix | Combo string |
|---|---|
| `combo1-maccs_avalon_scage2_mole`    | `maccs+avalon+scage2+mole` |
| `combo2-maccs_scage1_mole`           | `maccs+scage1+mole` |
| `combo3-maccs_scage1_scage2_mole`    | `maccs+scage1+scage2+mole` |

All commands below must use the combo string for your branch.

## Decision rule (keep / discard)

- **keep** iff `best_confirm_mean_val_auc > threshold_mean`
  where the threshold is the max over:
    - iter-200 phase-1 baseline (`results/<combo>/results.tsv` best keep row)
    - all prior phase-2 keep rows in `results/<combo>/hpo/results.tsv`
- **discard** otherwise → `git revert <commit>`. The discarded commit
  remains in history so the TSV row's commit hash always resolves.
- No std buffer. `threshold_std` stays in the TSV for reference only —
  use it (and the architecture_log holdout signal for the same iter)
  as a secondary sanity check on marginal wins, but the keep gate
  itself is just the mean comparison.

## Files

**Phase-2 lever surface**: phase-1 nailed down the model architecture
(layer classes, attention pool, head structure). Phase-2's job is to
find better training hyperparameters for that frozen architecture.
Following the bbb-combo1 pattern, the lever is therefore split across
TWO files — `optuna_combo.py` (search-side) and `best_train.py`
(training-side HP space). The model class definitions stay frozen.

- **`optuna_combo.py` — fully editable.** Modify any of:
  - `suggest_config()` — search space shape, ranges, log/categorical/int.
    Add new HP keys here when a new BASE_CONFIG kwarg is exposed.
  - sampler (`TPESampler`, `CmaEsSampler`, `RandomSampler`, ...)
  - pruner (`MedianPruner`, `NopPruner`, `SuccessiveHalvingPruner`, ...)
  - `objective()` — single-seed / multi-seed mean / staged variants
  - `run_seeds()` — eval protocol per trial
  - `SEARCH_SEEDS_DEFAULT`, `CONFIRM_SEEDS_DEFAULT`, `SPLIT_MODE`
    (constants override-able even though `evaluate_hpo.py` passes CLI
    flags that may override them).
- **`best_train.py` — editable on a restricted surface.** Phase-1 best
  was refactored here so Optuna trials can inject a config dict (commit
  `cc2a448`). Phase-2 may extend that surface as long as the two
  invariants below hold.
  - **Editable**:
    - `BASE_CONFIG` — add new HP keys (e.g. `lr_schedule`,
      `optimizer_type`, `label_smoothing`, `warmup_epochs`, ...) with
      defaults that reproduce the phase-1 behaviour when set.
    - `train_model` / `train_model_with_pruning` — add new optional
      arguments (default `None` or default-matching values) that
      consume the new BASE_CONFIG keys. New training hooks (LR
      scheduler step, warmup, label-smoothing in the loss, optimizer
      family switch, etc.) belong here.
    - `build_and_train` / `build_and_train_with_pruning` — wire the
      cfg keys into optimizer / loss / scheduler construction, then
      pass to `train_model(_with_pruning)`.
  - **Frozen (DO NOT edit)**:
    - Model class definitions: `SpatialGatingUnit`, `gMLPBlock`,
      `gMLP`, `MultiModalGMLPFromFlat`. Architecture is phase-1
      output; phase-2 only tunes training around it.
    - The phase-1 `build_and_train` public-API call signature when
      invoked without a `config` override must produce byte-identical
      training to the cc2a448 baseline. Any new BASE_CONFIG key MUST
      default to a value that turns off the new behaviour (e.g.
      `lr_schedule="constant"`, `label_smoothing=0.0`) so that
      `evaluate_combo.py` / `final_holdout_eval.py` (which call
      `build_and_train(..., config=None)` via `train.py`) keep
      reproducing phase-1 results bit-for-bit.
- Do **NOT** touch:
  - `train.py` — phase-1 active file, kept frozen for repro.
  - `prepare.py`, `evaluate_combo.py`, `final_holdout_eval.py`
  - `evaluate_hpo.py` — this iteration runner; `BUDGET` is the working
    source of truth for iter comparison (user can revise it but each
    revision must be documented in the BUDGET-dict comment).
  - `HOLDOUT_SUBSETS` in `optuna_combo.py` — frozen at `["nn05",
    "total"]` so the `architecture_log.md` column semantics stay
    stable across iters.
  - data paths anywhere.
- Edit `optuna_combo.py` and `best_train.py` with **partial Edit
  (old_string → new_string)**, never rewrite the whole file. Read the
  function first, then make a minimal diff.

## Iteration loop

For each iteration `N`:

1. Read `results/<combo>/hpo/results.tsv` last `keep=True` row (or the
   phase-1 baseline if empty) to know the current threshold. Also scan
   the `note` column of prior rows (keep or discard) for HP-region
   findings — agent should write these notes richly enough to inform
   the next iter, even when the iter was reverted.
2. Pick ONE search-design change. Examples:
   - narrow / widen continuous HP ranges (`lr`, `wd`, `dropout`, ...)
     based on previous study's top trials
   - swap `TPESampler` → `CmaEsSampler` for better local exploitation
   - increase `objective` aggregation from 3-seed to 5-seed mean
   - drop / re-include structural params (`d_model`/`d_ffn`/`depth`)
     from search
   - widen regularization range (`mod_drop_p`, `head_dropout`)
   - **expose a new training-side HP** in `best_train.py` BASE_CONFIG
     and search it via `optuna_combo.py`: LR schedule (cosine /
     warmup), optimizer family (AdamW / Adam / Lion), label smoothing,
     `es_metric` swap, etc. Default value must reproduce phase-1
     behaviour so the phase-1 build_and_train path stays byte-identical
     (see Files / `best_train.py` editable surface).
3. Apply via partial Edit on `optuna_combo.py` (and on `best_train.py`
   when the lever requires a new BASE_CONFIG key or new training hook).
4. **Haiku 4.5 sanity check.** Spawn a Haiku 4.5 agent to review the diff
   for syntax + obvious logic errors. If flagged, fix before continuing.
5. `git add optuna_combo.py && git commit -m "iter<N>: <short>"`.
6. Run evaluation IN BACKGROUND:
   ```
   conda run -n rapids-25.02 python evaluate_hpo.py \
     --combo <combo> --iter-id <N> --note "<short>; top-region <one-line summary>; next: <hint>"
   ```
   The `--note` value is the only place per-iter findings are recorded.
   Pack it: design change + top-trial HP region one-liner + suggestion
   for the next iter. This keeps the loop simple (one TSV, no extra
   files) while preventing learning loss on discard.
   Use Bash with `run_in_background=true`. **Do NOT poll or stream output.**
   Wait for the completion notification, then read the appended TSV row.
   Expected wall-time: ~45-55 min/iter (30 trials × 5 search seeds ×
   num_epochs=30 with pruning ~60% saved + top-3 × 10 confirm seeds ×
   num_epochs=50 with inline holdout). Budget lowered from 50 trials
   to 30 from iter5 onward (user decision; iter1-4 ran 50).
7. Inspect the new row's `keep` field:
   - `True`  → leave commit in place. Proceed to iter `N+1`.
   - `False` → **`git revert <commit>` only.** Do NOT use `git reset --hard`.
     The discarded commit must remain in history. The TSV row (including
     its rich `note`) stays — that's how findings persist across reverts.
   Holdout signal for every iter (keep or discard) is already in the
   confirm-phase output: `top_k_confirm[*]["holdout"]` in the study JSON
   and a per-study row in `architecture_log.md`. No separate retrain
   step — mirrors bbb-combo1's "val + holdout in one reevaluation pass"
   pattern.
8. The Optuna SQLite DB (`combo2_hpo.db` / per-worktree equivalent) is
   gitignored and grows with every iter. Don't delete it — it lets you
   inspect/replot any past study by study_name = `auto_iter<N>_<commit>`.

## Direction-switching and termination rules

- **Convergence (early termination):** after **5 consecutive discards**,
  stop the loop and write a one-paragraph summary to
  `results/<combo>/hpo/conclusion.md` describing what was tried and
  why the threshold proved unbeatable. This is a valid scientific
  finding: "iter-200 architecture is near-optimal for this combo's HP
  space".
- **Direction switch:** if 3 consecutive discards happen in the same
  family of change (e.g., 3 narrowing attempts in a row), switch to a
  qualitatively different design (e.g., change sampler instead of
  narrowing search space).
- **Run ceiling:** absolute max of 24 iterations regardless of keep/
  discard mix. If 24 reached without a definitive conclusion, summarize
  and stop. (Originally 12, extended to 18 from iter11+ to allow a
  bbb-combo1-style narrowing cycle of 5-6 iter on the iter10 low-wd
  cluster before declaring plateau; extended 18 -> 24 on 2026-05-30 by
  user decision to resume the loop past the iter18 holdout breakthrough.
  On resume the 5-consecutive-discard counter is reset to 0 from iter19
  -- the user's extend-and-continue directive overrides the trigger that
  had already fired at iter12-18, so the fresh segment iter19-24 gets a
  clean 5-consecutive-discard early-stop budget.)
- **Autonomy (default for this loop):** the agent chooses each iter's
  design lever using prior iter `note` rows + study JSON top-region
  summaries + architecture_log holdout signal as guide, then runs the
  full loop (edit → commit → evaluate_hpo → keep/discard → revert if
  needed → next iter) without per-iter user confirmation. User is
  alerted on milestones: a `keep=True` row, a 5-consecutive-discard
  early-termination trigger, a forced direction switch (3 consecutive
  same-family discards), or a hard failure (CUDA OOM, training
  divergence). Autonomy does not change *what* is allowed — only
  *who* makes the per-iter lever call (agent vs human).

## What CANNOT be optimized via this loop

The budget in `evaluate_hpo.py` (`BUDGET` dict) is the working source
of truth for iter comparison. The user may revise it (e.g., n_trials
50 → 30 from iter5 onward for wall-time) but each revision should be
documented in the BUDGET dict comment so prior-vs-current iter
comparisons are interpreted carefully (results.tsv n_trials column
records the actual value used per iter).

## Final reporting

When the loop terminates (convergence, ceiling, or human stop):

**Selection rule:** the phase-2 best is the `keep=True` row in
`results/<combo>/hpo/results.tsv` with the highest
`best_confirm_mean_val_auc`. Holdout signals (nn05 / total per-seed
mean±std for AUC/MCC/Accuracy) are tracked every iter and reported
alongside, but they do **not** gate selection. This matches the
bbb-combo1 pattern: val drives keep/discard and final selection;
holdout is the generalization report card.

Artifacts:

- Winning row in `results/<combo>/hpo/results.tsv`.
- Corresponding `study_auto_iter<N>_<commit>.json` carries the top-3
  confirm details plus the per-trial `holdout` block (per-seed mean±std
  for AUC/MCC/Accuracy on nn05 and total).
- `architecture_log.md` row for direct phase-1 vs phase-2 comparison.
- Write `results/<combo>/hpo/conclusion.md` with:
  - Number of iterations run, distribution of keep/discard
  - Best confirm mean ± std vs phase-1 baseline (val) and phase-2 best's
    nn05 / total per-seed mean vs phase-1 iter198 (for context, not a
    gate)
  - The winning HP configuration (or "no improvement found")
  - 1-2 sentences on what kinds of changes mattered (if any)

## Reference: phase-1 baseline (combo2)

From `results/maccs+scage1+mole/results.tsv` (iter198):
- `mean_val_auc` = 0.852661 ± 0.003551 (10 scaffold seeds)
- Threshold to beat (current keep rule, mean-only): **0.852661**

From `results/maccs+scage1+mole/holdout_eval/iter0198_2e8cbdc9f5.json`
(10-seed per-seed mean, **for tracking only**):
- nn05  ROC-AUC = 0.870011 ± 0.008095
- total ROC-AUC = 0.879823 ± 0.004205
- These are the phase-1 reference for holdout comparisons in the
  final report. They do NOT gate keep/discard or final selection (see
  Final reporting).

From the manual coarse pilot (cbc41bd, before this loop):
- Best confirm `mean_val_auc` = 0.852262 ± 0.003839
- Did NOT beat threshold. TPE converged on BASE_CONFIG defaults for
  structure; shifted only regularization (mod_drop_p 0.1 → 0.29,
  head_dropout 0.08 → 0.22, wd 1e-5 → 1e-4).
- Take-away for iter1: a different qualitative design is needed,
  not a narrower coarse re-run.
