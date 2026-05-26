# program_phase2.md — autoresearch combos-v2 Phase 2 agent instructions

You are the **Optuna search-space design engineer** for ONE feature combo of
the BBB multi-modal gMLP classifier. Phase 1 (architecture search up to
iter 200) is complete; the best architecture is frozen in `best_train.py`.
Your task is to find a hyperparameter configuration that beats the iter-197
phase-1 baseline.

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
    - iter-197 phase-1 baseline (`results/<combo>/results.tsv` best keep row)
    - all prior phase-2 keep rows in `results/<combo>/hpo/results.tsv`
- **discard** otherwise → `git revert <commit>`. The discarded commit
  remains in history so the TSV row's commit hash always resolves.
- No std buffer. `threshold_std` stays in the TSV for reference only —
  use it (and the architecture_log holdout signal for the same iter)
  as a secondary sanity check on marginal wins, but the keep gate
  itself is just the mean comparison.

## Files

**Phase-2 lever surface**: phase-1 nailed down the model architecture
(layer classes, multi-head SGU, pool skip-gate, R-Drop training loop).
Phase-2's job is to find better training hyperparameters for that frozen
architecture. Following the bbb-combo1 / combo2-phase2 pattern, the
lever is therefore split across TWO files — `optuna_combo.py`
(search-side) and `best_train.py` (training-side HP space). The model
class definitions stay frozen.

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
  was refactored here so Optuna trials can inject a config dict.
  Phase-2 may extend that surface as long as the two invariants below hold.
  - **Editable**:
    - `BASE_CONFIG` — add new HP keys (e.g. `lr_schedule`,
      `optimizer_type`, `label_smoothing`, `warmup_epochs`,
      `rdrop_alpha_max`, `rdrop_warmup_epochs`, ...) with defaults
      that reproduce the phase-1 behaviour when set.
    - `train_model` / `train_model_with_pruning` — add new optional
      arguments (default `None` or default-matching values) that
      consume the new BASE_CONFIG keys. New training hooks (LR
      scheduler step, label-smoothing in the loss, optimizer family
      switch, R-Drop alpha-warmup expose, etc.) belong here.
    - `build_and_train` / `build_and_train_with_pruning` — wire the
      cfg keys into optimizer / loss / scheduler construction, then
      pass to `train_model(_with_pruning)`.
    - `_build_lr_scheduler` and any analogous helpers introduced for
      new BASE_CONFIG keys.
  - **Frozen (DO NOT edit)**:
    - Model class definitions: `SpatialGatingUnit` (multi-head with
      `SGU_N_HEADS=2`, `DROP_V_PATH=0.08`), `gMLPBlock`, `gMLP`,
      `MultiModalGMLPFromFlat` (with `proj_scale`, `pool_skip_gate`,
      hardcoded `Dropout(0.10)` and `mod_drop_p=0.10`). Architecture
      is phase-1 output; phase-2 only tunes training around it.
    - The phase-1 `build_and_train` public-API call signature when
      invoked without a `config` override must produce byte-identical
      training to the iter197 baseline. Any new BASE_CONFIG key MUST
      default to a value that turns off the new behaviour (e.g.
      `lr_schedule="constant"`, `label_smoothing=0.0`) or matches
      the previously hard-coded value (e.g. `ema_decay=0.9993`,
      `ema_warmup_epochs=1`) so that `evaluate_combo.py` /
      `final_holdout_eval.py` (which call `build_and_train(...,
      config=None)` via `train.py`) keep reproducing phase-1 results
      bit-for-bit.
    - **Combo1-specific caveats**: The AdamW-override inside
      `train_model` (`lr = optimizer.param_groups[0]["lr"] * 1.25`,
      hardcoded `wd=0.003`, custom decay/no_decay param split where
      `{"alpha", "pool_skip_gate", "proj_scale"}` skip wd) is
      iter38/88/148/172 frozen state. As a result, `BASE_CONFIG.
      weight_decay` is **effectively unused** in the current baseline
      — Optuna searching it has no effect on the model. Similarly,
      the R-Drop two-pass loss with 5-epoch alpha warmup (iter80/
      iter106 frozen) is hardcoded. Phase-2 may untangle these later
      by exposing them as BASE_CONFIG keys with defaults that match
      the current hardcoded values (e.g. `rdrop_alpha_max=1.0`,
      `rdrop_warmup_epochs=5`, `adamw_wd=0.003`, `lr_multiplier=1.25`).
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
   - narrow / widen continuous HP ranges (`lr`, `lr_min_ratio`,
     `label_smoothing`, ...) based on previous study's top trials
   - swap `TPESampler` → `CmaEsSampler` for better local exploitation
   - increase `objective` aggregation from 3-seed to 5-seed mean
   - drop / re-include structural params (`d_model`/`d_ffn`/`depth`)
     from search
   - widen / narrow the LR-schedule subspace
   - **expose a new training-side HP** in `best_train.py` BASE_CONFIG
     and search it via `optuna_combo.py`: optimizer family (AdamW /
     Adam / Lion), R-Drop alpha-max / warmup-epochs expose, AdamW wd
     un-hardcode, head_dropout / mod_drop_p expose, `es_metric` swap,
     etc. Default value must reproduce phase-1 behaviour so the
     phase-1 build_and_train path stays byte-identical (see Files /
     `best_train.py` editable surface).
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
   num_epochs=50 with inline holdout). Combo1 budget defaults: 30 trials
   from iter1 (matches combo2 iter5-onward decision).
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
8. The Optuna SQLite DB (`combo1_hpo.db`) is gitignored and grows with
   every iter. Don't delete it — it lets you inspect/replot any past
   study by study_name = `auto_iter<N>_<commit>`.

## Direction-switching and termination rules

- **Convergence (early termination):** after **5 consecutive discards**,
  stop the loop and write a one-paragraph summary to
  `results/<combo>/hpo/conclusion.md` describing what was tried and
  why the threshold proved unbeatable. This is a valid scientific
  finding: "iter-197 architecture is near-optimal for this combo's HP
  space".
- **Direction switch:** if 3 consecutive discards happen in the same
  family of change (e.g., 3 narrowing attempts in a row), switch to a
  qualitatively different design (e.g., change sampler instead of
  narrowing search space).
- **Run ceiling:** absolute max of 18 iterations regardless of keep/
  discard mix. If 18 reached without a definitive conclusion, summarize
  and stop. (Ceiling matches combo2 phase-2; originally 12, extended
  to 18 from iter11+ in combo2 to allow a bbb-combo1-style narrowing
  cycle before declaring plateau.)
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
50 → 30 from iter5 onward in combo2 for wall-time) but each revision
should be documented in the BUDGET dict comment so prior-vs-current
iter comparisons are interpreted carefully (results.tsv n_trials
column records the actual value used per iter).

## Final reporting

When the loop terminates (convergence, ceiling, or human stop):

**Selection rule:** the phase-2 best is the `keep=True` row in
`results/<combo>/hpo/results.tsv` with the highest
`best_confirm_mean_val_auc`. Holdout signals (nn05 / total per-seed
mean±std for AUC/MCC/Accuracy) are tracked every iter and reported
alongside, but they do **not** gate selection. This matches the
bbb-combo1 / combo2-phase2 pattern: val drives keep/discard and final
selection; holdout is the generalization report card.

Artifacts:

- Winning row in `results/<combo>/hpo/results.tsv`.
- Corresponding `study_auto_iter<N>_<commit>.json` carries the top-3
  confirm details plus the per-trial `holdout` block (per-seed mean±std
  for AUC/MCC/Accuracy on nn05 and total).
- `architecture_log.md` row for direct phase-1 vs phase-2 comparison.
- Write `results/<combo>/hpo/conclusion.md` with:
  - Number of iterations run, distribution of keep/discard
  - Best confirm mean ± std vs phase-1 baseline (val) and phase-2 best's
    nn05 / total per-seed mean vs phase-1 iter197 (for context, not a
    gate)
  - The winning HP configuration (or "no improvement found")
  - 1-2 sentences on what kinds of changes mattered (if any)

## Reference: phase-1 baseline (combo1)

From `results/maccs+avalon+scage2+mole/results.tsv` (iter197, commit
`9bc22da482`):
- `mean_val_auc` = 0.854947 ± 0.002638 (10 scaffold seeds)
- Threshold to beat (current keep rule, mean-only): **0.854947**

From `results/maccs+avalon+scage2+mole/holdout_eval/iter0197_9bc22da482.json`
(10-seed per-seed mean, **for tracking only**):

| Subset | AUC (per-seed mean ± std) | MCC | Accuracy |
|---|---|---|---|
| nn05  | 0.856316 ± 0.007349 | 0.557481 ± 0.011201 | 0.857751 ± 0.005049 |
| total | 0.876385 ± 0.003684 | 0.609062 ± 0.010027 | 0.849887 ± 0.004363 |

- These are the phase-1 reference for holdout comparisons in the
  final report. They do NOT gate keep/discard or final selection (see
  Final reporting).

## iter1 starting state

`best_train.py` is a snapshot of iter197 (`9bc22da482`) refactored to
accept config-dict overrides. Five new phase-2 lever keys were added
to BASE_CONFIG with defaults that reproduce iter197 byte-for-byte:

- `lr_schedule="constant"`, `lr_warmup_epochs=0`, `lr_min_ratio=0.0`
- `label_smoothing=0.0`
- `ema_decay=0.9993`, `ema_warmup_epochs=1`

`optuna_combo.py::suggest_config()` initial state explores a 6-D
narrow pilot (lr + the five new phase-2 keys) with structural HP
pinned at iter197. **iter1 agent will redesign this as the first
lever choice** — the initial state is just a runnable baseline,
not a tuned starting point.
