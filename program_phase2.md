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

- **keep** iff `best_confirm_mean_val_auc > threshold_mean + threshold_std`
  where the threshold is the max over:
    - iter-200 phase-1 baseline (`results/<combo>/results.tsv` best keep row)
    - all prior phase-2 keep rows in `results/<combo>/hpo/results.tsv`
  with `threshold_std` taken from the same row.
- **discard** otherwise → `git revert <commit>`. The discarded commit
  remains in history so the TSV row's commit hash always resolves.
- The std buffer is intentional: confirm uses 10 fixed seeds, so even
  a noise-only winner would clear the bare mean. Buffer = "beat by at
  least one natural seed-variance".

## Files

- **EDIT ONLY** `optuna_combo.py`. Modify any of:
  - `suggest_config()` — search space shape, ranges, log/categorical/int
  - sampler (`TPESampler`, `CmaEsSampler`, `RandomSampler`, ...)
  - pruner (none currently; can add if you also hook train_model)
  - `objective()` — single-seed / multi-seed mean / staged variants
  - `run_seeds()` — eval protocol per trial
  - `SEARCH_SEEDS_DEFAULT`, `CONFIRM_SEEDS_DEFAULT`, `SPLIT_MODE`
    (constants override-able even though `evaluate_hpo.py` passes CLI flags
    that may override them)
- Do **NOT** touch:
  - `best_train.py` — frozen iter-200 architecture + `build_and_train`
    public API. New helpers (`*_with_pruning`) can be appended only
    when an iter explicitly hooks pruning; existing public API stays
    byte-identical so phase-1 paths reproduce.
  - `train.py` — phase-1 active file, kept frozen for repro.
  - `prepare.py`, `evaluate_combo.py`, `final_holdout_eval.py`
  - `evaluate_hpo.py` — this iteration runner; its frozen `BUDGET`
    (n_trials=50, top_k=3, search_num_epochs=30, search/confirm seeds)
    is the single source of truth for fair iter comparison.
  - `HOLDOUT_SUBSETS` in `optuna_combo.py` — frozen at `["nn05",
    "total"]` so the `architecture_log.md` column semantics stay
    stable across iters. (The constant lives in optuna_combo.py and
    is otherwise editable, but treat it as frozen for the loop.)
  - data paths anywhere.
- Edit `optuna_combo.py` with **partial Edit (old_string → new_string)**,
  never rewrite the whole file. Read the function first, then make a
  minimal diff.

## Iteration loop

For each iteration `N`:

1. Read `results/<combo>/hpo/results.tsv` last `keep=True` row (or the
   phase-1 baseline if empty) to know the current threshold. Also scan
   the `note` column of prior rows (keep or discard) for HP-region
   findings — agent should write these notes richly enough to inform
   the next iter, even when the iter was reverted.
2. Pick ONE search-design change. Examples:
   - narrow `lr` range based on previous study's top trials
   - swap `TPESampler` → `CmaEsSampler` for better local exploitation
   - increase `objective` aggregation from 3-seed to 5-seed mean
   - drop structural params (`d_model`/`d_ffn`/`depth`) from search
   - widen regularization range (`mod_drop_p`, `head_dropout`)
   - add a categorical for AdamW vs Lion vs SGD-style optimizer
     (would require also editing `best_train.py` — out of scope for now)
3. Apply via partial Edit on `optuna_combo.py`.
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
   Expected wall-time: ~2 hours (50 trials × 3 search seeds × num_epochs=30
   + top-3 × 10 confirm seeds × num_epochs=50).
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
- **Run ceiling:** absolute max of 40 iterations regardless of keep/
  discard mix. If 40 reached without a definitive conclusion, summarize
  and stop.
- **Autonomy mode (iter3+ onward):** the agent chooses each iter's
  design lever using prior iter `note` rows + study JSON top-region
  summaries as guide, then runs the full loop (commit → evaluate_hpo →
  holdout eval → keep/discard → revert if needed → next iter) without
  per-iter user confirmation. User is alerted only on milestones:
  a `keep=True` row, a 5-consecutive-discard early-termination trigger,
  a forced direction switch (3 consecutive same-family discards), or
  a hard failure (CUDA OOM, training divergence). The autonomy mode
  does not change *what* is allowed — only *who* makes the per-iter
  lever call (agent vs human).

## What CANNOT be optimized via this loop

The frozen budget in `evaluate_hpo.py` (`BUDGET` dict) is the single
source of truth for iter comparison. If you want to test "would a
larger budget help?" you must:
  1. Write a separate one-off study (not via this loop)
  2. Document the finding in `conclusion.md`
Changing `BUDGET` mid-loop would invalidate all prior keep/discard
comparisons.

## Final reporting

When the loop terminates (convergence, ceiling, or human stop):

- Last `keep=True` row in `results/<combo>/hpo/results.tsv` = winning
  HP configuration.
- Corresponding `study_auto_iter<N>_<commit>.json` has the full top-3
  confirm details (per-seed val_auc, params).
- Write `results/<combo>/hpo/conclusion.md` with:
  - Number of iterations run, distribution of keep/discard
  - Best confirm mean ± std vs phase-1 baseline (improvement or not)
  - The winning HP configuration (or "no improvement found")
  - 1-2 sentences on what kinds of changes mattered (if any)

## Reference: phase-1 baseline (combo2)

From `results/maccs+scage1+mole/results.tsv` (iter198):
- `mean_val_auc` = 0.852661 ± 0.003551 (10 scaffold seeds)
- Threshold to beat (iter1 of phase2): **0.856212**

From the manual coarse pilot (cbc41bd, before this loop):
- Best confirm `mean_val_auc` = 0.852262 ± 0.003839
- Did NOT beat threshold. TPE converged on BASE_CONFIG defaults for
  structure; shifted only regularization (mod_drop_p 0.1 → 0.29,
  head_dropout 0.08 → 0.22, wd 1e-5 → 1e-4).
- Take-away for iter1: a different qualitative design is needed,
  not a narrower coarse re-run.
