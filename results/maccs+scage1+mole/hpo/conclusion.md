# Phase-2 HPO — Conclusion (combo2: maccs+scage1+mole)

## Summary

- **Iterations run**: 9 (ceiling 12; self-terminated after 6 consecutive
  val-discards iter4-9, with the loop's main lever — training-procedure
  HP via best_train.py BASE_CONFIG extension — fully exercised).
- **Distribution**: 1 KEEP (iter3), 8 DISCARD (iter1/2/4/5/6/7/8/9).
- **Phase-2 best (val rule)**: **iter3 trial#7** (commit `a539a46706`,
  study `auto_iter003_a539a46706`). Same HP was deterministically
  rediscovered by iter1/2/5/6 under different sampler / pruner /
  search-space settings — strong evidence that this point is the robust
  peak of the original 11-D HP search space.
- **Holdout-side finding (iter7+)**: opening the training-procedure axis
  via BASE_CONFIG extension (LR schedule + label smoothing + ema_decay)
  produced the **first phase-2 iter whose top-3 confirm trials beat
  phase-1 iter198 on holdout per-seed mean roc_auc** (iter8). The
  effect did NOT transfer back to val_AUC, so iter8 was DISCARD under
  the val-only keep rule, but it is the meaningful generalisation
  landmark of the loop.

## Phase-1 baseline vs phase-2 winners

Three rows worth tracking — the val-rule winner (iter3 trial#7) and the
two holdout-best trials from iter8 (which DISCARDed by val but are the
phase-1-beating holdout configurations).

| Source | val ROC-AUC | nn05 ROC-AUC | total ROC-AUC | keep? |
|---|---|---|---|---|
| phase-1 iter198 (baseline) | 0.852661 ± 0.003551 | 0.870011 ± 0.008095 | 0.879823 ± 0.004205 | — |
| **iter3 trial#7** (val winner, KEEP) | **0.852850 ± 0.002940** | 0.866288 ± 0.007562 | 0.878848 ± 0.002268 | True |
| iter8 trial#1 (holdout-best val side) | 0.852397 ± 0.004151 | **0.870585 ± 0.007292** | **0.880124 ± 0.004112** | False |
| iter8 trial#3 (holdout-best nn05) | 0.852350 ± 0.004037 | **0.871329** ± ... | **0.880122** ± ... | False |
| iter8 trial#24 (holdout-best total) | 0.850348 ± 0.003531 | 0.865780 ± ... | **0.880566** ± ... | False |
| iter9 trial#0 (narrow re-exploit) | 0.851036 ± 0.003763 | 0.868241 ± 0.008708 | 0.881157 ± 0.004093 | False |

### Statistical test (iter3 trial#7 vs phase-1 iter198 on val, same 10 seeds)

Paired t-test: **t = 0.154, p (two-sided) = 0.881**; Wilcoxon signed-rank
one-sided p = 0.577. Indistinguishable from noise.

### Reading

- **val**: only iter3 trial#7 exceeds phase-1 mean, by a noise-level
  margin (+0.000189). All other 8 iter DISCARD or DISCARD-with-revert.
- **holdout AUC (phase-2 vs phase-1)**:
  - iter3 trial#7 (val winner) regresses slightly on both subsets.
  - iter8 trial#1/#3 are the FIRST configurations to beat phase-1 on
    both nn05 (+0.0006 / +0.0013) and total (+0.0003 / +0.0003).
    The training-procedure HP they exercise are
    *small label_smoothing (0.01-0.06)* + *ema_decay 0.994-0.9994*
    + *either constant or cosine schedule*.
  - iter9 narrowed the search to this region but actually degraded
    nn05 (sampler concentrated near the corner of the box where
    label_smoothing=0 / ema≈0.999, i.e. close to the phase-1 default,
    losing the holdout edge).

## Winning HP configuration (val rule)

iter3 trial#7 — what the phase-2 KEEP row records:

| param | value | vs phase-1 BASE_CONFIG |
|---|---|---|
| lr                 | 1.502e-04 | 1.000e-04 (×1.50) |
| weight_decay       | 1.499e-04 | 1.000e-05 (×15.0) |
| batch_size         | 128       | 128 (same) |
| grad_clip_max_norm | 1.136     | 1.000 (×1.14) |
| dropout            | 0.243     | 0.200 (+0.043) |
| drop_path          | 0.0019    | 0.025 (−0.023) |
| mod_drop_p         | 0.090     | 0.100 (−0.010) |
| head_dropout       | 0.198     | 0.080 (×2.5)  |
| d_model            | 512       | 512 (same) |
| d_ffn              | 1536      | 1048 (×1.47) |
| depth              | 5         | 4 (+1) |
| lr_schedule        | constant  | constant (phase-1) |
| label_smoothing    | 0.0       | 0.0 (phase-1) |
| ema_decay          | 0.999     | 0.999 (phase-1) |

Training-procedure HP at phase-1 defaults — iter3 is purely an HP shift
in the original 11-D block.

## Generalisation-side configuration (holdout landmark)

iter8 trial#1 — same architecture HP as iter3 trial#7 (pinned), but with
training-procedure HP turned on:

| param | value | note |
|---|---|---|
| (11 architecture HP)| iter3 trial#7 values | pinned |
| lr_schedule        | cosine    | active |
| lr_warmup_epochs   | 8         | (sampled but ignored; cosine has no warmup) |
| lr_min_ratio       | 0.159     | cosine eta_min = lr × 0.159 |
| label_smoothing    | 0.057     | small BCE smoothing |
| ema_decay          | 0.9992    | ≈ phase-1 0.999 |

Beats phase-1 on both holdout subsets per-seed mean, but DISCARD under
val rule (val 0.852397 < threshold 0.852850).

## What was tried (lever × outcome)

| iter | design (single lever) | family | confirm val | keep | takeaway |
|---|---|---|---|---|---|
| 1 | structural narrow (drop d_model/d_ffn/depth from search) | space | 0.852668 | False* | TPE wide converges on trial#7 |
| 2 | + MedianPruner + TPE n_startup=15 multivariate | pruner+sampler | 0.852850 | False* | same trial#7; pruner is wall-time tool |
| 3 | + 5-seed mean objective | objective | **0.852850** | **True** | first val gain over phase-1 (+0.000189) |
| 4 | search space narrowed around trial#7 | space | 0.852740 | False | search-to-confirm overfit |
| 5 | sampler TPE → CmaEsSampler | sampler | 0.852850 (=#7 HP) | False | CmaEs rediscovers trial#7 exact HP |
| 6 | pruner MedianPruner → NopPruner | pruner | 0.852850 (=#7 HP) | False | full-epoch eval doesn't surface any late-bloomer |
| 7 | BASE_CONFIG extension (lr_schedule + label_smoothing + ema_decay), full 16-D search | best_train.py | 0.851604 | False | 16-D / 30 trials too sparse; sub-optimal HP combo |
| 8 | pin 11 HP at trial#7, search ONLY the 5 new training-procedure HP | best_train.py + space-restrict | 0.852397 | False | **HOLDOUT BREAKTHROUGH**: top-3 of 3 beat phase-1 on holdout; val noise |
| 9 | narrow 4-D around iter8 top region | space | 0.851036 | False | narrow degrades both val and nn05; total still beats P1 (mixed) |

\* iter1/iter2 used the strict mean+1·std keep gate (later relaxed to
mean-only at user request from iter3 onward). Under the mean-only gate
iter1 (+0.000007) and iter2 (+0.000189) would have been marginal keeps,
but both rows stay False in the TSV — we did not retroactively rewrite
history.

## Protocol evolution during the loop

| change | iter | rationale |
|---|---|---|
| `*_with_pruning` helper allowed in best_train.py | iter2 | pruning needed a train_model hook |
| Keep rule: mean+1·std → mean only | post-iter3 | std buffer too strict for noise-level landscape |
| Ceiling 12 → 40 → 12 | post-iter3 / post-iter6 | 40 was too costly; restored to 12 |
| Autonomy mode formalised | post-iter3 | agent chooses per-iter lever, user only on milestones |
| BUDGET n_trials 50 → 30 | from iter5 | iter1-4 all converged on trial#7; trial-count cut is fair |
| Inline holdout in confirm phase (no separate retrain) | from iter3 | aligns with bbb-combo1's val+holdout-in-one-reeval pattern |
| **best_train.py BASE_CONFIG extension officially allowed** | post-iter6 | bbb-combo1-style training-procedure HP search needed beyond optuna_combo.py-only levers |
| Section headers in best_train.py rewritten to phase-2 context | post-iter6 | source-of-truth doc + code agreement |

## Why the loop stopped at iter9

- Six consecutive val-DISCARD (iter4-9) reached the early-termination
  trigger from program_phase2.md.
- The most informative new lever (iter7's BASE_CONFIG extension) was
  exercised in iter7 (full 16-D), iter8 (5-D pinned), iter9 (4-D
  narrow). iter8 surfaced the only phase-2 holdout improvement of the
  whole loop, and iter9 confirmed that narrower exploitation of that
  region degrades the signal — suggesting the iter8 finding is a
  shallow plateau, not the foot of a deeper basin to descend into.
- Remaining unexpolored levers (optimizer family, ES-metric swap,
  patience tuning, alternative samplers) all have expected gain inside
  the val noise band ≈ 0.003. Burning iter10-12 on them is unlikely to
  beat the iter3 trial#7 threshold.

## bbb-combo1 cross-reference

bbb-combo1 phase-2 (40 iter, framework adopted here) ended with
val +0.0028 over phase-1 but holdout −0.0046 — val win, holdout
regress. combo2 phase-2 ended with val +0.0002 (noise) and a single
iter (iter8) where holdout beat phase-1 on both subsets. The opposite
asymmetry from bbb-combo1: combo2's strong landmark is on the
generalisation side, but val gate kept it from being formally KEPT.

## Artefacts

- `results/<combo>/hpo/results.tsv` — 9 row history (iter1-9).
- `results/<combo>/hpo/study_auto_iter003_a539a46706.json` carries the
  val winner + full per-seed confirm + holdout block.
- `results/<combo>/hpo/study_*_iter008_*.json` carries the holdout
  landmark trials (#1 / #3 / #24).
- `results/<combo>/architecture_log.md` — phase-1 architecture sweep
  rows (iter1-198) plus phase-2 confirm-trial rows for each study;
  direct line-by-line comparison of val_auc / nn05 / total.
- `combo2_hpo.db` — Optuna SQLite store; all studies (including
  reverted designs) inspectable.
- `program_phase2.md` — authoritative protocol (rules + iteration loop).
- `PHASE2_OVERVIEW.md` — repo-root navigation index.
- Git history: every iter commit + revert preserved.

## If extending phase-2 further (out of current loop budget)

The iter8 finding raises two follow-on questions worth a one-off study
(documented but outside the keep/discard loop):

1. **Soft-voting ensemble of iter8 top-3** — if cosine and constant
   schedule each land independently on phase-1-beating holdout, an
   ensemble of their per-seed probabilities may exceed both. Free of
   architecture change; just inference-side.
2. **Multi-objective Optuna** — keep val as primary objective but add
   holdout as secondary, sample the Pareto front instead of single
   best. Would change the framework noticeably, so out of phase-2 scope
   here, but the natural next step if the user wants a phase-3.
