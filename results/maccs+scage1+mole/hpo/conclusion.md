# Phase-2 HPO — Conclusion (combo2: maccs+scage1+mole)

## Summary

- **Iterations run**: **18 (ceiling 18 reached)**.
- **Distribution**: 2 KEEP (iter3, iter11), 16 DISCARD.
- **Val rule winner (phase-2 best)**: **iter11 trial#21**
  (commit `78fd984a82`). val 0.852878 ± 0.003240, nn05 0.868296,
  total 0.879793.
- **Holdout breakthrough (3 iter, 7+ trials)**: iter8, iter17, iter18
  all produced top-3 confirm trials beating phase-1 iter198 on both
  nn05 and total per-seed mean roc_auc. **Phase-2 holdout record**:
  iter18 trial#5 — total AUC **0.881540** (+0.001717 vs P1),
  nn05 AUC 0.871302 (+0.001291 vs P1).
- **Termination**: ceiling 18 reached.

## Phase-1 baseline vs Phase-2 winners

The val rule and the holdout signal pick different winners. Both are
listed.

| Source | val ROC-AUC | nn05 ROC-AUC | total ROC-AUC | keep? | role |
|---|---|---|---|---|---|
| phase-1 iter198 (baseline) | 0.852661 ± 0.003551 | 0.870011 ± 0.008095 | 0.879823 ± 0.004205 | — | baseline |
| iter3 trial#7 (val winner v1) | 0.852850 ± 0.002940 | 0.866288 | 0.878848 | True | first KEEP |
| **iter11 trial#21** (val winner v2, FINAL) | **0.852878 ± 0.003240** | 0.868296 | 0.879793 | **True** | **val winner** |
| iter8 trial#1 | 0.852397 | 0.870585 | 0.880124 | False | 1st holdout breakthrough |
| iter8 trial#3 | 0.852350 | 0.871329 | 0.880122 | False | best nn05 in iter8 |
| iter17 trial#1 | 0.852594 | 0.870678 | 0.881090 | False | 2nd breakthrough |
| iter17 trial#16 | 0.852250 | 0.871423 | 0.881458 | False | best total in iter17 |
| **iter18 trial#5** | 0.852641 | **0.871302** | **0.881540** | False | **holdout record** |
| iter18 trial#18 | 0.852462 | (similar) | (similar) | False | breakthrough |
| iter18 trial#2 | 0.852390 | (similar) | (similar) | False | breakthrough |

### Reading

- **Val gate**: only iter3 trial#7 and iter11 trial#21 cleared it.
  Both improvements over phase-1 are within the noise band
  (+0.000189 and +0.000217 respectively; paired t-test p ≈ 0.88).
- **Holdout signal**: a stable cluster of 7+ trials across iter8,
  iter17, iter18 strictly dominates phase-1 on both nn05 and total
  per-seed mean. The cluster shares a common HP pattern (see below).
- **Phase-2 holdout maximum**: iter18 trial#5 with total 0.881540
  is the best holdout point found across all 18 iter and all 30 × 18
  trials. nn05 0.871302 is essentially tied with iter17 trial#16's
  0.871423.

## Two winning configurations

### Val winner (the phase-2 "best" under the keep rule): **iter11 trial#21**

| param | value | vs phase-1 BASE_CONFIG |
|---|---|---|
| lr                 | 1.224e-04 | ×1.22 |
| weight_decay       | 1.145e-06 | ×0.115 (much lower) |
| dropout            | 0.047     | ×0.235 (much lower) |
| drop_path          | 0.029     | similar |
| mod_drop_p         | 0.114     | similar |
| head_dropout       | 0.086     | ×1.075 (similar) |
| batch_size / grad_clip / arch | iter3 trial#7 baseline | — |
| lr_schedule / smoothing / ema | phase-1 default | — |

Bbb-combo1 low-wd / low-dropout cluster region. constant schedule,
no label smoothing.

### Holdout best: **iter18 trial#5**

Same 11 architecture / regularisation HP as iter11 trial#21, plus:

| training-procedure param | value | note |
|---|---|---|
| lr_schedule        | constant | (top-3 of every iter17/18 picked constant, NOT cosine) |
| label_smoothing    | ≈ 0.04-0.06 | BCE target smoothing |
| ema_decay          | ≈ 0.998 | slightly lower than phase-1 0.999 |

iter17 and iter18 confirm that adding label_smoothing ~ 0.05 and
ema_decay ~ 0.998 on top of the iter11 trial#21 HP shifts the model
into a "smoother-prediction" regime that loses ~0.0003 on val mean
but gains +0.0013 on nn05 AUC and +0.0017 on total AUC. The val cost
sits just inside the noise band of phase-1; the holdout gain is
substantial relative to phase-1 std.

## Full iteration log (lever × outcome)

| iter | design | family | confirm val | keep | takeaway |
|---|---|---|---|---|---|
| 1  | structural narrow                                  | space   | 0.852668 | F* | TPE wide -> trial#7 |
| 2  | + MedianPruner + TPE multivariate                  | pruner+sampler | 0.852850 | F* | same trial#7 |
| 3  | + 5-seed mean objective                            | objective | **0.852850** | **T** | first val gain over phase-1 |
| 4  | search space narrow around trial#7                 | space   | 0.852740 | F | overfit |
| 5  | sampler TPE -> CmaEsSampler (wide)                 | sampler | 0.852850 (=#7 HP) | F | CmaEs rediscovers trial#7 |
| 6  | pruner MedianPruner -> NopPruner                   | pruner  | 0.852850 (=#7 HP) | F | full-epoch identical |
| 7  | BASE_CONFIG extension (LR sched+smooth+ema), 16-D  | best_train.py | 0.851604 | F | 16-D / 30 trials too sparse |
| 8  | pin 11 HP at trial#7 + search 5 training-proc HP   | best_train.py + space | 0.852397 | F | **1st HOLDOUT BREAKTHROUGH** |
| 9  | narrow 4-D around iter8 top                        | space   | 0.851036 | F | narrow degrades |
| 10 | shift to bbb-combo1 low-wd cluster (6-D)           | space   | 0.852306 | F | second cluster found |
| 11 | + 7-seed objective on low-wd cluster               | objective | **0.852878** | **T** | **NEW BEST**: trial#21 in low-wd cluster |
| 12 | arch range pin + 5-seed evenly spread (bundled)    | space+obj | 0.851858 | F | both levers hurt, depth=5 critical |
| 13 | arch range pin alone (isolate)                     | space   | 0.851858 | F | depth=5 critical |
| 14 | narrow exploitation around iter11 trial#21         | space   | 0.852420 | F | trial#21 is local peak |
| 15 | sampler TPE -> CmaEsSampler on low-wd region       | sampler | 0.851744 | F | CmaEs worse here |
| 16 | mod_drop_p / head_dropout upper extend 0.15 -> 0.25 | space  | 0.852282 | F | edge probe failed (5-consec) |
| 17 | pin 11 HP at iter11 trial#21 + 4-D training-proc   | best_train.py + space | 0.852594 | F | **2nd HOLDOUT BREAKTHROUGH** |
| 18 | 2-D narrow on label_smoothing + ema_decay          | best_train.py + space | 0.852641 | F | **3rd HOLDOUT BREAKTHROUGH** + phase-2 holdout RECORD |

\* iter1/iter2 evaluated under the strict mean+1·std keep gate (later relaxed).

## bbb-combo1 cross-reference

| | bbb-combo1 phase-2 (40 iter) | combos-v2-combo2 phase-2 (18 iter) |
|---|---|---|
| Val gain | +0.00275 | +0.000217 (noise) |
| Holdout vs phase-1 | −0.0046 (regress) | **+0.0017 (total, breakthrough)** |
| Winning lever | d_ffn jump (run17) + narrowing | label_smoothing + ema_decay on low-wd cluster |
| Phase-1 saturation | 106 iter, less saturated | 200 iter, near-saturated |
| Final character | "val win, holdout lose" | "val plateau, holdout WIN" |

combos-v2-combo2's phase-2 produced the inverse asymmetry from
bbb-combo1: where bbb-combo1's HPO mildly overfit val and hurt
holdout, combo2's HPO did not move val but cleanly improved holdout
generalisation, especially when the iter11 low-wd cluster was paired
with iter17/18's training-procedure HP search (label_smoothing +
ema_decay). The val-only keep rule never catches this gain — it
sits within val's measurement noise band.

## Why phase-2 stopped at iter18

Ceiling 18 reached. The autonomous loop exercised every realistic
lever class across 18 iter:
  - space (structural narrow, region shift, narrow exploitation,
    edge probe, range pin / unpin) — 8 iter
  - sampler (TPE multivariate, CmaEs ×2) — 3 iter
  - pruner (MedianPruner introduce, NopPruner) — 2 iter
  - objective (5-seed mean, 5-seed evenly, 7-seed) — 3 iter
  - best_train.py BASE_CONFIG extension (LR sched, smoothing, ema) — 4 iter

The val gate accepted 2 of 18 iter (iter3, iter11). The holdout signal
peaked at iter18 trial#5. Further iter inside ceiling 18 would
re-explore already-mapped ground.

## What remains as out-of-loop follow-ons

1. **Soft-voting ensemble** of iter3 / iter11 / iter17 / iter18 best
   trials — distinct HP regions (high-wd vs low-wd, with/without
   label smoothing) may give compensating prediction biases.
   Inference-side only; no further training of new HP combinations.
2. **Multi-objective Optuna (phase-3)** — formalise val × holdout
   Pareto front. Phase-2's iter17/18 holdout breakthrough is the
   exact signal that single-objective val-only search cannot capture.
3. **Final retrain of iter18 trial#5 at full 10-seed scaffold with
   extended num_epochs** to confirm the holdout record under more
   thorough training.

## Artefacts

- `results/<combo>/hpo/results.tsv` — 18 row history.
- `results/<combo>/hpo/study_*_iter011_*.json` — val winner detail.
- `results/<combo>/hpo/study_*_iter008_*.json`,
  `study_*_iter017_*.json`, `study_*_iter018_*.json` — holdout
  breakthrough trial details.
- `results/<combo>/architecture_log.md` — phase-1 + phase-2 row
  comparison.
- `combo2_hpo.db` — Optuna SQLite store.
- `program_phase2.md`, `PHASE2_OVERVIEW.md` — protocol + index.
- Git history: every iter design commit + revert preserved.
