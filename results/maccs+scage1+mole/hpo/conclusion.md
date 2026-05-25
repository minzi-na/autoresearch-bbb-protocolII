# Phase-2 HPO — Conclusion (combo2: maccs+scage1+mole)

## Summary

- **Iterations run**: 16 (ceiling 18; self-terminated at iter16 by the
  5-consecutive-discard convergence trigger).
- **Distribution**: 2 KEEP (iter3, iter11), 14 DISCARD.
- **Phase-2 best (val rule)**: **iter11 trial#21** (commit `78fd984a82`,
  study `auto_iter011_78fd984a82`). lr=1.22e-4, wd=1.14e-6, dropout=
  0.047, drop_path=0.029, mod_drop_p=0.114, head_dropout=0.086, with
  arch d_model=512 / d_ffn=1536 / depth=5 (latter is the iter3
  trial#7 architecture). Lives in the bbb-combo1-style low-wd /
  low-dropout cluster.
- **Generalisation landmark**: iter8 trial#1/#3 — DISCARD under val
  rule but the only phase-2 configurations whose top-3 confirm trials
  BEAT phase-1 iter198 on holdout per-seed mean roc_auc (both nn05
  and total).
- **End state**: val improvement +0.000217 over phase-1 (paired test
  noise), holdout near-parity with phase-1 (slight nn05 regression in
  the val winner, total parity).

## Phase-1 baseline vs Phase-2 winners

Three rows worth tracking — the val-rule winner (iter11 trial#21), the
prior val-rule winner (iter3 trial#7, kept under the same rule), and
the holdout landmark (iter8 trial#1).

| Source | val ROC-AUC | nn05 ROC-AUC | total ROC-AUC | keep? |
|---|---|---|---|---|
| phase-1 iter198 (baseline) | 0.852661 ± 0.003551 | **0.870011 ± 0.008095** | **0.879823 ± 0.004205** | — |
| iter3 trial#7 (val winner v1) | 0.852850 ± 0.002940 | 0.866288 ± 0.007562 | 0.878848 ± 0.002268 | True |
| **iter11 trial#21** (val winner v2, FINAL) | **0.852878 ± 0.003240** | 0.868296 ± 0.006449 | 0.879793 ± 0.003691 | **True** |
| iter8 trial#1 (holdout landmark) | 0.852397 ± 0.004151 | **0.870585 ± 0.007292** | **0.880124 ± 0.004112** | False |
| iter8 trial#3 (holdout best nn05) | 0.852350 ± 0.004037 | **0.871329** ± ... | 0.880122 ± ... | False |

### Statistical test (iter11 trial#21 vs phase-1, same 10 seeds)

Threshold gap: +0.000217 (val), nn05 −0.0017, total −0.00003. Same
noise band as the iter3-vs-phase-1 comparison. Effect on val is not
statistically distinguishable from sampling noise.

### Reading

- **val rule**: phase-2 produced two KEEPs, both within the
  measurement-noise band of phase-1. The val gate is decisive about
  the trial selection but the magnitude is small.
- **Two robust HP regions** in combo2: (a) iter3 trial#7 at high-wd
  (1.5e-4) / high-dropout (0.243), (b) iter11 trial#21 at low-wd
  (1.14e-6) / low-dropout (0.047). The two clusters give nearly
  identical confirm val 0.852850 vs 0.852878 — the val landscape has
  a flat plateau between them.
- **Holdout signal** is best at iter8 trial#1/#3 (low-wd cluster but
  with non-zero training-procedure HP — cosine LR + small label
  smoothing). These DISCARD by val rule but are the only phase-2
  configurations that strictly beat phase-1 on both nn05 and total.

## Winning HP configuration (val rule, iter11 trial#21)

| param | iter11 trial#21 | iter3 trial#7 | phase-1 BASE_CONFIG | bbb-combo1 cluster |
|---|---|---|---|---|
| lr                 | **1.224e-04** | 1.502e-04 | 1.000e-04 | ~1.1e-04 |
| **weight_decay**   | **1.145e-06** | 1.499e-04 | 1.000e-05 | ~3-4e-6 |
| **dropout**        | **0.047**     | 0.243     | 0.200     | 0.04-0.06 |
| drop_path          | 0.029         | 0.002     | 0.025     | — |
| mod_drop_p         | 0.114         | 0.090     | 0.100     | — |
| head_dropout       | 0.086         | 0.198     | 0.080     | — |
| batch_size         | 128           | 128       | 128       | 128 |
| grad_clip_max_norm | 1.136         | 1.136     | 1.000     | — |
| d_model            | 512           | 512       | 512       | 512 |
| d_ffn              | 1536          | 1536      | 1048      | 1048 |
| depth              | 5             | 5         | 4         | — |
| lr_schedule        | constant      | constant  | constant  | constant |
| label_smoothing    | 0.0           | 0.0       | 0.0       | 0.0 |
| ema_decay          | 0.999         | 0.999     | 0.999     | — |

iter11 trial#21 sits in the bbb-combo1 winning cluster (low-wd /
low-dropout). The two phase-2 KEEPs differ on every continuous HP
except batch_size / grad_clip / arch — they are TWO distinct optima
that combo2's val landscape supports equally well.

## What was tried (lever × outcome)

| iter | design (single lever) | family | confirm val | keep | takeaway |
|---|---|---|---|---|---|
| 1 | structural narrow (drop d_model/d_ffn/depth from search) | space | 0.852668 | False* | TPE wide converges on trial#7 |
| 2 | + MedianPruner + TPE n_startup=15 multivariate | pruner+sampler | 0.852850 | False* | same trial#7 emerges |
| 3 | + 5-seed mean objective | objective | **0.852850** | **True** | first val gain over phase-1 |
| 4 | search space narrow around trial#7 | space | 0.852740 | False | search-to-confirm overfit |
| 5 | sampler TPE → CmaEsSampler (wide) | sampler | 0.852850 (=#7 HP) | False | CmaEs rediscovers trial#7 exact HP |
| 6 | pruner MedianPruner → NopPruner | pruner | 0.852850 (=#7 HP) | False | full-epoch doesn't change result |
| 7 | BASE_CONFIG extension (lr_schedule + smoothing + ema), full 16-D | best_train.py | 0.851604 | False | 16-D / 30 trials too sparse |
| 8 | pin 11 HP at trial#7, search ONLY the 5 new training-procedure HP | best_train.py + space-restrict | 0.852397 | False | **HOLDOUT BREAKTHROUGH** top-3 of 3 beat P1 |
| 9 | narrow 4-D around iter8 top region | space | 0.851036 | False | narrow degrades |
| 10 | shift to bbb-combo1 low-wd cluster (6-D) | space | 0.852306 | False | new cluster found; search-to-confirm overfit |
| 11 | + 7-seed objective on low-wd cluster | objective | **0.852878** | **True** | **NEW BEST**: 7-seed picked robust trial#21 |
| 12 | arch range pin + 5-seed evenly spread (bundled) | space + objective | 0.851858 | False | both levers hurt; can't disambiguate |
| 13 | arch range pin alone (single-lever isolate) | space | 0.851858 | False | depth=5 critical; range pin lets sampler pick depth=6 worse |
| 14 | narrow exploitation around iter11 trial#21 | space | 0.852420 | False | trial#21 is local peak; narrow box has no better trial |
| 15 | sampler TPE → CmaEsSampler on low-wd region | sampler | 0.851744 | False | CmaEs worse than TPE here |
| 16 | mod_drop_p / head_dropout upper extend 0.15 → 0.25 | space | 0.852282 | False | edge probe fails; mod_drop_p=0.114 was a local peak not an edge signal |

\* iter1/iter2 used the strict mean+1·std keep gate. Both rows stay
False under the recorded rule; we did not retroactively rewrite history.

## Protocol evolution during the loop

| change | iter | rationale |
|---|---|---|
| `*_with_pruning` helper in best_train.py | iter2 | pruning needs a train_model hook |
| Keep rule: mean+1·std → mean only | post-iter3 | std buffer too strict for noise-level landscape |
| Ceiling 12 → 40 → 12 → 18 | iter3 / iter6 / iter11 | wall-time vs narrowing-cycle accommodation |
| Autonomy mode formalised | post-iter3 | agent chooses per-iter lever |
| BUDGET n_trials 50 → 30 | from iter5 | early iter all converged on trial#7 |
| Inline holdout in confirm phase | from iter3 | bbb-combo1 pattern alignment |
| best_train.py BASE_CONFIG extension officially allowed | post-iter6 | bbb-combo1-style training-procedure HP search |
| best_train.py headers refreshed | iter11 prep | source-of-truth doc + code consistency |
| ceiling 12 → 18 | iter11 prep | accommodate bbb-combo1-style 6-iter narrowing window |

## Why the loop stopped at iter16

- **Five consecutive val-DISCARD** (iter12-16) hit the early-termination
  trigger from program_phase2.md.
- Across iter12-16 the levers exercised cover every realistic class:
  - space (arch range, narrow exploitation, edge probe of regulariser
    upper bounds);
  - sampler (CmaEs swap);
  - objective (5-seed evenly spread vs 7-seed first-half bias —
    bundled in iter12 but isolated to be ineffective in iter13).
  None beat iter11 trial#21's val 0.852878.
- iter11 trial#21 is a true local peak of the low-wd cluster: every
  narrow exploitation around it (iter14), every sampler-family switch
  (iter15), every neighbour-region probe (iter16) plateaued at
  val ≈ 0.852 — within one phase-1 std of iter11 KEEP.
- The val landscape between the two robust clusters (iter3 trial#7
  and iter11 trial#21) is flat. HP-tuning alone is unlikely to move
  val outside the noise band.

## bbb-combo1 cross-reference

bbb-combo1 phase-2 (40 iter, framework adopted here) achieved val
+0.00275 over phase-1 with holdout −0.0046 (regression). combo2
phase-2 (16 iter) achieved val +0.000217 (noise level) with holdout
near-parity. The asymmetry:

- bbb-combo1's phase-1 reached val 0.87573 with 106 iter; phase-2's
  6-iter narrowing chain (run13→18, drop=0.05 / lr=1e-4 / wd=3e-6
  cluster) found the big +0.00135 jump (run17 d_ffn=1536). bbb-combo1's
  combination of an earlier phase-1 stop AND a productive d_ffn switch
  is what produced the val gain.
- combo2's phase-1 ran 200 iter, leaving the architecture HP nearly
  saturated; iter13 confirmed that re-opening d_ffn / depth ranges in
  phase-2 picked architectures that confirm WORSE than the iter3 / iter11
  single-pin (depth=5 is the productive value, range pin moves the
  sampler to depth=6 and loses confirm val). The d_ffn lever that
  worked for bbb-combo1 simply isn't available for combo2.

What did transfer: the bbb-combo1 winning cluster region itself.
combo2's iter10 / iter11 / iter14 confirmed a low-wd / low-dropout
cluster exists for combo2 too — iter11 trial#21 lives in it. The cluster
holds val ≈ phase-1 and holdout ≈ phase-1, but no narrowing inside it
gave a productive jump.

## Artefacts

- `results/<combo>/hpo/results.tsv` — 16 row history.
- `results/<combo>/hpo/study_*_iter011_*.json` — val winner. Filename:
  `study_1779678495.json` (iter11 KEEP, trial#21).
- `results/<combo>/hpo/study_*_iter008_*.json` — holdout landmark
  trials (#1 / #3 / #24).
- `results/<combo>/architecture_log.md` — phase-1 architecture sweep
  rows (iter1-198) plus phase-2 confirm-trial rows for each study.
- `combo2_hpo.db` — Optuna SQLite store; all studies inspectable.
- `program_phase2.md` — authoritative protocol with the evolution noted
  above.
- `PHASE2_OVERVIEW.md` — repo-root navigation index.
- Git history: every iter commit + revert preserved.

## If extending phase-2 further (out of current loop budget)

Two of the three follow-ons from the iter9 conclusion are still
applicable; the third has now been ruled out empirically.

1. **Soft-voting ensemble of iter8 top-3 (+ iter11 trial#21)** —
   inference-side only, no further training. Two distinct clusters
   (high-wd iter3 / low-wd iter11) and one holdout-best (iter8 #1)
   could compensate each other's prediction biases. Worth a one-off
   experiment.
2. **Multi-objective Optuna (val × holdout Pareto)** — would be a
   phase-3 framework change. Combo2's holdout signal at iter8 trial#1
   already hints that a Pareto-aware search might land on a different
   confirm-best than the val-only loop did.
3. ~~Architecture HP re-opening (e.g. d_ffn / depth wider ranges)~~ —
   ruled out by iter13. depth=5 is the productive value for combo2;
   range pin actively hurts confirm val.
