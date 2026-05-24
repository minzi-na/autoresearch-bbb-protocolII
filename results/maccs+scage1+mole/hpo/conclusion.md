# Phase-2 HPO — Conclusion (combo2: maccs+scage1+mole)

## Summary

- **Iterations run**: 6 (ceiling 12; loop terminated early by user decision after iter6).
- **Distribution**: 1 KEEP (iter3), 5 DISCARD (iter1/2/4/5/6). A 7th iter
  was committed but aborted before evaluation due to a design flaw the
  user flagged; that commit is reverted.
- **Phase-2 best**: **iter3 trial#7** (commit `a539a46706`, study
  `auto_iter003_a539a46706`). Same HP was deterministically rediscovered
  by every other iter (iter1/2/5/6) under different sampler / pruner /
  search-space settings — strong evidence that this point is the robust
  peak of the wide HP search space.

## Phase-1 baseline vs phase-2 best (per-seed mean ± std, 10 seeds)

| Metric | Phase-1 iter198 | Phase-2 iter3 trial#7 | Δ (phase2 − phase1) |
|---|---|---|---|
| val ROC-AUC      | 0.852661 ± 0.003551 | **0.852850 ± 0.002940** | **+0.000189** |
| nn05 ROC-AUC     | **0.870011 ± 0.008095** | 0.866288 ± 0.007562 | −0.003723 |
| nn05 MCC         | 0.564617 ± 0.023596 | 0.568680 ± 0.022707 | +0.004063 |
| nn05 Accuracy    | 0.860486 ± 0.008429 | 0.861398 ± 0.007218 | +0.000912 |
| total ROC-AUC    | **0.879823 ± 0.004205** | 0.878848 ± 0.002268 | −0.000975 |
| total MCC        | 0.610760 ± 0.016543 | 0.617523 ± 0.015402 | +0.006763 |
| total Accuracy   | 0.851013 ± 0.006107 | 0.853266 ± 0.006251 | +0.002253 |

### Statistical test on val_AUC

Paired t-test (same 10 seeds, phase-1 iter198 vs phase-2 iter3 trial#7):
**t = 0.154, p (two-sided) = 0.881**; Wilcoxon signed-rank one-sided
p = 0.577. Effect indistinguishable from noise.

### Reading

- **val**: trivial mean gain (+0.000189 ≈ 5% of one std), not
  statistically distinguishable from phase-1 under paired analysis.
- **holdout AUC**: small regression on both subsets (within phase-1
  std band). Phase-2 HPO did *not* improve generalisation AUC.
- **holdout MCC / Accuracy**: small improvements (within std band).
  Probability calibration around the 0.5 threshold is slightly better
  in iter3 than phase-1.

Net: the phase-2 winner is a near-equivalent of phase-1 iter198 with a
slightly different threshold-side trade-off, not a generalisation win.

## Winning HP configuration

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

Structure stayed close to BASE_CONFIG (same d_model/batch_size, +1
depth, wider d_ffn). The bulk of the shift was in regularisation
(wider weight_decay, head_dropout 2.5×, slightly higher dropout,
near-zero drop_path).

## What was tried (lever × outcome)

| iter | design (single lever) | confirm val | keep | takeaway |
|---|---|---|---|---|
| 1 | structural narrow (drop d_model/d_ffn/depth from search) | 0.852668 | False* | wide TPE converges on trial#7 anyway |
| 2 | + MedianPruner + TPE n_startup=15 multivariate | 0.852850 | False* | same trial#7 emerges; pruner saves wall-time only |
| 3 | + 5-seed mean objective | **0.852850** | **True** | search SNR up; trial#7 confirm-stable; first val gain over phase-1 (+0.000189) |
| 4 | search space narrowed around trial#7 | 0.852740 | False | best is now a different HP but worse on confirm → search-to-confirm overfit, no local refinement |
| 5 | sampler swap TPE → CmaEsSampler | 0.852850 | False | CmaEs rediscovers trial#7 exact HP; not a TPE artefact |
| 6 | pruner removed (NopPruner) | 0.852850 | False | full-epoch eval doesn't surface any late-bloomer; pruning wasn't the issue |
| (7) | search seeds confirm-disjoint pool | — | (aborted) | design flaw: doesn't change the confirm-side optimum |

* iter1/iter2 used the strict mean+1·std keep gate (later relaxed to
  mean-only at user request from iter3 onward). Under the mean-only
  gate iter1 (+0.000007) and iter2 (+0.000189) would have been marginal
  keeps, but both rows stay False in the TSV — we did not retroactively
  rewrite history.

## Why the loop stopped early

Five different lever families (search space, sampler, pruner,
objective seed-count, sampler+pruner bundle) all converged onto the
same trial#7 HP with identical 10-seed val per-seed lists. The
hypothesis space "different sampler/pruner/search-space settings find
a better HP than trial#7" was empirically falsified by iter5
(CmaEsSampler, structurally different from TPE) and iter6 (NopPruner,
all 30 trials full-epoch). The remaining theoretical levers (search
space extension into untouched extreme-regularisation regions; objective
stability penalty; non-TPE/CmaEs samplers like Random/BoTorch) all have
the same structural ceiling: they can only affect *which point inside
this HP space is preferred by the sampler*, and our confirm 10-seed
re-evaluation has now seen the entire trial#7 basin and its
neighbourhoods. There is no obvious lever left whose expected effect
is larger than the noise band (`std ≈ 0.003`) we are already inside.

User decision (iter7 abort, terminate): stop spending wall-time on
levers whose expected gain is within the noise band, and treat
`trial#7 HP` as the phase-2 outcome.

## bbb-combo1 cross-reference

bbb-combo1 phase-2 (40 iter, framework adopted here) ended with
val +0.0028 over phase-1 but holdout −0.0046 — a similar "val win,
holdout regress" pattern. The phase-2 HPO regime tends to mildly
overfit val without transfer to holdout when phase-1 already exhausted
the architecture-level search. Combo2's pattern is consistent with
that: trial#7 is a slightly different HP than phase-1 BASE_CONFIG with
near-zero net effect on either val or holdout.

## Artefacts

- `results/<combo>/hpo/results.tsv` — 6 row history (iter1–6).
- `results/<combo>/hpo/study_auto_iter003_a539a46706.json` carries the
  winning HP + full per-seed confirm + holdout block.
  (Filename on disk: `study_1779611433.json`.)
- `results/<combo>/architecture_log.md` — phase-1 architecture sweep
  rows (iter1–198) plus phase-2 confirm-trial rows for each study;
  direct line-by-line comparison of val_auc / nn05 / total.
- `combo2_hpo.db` — Optuna SQLite store; all studies (including
  aborted iter7 deletion-recovered state and the original
  `combo2_phase2_coarse` pilot study) are inspectable.
- Git history: every iter commit + revert preserved. iter3 design
  commit `a539a46706` stays in place (no revert); iter1/2/4/5/6/7
  were each reverted in their own commit.
