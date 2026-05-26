# Phase-2 conclusion — combo1 (maccs+avalon+scage2+mole)

**Termination:** 5-consecutive-discard early-termination triggered after
iter5. Total 5 phase-2 iterations run (iter1-5), 0 keeps, 5 discards.
Phase-1 iter197 baseline (`mean_val_auc=0.854947 ± 0.002638`, commit
`9bc22da482`) was not surpassed.

## Iter summary

| iter | family | best confirm val | Δ vs threshold | keep |
|------|--------|------------------|---------------|------|
| 1 | narrow-existing-HP (5-D refinement around iter197 anchor) | 0.854310 ± 0.002905 | -0.000637 | False |
| 2 | narrow-existing-HP (pin cosine + tighter lr/lr_min/ema/ls) | 0.854503 ± 0.003208 | -0.000444 | False |
| 3 | expose-new-HP (un-hardcode `adamw_wd` from iter148-frozen 0.003) | 0.854237 ± 0.002717 | -0.000710 | False |
| 4 | narrow-new-HP (tight exploitation around iter3 top cluster) | 0.854504 ± 0.002991 | -0.000443 | False |
| 5 | expose-new-HP (un-hardcode `rdrop_alpha_max` from iter80-frozen 1.0) | 0.854136 ± 0.002905 | -0.000811 | False |

**Best phase-2 attempt:** iter4 confirm 0.854504 ± 0.002991 (iter2 tied at 0.854503).
Both fell short of phase-1 by ~0.000443 — within one σ of the confirm-mean
sampling noise (σ ≈ 0.003 for the 10-seed scaffold mean).

## What was searched

Five training-procedure levers were exposed in `best_train.py` BASE_CONFIG
and explored across the iters (all with phase-1 byte-identical defaults
so `build_and_train(config=None)` keeps reproducing iter197 bit-for-bit):

1. **Phase-2 starter five** (iter1-2): `lr_schedule`, `lr_warmup_epochs`,
   `lr_min_ratio`, `label_smoothing`, `ema_decay` (+ `lr` from iter1
   already in BASE_CONFIG).
2. **`adamw_wd`** (iter3): un-hardcoded the AdamW decay-group weight
   decay (was iter148-frozen at 0.003). Searched 1e-3..1e-2 log.
3. **`rdrop_alpha_max`** + **`rdrop_warmup_epochs`** (iter5):
   un-hardcoded the R-Drop consistency-loss coefficient (was iter80-frozen
   at 1.0) and its linear-warmup length (was iter106-frozen at 5).
   Searched alpha_max 0.3..2.0 log; warmup pinned at 5.

## Top-region pattern (across iters)

All five iters' top trials converged on a narrow corner of the search space:

| HP | iter1-5 top-trial cluster | Phase-1 iter197 | Notes |
|----|---|---|---|
| `lr_schedule` | `cosine` (unanimous from iter1) | `constant` | Clearest phase-2 finding |
| `lr_warmup_epochs` | 0 (cosine no-warmup preferred) | 0 (no scheduler) | |
| `lr` | 1.07e-4..1.39e-4 (effective AdamW 1.34..1.74e-4) | 1e-4 (eff. 1.25e-4) | Slightly higher base lr preferred with cosine |
| `lr_min_ratio` | 0.12..0.28 (cluster ~0.20-0.25) | n/a | Cosine decays to ~20% of peak lr |
| `ema_decay` | 0.9982..0.9994 (cluster ~0.9986) | 0.9993 | Slightly LOWER decay preferred (faster forgetting) |
| `label_smoothing` | 0.02..0.05 (cluster ~0.03-0.04) | 0 | Small smoothing helps slightly |
| `adamw_wd` | 0.0011..0.0027 (cluster ~0.0015-0.002) | 0.003 (hardcoded) | LOWER than phase-1 frozen value |
| `rdrop_alpha_max` | 0.93..1.07 (top 5) | 1.0 (hardcoded) | Phase-1 pick was already optimal |

Phase-1's iter80 (R-Drop alpha=1.0) and iter148 (wd=0.003) choices were
made *before* the phase-2 starter five were available; the phase-2 top
region shows that under the cosine + lower-ema_decay phase-2 stack the
locally optimal `adamw_wd` shifts slightly lower (0.0015-0.002 vs
phase-1's 0.003 narrow sweep), while `rdrop_alpha_max=1.0` remains the
correct pick.

## Holdout signal (best phase-2 trial — iter4 trial#0)

| Subset | Phase-2 best (iter4 #0) | Phase-1 iter197 | Δ |
|--------|-------------------------|------------------|---|
| nn05 AUC  | 0.85586 ± 0.006741 | 0.856316 ± 0.007349 | −0.0005 |
| nn05 MCC  | 0.553557 | 0.557481 ± 0.011201 | −0.004 |
| nn05 Acc  | 0.857411 | 0.857751 ± 0.005049 | −0.0003 |
| total AUC | 0.875035 ± 0.003614 | 0.876385 ± 0.003684 | −0.0014 |
| total MCC | 0.608873 | 0.609062 ± 0.010027 | −0.0002 |
| total Acc | 0.849887 | 0.849887 ± 0.004363 | 0.0000 |

Holdout AUC is marginally lower than phase-1 across both subsets;
MCC/Acc are statistically indistinguishable. Phase-2 did not yield a
generalization improvement either.

## Conclusion

For combo1 (`maccs+avalon+scage2+mole`), the phase-1 iter197 architecture
+ HP stack is at or extremely close to the local optimum reachable via
the phase-2 lever surface (training-procedure HP only; architecture
frozen). The maximum phase-2 improvement attempted across 5 iters was a
val mean +0 (best phase-2 confirm 0.854504 vs threshold 0.854947 = -0.0004
below). The search-vs-confirm gap (~0.001 in every iter) suggests we are
at the sampling-noise floor of the 10-seed scaffold val mean — even if a
slightly better config exists, the 30-trial × 5-search-seed Optuna study
cannot reliably surface it above the noise.

This matches the bbb-combo1 and combo2 phase-2 patterns: phase-1
architecture search produces an HP-sensitivity well that phase-2 HPO
cannot statistically escape.

**Selection:** No phase-2 keep row exists. The recommended deployment
configuration is the phase-1 iter197 baseline (commit `9bc22da482`).

## Artifacts

- `results/maccs+avalon+scage2+mole/hpo/results.tsv` — 5 phase-2 iter rows
- `study_1779773386.json` (iter1) ... `study_1779818903.json` (iter5)
- `trials_<stamp>.tsv` per iter
- `architecture_log.md` — phase-1 + phase-2 unified comparison
- `combo1_hpo.db` (gitignored) — Optuna SQLite traces of all 5 studies
