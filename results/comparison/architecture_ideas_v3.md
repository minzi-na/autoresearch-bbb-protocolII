# Architecture Optimization Ideas — autoresearch_combos_v2

Component-by-component improvement catalog for the multi-modal gMLP BBB
classifier. Read this when picking the next iteration's change in
`program.md` step 2.

Adapted from the prior BBB autoresearch playbook. Differences from before:
- Pool is **8:2 train/val only**, scaffold split — no internal test bucket.
- Generalization track is the 5 holdout subsets (reporting-only via
  `final_holdout_eval.py`); keep/discard is the **v3 asymmetric composite
  gate** (val_AUC primary, val_MCC tiebreaker — see next section).
- All 3 combos use the `maccs/avalon/scage1/scage2/mole` family — none
  contain `rdkit`/`ecfp`/`tt`. Ignore prior rdkit-scaling discussion.

---

## Keep gate (v3) — what to optimize for

**TL;DR.** A new iter is `keep` only if it shows a **clear gain** on val_AUC,
OR (val_AUC is within noise AND val_MCC strictly improves). The v2 policy of
"AUC tie also keeps" is gone — it led to late-iter drift where val_MCC
silently regressed while val_AUC was tied (see
`results/comparison/pre_post_baseline_comparison_100.md` section 6).

Concretely (`evaluate_combo.py` v3, default eps_auc = eps_mcc = 1e-3):

```
ref = (val_AUC, val_MCC) of the most recent keep=True row.
let d_auc = mean_val_auc - ref_AUC ;  d_mcc = mean_val_mcc - ref_MCC
keep iff:
    d_auc > +eps_auc                                # (a) clear AUC improvement
    OR  (|d_auc| <= eps_auc  AND  d_mcc > +eps_mcc) # (b) AUC tied, MCC strict
discard if:
    d_auc < -eps_auc                                # AUC regression beyond noise
    OR  |d_auc| <= eps_auc AND d_mcc <= +eps_mcc    # no-progress tie
```

**Implications for your proposals.**

1. **Move val_AUC by > 1e-3 or val_MCC by > 1e-3 — otherwise the iter is
   rejected by the gate.** 10-seed val mean has SE ≈ 0.001 (AUC) and
   ≈ 0.002–0.005 (MCC). Sub-noise nudges no longer count as progress.

2. **Avoid one-step hyperparam fine sweeps.** Sequences like
   `EMA decay 0.88 → 0.87 → 0.86 → ... → 0.83`, or
   `weight_decay 0.01 → 0.005 → 0.003`, or
   `dropout 0.20 → 0.18 → 0.15` are now anti-patterns: each step
   typically moves val_AUC by ~1e-4–3e-4 and val_MCC noisily, so all but
   the lucky tie-break attempts get rejected. If you want to search a
   scalar hyperparam, take coarser jumps (e.g. EMA `0.99 → 0.9 → 0.8`
   directly) or treat it as one decision, not five.

3. **Prefer structural / qualitative changes over scalar tuning.** New
   pooling variant, new SGU mixing scheme, new conditioning layer, new
   loss formulation, new regularization mechanism — these are likely to
   produce |Δ val_AUC| > 1e-3 in either direction so the gate has real
   signal to keep or reject.

4. **MCC matters in the noise band.** If your proposed change is
   designed to be val_AUC-neutral (e.g. better-calibrated head, focal
   loss, threshold-aware loss, class-weighted training), make sure
   val_MCC moves visibly. In the noise band MCC tiebreaker is the only
   way to keep.

5. **AUC-MCC tradeoffs are still allowed.** If your change pushes val_AUC
   up clearly (Δ > eps_auc) but pulls val_MCC down, the gate still keeps
   it — this preserves the early-iter freedom to explore architectures
   with sharp AUC-MCC tradeoffs (like combo1 iter6/iter12 type moves in
   the v2 run). MCC is only a tiebreaker, not a veto.

**What this changes about how to read this catalog.** The Priority table
below stays valid in terms of *which* component to touch, but skip any
proposal that boils down to a one-step scalar nudge unless you have a
specific reason to expect > 1e-3 movement. Structural changes are now
strictly preferred at all priorities.

---

## Current architecture

```
flat feature vector
    ↓ split by modality
[Linear(in_dim → d_model)] × N modalities    (per-modality projection)
    ↓ stack
(B, seq_len, d_model)
    ↓
[gMLPBlock × depth=4]                         (backbone)
  └─ LayerNorm → Linear(d_model → d_ffn*2) → GELU
     → SGU: split u,v; norm v; Conv1d mixing; u*v
     → Linear(d_ffn → d_model) → residual add
    ↓
gated weighted pool (learned softmax α)       (aggregation)
    ↓
Dropout(0.2) → Linear(d_model → 1)
```

Already in baseline: `es_metric=val_auc`. Do **not** waste an iteration on
"switching ES from val_loss to val_auc" — it is the default.

---

## A. Per-modality Projection — risk: low

**Problem.** MACCS/Avalon are 0/1 bits, MolE/Scage1/Scage2 are pretrained
embeddings with very different scales. A single `Linear` per modality maps
all of these into d_model without normalization → token-scale mismatch
that hurts early backbone stability.

**Ideas.**

1. **LayerNorm after projection** (simple, often a free win):
   ```python
   self.proj_norm = nn.ModuleDict({n: nn.LayerNorm(d_model) for n in self.mod_names})
   # forward:  token = self.proj_norm[name](self.proj[name](chunk))
   ```

2. **2-layer projection** (more capacity per modality):
   ```python
   nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
   ```

3. **Per-modality scale parameter** (learnable, init 1.0):
   ```python
   self.proj_scale = nn.Parameter(torch.ones(len(mod_names)))
   # forward: tokens[i] = self.proj_scale[i] * tokens[i]
   ```

---

## B. Spatial Gating Unit — risk: medium

**Problem.** `Conv1d(seq_len, seq_len, kernel_size=1)` is a learnable
seq_len × seq_len mixing matrix shared across all hidden channels. Static
per-position weights; no input-dependent mixing.

**Ideas.**

1. **Learnable residual scale on the spatial path** (safe-init at zero):
   ```python
   self.gate_scale = nn.Parameter(torch.zeros(1))
   # forward: v_mixed = v + self.gate_scale.exp() * spatial_proj(v) - v
   #          == self.gate_scale.exp() * spatial_proj(v) + (1 - exp)*v
   # init exp(0)=1.0 — identity to original SGU when scale=0
   ```

2. **Multi-head SGU** — split channels into n heads, each with its own
   `Conv1d(seq_len, seq_len)`; concat:
   ```python
   # d_ffn must be divisible by n_heads. Init each head's bias=1.0.
   ```

3. **Attention-based SGU** — content-dependent mixing (higher risk, more
   parameters):
   ```python
   # Q = Wq(v); K = Wk(v); V = Wv(v)   shape (B, seq_len, d_ffn)
   # attn = softmax(Q @ K.transpose / sqrt(d_ffn))
   # out  = attn @ V
   # u * out
   ```
   Use small attn_drop (0.1). Initialize Wv to identity if you want to
   start near baseline.

4. **Diagonal-masked attention SGU**: mask the diagonal of the attention
   matrix so a token cannot attend to itself, forcing genuine cross-modal
   mixing. Useful when seq_len is small (3–4).

---

## C. Pooling — risk: low → medium

**Problem.** `α = softmax(learnable_param)` gives the same weights to every
sample. Per-molecule modality importance is ignored.

**Ideas.**

1. **Attention pooling** (input-dependent weights):
   ```python
   self.pool_query = nn.Parameter(torch.zeros(d_model))
   # forward: scores = (X @ self.pool_query) / sqrt(d_model)  # (B, seq_len)
   #          w = softmax(scores, dim=1)
   #          pool = (w.unsqueeze(-1) * X).sum(dim=1)
   ```

2. **CLS token** — prepend a learnable token, take its final state.
   Requires updating SGU `seq_len` to `N+1`.

3. **Multi-query pool** — k learnable queries, concat their pooled outputs,
   project back to d_model:
   ```python
   # queries: (k, d_model); pooled: (B, k, d_model)
   # out = Linear(k*d_model, d_model)(pooled.flatten(1))
   ```

4. **Sigmoid skip-gate mix** — convex combine `mean_pool` and `gated_pool`
   with a sigmoid gate (init 0 → 50/50):
   ```python
   self.skip_gate = nn.Parameter(torch.zeros(1))
   # mix = sigmoid(skip_gate) * gated + (1 - sigmoid(skip_gate)) * mean
   ```

---

## D. Training dynamics — risk: low

**Problem.** Plain Adam + lr=1e-4 + no scheduler + no clipping. Class
imbalance (BBB+:BBB- ≈ 2.27:1 on the merged pool) is not handled.

`BASE_CONFIG` is frozen, so you can only change things **inside**
`train_model()` (optimizer instance, schedulers, loss formulation,
clipping), not the values in `BASE_CONFIG`.

**Ideas.**

1. **AdamW + gradient clipping** (usually a free win):
   ```python
   optimizer = optim.AdamW(model.parameters(),
                           lr=BASE_CONFIG["lr"],
                           weight_decay=BASE_CONFIG["weight_decay"])
   # after loss.backward():
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Warmup + cosine annealing**:
   ```python
   warmup_epochs = 5
   def lr_lambda(epoch):
       if epoch < warmup_epochs:
           return (epoch + 1) / warmup_epochs
       p = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
       return 0.5 * (1 + math.cos(math.pi * p)) * 0.99 + 0.01
   scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
   # at end of each epoch: scheduler.step()
   ```

3. **Label smoothing**:
   ```python
   smooth = 0.1
   y_smooth = y * (1 - smooth) + smooth * 0.5
   loss = F.binary_cross_entropy_with_logits(pred, y_smooth)
   ```

4. **pos_weight from train batch class ratio** (counters imbalance):
   ```python
   n_pos = (y_train_full == 1).float().sum()
   n_neg = (y_train_full == 0).float().sum()
   pw = (n_neg / n_pos).to(device)              # ~0.44 on this pool
   loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
   ```
   Search around the auto value: 0.3 / 0.44 / 0.6 / 1.0.

5. **EMA of weights** for early-stop snapshot (more stable best-of-epoch
   selection in scaffold split):
   ```python
   # maintain ema_state with decay 0.999; at val time use ema_state to score
   ```

---

## E. Cross-modal FiLM / AdaLN — risk: medium

**Concept.** Before the backbone, let each modality token receive a
conditioning signal from the other modalities and adjust its own
scale/bias. SGU mixes inside the backbone; FiLM pre-conditions the input.

**Combo-specific fp / embed partitions** (used by category-based FiLM):

| Combo | fp tokens | embed tokens |
|-------|-----------|--------------|
| combo1: `maccs+avalon+scage2+mole` | maccs, avalon | scage2, mole |
| combo2: `maccs+scage1+mole`        | maccs         | scage1, mole |
| combo3: `maccs+scage1+scage2+mole` | maccs         | scage1, scage2, mole |

**Implementation sketch:**

```python
class CrossModalFiLM(nn.Module):
    def __init__(self, d_model, fp_idx, emb_idx):
        super().__init__()
        self.embed_to_fp = nn.Linear(d_model, d_model * 2)
        self.fp_to_embed = nn.Linear(d_model, d_model * 2)
        # safe init: gamma path → 0 ⇒ FiLM starts as identity
        nn.init.zeros_(self.embed_to_fp.weight); nn.init.zeros_(self.embed_to_fp.bias)
        nn.init.zeros_(self.fp_to_embed.weight); nn.init.zeros_(self.fp_to_embed.bias)
        self.fp_idx, self.emb_idx = fp_idx, emb_idx

    def forward(self, X):
        fp_sum  = X[:, self.fp_idx,  :].mean(1)
        emb_sum = X[:, self.emb_idx, :].mean(1)
        gf, bf = self.embed_to_fp(emb_sum).chunk(2, dim=-1)
        ge, be = self.fp_to_embed(fp_sum).chunk(2, dim=-1)
        X = X.clone()
        X[:, self.fp_idx,  :] = (1 + gf.unsqueeze(1)) * X[:, self.fp_idx,  :] + bf.unsqueeze(1)
        X[:, self.emb_idx, :] = (1 + ge.unsqueeze(1)) * X[:, self.emb_idx, :] + be.unsqueeze(1)
        return X
```

**AdaLN variant** — apply after a LayerNorm with `elementwise_affine=False`
for more stable scaling.

For combo2 the fp set is only one token (maccs), so `fp_sum` collapses to
that single token's vector — FiLM is asymmetric. Consider degrading to
`global_pool → modality-specific FiLM` for combo2.

---

## F. Regularization — risk: low

**Problem.** Only `Dropout(0.2)` at head input. ~9786-mol pool with
multi-modal input is at risk of single-modality overfit.

**Ideas.**

1. **Modality dropout** — zero out an entire modality token with prob p
   during training (1.0/n_modalities is a soft default for p):
   ```python
   if self.training and self.mod_drop_p > 0:
       mask = (torch.rand(B, self.seq_len, device=X.device) > self.mod_drop_p).float()
       X = X * mask.unsqueeze(-1)
   ```

2. **Stochastic depth on gMLPBlocks** — skip a block with prob p:
   ```python
   if self.training and torch.rand(1) < self.drop_path:
       return residual
   return out
   ```
   Depth is only 4, so set p small (0.05–0.10).

3. **DropPath inside SGU** — drop the `u * v` mixed path entirely, keep `u`
   only. Mild architecture regularization.

---

## Priority table

Run in this order; switch direction after 3 consecutive discards in the
same row's "component".

| # | Change | Component | Risk | Notes |
|---|--------|-----------|------|-------|
| 1  | AdamW + grad-clip(1.0)                                    | D | low    | usually a free win |
| 2  | Warmup(5) + cosine annealing                              | D | low    | pairs with #1 |
| 3  | pos_weight = n_neg/n_pos (auto-from-batch)                | D | low    | direct counter to 2.27:1 imbalance |
| 4  | Label smoothing (smooth=0.1)                              | D | low    | pair with #3 carefully |
| 5  | LayerNorm after per-modality projection                   | A | low    | independent variable test |
| 6  | Modality dropout (p≈0.15)                                 | F | low    | most useful on combo3 (4 modalities) |
| 7  | Sigmoid skip-gate mix between gated and mean pool         | C | low    | safe init = 50/50 |
| 8  | Cross-modal FiLM (category-based, safe-init)              | E | medium | combo-aware fp/emb partition |
| 9  | Attention pooling                                         | C | medium | replaces gated_pool |
| 10 | Learnable residual scale on SGU spatial path              | B | medium | init exp(0)=1.0 |
| 11 | 2-layer per-modality projection                           | A | medium | param↑, risk overfitting |
| 12 | Multi-head SGU (n_heads=2 or 4)                           | B | medium | small seq_len → only mild benefit |
| 13 | AdaLN variant of #8                                       | E | medium | apply after LN(elementwise_affine=False) |
| 14 | CLS token pooling (update SGU seq_len)                    | C | medium | bookkeeping-heavy |
| 15 | Stochastic depth on gMLPBlocks (p=0.05–0.10)              | F | medium | depth=4 caps the gain |
| 16 | Diagonal-masked attention SGU                             | B | high   | helps when seq_len ≤ 4 |
| 17 | EMA of model weights for early-stop snapshot              | D | medium | adds complexity to train_model |

Apply one change at a time. The only exception is `#1 + #2` (AdamW +
warmup/cosine) — these are a training-dynamics package and may be tried
together to save an iteration.

---

## Combo-specific notes

| Combo | seq_len | fp / embed |
|-------|---------|------------|
| combo1 `maccs+avalon+scage2+mole`      | 4 | 2 fp + 2 embed |
| combo2 `maccs+scage1+mole`             | 3 | 1 fp + 2 embed |
| combo3 `maccs+scage1+scage2+mole`      | 4 | 1 fp + 3 embed |

`scage2` is the new modality that did not appear in the prior BBB
autoresearch combos. Treat it as a second pretrained embedding analogous
to `scage1` — same dim (512), different training corpus.
