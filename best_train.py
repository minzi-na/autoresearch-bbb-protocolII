#!/usr/bin/env python3
"""
best_train.py — phase-1 best architecture snapshot for Phase-2 HPO.

Created from train.py at commit 9bc22da482 ("iter197: EMA decay 0.9995
-> 0.9993 on lr=1.25e-4 stack") as a snapshot of the iter197 best
architecture, refactored so Optuna trials can inject a config dict
instead of mutating module-level constants. evaluate_combo.py /
final_holdout_eval.py keep importing train.py (NOT this file), so
phase-1 paths are not affected by anything that happens here.

Phase-2 edit surface (see program_phase2.md for authoritative rules):

  FROZEN (phase-1 output; never edit in phase-2):
    - Model class definitions: SpatialGatingUnit (multi-head),
      gMLPBlock, gMLP, MultiModalGMLPFromFlat. Phase-1 architecture
      sweep is the single source of truth for these. Constants
      hardcoded inside model classes (SGU_N_HEADS=2, DROP_V_PATH=0.08,
      self.drop=Dropout(0.10), self.mod_drop_p=0.10) are part of
      iter197 frozen state.
    - device handling.
    - The phase-1 behaviour of build_and_train(config=None) — that
      call MUST keep reproducing iter197 byte-for-byte after any
      phase-2 change. Adding new BASE_CONFIG keys is allowed only if
      their default values turn the new behaviour OFF (e.g.
      lr_schedule="constant", label_smoothing=0.0) or match the
      previously hard-coded value (ema_decay=0.9993,
      ema_warmup_epochs=1).
    - The AdamW-override inside train_model (lr=Adam_lr * 1.25,
      hardcoded wd=0.003, decay/no_decay split where the listed
      param names {"alpha", "pool_skip_gate", "proj_scale"} skip wd):
      this is iter38/88/148/172 frozen training-procedure state.
      BASE_CONFIG.weight_decay is therefore EFFECTIVELY UNUSED in
      this baseline — Optuna searching it has no effect. Phase-2 may
      later untangle this override via a separate lever extension.
    - R-Drop two-pass loss with 5-epoch alpha warmup (iter80/106
      frozen): also untouched here. Phase-2 may later expose
      rdrop_alpha_max / rdrop_warmup_epochs.

  EDITABLE (phase-2 lever surface):
    - BASE_CONFIG: add new HP keys with phase-1-reproducing defaults
      (see the Phase-2 iter1+ block at the end of the dict).
    - train_model / train_model_with_pruning: new optional arguments
      (with defaults that disable the new behaviour) + new training
      hooks (LR scheduler step, label smoothing in loss target,
      EMA decay sourced from cfg, ...).
    - build_and_train / build_and_train_with_pruning: wire the new
      cfg keys into the train_model variants.
    - _build_lr_scheduler (and any analogous helpers introduced for
      new BASE_CONFIG keys).

Phase-2 SHOULD touch this file only in concert with a matching change
in optuna_combo.py (suggest_config), and only via partial Edit — never
rewrite the file as a whole.
"""

import random
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  BASE_CONFIG + device — phase-2 EDIT SURFACE (with constraints)
#
#  Phase-1 view: this block was frozen.
#  Phase-2 view: BASE_CONFIG is the public training-procedure HP space.
#    - You MAY add new keys (LR schedule, label smoothing, EMA decay,
#      optimizer family, ...) so suggest_config() in optuna_combo.py can
#      search them.
#    - Each new key's DEFAULT must reproduce phase-1 byte-for-byte when
#      build_and_train is invoked with config=None (i.e. the new
#      behaviour is OFF by default, or the default matches the value
#      previously hard-coded inside train_model).
#    - Existing keys' default values are FROZEN at phase-1 values.
#      Do not change them; only add new keys.
#  Device handling is FROZEN.
# ═══════════════════════════════════════════════════════════════════════════════

BASE_CONFIG = {
    # ─── Phase-1 iter197 frozen values ───────────────────────────────────────
    "d_model":        512,
    "d_ffn":          1048,
    "depth":          4,
    "dropout":        0.2,
    "use_gated_pool": True,
    "lr":             1e-4,
    "weight_decay":   1e-5,
    "num_epochs":     50,
    "patience":       10,
    "batch_size":     128,
    "es_metric":      "val_auc",
    # ─── Phase-2 iter1+: training-procedure HP ───────────────────────────────
    # Defaults chosen so build_and_train(config=None) reproduces iter197
    # byte-for-byte. ema_decay / ema_warmup_epochs match the values
    # previously hard-coded inside train_model (lines 265/267 of train.py).
    # Other keys default to OFF/identity.
    "lr_schedule":         "constant",  # "constant" / "cosine" / "warmup_cosine"
    "lr_warmup_epochs":    0,           # int >= 0; ignored when schedule="constant"
    "lr_min_ratio":        0.0,         # cosine eta_min = lr * lr_min_ratio
    "label_smoothing":     0.0,         # BCE target smoothing in [0, 1)
    "ema_decay":           0.9993,      # was hardcoded; now overridable
    "ema_warmup_epochs":   1,           # was hardcoded; now overridable
    # iter8: un-hardcode head dropout (was Dropout(0.10) in MultiModalGMLPFromFlat
    # since iter86). Default 0.10 = phase-1 byte-identical.
    "head_dropout":        0.10,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        import os as _os
        _os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  FROZEN in phase-2 — model architecture (set by phase-1 iter1-200 sweep)
#
#  Phase-1 view: this block was the agent's single edit surface.
#  Phase-2 view: the model architecture is now an output of phase-1, not
#  an input to phase-2. SpatialGatingUnit, gMLPBlock, gMLP, and
#  MultiModalGMLPFromFlat are FROZEN. Do not modify constructor
#  arguments, forward(), submodule shapes, init schemes, or anything
#  that would change the computation graph or its initial weights.
#  Constants hardcoded inside the classes (SGU_N_HEADS=2,
#  DROP_V_PATH=0.08, self.drop=Dropout(0.10), self.mod_drop_p=0.10)
#  are part of iter197 frozen state.
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialGatingUnit(nn.Module):
    # iter12: multi-head SGU — split the d_ffn hidden dim into n_heads
    # disjoint groups, each with its own seq_len×seq_len spatial mixer.
    # d_ffn=1048 is divisible by 2 (524) and 4 (262); 2 is the conservative
    # choice given seq_len=4 (per-head mixer is still small).
    SGU_N_HEADS = 2
    # iter22: per-sample DropPath on the v-mixing path. When dropped,
    # v_out is replaced by 1.0 (identity multiplier), so SGU output collapses
    # to `u` alone — bypassing spatial mixing. Train-time only.
    # iter93: fine sweep 0.10 -> 0.08 on R-Drop stack (iter64 was 0.08 without
    # R-Drop, failed; iter85 was 0.05 with R-Drop, failed).
    DROP_V_PATH = 0.08

    def __init__(self, d_ffn, seq_len):
        super().__init__()
        assert d_ffn % self.SGU_N_HEADS == 0, \
            f"d_ffn={d_ffn} not divisible by n_heads={self.SGU_N_HEADS}"
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.ModuleList([
            nn.Conv1d(seq_len, seq_len, kernel_size=1)
            for _ in range(self.SGU_N_HEADS)
        ])
        for proj in self.spatial_proj:
            nn.init.constant_(proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v_chunks = v.chunk(self.SGU_N_HEADS, dim=-1)
        v_out = torch.cat(
            [proj(vc) for proj, vc in zip(self.spatial_proj, v_chunks)],
            dim=-1,
        )
        if self.training and self.DROP_V_PATH > 0:
            keep = (torch.rand(x.size(0), 1, 1, device=x.device)
                    >= self.DROP_V_PATH).float()
            v_out = keep * v_out + (1.0 - keep)
        return u * v_out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return x + residual


class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)


class MultiModalGMLPFromFlat(nn.Module):
    def __init__(self, mod_dims: OrderedDict, d_model=512, d_ffn=1024,
                 depth=4, dropout=0.2, use_gated_pool=True,
                 head_dropout=0.10):
        super().__init__()
        self.mod_names = list(mod_dims.keys())
        self.mod_dims  = [mod_dims[n] for n in self.mod_names]
        self.seq_len   = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj = nn.ModuleDict({
            name: nn.Linear(in_dim, d_model)
            for name, in_dim in zip(self.mod_names, self.mod_dims)
        })
        # iter20: per-modality learnable scale (init=1.0, identity at start)
        self.proj_scale = nn.Parameter(torch.ones(self.seq_len))
        self.backbone = gMLP(seq_len=self.seq_len, d_model=d_model,
                             d_ffn=d_ffn, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        if use_gated_pool:
            self.alpha = nn.Parameter(torch.zeros(self.seq_len))
        # iter114: sigmoid skip-gate between gated and mean pool. init=0 so
        # sigmoid(0)=0.5 => 50/50 mix at start; lets training shift toward
        # the better aggregation. iter7/iter43 failed pre-stack.
        # iter116: upgrade scalar skip-gate to per-feature (d_model gates),
        # so each channel can pick its own mix of gated vs mean. Init=0 still
        # gives 50/50 at start.
        self.pool_skip_gate = nn.Parameter(torch.zeros(d_model))
        self.head = nn.Linear(d_model, 1)
        # iter86: head dropout 0.20 -> 0.10 on R-Drop stack. iter56 tried this
        # without R-Drop and failed; R-Drop's consistency reg may compensate
        # for the reduced explicit head dropout.
        # iter8 (phase-2): expose via constructor kwarg; default 0.10 keeps
        # phase-1 byte-identical when called from train.py / evaluate_combo.py.
        self.drop = nn.Dropout(head_dropout)
        # iter6: per-sample modality token dropout (zero a whole modality
        # token with prob p) — encourages cross-modal redundancy / prevents
        # single-modality overfit. Active in training only.
        # iter83: mod_drop 0.15 -> 0.10 on top of R-Drop. R-Drop provides
        # consistency-based regularization; reducing explicit token-zero
        # noise may free up signal that R-Drop is already protecting.
        self.mod_drop_p = 0.10

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        X = torch.stack(tokens, dim=1)
        # iter142: ablate iter20's proj_scale (skip its application). iter51's
        # ablation failed pre-R-Drop+skipgate; the gain may now come from the
        # downstream gates, making per-modality scale redundant.
        _ = self.proj_scale
        if self.training and self.mod_drop_p > 0:
            B = X.size(0)
            mask = (torch.rand(B, self.seq_len, device=X.device)
                    > self.mod_drop_p).float()
            X = X * mask.unsqueeze(-1)
        X = self.backbone(X)
        if self.use_gated_pool:
            w = torch.softmax(self.alpha, dim=0)
            gated = (X * w.view(1, -1, 1)).sum(dim=1)
            mean_pool = X.mean(dim=1)
            g = torch.sigmoid(self.pool_skip_gate)
            Xp = g * gated + (1.0 - g) * mean_pool
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  EDITABLE in phase-2 — training / eval
#
#  This block is the phase-2 lever for new training-procedure HP
#  (LR scheduler step, label smoothing in BCE target, EMA decay
#  sourced from cfg, ...). When adding a new hook:
#    1. Thread it as an OPTIONAL argument with a default that disables
#       the new behaviour (e.g. scheduler=None, label_smoothing=0.0).
#    2. Inside the loop, branch only when the value is non-default —
#       the phase-1 byte-identical code path must still execute when
#       all new args take their defaults.
#    3. Expose the same HP in BASE_CONFIG (above) and in
#       optuna_combo.py:suggest_config so Optuna can search it.
#  Do NOT touch the AdamW override semantics (iter38/88/148/172), the
#  R-Drop two-pass logic (iter80/106), the EMA bookkeeping shape, the
#  early-stopping logic, the val-pass shape, or the returned
#  train_info schema in a way that breaks the phase-1 byte-identical
#  guarantee.
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, optimizer, train_loader, val_loader, loss_fn,
                num_epochs=50, patience=10, es_metric="val_auc",
                ema_decay=0.9993, ema_warmup_epochs=1,
                label_smoothing=0.0,
                lr_schedule="constant", lr_warmup_epochs=0, lr_min_ratio=0.0):
    if es_metric == "val_loss":
        best_score = float("inf")
        is_better  = lambda new, cur: new < cur
    elif es_metric == "val_auc":
        best_score = float("-inf")
        is_better  = lambda new, cur: new > cur
    else:
        raise ValueError(f"Unknown es_metric: {es_metric}")

    # iter38: replace the passed-in Adam (wd=1e-5 ~ effectively 0) with AdamW
    # using decoupled wd=0.01 — a real weight-decay regularizer to complement
    # EMA / DropPath / mod_drop, with the same lr as before.
    lr = optimizer.param_groups[0]["lr"] * 1.25
    # iter88: AdamW wd 0.01 -> 0.005 on R-Drop stack. iter40 was a failure
    # without R-Drop; R-Drop adds reg, so the optimal explicit wd may shift
    # lower (similar to mod_drop / head dropout reductions in iter83/iter86).
    # iter148: AdamW wd 0.005 -> 0.003 (sweep down post proj_scale ablation).
    # iter172: split parameters into decay / no-decay groups (standard
    # transformer convention). Norms, biases, and small learnable scalars
    # (alpha, pool_skip_gate, proj_scale) skip wd; matrices keep wd=0.003.
    # iter58 tried a similar split pre-R-Drop, failed; retry on iter148 stack.
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "norm" in n.lower() or n in (
            "alpha", "pool_skip_gate", "proj_scale"
        ):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.003},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    # iter17: EMA of weights — validate and snapshot ES from the EMA copy;
    # training keeps running on the online weights.
    # iter27: warmup the EMA — during the first ema_warmup_epochs epochs,
    # ema_state tracks the online state exactly (no smoothing). After warmup,
    # exponential averaging at decay begins.
    # iter32/iter197: ema_decay swept 0.999 -> 0.9995 -> 0.9993.
    # Phase-2 iter1+: both ema_decay and ema_warmup_epochs are now caller-
    # provided. Defaults (0.9993, 1) reproduce phase-1 byte-for-byte.
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Phase-2 iter1+: LR scheduler built after AdamW (because AdamW is
    # constructed inside this function). scheduler=None when lr_schedule
    # ="constant" -> phase-1 byte-identical.
    scheduler = _build_lr_scheduler(
        optimizer,
        num_epochs=num_epochs,
        lr_schedule=lr_schedule,
        lr_warmup_epochs=lr_warmup_epochs,
        lr_min_ratio=lr_min_ratio,
    )

    best_state = None
    best_epoch = -1
    bad = 0
    epoch_log = []

    for epoch in range(num_epochs):
        model.train()
        tr_loss_sum, tr_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Phase-2 iter1+: optional label smoothing. label_smoothing=0.0
            # means the targets are untouched -> phase-1 byte-identical BCE.
            if label_smoothing > 0.0:
                y_target = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                y_target = y
            optimizer.zero_grad()
            # iter80: R-Drop — two forward passes (different dropout masks)
            # with a symmetric Bernoulli-KL consistency term added to the BCE.
            # Forces predictions to be invariant to the stochastic forward
            # (mod_drop / DROP_V_PATH / Dropout) — independent of regularizer
            # strength, complements them with a self-consistency constraint.
            logits1 = model(x)
            logits2 = model(x)
            bce_loss = 0.5 * (loss_fn(logits1, y_target) + loss_fn(logits2, y_target))
            eps = 1e-7
            p1 = torch.sigmoid(logits1).clamp(eps, 1.0 - eps)
            p2 = torch.sigmoid(logits2).clamp(eps, 1.0 - eps)
            kl_12 = (p1 * (p1.log() - p2.log())
                     + (1 - p1) * ((1 - p1).log() - (1 - p2).log())).mean()
            kl_21 = (p2 * (p2.log() - p1.log())
                     + (1 - p2) * ((1 - p2).log() - (1 - p1).log())).mean()
            # iter106: warmup R-Drop alpha 0 -> 1.0 over first 5 epochs (linear).
            # Defer consistency penalty until model has learned useful features,
            # avoiding penalizing noise-driven predictions in epoch 0-1.
            rdrop_alpha = 1.0 * min(1.0, (epoch + 1) / 5.0)
            loss = bce_loss + rdrop_alpha * 0.5 * (kl_12 + kl_21)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                msd = model.state_dict()
                in_warmup = epoch < ema_warmup_epochs
                for k in ema_state:
                    if msd[k].dtype.is_floating_point and not in_warmup:
                        ema_state[k].mul_(ema_decay).add_(
                            msd[k].detach(), alpha=1.0 - ema_decay)
                    else:
                        ema_state[k].copy_(msd[k])
            tr_loss_sum += loss.item()
            tr_batches  += 1
        train_loss = tr_loss_sum / max(tr_batches, 1)

        # Validate using EMA weights (swap in, then restore).
        online_state = deepcopy(model.state_dict())
        model.load_state_dict(ema_state)
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        y_true_v, y_prob_v = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x_d, y_d = x.to(device), y.to(device)
                logits = model(x_d)
                val_loss_sum += loss_fn(logits, y_d).item()
                val_batches  += 1
                probs = torch.sigmoid(logits).cpu().numpy()
                y_prob_v.extend(probs.tolist())
                y_true_v.extend(y.numpy().tolist())
        model.load_state_dict(online_state)
        val_loss = val_loss_sum / max(val_batches, 1)
        has_both = len(set(y_true_v)) > 1
        val_auc = float(roc_auc_score(y_true_v, y_prob_v)) if has_both else 0.0
        val_pred = (np.array(y_prob_v) > 0.5).astype(int)
        val_mcc = float(matthews_corrcoef(y_true_v, val_pred))

        epoch_log.append({
            "epoch":      int(epoch),
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_auc":    round(val_auc, 6),
            "val_mcc":    round(val_mcc, 6),
        })

        score = val_auc if es_metric == "val_auc" else val_loss
        if is_better(score, best_score):
            best_score = score
            best_state = deepcopy(ema_state)
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        # Phase-2 iter1+: LR scheduler step at epoch end. scheduler=None is
        # the default (-> phase-1 byte-identical: no LR change).
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "es_metric":    es_metric,
        "best_epoch":   int(best_epoch),
        "best_score":   round(float(best_score), 6),
        "n_epochs_run": len(epoch_log),
        "epoch_log":    epoch_log,
    }


def eval_model(model, loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            y_prob.extend(probs)
            y_true.extend(y.numpy())

    y_pred = (np.array(y_prob) > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, _, _ = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    has_both = len(set(y_true)) > 1
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 6),
        "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":      round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1":          round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc":     round(float(roc_auc_score(y_true, y_prob) if has_both else 0.0), 6),
        "mcc":         round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "auprc":       round(float(average_precision_score(y_true, y_prob) if has_both else 0.0), 6),
        "specificity": round(float(specificity), 6),
    }, y_true, y_prob


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API — phase-1 byte-identical guarantee
#
#  build_and_train(...) is the entry point that evaluate_combo.py and
#  final_holdout_eval.py call via train.py (NOT this file), so they are
#  unaffected by changes here. Phase-2 (optuna_combo.py) DOES call
#  best_train.build_and_train with a `config` override.
#
#  Editable in phase-2:
#    - cfg-driven wiring: forwarding new BASE_CONFIG keys to train_model
#      kwargs.
#
#  Frozen:
#    - The CALL SIGNATURE of build_and_train (positional args + the
#      `config` kwarg).
#    - The RETURN tuple shape: (model, train_info, val_metrics, scaler).
#    - The behaviour when invoked as build_and_train(..., config=None):
#      MUST reproduce iter197 byte-for-byte. New code paths (scheduler,
#      smoothing, ...) must be gated on cfg values whose defaults
#      disable them.
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_train(
    combo,
    X_pool: np.ndarray, y_pool: np.ndarray,
    train_idx, val_idx,
    fp_dim: dict,
    seed: int,
    config: dict = None,
):
    """
    Train one (combo, seed) run. Returns (model, train_info, val_metrics, scaler).

    Used by Phase-2 HPO (optuna_combo.py) for both search and confirm phases.
    Phase-1 paths (evaluate_combo.py / final_holdout_eval.py) still import
    train.py:build_and_train (NOT this one).

    config: optional dict of overrides on top of BASE_CONFIG (Phase 2 HPO).
        When None or empty, behavior is identical to the frozen iter197
        baseline. Missing keys fall back to BASE_CONFIG defaults.
    """
    cfg = {**BASE_CONFIG, **(config or {})}

    set_seed(seed)

    X_train = torch.tensor(X_pool[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_pool[train_idx], dtype=torch.float32)
    X_val   = torch.tensor(X_pool[val_idx],   dtype=torch.float32)
    y_val   = torch.tensor(y_pool[val_idx],   dtype=torch.float32)

    scaler = None
    if "rdkit" in combo:
        offset = 0
        for t in combo:
            if t == "rdkit":
                rd_start, rd_end = offset, offset + fp_dim["rdkit"]
                break
            offset += fp_dim[t]
        scaler = StandardScaler()
        scaler.fit(X_train[:, rd_start:rd_end].numpy())
        X_train[:, rd_start:rd_end] = torch.tensor(
            scaler.transform(X_train[:, rd_start:rd_end].numpy()), dtype=torch.float32)
        X_val[:, rd_start:rd_end] = torch.tensor(
            scaler.transform(X_val[:, rd_start:rd_end].numpy()), dtype=torch.float32)

    bs = cfg["batch_size"]
    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train), batch_size=bs, shuffle=True, drop_last=False)
    val_loader = data.DataLoader(
        data.TensorDataset(X_val, y_val), batch_size=bs, shuffle=False)

    mod_dims = OrderedDict((t, fp_dim[t]) for t in combo)
    model = MultiModalGMLPFromFlat(
        mod_dims=mod_dims,
        d_model=cfg["d_model"],
        d_ffn=cfg["d_ffn"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
        use_gated_pool=cfg["use_gated_pool"],
        head_dropout=cfg.get("head_dropout", 0.10),
    ).to(device)

    # The Adam optimizer here is read for its lr only — train_model
    # replaces it with AdamW(lr * 1.25, wd=0.003) (iter38/88/148/172
    # frozen). BASE_CONFIG.weight_decay is therefore effectively unused
    # in this baseline.
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss()

    model, train_info = train_model(
        model, optimizer, train_loader, val_loader, loss_fn,
        num_epochs=cfg["num_epochs"],
        patience=cfg["patience"],
        es_metric=cfg["es_metric"],
        ema_decay=cfg.get("ema_decay", 0.9993),
        ema_warmup_epochs=cfg.get("ema_warmup_epochs", 1),
        label_smoothing=cfg.get("label_smoothing", 0.0),
        lr_schedule=cfg.get("lr_schedule", "constant"),
        lr_warmup_epochs=cfg.get("lr_warmup_epochs", 0),
        lr_min_ratio=cfg.get("lr_min_ratio", 0.0),
    )

    val_metrics, _, _ = eval_model(model, val_loader)
    return model, train_info, val_metrics, scaler


def _build_lr_scheduler(optimizer, *, num_epochs,
                        lr_schedule="constant",
                        lr_warmup_epochs=0, lr_min_ratio=0.0):
    """Construct an LR scheduler from the phase-2 BASE_CONFIG keys.

    Returns None when lr_schedule == "constant" so the training loop's
    `if scheduler is not None: scheduler.step()` skips entirely and
    behaves byte-identical to phase-1.

    Supported schedules:
      - "constant"      -> None (phase-1 default).
      - "cosine"        -> CosineAnnealingLR over num_epochs.
                            eta_min = lr * lr_min_ratio.
      - "warmup_cosine" -> linear warmup over lr_warmup_epochs (lr_i/warmup),
                            then cosine annealing on the remainder down to
                            lr * lr_min_ratio.
    """
    if lr_schedule == "constant":
        return None

    total = max(1, int(num_epochs))
    min_ratio = float(lr_min_ratio)

    if lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        # Use the AdamW lr (read from optimizer) to compute eta_min so the
        # ratio is correct regardless of upstream lr multipliers.
        lr0 = float(optimizer.param_groups[0]["lr"])
        eta_min = lr0 * min_ratio
        return CosineAnnealingLR(optimizer, T_max=total, eta_min=eta_min)

    if lr_schedule == "warmup_cosine":
        from torch.optim.lr_scheduler import LambdaLR
        import math
        warmup = int(lr_warmup_epochs)

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                # Linear warmup from 1/warmup -> 1.0 over `warmup` epochs.
                return float(epoch + 1) / float(warmup)
            # Cosine annealing from 1.0 -> min_ratio over the remainder.
            progress = (epoch - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown lr_schedule: {lr_schedule!r}")


def apply_rdkit_scaler(X: np.ndarray, combo, fp_dim: dict, scaler):
    """Apply a fitted scaler to the rdkit slice of a feature matrix. Returns new array."""
    if scaler is None or "rdkit" not in combo:
        return X
    offset = 0
    for t in combo:
        if t == "rdkit":
            rd_start, rd_end = offset, offset + fp_dim["rdkit"]
            break
        offset += fp_dim[t]
    X_out = X.copy()
    X_out[:, rd_start:rd_end] = scaler.transform(X_out[:, rd_start:rd_end])
    return X_out


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase-2 HPO additions — Optuna pruning hooks
#
#  These functions exist solely to support optuna_combo.py:
#    - train_model_with_pruning / build_and_train_with_pruning mirror
#      train_model / build_and_train with an added trial.report +
#      should_prune call per epoch. optuna is imported lazily so the
#      phase-1 path has no optuna dependency.
#  Phase-1 code paths (evaluate_combo.py, final_holdout_eval.py) keep
#  importing train.py (NOT best_train.py), so anything added below has
#  zero impact on phase-1 reproducibility.
#
#  Phase-2 editable: append new helpers parallel to the *_with_pruning
#  pair when a new HPO lever needs a hook not expressible inside
#  train_model. Do NOT remove or break the existing *_with_pruning
#  signature (optuna_combo.py depends on it).
# ═══════════════════════════════════════════════════════════════════════════════

def train_model_with_pruning(model, optimizer, train_loader, val_loader, loss_fn,
                             trial,
                             num_epochs=50, patience=10, es_metric="val_auc",
                             ema_decay=0.9993, ema_warmup_epochs=1,
                             label_smoothing=0.0,
                             lr_schedule="constant",
                             lr_warmup_epochs=0, lr_min_ratio=0.0):
    """Mirror of train_model with Optuna pruning. Reports best-so-far score
    each epoch and raises TrialPruned if the pruner says so."""
    import optuna

    if es_metric == "val_loss":
        best_score = float("inf")
        is_better  = lambda new, cur: new < cur
    elif es_metric == "val_auc":
        best_score = float("-inf")
        is_better  = lambda new, cur: new > cur
    else:
        raise ValueError(f"Unknown es_metric: {es_metric}")

    # Same AdamW override as train_model (iter38/88/148/172 frozen).
    lr = optimizer.param_groups[0]["lr"] * 1.25
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "norm" in n.lower() or n in (
            "alpha", "pool_skip_gate", "proj_scale"
        ):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.003},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    scheduler = _build_lr_scheduler(
        optimizer,
        num_epochs=num_epochs,
        lr_schedule=lr_schedule,
        lr_warmup_epochs=lr_warmup_epochs,
        lr_min_ratio=lr_min_ratio,
    )

    best_state = None
    best_epoch = -1
    bad = 0
    epoch_log = []

    for epoch in range(num_epochs):
        model.train()
        tr_loss_sum, tr_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if label_smoothing > 0.0:
                y_target = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                y_target = y
            optimizer.zero_grad()
            logits1 = model(x)
            logits2 = model(x)
            bce_loss = 0.5 * (loss_fn(logits1, y_target) + loss_fn(logits2, y_target))
            eps = 1e-7
            p1 = torch.sigmoid(logits1).clamp(eps, 1.0 - eps)
            p2 = torch.sigmoid(logits2).clamp(eps, 1.0 - eps)
            kl_12 = (p1 * (p1.log() - p2.log())
                     + (1 - p1) * ((1 - p1).log() - (1 - p2).log())).mean()
            kl_21 = (p2 * (p2.log() - p1.log())
                     + (1 - p2) * ((1 - p2).log() - (1 - p1).log())).mean()
            rdrop_alpha = 1.0 * min(1.0, (epoch + 1) / 5.0)
            loss = bce_loss + rdrop_alpha * 0.5 * (kl_12 + kl_21)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                msd = model.state_dict()
                in_warmup = epoch < ema_warmup_epochs
                for k in ema_state:
                    if msd[k].dtype.is_floating_point and not in_warmup:
                        ema_state[k].mul_(ema_decay).add_(
                            msd[k].detach(), alpha=1.0 - ema_decay)
                    else:
                        ema_state[k].copy_(msd[k])
            tr_loss_sum += loss.item()
            tr_batches  += 1
        train_loss = tr_loss_sum / max(tr_batches, 1)

        online_state = deepcopy(model.state_dict())
        model.load_state_dict(ema_state)
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        y_true_v, y_prob_v = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x_d, y_d = x.to(device), y.to(device)
                logits = model(x_d)
                val_loss_sum += loss_fn(logits, y_d).item()
                val_batches  += 1
                probs = torch.sigmoid(logits).cpu().numpy()
                y_prob_v.extend(probs.tolist())
                y_true_v.extend(y.numpy().tolist())
        model.load_state_dict(online_state)
        val_loss = val_loss_sum / max(val_batches, 1)
        has_both = len(set(y_true_v)) > 1
        val_auc = float(roc_auc_score(y_true_v, y_prob_v)) if has_both else 0.0
        val_pred = (np.array(y_prob_v) > 0.5).astype(int)
        val_mcc = float(matthews_corrcoef(y_true_v, val_pred))

        epoch_log.append({
            "epoch":      int(epoch),
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_auc":    round(val_auc, 6),
            "val_mcc":    round(val_mcc, 6),
        })

        score = val_auc if es_metric == "val_auc" else val_loss
        if is_better(score, best_score):
            best_score = score
            best_state = deepcopy(ema_state)
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        # Study direction is maximize; flip sign for val_loss so the reported
        # signal is monotone-increasing in either es_metric setting.
        reported = best_score if es_metric == "val_auc" else -best_score
        trial.report(float(reported), step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "es_metric":    es_metric,
        "best_epoch":   int(best_epoch),
        "best_score":   round(float(best_score), 6),
        "n_epochs_run": len(epoch_log),
        "epoch_log":    epoch_log,
    }


def build_and_train_with_pruning(
    combo,
    X_pool: np.ndarray, y_pool: np.ndarray,
    train_idx, val_idx,
    fp_dim: dict,
    seed: int,
    trial,
    config: dict = None,
):
    """Mirror of build_and_train that uses train_model_with_pruning.
    Only used by phase-2 HPO; phase-1 paths continue to call build_and_train."""
    cfg = {**BASE_CONFIG, **(config or {})}

    set_seed(seed)

    X_train = torch.tensor(X_pool[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_pool[train_idx], dtype=torch.float32)
    X_val   = torch.tensor(X_pool[val_idx],   dtype=torch.float32)
    y_val   = torch.tensor(y_pool[val_idx],   dtype=torch.float32)

    scaler = None
    if "rdkit" in combo:
        offset = 0
        for t in combo:
            if t == "rdkit":
                rd_start, rd_end = offset, offset + fp_dim["rdkit"]
                break
            offset += fp_dim[t]
        scaler = StandardScaler()
        scaler.fit(X_train[:, rd_start:rd_end].numpy())
        X_train[:, rd_start:rd_end] = torch.tensor(
            scaler.transform(X_train[:, rd_start:rd_end].numpy()), dtype=torch.float32)
        X_val[:, rd_start:rd_end] = torch.tensor(
            scaler.transform(X_val[:, rd_start:rd_end].numpy()), dtype=torch.float32)

    bs = cfg["batch_size"]
    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train), batch_size=bs, shuffle=True, drop_last=False)
    val_loader = data.DataLoader(
        data.TensorDataset(X_val, y_val), batch_size=bs, shuffle=False)

    mod_dims = OrderedDict((t, fp_dim[t]) for t in combo)
    model = MultiModalGMLPFromFlat(
        mod_dims=mod_dims,
        d_model=cfg["d_model"],
        d_ffn=cfg["d_ffn"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
        use_gated_pool=cfg["use_gated_pool"],
        head_dropout=cfg.get("head_dropout", 0.10),
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss()

    model, train_info = train_model_with_pruning(
        model, optimizer, train_loader, val_loader, loss_fn,
        trial=trial,
        num_epochs=cfg["num_epochs"],
        patience=cfg["patience"],
        es_metric=cfg["es_metric"],
        ema_decay=cfg.get("ema_decay", 0.9993),
        ema_warmup_epochs=cfg.get("ema_warmup_epochs", 1),
        label_smoothing=cfg.get("label_smoothing", 0.0),
        lr_schedule=cfg.get("lr_schedule", "constant"),
        lr_warmup_epochs=cfg.get("lr_warmup_epochs", 0),
        lr_min_ratio=cfg.get("lr_min_ratio", 0.0),
    )

    val_metrics, _, _ = eval_model(model, val_loader)
    return model, train_info, val_metrics, scaler
