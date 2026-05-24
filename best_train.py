#!/usr/bin/env python3
"""
best_train.py — phase-1 best architecture snapshot for Phase-2 HPO.

Created from train.py at commit cc2a448 ("Add HPO-friendly best_train.py
for Phase 2") as a snapshot of the iter198 best architecture, refactored
so Optuna trials can inject a config dict instead of mutating the
module-level BASE_CONFIG. evaluate_combo.py / final_holdout_eval.py
keep importing train.py (NOT this file), so phase-1 paths are not
affected by anything that happens here.

Phase-2 edit surface (see program_phase2.md → Files section for the
authoritative rules):

  FROZEN (phase-1 output; never edit in phase-2):
    - Model class definitions: SpatialGatingUnit, gMLPBlock, gMLP,
      MultiModalGMLPFromFlat. Phase-1 architecture sweep is the
      single source of truth for these.
    - device handling.
    - The phase-1 behaviour of build_and_train(config=None) — that
      call MUST keep reproducing iter198 byte-for-byte after any
      phase-2 change. Adding new BASE_CONFIG keys is allowed only if
      their default values turn the new behaviour OFF (e.g.
      lr_schedule="constant", label_smoothing=0.0, ema_decay=0.999).

  EDITABLE (phase-2 lever surface):
    - BASE_CONFIG: add new HP keys with phase-1-reproducing defaults
      (see Phase-2 iter7+ block at the end of the dict).
    - train_model / train_model_with_pruning: new optional arguments
      (with defaults that disable the new behaviour) + new training
      hooks (LR scheduler step, label smoothing in loss target,
      optimizer-family branch, ...).
    - build_and_train / build_and_train_with_pruning: wire the new
      cfg keys into optimizer / loss / scheduler construction, then
      forward to the train_model variants.
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
#    - You MAY add new keys (LR schedule, label smoothing, ema decay,
#      optimizer family, ...) so suggest_config() in optuna_combo.py can
#      search them.
#    - Each new key's DEFAULT must reproduce phase-1 byte-for-byte when
#      build_and_train is invoked with config=None (i.e. the new
#      behaviour is OFF by default). See the "Phase-2 iter7+" block
#      inside the dict for examples (lr_schedule="constant",
#      label_smoothing=0.0, ema_decay=0.999).
#    - Existing keys' default values are FROZEN at phase-1 values.
#      Do not change them; only add new keys.
#  Device handling is FROZEN.
# ═══════════════════════════════════════════════════════════════════════════════

BASE_CONFIG = {
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
    # Exposed for Phase 2 HPO. Defaults match prior hard-coded values.
    "drop_path":           0.025,
    "mod_drop_p":          0.1,
    "head_dropout":        0.08,
    "grad_clip_max_norm":  1.0,
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
#
#  The constructor kwargs (drop_path, mod_drop_p, head_dropout, ...) ARE
#  HP knobs that BASE_CONFIG can override — that is intentional and not
#  considered an edit to the model definition.
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        self.gate_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        s = self.gate_scale.exp()
        v = s * self.spatial_proj(v) + (1 - s) * v
        return u * v


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, drop_path=0.025):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.drop_path = drop_path

    def forward(self, x):
        residual = x
        if self.training and self.drop_path > 0 and torch.rand(1).item() < self.drop_path:
            return residual
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return x + residual


class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6,
                 drop_path=0.025):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len, drop_path=drop_path)
              for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)


class MultiModalGMLPFromFlat(nn.Module):
    def __init__(self, mod_dims: OrderedDict, d_model=512, d_ffn=1024,
                 depth=4, dropout=0.2, use_gated_pool=True,
                 drop_path=0.025, mod_drop_p=0.1, head_dropout=0.08):
        super().__init__()
        self.mod_names = list(mod_dims.keys())
        self.mod_dims  = [mod_dims[n] for n in self.mod_names]
        self.seq_len   = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj = nn.ModuleDict({
            name: nn.Linear(in_dim, d_model)
            for name, in_dim in zip(self.mod_names, self.mod_dims)
        })
        self.backbone = gMLP(seq_len=self.seq_len, d_model=d_model,
                             d_ffn=d_ffn, num_layers=depth,
                             drop_path=drop_path)
        self.final_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)
        if use_gated_pool:
            self.alpha = nn.Parameter(torch.zeros(self.seq_len))
            self.n_attn_heads = 1
            self.pool_query = nn.Parameter(torch.zeros(self.n_attn_heads, d_model // self.n_attn_heads))
            self.pool_logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.6]))
            self.gated_norm = nn.LayerNorm(d_model)
            self.mean_norm = nn.LayerNorm(d_model)
            self.max_norm = nn.LayerNorm(d_model)
            self.attn_norm = nn.LayerNorm(d_model)
            self.attn_head_proj = nn.Linear(d_model, d_model)
            nn.init.eye_(self.attn_head_proj.weight)
            nn.init.zeros_(self.attn_head_proj.bias)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, 1),
        )
        self.drop = nn.Dropout(dropout)
        self.mod_drop_p = mod_drop_p

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        X = torch.stack(tokens, dim=1)
        if self.training and self.mod_drop_p > 0:
            B = X.shape[0]
            mask = (torch.rand(B, self.seq_len, device=X.device) > self.mod_drop_p).float()
            X = X * mask.unsqueeze(-1)
        X = self.backbone(X)
        X = self.final_norm(X)
        if self.use_gated_pool:
            w = torch.softmax(self.alpha, dim=0)
            gated = (X * w.view(1, -1, 1)).sum(dim=1)
            mean = X.mean(dim=1)
            max_pool = X.max(dim=1).values
            B, L, D = X.shape
            H = self.n_attn_heads
            X_h = X.view(B, L, H, D // H)
            attn_scores = (X_h * self.pool_query.view(1, 1, H, -1)).sum(dim=-1) / ((D // H) ** 0.5)
            attn_w = torch.softmax(attn_scores, dim=1)
            attn_pool = (X_h * attn_w.unsqueeze(-1)).sum(dim=1).reshape(B, D)
            gated = self.gated_norm(gated)
            mean = self.mean_norm(mean)
            max_pool = self.max_norm(max_pool)
            attn_pool = self.attn_norm(attn_pool)
            attn_pool = self.attn_head_proj(attn_pool)
            pw = torch.softmax(self.pool_logits, dim=0)
            Xp = pw[0] * gated + pw[1] * mean + pw[2] * max_pool + pw[3] * attn_pool
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  EDITABLE in phase-2 — training / eval
#
#  This block is the phase-2 lever for new training-procedure HP
#  (LR scheduler step, label smoothing in BCE target, optimizer-family
#  branches, EMA decay sourced from cfg, ...). When adding a new hook:
#    1. Thread it as an OPTIONAL argument with a default that disables
#       the new behaviour (e.g. scheduler=None, label_smoothing=0.0).
#    2. Inside the loop, branch only when the value is non-default —
#       the phase-1 byte-identical code path must still execute when
#       all new args take their defaults.
#    3. Expose the same HP in BASE_CONFIG (above) and in
#       optuna_combo.py:suggest_config so Optuna can search it.
#  Do NOT touch the optimizer construction, the EMA bookkeeping
#  semantics, the early-stopping logic, the val-pass shape, or the
#  returned train_info schema in a way that breaks the phase-1
#  byte-identical guarantee.
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, optimizer, train_loader, val_loader, loss_fn,
                num_epochs=50, patience=10, es_metric="val_auc",
                grad_clip_max_norm=1.0):
    if es_metric == "val_loss":
        best_score = float("inf")
        is_better  = lambda new, cur: new < cur
    elif es_metric == "val_auc":
        best_score = float("-inf")
        is_better  = lambda new, cur: new > cur
    else:
        raise ValueError(f"Unknown es_metric: {es_metric}")

    best_state = None
    best_epoch = -1
    bad = 0
    epoch_log = []

    ema_decay = 0.999
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(num_epochs):
        model.train()
        tr_loss_sum, tr_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                    else:
                        ema_state[k].copy_(v.detach())
            tr_loss_sum += loss.item()
            tr_batches  += 1
        train_loss = tr_loss_sum / max(tr_batches, 1)

        train_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
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
        model.load_state_dict(train_state)
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
            best_state = {k: v.detach().clone() for k, v in ema_state.items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

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
#  unaffected by changes here. Phase-2 (optuna_combo.py and
#  phase2_holdout integration) DOES call best_train.build_and_train
#  with a `config` override.
#
#  Editable in phase-2:
#    - cfg-driven wiring: scheduler construction via _build_lr_scheduler,
#      optimizer/loss kwargs branched on new BASE_CONFIG keys, etc.
#    - forwarding new optional kwargs to train_model variants.
#
#  Frozen:
#    - The CALL SIGNATURE of build_and_train (positional args + the
#      `config` kwarg).
#    - The RETURN tuple shape: (model, train_info, val_metrics, scaler).
#    - The behaviour when invoked as build_and_train(..., config=None):
#      MUST reproduce the iter198 baseline byte-for-byte. New code paths
#      (scheduler, smoothing, ...) must be gated on cfg values whose
#      defaults disable them.
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

    Used by evaluate_combo.py for iteration eval and by final_holdout_eval.py
    for the 5-subset breakdown.

    config: optional dict of overrides on top of BASE_CONFIG (Phase 2 HPO).
        When None or empty, behavior is identical to the frozen BASE_CONFIG
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
        drop_path=cfg["drop_path"],
        mod_drop_p=cfg["mod_drop_p"],
        head_dropout=cfg["head_dropout"],
    ).to(device)

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
    )
    loss_fn = nn.BCEWithLogitsLoss()

    model, train_info = train_model(
        model, optimizer, train_loader, val_loader, loss_fn,
        num_epochs=cfg["num_epochs"],
        patience=cfg["patience"],
        es_metric=cfg["es_metric"],
        grad_clip_max_norm=cfg["grad_clip_max_norm"],
    )

    val_metrics, _, _ = eval_model(model, val_loader)
    return model, train_info, val_metrics, scaler


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
#  Phase-2 HPO additions — Optuna pruning hooks (iter2+) and any phase-2
#  helpers that mirror but do not replace the phase-1 functions above.
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
                             grad_clip_max_norm=1.0):
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

    best_state = None
    best_epoch = -1
    bad = 0
    epoch_log = []

    ema_decay = 0.999
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(num_epochs):
        model.train()
        tr_loss_sum, tr_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                    else:
                        ema_state[k].copy_(v.detach())
            tr_loss_sum += loss.item()
            tr_batches  += 1
        train_loss = tr_loss_sum / max(tr_batches, 1)

        train_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
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
        model.load_state_dict(train_state)
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
            best_state = {k: v.detach().clone() for k, v in ema_state.items()}
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
        drop_path=cfg["drop_path"],
        mod_drop_p=cfg["mod_drop_p"],
        head_dropout=cfg["head_dropout"],
    ).to(device)

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
    )
    loss_fn = nn.BCEWithLogitsLoss()

    model, train_info = train_model_with_pruning(
        model, optimizer, train_loader, val_loader, loss_fn,
        trial=trial,
        num_epochs=cfg["num_epochs"],
        patience=cfg["patience"],
        es_metric=cfg["es_metric"],
        grad_clip_max_norm=cfg["grad_clip_max_norm"],
    )

    val_metrics, _, _ = eval_model(model, val_loader)
    return model, train_info, val_metrics, scaler
