#!/usr/bin/env python3
"""
train.py — model + training loop for autoresearch_combos_v2.

This is the SINGLE FILE THE AGENT EDITS.

Editable:
  - SpatialGatingUnit, gMLPBlock, gMLP, MultiModalGMLPFromFlat (the model)
  - train_model() and eval_model() (training/eval logic)

Do NOT edit:
  - BASE_CONFIG (frozen below)
  - the import / public-API surface used by evaluate_combo.py and
    final_holdout_eval.py: build_and_train(...) -> (model, train_info, val_metrics)
  - device handling
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
#  FROZEN — DO NOT EDIT
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
#  EDITABLE — model definition
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialGatingUnit(nn.Module):
    # iter12: multi-head SGU — split the d_ffn hidden dim into n_heads
    # disjoint groups, each with its own seq_len×seq_len spatial mixer.
    # d_ffn=1048 is divisible by 2 (524) and 4 (262); 2 is the conservative
    # choice given seq_len=4 (per-head mixer is still small).
    SGU_N_HEADS = 2

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
                 depth=4, dropout=0.2, use_gated_pool=True):
        super().__init__()
        self.mod_names = list(mod_dims.keys())
        self.mod_dims  = [mod_dims[n] for n in self.mod_names]
        self.seq_len   = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj = nn.ModuleDict({
            name: nn.Linear(in_dim, d_model)
            for name, in_dim in zip(self.mod_names, self.mod_dims)
        })
        # iter14: prepend a learnable CLS token before the backbone and pool
        # from its final state. Backbone (and SGU Conv1d) is built with
        # seq_len+1 to accommodate it.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = gMLP(seq_len=self.seq_len + 1, d_model=d_model,
                             d_ffn=d_ffn, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        # iter6: per-sample modality token dropout (zero a whole modality
        # token with prob p) — encourages cross-modal redundancy / prevents
        # single-modality overfit. Active in training only.
        self.mod_drop_p = 0.15

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        X = torch.stack(tokens, dim=1)
        if self.training and self.mod_drop_p > 0:
            B = X.size(0)
            mask = (torch.rand(B, self.seq_len, device=X.device)
                    > self.mod_drop_p).float()
            X = X * mask.unsqueeze(-1)
        # Prepend CLS (never dropped). Resulting sequence length = seq_len+1.
        cls = self.cls_token.expand(X.size(0), -1, -1)
        X = torch.cat([cls, X], dim=1)
        X = self.backbone(X)
        if self.use_gated_pool:
            Xp = X[:, 0, :]
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  EDITABLE — training / eval
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, optimizer, train_loader, val_loader, loss_fn,
                num_epochs=50, patience=10, es_metric="val_auc"):
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

    for epoch in range(num_epochs):
        model.train()
        tr_loss_sum, tr_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item()
            tr_batches  += 1
        train_loss = tr_loss_sum / max(tr_batches, 1)

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
            best_state = deepcopy(model.state_dict())
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
#  FROZEN — public API used by evaluate_combo.py / final_holdout_eval.py
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_train(
    combo,
    X_pool: np.ndarray, y_pool: np.ndarray,
    train_idx, val_idx,
    fp_dim: dict,
    seed: int,
):
    """
    Train one (combo, seed) run. Returns (model, train_info, val_metrics, scaler).

    Used by evaluate_combo.py for iteration eval and by final_holdout_eval.py
    for the 5-subset breakdown. Signature is part of the frozen API — agents
    must not change it.
    """
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

    bs = BASE_CONFIG["batch_size"]
    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train), batch_size=bs, shuffle=True, drop_last=False)
    val_loader = data.DataLoader(
        data.TensorDataset(X_val, y_val), batch_size=bs, shuffle=False)

    mod_dims = OrderedDict((t, fp_dim[t]) for t in combo)
    model = MultiModalGMLPFromFlat(
        mod_dims=mod_dims,
        d_model=BASE_CONFIG["d_model"],
        d_ffn=BASE_CONFIG["d_ffn"],
        depth=BASE_CONFIG["depth"],
        dropout=BASE_CONFIG["dropout"],
        use_gated_pool=BASE_CONFIG["use_gated_pool"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=BASE_CONFIG["lr"],
        weight_decay=BASE_CONFIG["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss()

    model, train_info = train_model(
        model, optimizer, train_loader, val_loader, loss_fn,
        num_epochs=BASE_CONFIG["num_epochs"],
        patience=BASE_CONFIG["patience"],
        es_metric=BASE_CONFIG["es_metric"],
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
