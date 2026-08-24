"""
Baseline model definitions and training helpers for autoresearch_combos_v2.

Models: mlp2, mlp3, mlp4 (SimpleMLP with depth=2/3/4), xgboost, lightgbm.

Hyperparameters and architecture are kept identical to the April 2026 baseline
comparison (see /home/minji/BBB/scage/BBB/baseline_ext_augment.py) so that
results align with the previous comparison study. The only adaptation is that
training/eval data here comes from autoresearch_combos_v2's pool cache and
8:2 scaffold split (single train/val), not the April 8:1:1 train/val/test.
"""

import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import xgboost as xgb
import lightgbm as lgb


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mirror BASE_CONFIG in autoresearch_combos_v2/train.py for fair comparison
HIDDEN_DIM   = 1024
DROPOUT      = 0.2
LR           = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS   = 50
PATIENCE     = 10
BATCH_SIZE   = 128

MLP_DEPTHS = {"mlp2": 2, "mlp3": 3, "mlp4": 4}
ALL_MODELS = ["mlp2", "mlp3", "mlp4", "xgboost", "lightgbm"]


class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=HIDDEN_DIM, depth=2, dropout=DROPOUT):
        super().__init__()
        layers, prev = [], in_features
        for _ in range(depth):
            layers.extend([nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            prev = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mlp(X_train, y_train, X_val, y_val, depth: int, seed: int):
    set_seed(seed)
    model = SimpleMLP(X_train.shape[1], HIDDEN_DIM, depth, DROPOUT).to(DEVICE)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos > 0:
        pw = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val,   dtype=torch.float32)
    yv = torch.tensor(y_val,   dtype=torch.float32)

    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(Xv, yv), batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    n_epochs_run = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            optimizer.step()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                vloss += loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        vloss /= max(len(val_loader), 1)

        n_epochs_run = epoch + 1
        if vloss < best_val_loss:
            best_val_loss = vloss
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": float(best_val_loss), "n_epochs_run": int(n_epochs_run)}


def predict_mlp_probs(model, X_np):
    model.eval()
    X = torch.tensor(X_np, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(torch.sigmoid(model(X[i:i+256].to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def train_xgb(X_train, y_train, seed: int):
    model = xgb.XGBClassifier(
        random_state=seed,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)
    return model


def train_lgbm(X_train, y_train, seed: int):
    model = lgb.LGBMClassifier(
        random_state=seed,
        verbose=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)
    return model


def train_one(model_name: str, X_train, y_train, X_val, y_val, seed: int):
    """Dispatch trainer by model name. Returns (model, train_info_dict)."""
    if model_name in MLP_DEPTHS:
        return train_mlp(X_train, y_train, X_val, y_val, MLP_DEPTHS[model_name], seed)
    if model_name == "xgboost":
        return train_xgb(X_train, y_train, seed), {}
    if model_name == "lightgbm":
        return train_lgbm(X_train, y_train, seed), {}
    raise ValueError(f"Unknown model: {model_name}")


def predict_probs(model_name: str, model, X):
    """Dispatch inference by model name. Returns 1D numpy array of P(label=1)."""
    if model_name in MLP_DEPTHS:
        return predict_mlp_probs(model, X)
    if model_name in ("xgboost", "lightgbm"):
        return model.predict_proba(X)[:, 1]
    raise ValueError(f"Unknown model: {model_name}")
