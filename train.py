"""
PyTorch-based tabular binary classification for obstructive_cad.

Uses residual MLP with early stopping. Compatible with prepare.load_tabular_data
and prepare.evaluate (BCE loss, ROC AUC).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import load_tabular_data, evaluate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "2601_ground_truth_input_data_mapped.csv"
LOG1P_TRANSFORM = True
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Architecture
HIDDEN = [256, 128, 64, 32]
DROPOUT = 0.25

# Training
EPOCHS = 500
BATCH_SIZE = 64
LR = 8e-4
PATIENCE = 30


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(F.gelu(self.linear(x)))


class ResidualMLP(nn.Module):
    """Residual MLP for binary classification."""

    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(ResidualBlock(h, dropout))
            prev = h
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(x)
        return self.out(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)

    print(f"device: {DEVICE}")
    print("loading data...")

    data = load_tabular_data(DATA_PATH, train_frac=0.8, seed=SEED, log1p=LOG1P_TRANSFORM)
    x_train = data.x_train.to(device)
    x_val = data.x_val.to(device)
    y_train = data.y_train.to(device).unsqueeze(1)
    y_val = data.y_val

    n_train = x_train.shape[0]
    n_pos = int(y_train.sum().item())
    n_neg = n_train - n_pos
    bce_pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)

    print(f"rows: train={n_train} val={x_val.shape[0]}")
    print(f"features used: {len(data.feature_names)}")
    print("target: obstructive_cad (binary)")
    print("excluded from features: population_risk, source")

    in_dim = x_train.shape[1]
    model = ResidualMLP(in_dim, HIDDEN, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            x_b = x_train[idx]
            y_b = y_train[idx]
            logits = model(x_b).unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y_b, pos_weight=bce_pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        metrics = evaluate(model, data.x_val, data.y_val, device, bce_pos_weight)
        val_loss = metrics["loss"]
        val_auc = metrics["auc"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch <= 5) or (epoch % 50 == 0) or (patience_counter == 0):
            print(f"  epoch {epoch}: val_loss={val_loss:.6f} val_auc={val_auc:.6f}")

        if patience_counter >= PATIENCE:
            print(f"early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print("---")
    print(f"best_epoch:       {best_epoch}")
    print(f"best_val_loss:    {best_val_loss:.6f}")
    print(f"best_val_auc:     {best_val_auc:.6f}")
    print(f"runtime_seconds:  {elapsed:.1f}")


if __name__ == "__main__":
    main()
