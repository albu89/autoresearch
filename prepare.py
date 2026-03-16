"""
Evaluation and data loading for autoresearch (tabular binary classification).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC using Mann-Whitney U statistic (no sklearn)."""
    order = np.argsort(y_score)  # ascending: lowest score gets rank 1
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.empty(len(y_true), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_true) + 1)
    sum_rank_pos = np.sum(ranks[y_true == 1])
    u = sum_rank_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TABULAR_CACHE_DIR = os.path.join(CACHE_DIR, "tabular")
BLOCKED_FEATURES = {"population_risk", "source"}


@dataclass
class TabularData:
    x_train: torch.Tensor
    x_val: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    feature_names: list[str]


def _split_train_val(n_rows: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    n_train = int(n_rows * train_frac)
    return idx[:n_train], idx[n_train:]


def _load_and_preprocess(path: str, train_frac: float, seed: int, log1p: bool) -> TabularData:
    """Load CSV, preprocess, split; does not use cache."""
    df = pd.read_csv(path)
    if "obstructive_cad" not in df.columns:
        raise ValueError("CSV is missing required target column: obstructive_cad")

    y = pd.to_numeric(df["obstructive_cad"], errors="coerce")
    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(np.float32)

    drop_cols = {"obstructive_cad", "patient_id"} | BLOCKED_FEATURES
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    for blocked in BLOCKED_FEATURES:
        if blocked in feature_df.columns:
            raise RuntimeError(f"{blocked} leaked into feature matrix.")

    num_df = feature_df.select_dtypes(include=[np.number]).copy()
    cat_df = feature_df.select_dtypes(exclude=[np.number]).copy()

    if num_df.shape[1] > 0:
        num_df = num_df.fillna(num_df.median(numeric_only=True))
        if log1p:
            for col in num_df.columns:
                if num_df[col].min() >= 0:
                    num_df[col] = np.log1p(num_df[col])
    if cat_df.shape[1] > 0:
        cat_df = cat_df.fillna("__missing__").astype(str)
        cat_df = pd.get_dummies(cat_df, dummy_na=False)

    x_df = pd.concat([num_df, cat_df], axis=1)
    if x_df.shape[1] == 0:
        raise ValueError("No usable training features after preprocessing.")

    x = x_df.to_numpy(dtype=np.float32)
    y_np = y.to_numpy(dtype=np.float32)

    train_idx, val_idx = _split_train_val(len(x), train_frac=train_frac, seed=seed)
    x_train = x[train_idx]
    x_val = x[val_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    return TabularData(
        x_train=torch.from_numpy(x_train),
        x_val=torch.from_numpy(x_val),
        y_train=torch.from_numpy(y_np[train_idx]),
        y_val=torch.from_numpy(y_np[val_idx]),
        feature_names=x_df.columns.tolist(),
    )


def load_tabular_data(path: str, train_frac: float, seed: int, log1p: bool) -> TabularData:
    """
    Load tabular dataset. Uses cache at ~/.cache/autoresearch/tabular/ if available.
    Cache key: basename of path + train_frac + seed + log1p.
    """
    os.makedirs(TABULAR_CACHE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    cache_name = f"{base}_tf{train_frac}_s{seed}_log1p{log1p}.pt"
    cache_path = os.path.join(TABULAR_CACHE_DIR, cache_name)

    if os.path.exists(cache_path):
        state = torch.load(cache_path, map_location="cpu", weights_only=False)
        return TabularData(**state)

    data = _load_and_preprocess(path, train_frac=train_frac, seed=seed, log1p=log1p)
    torch.save(
        {
            "x_train": data.x_train,
            "x_val": data.x_val,
            "y_train": data.y_train,
            "y_val": data.y_val,
            "feature_names": data.feature_names,
        },
        cache_path,
    )
    return data


@torch.no_grad()
def evaluate(model, x, y, device, bce_pos_weight):
    """
    Evaluate binary classifier: BCE loss, accuracy, ROC AUC.
    """
    model.eval()
    x_d = x.to(device)
    y_d = y.to(device)
    logits = model(x_d)
    loss = F.binary_cross_entropy_with_logits(logits, y_d, pos_weight=bce_pos_weight)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_np = y.numpy()
    acc = float(((probs >= 0.5).astype(np.float32) == y_np).mean())
    auc = float(roc_auc_score(y_np, probs))
    return {"loss": float(loss.item()), "acc": acc, "auc": auc}
