"""
TabPFN-based tabular binary classification script.

Changes vs previous setup:
- replaces custom PyTorch residual MLP training loop with TabPFN
- uses hosted TabPFN client
- keeps the existing data loading path via prepare.load_tabular_data
- evaluates validation log loss and ROC AUC
- includes a placeholder for the TabPFN API key

Install:
    uv add tabpfn-client
or:
    pip install --upgrade tabpfn-client

Usage:
    uv run train.py

    Set TABPFN_API_KEY via:
    - .env file in project root (gitignored)
    - export TABPFN_API_KEY='your_key'
    - --api-key your_key
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from prepare import load_tabular_data

try:
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier
except ImportError as exc:
    raise ImportError(
        "tabpfn-client is required. Install it with `uv add tabpfn-client` "
        "or `pip install --upgrade tabpfn-client`."
    ) from exc


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_PATH = "2601_ground_truth_input_data_mapped.csv"
RANDOM_SEED = 42
LOG1P_TRANSFORM = True

# Placeholder API key. Prefer setting the environment variable instead:
#   export TABPFN_API_KEY='your_real_key_here'
TABPFN_API_KEY = "<YOUR_TABPFN_API_KEY_HERE>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TabPFN on obstructive_cad.")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--log1p", type=lambda x: x.lower() != "false", default=LOG1P_TRANSFORM)
    p.add_argument("--api-key", type=str, default=None, help="Optional TabPFN API key.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def to_numpy(x) -> np.ndarray:
    """Convert torch.Tensor or array-like to a NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = np.clip(y_prob.astype(np.float64), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC AUC for binary classification without sklearn.
    Uses rank statistics with average ranks for ties.
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n = y_true.shape[0]
    n_pos = int(y_true.sum())
    n_neg = int(n - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def resolve_api_key(cli_key: str | None) -> str:
    api_key = cli_key or os.getenv("TABPFN_API_KEY") or TABPFN_API_KEY
    if not api_key or api_key == "<YOUR_TABPFN_API_KEY_HERE>":
        raise ValueError(
            "No valid TabPFN API key provided. Set TABPFN_API_KEY in your environment, "
            "pass --api-key, or replace the TABPFN_API_KEY placeholder in the script."
        )
    return api_key


def main() -> None:
    args = parse_args()
    t0 = time.time()

    set_seed(args.seed)

    api_key = resolve_api_key(args.api_key)
    tabpfn_client.set_access_token(api_key)

    print("model: TabPFNClassifier (hosted API)")
    print("loading data...")

    data = load_tabular_data(DATA_PATH, train_frac=0.75, seed=args.seed, log1p=args.log1p)

    x_train = to_numpy(data.x_train).astype(np.float32)
    y_train = to_numpy(data.y_train).astype(np.int64)

    x_val = to_numpy(data.x_val).astype(np.float32)
    y_val = to_numpy(data.y_val).astype(np.int64)

    print(f"rows: train={x_train.shape[0]} val={x_val.shape[0]}")
    print(f"features used: {len(data.feature_names)}")
    print("target: obstructive_cad (binary)")
    print("excluded from features: population_risk, source")

    model = TabPFNClassifier()

    print("fitting TabPFN...")
    model.fit(x_train, y_train)

    print("running validation...")
    val_proba = model.predict_proba(x_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(np.int64)

    val_loss = binary_log_loss(y_val, val_proba)
    val_auc = roc_auc_score_binary(y_val, val_proba)
    val_acc = float((val_pred == y_val).mean())

    elapsed = time.time() - t0
    print("---")
    print(f"best_val_loss:    {val_loss:.6f}")
    print(f"best_val_auc:     {val_auc:.6f}")
    print(f"val_accuracy:     {val_acc:.6f}")
    print(f"runtime_seconds:  {elapsed:.1f}")


if __name__ == "__main__":
    main()