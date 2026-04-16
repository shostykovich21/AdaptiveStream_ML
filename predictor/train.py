"""
Train predictor model(s) on synthetic burst data.
Saves checkpoints to ../models/{name}_predictor.pt
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path

from models import MODELS
from data import generate_series

MODEL_DIR  = Path(__file__).parent.parent / "models"
K          = 30
N_SERIES   = 150   # was 100
EPOCHS     = 40    # was 30
BATCH_SIZE = 256
LR         = 0.001


# original single-shape generator — replaced by generate_series() from data.py
# which covers 7 shape types (wall, plateau, cliff, double_peak, sawtooth, ramp, noise)
#
# def generate_burst_series(n=300, baseline=100, burst_mult=8, burst_dur=30, noise=10):
#     series = []
#     i = 0
#     while i < n:
#         if np.random.random() < 0.15 and i + burst_dur < n:
#             third = burst_dur // 3
#             ramp_up   = np.linspace(baseline, baseline * burst_mult, third)
#             hold      = np.full(third, baseline * burst_mult)
#             ramp_down = np.linspace(baseline * burst_mult, baseline, third)
#             burst = np.concatenate([ramp_up, hold, ramp_down])
#             series.extend(burst + np.random.normal(0, noise, len(burst)))
#             i += len(burst)
#         else:
#             series.append(baseline + np.random.normal(0, noise))
#             i += 1
#     return np.array(series[:n])


def windows_from_series(values, k):
    """
    Build (X, Y) pairs from a single series using per-window normalisation —
    matching exactly what predictor_server.py does at inference time.

    Each window is normalised with its own mean/std (not the full-series stats),
    so no future values leak into any training sample.
    """
    X, Y = [], []
    for j in range(k, len(values) - 1):
        window = values[j - k:j].astype(np.float32)
        mu  = window.mean()
        std = window.std() + 1e-8
        X.append((window - mu) / std)
        Y.append((values[j] - mu) / std)   # target normalised with same window stats
    return X, Y


def build_dataset(n_series, k, base_seed=42, train_ratio=0.8):
    """
    Split at the series level so no series appears in both train and test.
    Previously the split was on the flattened window pool which would break
    if n_series didn't divide evenly at the 80% boundary.
    """
    n_train = int(n_series * train_ratio)

    all_X_tr, all_Y_tr = [], []
    all_X_te, all_Y_te = [], []

    for i in range(n_series):
        values, _ = generate_series(n=300, baseline=100, noise_std=10,
                                    seed=base_seed + i)
        # was: values = generate_burst_series()
        X_s, Y_s = windows_from_series(values, k)
        if i < n_train:
            all_X_tr.extend(X_s); all_Y_tr.extend(Y_s)
        else:
            all_X_te.extend(X_s); all_Y_te.extend(Y_s)

    def to_tensors(X, Y):
        return (torch.FloatTensor(np.array(X)).unsqueeze(-1),
                torch.FloatTensor(np.array(Y)).unsqueeze(-1))

    return to_tensors(all_X_tr, all_Y_tr), to_tensors(all_X_te, all_Y_te)


def train_model(model, X, Y):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()
    model.train()
    last_loss = 0.0

    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X))
        total, batches = 0.0, 0
        for i in range(0, len(X), BATCH_SIZE):
            idx  = perm[i:i + BATCH_SIZE]
            pred = model(X[idx])
            loss = loss_fn(pred, Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total  += loss.item()
            batches += 1
        last_loss = total / batches
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{EPOCHS}  loss={last_loss:.6f}")

    return last_loss


def evaluate_model(model, X, Y):
    model.eval()
    with torch.no_grad():
        preds  = model(X).squeeze(-1).numpy()
    actual = Y.squeeze(-1).numpy()
    mae    = float(np.mean(np.abs(preds - actual)))
    rmse   = float(np.sqrt(np.mean((preds - actual) ** 2)))
    dir_acc = float(np.mean(
        np.sign(np.diff(preds)) == np.sign(np.diff(actual))
    )) * 100
    return mae, rmse, dir_acc


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Building dataset: {N_SERIES} series × 300 steps, K={K} ...")
    (X_tr, Y_tr), (X_te, Y_te) = build_dataset(N_SERIES, K)
    print(f"Train: {len(X_tr):,}  |  Test: {len(X_te):,}\n")

    results = {}

    # original single-model block:
    # model     = LSTMPredictor()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_fn   = nn.MSELoss()
    # model.train()
    # for epoch in range(EPOCHS):
    #     perm = torch.randperm(len(X))
    #     total_loss, batches = 0, 0
    #     for i in range(0, len(X), BATCH_SIZE):
    #         idx = perm[i:i+BATCH_SIZE]
    #         pred = model(X[idx])
    #         loss = loss_fn(pred, Y[idx])
    #         optimizer.zero_grad(); loss.backward(); optimizer.step()
    #         total_loss += loss.item(); batches += 1
    #     if (epoch + 1) % 5 == 0:
    #         print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/batches:.6f}")
    # torch.save(model.state_dict(), MODEL_DIR / "lstm_predictor.pt")

    for name, ModelClass in MODELS.items():
        print(f"{'─'*50}\n  {name.upper()}")
        model   = ModelClass()
        t0      = time.time()
        train_model(model, X_tr, Y_tr)
        elapsed = time.time() - t0

        mae, rmse, dir_acc = evaluate_model(model, X_te, Y_te)
        results[name] = dict(mae=mae, rmse=rmse, dir_acc=dir_acc, secs=elapsed)

        save_path = MODEL_DIR / f"{name}_predictor.pt"
        torch.save(model.state_dict(), save_path)
        print(f"  → {save_path.name}  MAE={mae:.4f}  DirAcc={dir_acc:.1f}%  ({elapsed:.1f}s)\n")

    print(f"\n{'='*60}")
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'DirAcc':>9} {'Secs':>7}")
    print(f"{'─'*60}")
    best = min(results, key=lambda n: results[n]["mae"])
    for name, r in sorted(results.items(), key=lambda x: x[1]["mae"]):
        star = " ★" if name == best else ""
        print(f"{name:<12} {r['mae']:>8.4f} {r['rmse']:>8.4f}"
              f" {r['dir_acc']:>8.1f}% {r['secs']:>6.1f}s{star}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
