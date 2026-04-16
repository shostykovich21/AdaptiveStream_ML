"""
Train predictor model(s) on synthetic burst data.
Saves checkpoints to ../models/{name}_predictor.pt
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path

from models import MODELS
from data import generate_series

MODEL_DIR  = Path(__file__).parent.parent / "models"
K          = 30
N_SERIES   = 150
BATCH_SIZE = 256
LR         = 0.001

# time-based training — same wall-clock budget for every architecture
# (TCN/DLinear complete far more epochs than LSTM/GRU in equal time)
TIME_BUDGET     = 150
CHECKPOINT_SECS = [30, 60, 120, 150]

# early stopping — checked every ES_INTERVAL seconds, triggers after
# PATIENCE consecutive checks with no improvement > MIN_DELTA
ES_INTERVAL = 5.0
ES_PATIENCE = 5
ES_MIN_DELTA = 1e-4

# was: EPOCHS = 40 (fixed, unfair across architectures)


# original single-shape generator — replaced by generate_series() from data.py
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
    X, Y = [], []
    for j in range(k, len(values) - 1):
        window = values[j - k:j].astype(np.float32)
        mu  = window.mean()
        std = max(float(window.std()), mu * 0.05) + 1e-8
        X.append((window - mu) / std)
        Y.append((values[j] - mu) / std)
    return X, Y


def build_dataset(n_series, k, base_seed=42, train_ratio=0.7, val_ratio=0.15):
    # three-way series-level split: 70% train / 15% val / 15% test
    # val used for early stopping only — test never touched during training
    # was: 80/20 train/test with early stopping on test set (leakage)
    n_train = int(n_series * train_ratio)
    n_val   = int(n_series * val_ratio)

    all_X_tr, all_Y_tr = [], []
    all_X_va, all_Y_va = [], []
    all_X_te, all_Y_te = [], []

    for i in range(n_series):
        values, _ = generate_series(n=300, baseline=100, noise_std=10,
                                    seed=base_seed + i)
        # was: values = generate_burst_series()
        X_s, Y_s = windows_from_series(values, k)
        if i < n_train:
            all_X_tr.extend(X_s); all_Y_tr.extend(Y_s)
        elif i < n_train + n_val:
            all_X_va.extend(X_s); all_Y_va.extend(Y_s)
        else:
            all_X_te.extend(X_s); all_Y_te.extend(Y_s)

    def to_tensors(X, Y):
        return (torch.FloatTensor(np.array(X)).unsqueeze(-1),
                torch.FloatTensor(np.array(Y)).unsqueeze(-1))

    return (to_tensors(all_X_tr, all_Y_tr),
            to_tensors(all_X_va, all_Y_va),
            to_tensors(all_X_te, all_Y_te))


def evaluate_model(model, X, Y):
    model.eval()
    with torch.no_grad():
        preds  = model(X).squeeze(-1).numpy()
    actual = Y.squeeze(-1).numpy()
    mae     = float(np.mean(np.abs(preds - actual)))
    rmse    = float(np.sqrt(np.mean((preds - actual) ** 2)))
    # sign comparison in normalised space — was np.diff which compared
    # direction between consecutive predictions (wrong)
    dir_acc = float(np.mean(np.sign(preds) == np.sign(actual))) * 100
    model.train()
    return mae, rmse, dir_acc


def train_timed(model, X_tr, Y_tr, X_va, Y_va, X_te, Y_te):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Huber loss: quadratic for |error|<1, linear beyond — less blown up by
    # large normalised targets at shape transitions (wall spikes etc.)
    # was: nn.MSELoss()
    loss_fn   = nn.HuberLoss(delta=1.0)

    # checkpoints: {t_sec: (mae, rmse, dir_acc, early_stopped)}
    checkpoints = {}
    remaining   = list(CHECKPOINT_SECS)
    epoch       = 0
    t_start     = time.time()

    # early stopping state
    best_mae    = float("inf")
    best_state  = copy.deepcopy(model.state_dict())
    no_improve  = 0
    last_es_t   = t_start
    stopped_at  = None    # wall-seconds when ES triggered
    stopped_m   = None    # (mae, rmse, dir_acc) at best restored state

    while True:
        elapsed = time.time() - t_start

        # fill checkpoint slots that have passed
        while remaining and elapsed >= remaining[0]:
            t = remaining.pop(0)
            if stopped_at is not None:
                checkpoints[t] = (*stopped_m, True)
                print(f"    @{t:>3}s  [early stop @{stopped_at:.0f}s]  "
                      f"val_MAE={best_mae:.4f}  test_MAE={stopped_m[0]:.4f}  DirAcc={stopped_m[2]:.1f}%*")
            else:
                m_te = evaluate_model(model, X_te, Y_te)
                m_va = evaluate_model(model, X_va, Y_va)
                checkpoints[t] = (*m_te, False)
                print(f"    @{t:>3}s  epoch={epoch:>4}  "
                      f"val_MAE={m_va[0]:.4f}  test_MAE={m_te[0]:.4f}  DirAcc={m_te[2]:.1f}%")

        if elapsed >= TIME_BUDGET or stopped_at is not None:
            break

        # early stopping on val set — test set never seen during training
        if time.time() - last_es_t >= ES_INTERVAL:
            last_es_t = time.time()
            current_mae = evaluate_model(model, X_va, Y_va)[0]
            if current_mae < best_mae - ES_MIN_DELTA:
                best_mae   = current_mae
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= ES_PATIENCE:
                    model.load_state_dict(best_state)
                    stopped_at = time.time() - t_start
                    stopped_m  = evaluate_model(model, X_te, Y_te)
                    print(f"    early stop at {stopped_at:.1f}s  "
                          f"(best val_MAE={best_mae:.4f})")
                    # fill all remaining checkpoints immediately
                    for t in remaining:
                        checkpoints[t] = (*stopped_m, True)
                        print(f"    @{t:>3}s  [early stop @{stopped_at:.0f}s]  "
                              f"val_MAE={best_mae:.4f}  test_MAE={stopped_m[0]:.4f}  DirAcc={stopped_m[2]:.1f}%*")
                    remaining.clear()
                    break

        # one epoch
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), BATCH_SIZE):
            idx  = perm[i:i + BATCH_SIZE]
            loss = loss_fn(model(X_tr[idx]), Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch += 1

    # always restore best val weights — not just on early stop
    model.load_state_dict(best_state)
    return checkpoints, epoch


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Building dataset: {N_SERIES} series × 300 steps, K={K} ...")
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te) = build_dataset(N_SERIES, K)
    print(f"Train: {len(X_tr):,}  |  Val: {len(X_va):,}  |  Test: {len(X_te):,}")
    print(f"Budget: {TIME_BUDGET}s/model  |  ES patience: {ES_PATIENCE}×{ES_INTERVAL}s\n")

    all_checkpoints = {}   # {name: {t: (mae, rmse, dir_acc, early_stopped)}}

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
        print(f"{'─'*52}\n  {name.upper()}")
        model = ModelClass()
        checkpoints, epochs_done = train_timed(model, X_tr, Y_tr, X_va, Y_va, X_te, Y_te)
        all_checkpoints[name] = checkpoints

        save_path = MODEL_DIR / f"{name}_predictor.pt"
        torch.save(model.state_dict(), save_path)
        saved_val_mae, _, saved_val_dacc = evaluate_model(model, X_va, Y_va)
        print(f"  → {save_path.name}  epochs={epochs_done}  "
              f"saved val_MAE={saved_val_mae:.4f}  DirAcc={saved_val_dacc:.1f}%\n")

    # ── Comparison table ──────────────────────────────────────────────────────
    secs = CHECKPOINT_SECS
    col  = 16

    print(f"\n{'='*(12 + col * len(secs))}")
    print(f"{'Model':<12}" +
          "".join(f"@{s}s".center(col) for s in secs))
    print(f"{'─'*(12 + col * len(secs))}")

    for name in MODELS:
        if name not in all_checkpoints:
            continue
        row = f"{name:<12}"
        for s in secs:
            if s not in all_checkpoints[name]:
                row += "—".center(col)
                continue
            mae, _, da, es = all_checkpoints[name][s]
            star = "*" if es else " "
            row += f"{mae:.3f}/{da:.0f}%{star}".center(col)
        print(row)

    print(f"{'='*(12 + col * len(secs))}")
    print("* = early stopping active, weights restored to best epoch\n")


if __name__ == "__main__":
    main()
