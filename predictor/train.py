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
from data import generate_series, generate_series_with_lag
from data2 import generate_series2, generate_series_with_lag2
from config import K, MODEL_DIR, N_SERIES, BASE_SEED, TRAIN_RATIO, VAL_RATIO

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
    for j in range(k, len(values)):
        window = values[j - k:j].astype(np.float32)
        mu  = window.mean()
        std = max(float(window.std()), mu * 0.05) + 1e-8
        X.append((window - mu) / std)
        Y.append((values[j] - mu) / std)
    return X, Y


def windows_from_series_with_lag(values, lag, k):
    """Build [K, 2] windows: (normalised_rate, normalised_lag) pairs."""
    X, Y = [], []
    for j in range(k, len(values)):
        rate_win = values[j - k:j].astype(np.float32)
        lag_win  = lag[j - k:j].astype(np.float32)

        mu_r  = rate_win.mean()
        std_r = max(float(rate_win.std()), mu_r * 0.05) + 1e-8
        norm_r = (rate_win - mu_r) / std_r

        lag_std = float(lag_win.std())
        norm_l  = np.zeros_like(lag_win) if lag_std < 1e-6 \
                  else (lag_win - lag_win.mean()) / (lag_std + 1e-8)

        X.append(np.stack([norm_r, norm_l], axis=-1))  # [K, 2]
        Y.append((values[j] - mu_r) / std_r)
    return X, Y


def build_dataset(n_series=N_SERIES, k=K, base_seed=BASE_SEED,
                  train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                  use_lag=False, log_uniform=False):
    n_train = int(n_series * train_ratio)
    n_val   = int(n_series * val_ratio)

    all_X_tr, all_Y_tr = [], []
    all_X_va, all_Y_va = [], []
    all_X_te, all_Y_te = [], []

    for i in range(n_series):
        if log_uniform:
            if use_lag:
                values, lag, _, _ = generate_series_with_lag2(
                    n=600, seed=base_seed + i)
                X_s, Y_s = windows_from_series_with_lag(values, lag, k)
            else:
                values, _, _ = generate_series2(n=600, seed=base_seed + i)
                X_s, Y_s = windows_from_series(values, k)
        elif use_lag:
            values, lag, _ = generate_series_with_lag(
                n=300, baseline=100, noise_std=10, seed=base_seed + i)
            X_s, Y_s = windows_from_series_with_lag(values, lag, k)
        else:
            values, _ = generate_series(n=300, baseline=100, noise_std=10,
                                        seed=base_seed + i)
            X_s, Y_s = windows_from_series(values, k)

        if i < n_train:
            all_X_tr.extend(X_s); all_Y_tr.extend(Y_s)
        elif i < n_train + n_val:
            all_X_va.extend(X_s); all_Y_va.extend(Y_s)
        else:
            all_X_te.extend(X_s); all_Y_te.extend(Y_s)

    def to_tensors(X, Y):
        X_arr = np.array(X)
        X_t   = torch.FloatTensor(X_arr)
        if X_t.dim() == 2:          # [N, K] — no lag, add feature dim
            X_t = X_t.unsqueeze(-1)
        # [N, K, features] — correct for both 1-feature and 2-feature cases
        return X_t, torch.FloatTensor(np.array(Y)).unsqueeze(-1)

    return (to_tensors(all_X_tr, all_Y_tr),
            to_tensors(all_X_va, all_Y_va),
            to_tensors(all_X_te, all_Y_te))


def evaluate_model(model, X, Y):
    model.eval()
    with torch.no_grad():
        preds  = model(X).squeeze(-1).numpy()
    actual = Y.squeeze(-1).numpy()
    mae    = float(np.mean(np.abs(preds - actual)))
    rmse   = float(np.sqrt(np.mean((preds - actual) ** 2)))
    # DirAcc: does the model predict up/down vs the last window value?
    # X[:,-1,0] is the last normalised value in the window (equivalent to "current")
    # matches streaming eval which compares predicted > current vs actual_next > current
    last    = X[:, -1, 0].numpy()
    dir_acc = float(np.mean(np.sign(preds - last) == np.sign(actual - last))) * 100
    model.train()
    return mae, rmse, dir_acc


def train_timed(model, X_tr, Y_tr, X_va, Y_va):
    # test set is never passed in — fully sealed until evaluate_stream.py
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Huber loss: quadratic for |error|<1, linear beyond — less blown up by
    # large normalised targets at shape transitions (wall spikes etc.)
    # was: nn.MSELoss()
    loss_fn   = nn.HuberLoss(delta=1.0)

    # checkpoints: {t_sec: (mae, rmse, dir_acc, early_stopped)} — all val-set
    checkpoints = {}
    remaining   = list(CHECKPOINT_SECS)
    epoch       = 0
    t_start     = time.time()

    # early stopping state
    best_mae    = float("inf")
    best_state  = copy.deepcopy(model.state_dict())
    no_improve  = 0
    last_es_t   = t_start
    stopped_at  = None
    stopped_m   = None   # val metrics at best restored state

    while True:
        elapsed = time.time() - t_start

        # fill checkpoint slots that have passed
        while remaining and elapsed >= remaining[0]:
            t = remaining.pop(0)
            if stopped_at is not None:
                checkpoints[t] = (*stopped_m, True)
                print(f"    @{t:>3}s  [early stop @{stopped_at:.0f}s]  "
                      f"val_MAE={stopped_m[0]:.4f}  DirAcc={stopped_m[2]:.1f}%*")
            else:
                m_va = evaluate_model(model, X_va, Y_va)
                checkpoints[t] = (*m_va, False)
                print(f"    @{t:>3}s  epoch={epoch:>4}  "
                      f"val_MAE={m_va[0]:.4f}  DirAcc={m_va[2]:.1f}%")

        if elapsed >= TIME_BUDGET or stopped_at is not None:
            break

        # early stopping on val set
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
                    stopped_m  = evaluate_model(model, X_va, Y_va)
                    print(f"    early stop at {stopped_at:.1f}s  "
                          f"(best val_MAE={best_mae:.4f})")
                    for t in remaining:
                        checkpoints[t] = (*stopped_m, True)
                        print(f"    @{t:>3}s  [early stop @{stopped_at:.0f}s]  "
                              f"val_MAE={stopped_m[0]:.4f}  DirAcc={stopped_m[2]:.1f}%*")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag", action="store_true",
                        help="Add Kafka lag as a second input feature [K,2]")
    parser.add_argument("--log-uniform", action="store_true",
                        help="Use log-uniform baseline dataset (data2.py): "
                             "500 series×600 steps, baseline∈[10,500k]")
    parser.add_argument("--model-dir", default=None,
                        help="Directory to save model checkpoints "
                             "(default: models/ at project root)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    input_size = 2 if args.lag else 1
    features   = "rate+lag" if args.lag else "rate only"
    n_series   = 500 if args.log_uniform else N_SERIES
    steps      = 600 if args.log_uniform else 300
    dataset_tag = "log-uniform" if args.log_uniform else "fixed-baseline"
    print(f"Building dataset: {n_series} series × {steps} steps, K={K}, "
          f"features={features}, scale={dataset_tag} ...")
    (X_tr, Y_tr), (X_va, Y_va), _ = build_dataset(
        n_series=n_series, use_lag=args.lag, log_uniform=args.log_uniform)
    print(f"Train: {len(X_tr):,}  |  Val: {len(X_va):,}  |  "
          f"input shape: {tuple(X_tr.shape[1:])}")
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
        model = ModelClass(input_size=input_size)
        checkpoints, epochs_done = train_timed(model, X_tr, Y_tr, X_va, Y_va)
        all_checkpoints[name] = checkpoints

        save_path = model_dir / f"{name}_predictor.pt"
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
    print("  (val_MAE/val_DirAcc — test reserved for evaluate_stream.py)")
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
