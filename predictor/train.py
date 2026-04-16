"""
Train LSTM predictor on synthetic burst data.
Saves model to models/lstm_predictor.pt
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, "/home/aayushvbarhate/adaptivestream/predictor")
from predictor_server import LSTMPredictor

def generate_burst_series(n=300, baseline=100, burst_mult=8, burst_dur=30, noise=10):
    series = []
    i = 0
    while i < n:
        if np.random.random() < 0.15 and i + burst_dur < n:
            third = burst_dur // 3
            ramp_up = np.linspace(baseline, baseline * burst_mult, third)
            hold = np.full(third, baseline * burst_mult)
            ramp_down = np.linspace(baseline * burst_mult, baseline, third)
            burst = np.concatenate([ramp_up, hold, ramp_down])
            series.extend(burst + np.random.normal(0, noise, len(burst)))
            i += len(burst)
        else:
            series.append(baseline + np.random.normal(0, noise))
            i += 1
    return np.array(series[:n])

def main():
    K = 30
    N_SERIES = 100
    EPOCHS = 30
    BATCH_SIZE = 256

    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Generating {N_SERIES} training series...")
    train_series = [generate_burst_series() for _ in range(N_SERIES)]

    # Build training data (normalized per-series)
    all_X, all_Y = [], []
    for series in train_series:
        mu, std = series.mean(), series.std() + 1e-8
        normed = (series - mu) / std
        for i in range(K, len(normed) - 1):
            all_X.append(normed[i-K:i])
            all_Y.append(normed[i])

    X = torch.FloatTensor(np.array(all_X)).unsqueeze(-1)
    Y = torch.FloatTensor(np.array(all_Y)).unsqueeze(-1)
    print(f"Training samples: {len(X)}")

    model = LSTMPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X))
        total_loss, batches = 0, 0
        for i in range(0, len(X), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            pred = model(X[idx])
            loss = loss_fn(pred, Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/batches:.6f}")

    save_path = "/home/aayushvbarhate/adaptivestream/models/lstm_predictor.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
