"""
All predictor model architectures.

Interface contract (all models share this):
  Input:  [batch, seq_len, 1]  — normalized rate history
  Output: [batch, 1]           — predicted next normalized rate
"""

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUPredictor(nn.Module):
    """
    GRU — simpler than LSTM (no cell state), trains faster, often same accuracy.
    Good ablation: if GRU matches LSTM, the cell state adds no value here.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network with dilated convolutions.
    Fully parallel (no sequential hidden state) — trains faster than RNNs.
    Dilations [1, 2, 4, 8] give receptive field of ~30 steps, covering the full window.
    """
    def __init__(self, seq_len=30, channels=32, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = 1
        for dilation in [1, 2, 4, 8]:
            padding = dilation * (kernel_size - 1) // 2  # keeps sequence length constant
            layers += [
                nn.Conv1d(in_ch, channels, kernel_size,
                          dilation=dilation, padding=padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = channels
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)         # [B, 1, seq] — Conv1d expects channels first
        out = self.net(x)              # [B, channels, seq]
        return self.fc(out[:, :, -1])  # last timestep → [B, 1]


class DLinear(nn.Module):
    """
    Decomposition Linear (from 'Are Transformers Effective for Time Series?' 2022).
    Splits input into trend + residual, applies separate linear to each.
    Surprisingly competitive with deep models on many benchmarks.
    Key experiment: if DLinear matches LSTM, the sequence modelling is overkill.
    """
    def __init__(self, seq_len=30, kernel=5):
        super().__init__()
        self.kernel = kernel
        self.trend_linear = nn.Linear(seq_len, 1)
        self.resid_linear = nn.Linear(seq_len, 1)

    def _moving_avg(self, x):
        # x: [B, seq, 1]
        pad = self.kernel // 2
        x_padded = torch.cat([x[:, :pad, :], x, x[:, -pad:, :]], dim=1)
        # unfold(dim=1, size=kernel, step=1) → [B, seq, 1, kernel]
        return x_padded.unfold(1, self.kernel, 1).mean(-1)  # [B, seq, 1]

    def forward(self, x):
        trend = self._moving_avg(x)                         # [B, seq, 1]
        resid = x - trend                                   # [B, seq, 1]
        trend_out = self.trend_linear(trend.squeeze(-1))    # [B, 1]
        resid_out = self.resid_linear(resid.squeeze(-1))    # [B, 1]
        return trend_out + resid_out


# Registry — used by train_all.py and evaluate_stream.py
MODELS = {
    "lstm":    LSTMPredictor,
    "gru":     GRUPredictor,
    "tcn":     TCNPredictor,
    "dlinear": DLinear,
}
