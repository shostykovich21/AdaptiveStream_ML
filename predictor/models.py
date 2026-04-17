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
    Temporal Convolutional Network with causal dilated convolutions.
    Causal (left-only) padding ensures output[t] depends only on input[0..t],
    so the last output has full access to the entire 30-step window.
    With kernel_size=3 and dilations [1,2,4,8]: RF = 1+2+4+8+16 = 31 ≥ 30. ✓

    Symmetric padding (the previous approach) gave a receptive field of only
    ~16 steps at the last position because right-side zeros displaced left-side reach.
    """
    def __init__(self, seq_len=30, channels=32, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = 1
        for dilation in [1, 2, 4, 8]:
            left_pad = dilation * (kernel_size - 1)   # causal: pad left only
            layers += [
                nn.ConstantPad1d((left_pad, 0), 0),
                nn.Conv1d(in_ch, channels, kernel_size, dilation=dilation, padding=0),
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
        return x_padded.unfold(1, self.kernel, 1).mean(-1)  # [B, seq, 1]

    def forward(self, x):
        trend = self._moving_avg(x)                         # [B, seq, 1]
        resid = x - trend                                   # [B, seq, 1]
        trend_out = self.trend_linear(trend.squeeze(-1))    # [B, 1]
        resid_out = self.resid_linear(resid.squeeze(-1))    # [B, 1]
        return trend_out + resid_out


class MLPPredictor(nn.Module):
    """
    Flatten window → dense layers, no temporal structure.
    Ablation: if MLP matches RNNs/TCN, sequence modelling adds no value for this window size.
    """
    def __init__(self, seq_len=30, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x.squeeze(-1))  # [B, seq] → [B, 1]


class AttnPredictor(nn.Module):
    """
    Lite self-attention over the 30-step window with learned positional embeddings.
    Without positional encoding, self-attention is permutation-invariant in interior
    tokens — verified empirically that permuting interior tokens leaves output unchanged.
    Positional embedding fixes this and makes the model properly order-aware.
    Ablation: tests whether attended global context helps beyond TCN's local receptive field.
    """
    def __init__(self, seq_len=30, d_model=32, n_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb    = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.fc   = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        h = self.input_proj(x) + self.pos_emb(pos)           # [B, T, d_model]
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out)
        return self.fc(h[:, -1, :])                           # last timestep → [B, 1]


class TiDEPredictor(nn.Module):
    """
    Time-series Dense Encoder (TiDE, 2023).
    MLP encoder-decoder with a linear residual skip from input to output.
    No recurrence or convolution — pure dense layers, very fast inference.
    """
    def __init__(self, seq_len=30, hidden=64, enc_depth=2, dec_depth=1, dropout=0.2):
        super().__init__()
        enc = []
        in_dim = seq_len
        for _ in range(enc_depth):
            enc += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        self.encoder = nn.Sequential(*enc)

        dec = []
        for _ in range(dec_depth):
            dec += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        dec.append(nn.Linear(hidden, 1))
        self.decoder = nn.Sequential(*dec)

        self.residual = nn.Linear(seq_len, 1)  # direct skip connection

    def forward(self, x):
        flat = x.squeeze(-1)                                   # [B, seq]
        return self.decoder(self.encoder(flat)) + self.residual(flat)


class FITSPredictor(nn.Module):
    """
    Frequency Interpolation Time Series (FITS, ICLR 2024) — lite version.
    Low-pass filters the input in the FFT domain (keeps cut_freq components),
    applies a learned complex-valued projection, then maps to a real prediction.
    ~10k parameters; orthogonal approach to all other models here.
    """
    def __init__(self, seq_len=30, cut_freq=10):
        super().__init__()
        self.cut_freq  = min(cut_freq, seq_len // 2 + 1)
        self.freq_proj = nn.Linear(self.cut_freq, self.cut_freq, dtype=torch.cfloat)
        self.fc        = nn.Linear(self.cut_freq * 2, 1)

    def forward(self, x):
        # x: [B, seq, 1]
        xf = torch.fft.rfft(x.squeeze(-1), dim=-1)[:, :self.cut_freq]  # [B, cut_freq] complex
        xf = self.freq_proj(xf)                                          # [B, cut_freq] complex
        out = torch.cat([xf.real, xf.imag], dim=-1)                     # [B, 2*cut_freq]
        return self.fc(out)                                              # [B, 1]


class _NBEATSBlock(nn.Module):
    """Single N-BEATS block: FC stack → backcast (input reconstruction) + forecast."""
    def __init__(self, seq_len, hidden):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden, seq_len)
        self.forecast = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.backcast(h), self.forecast(h)


class NBEATSPredictor(nn.Module):
    """
    N-BEATS lite (Neural Basis Expansion Analysis, 2019).
    Stack of FC blocks with backward/forward residual links.
    Each block reconstructs its input (backcast subtracted) and adds a forecast.
    Ablation: interpretable basis expansion vs. black-box recurrent/conv models.
    """
    def __init__(self, seq_len=30, hidden=64, n_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            _NBEATSBlock(seq_len, hidden) for _ in range(n_blocks)
        ])

    def forward(self, x):
        residual = x.squeeze(-1)                              # [B, seq]
        forecast = torch.zeros(residual.shape[0], 1, device=x.device)
        for block in self.blocks:
            backcast, f = block(residual)
            residual = residual - backcast
            forecast = forecast + f
        return forecast                                       # [B, 1]


# Registry — used by train.py and evaluate_stream.py
MODELS = {
    "lstm":    LSTMPredictor,
    "gru":     GRUPredictor,
    "tcn":     TCNPredictor,
    "dlinear": DLinear,
    "mlp":     MLPPredictor,
    "attn":    AttnPredictor,
    "tide":    TiDEPredictor,
    "fits":    FITSPredictor,
    "nbeats":  NBEATSPredictor,
}
