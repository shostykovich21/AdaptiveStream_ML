"""
data2.py — Log-uniform synthetic dataset generator.

Key differences from data.py (fixed baseline=100):
  - baseline = 10 ** Uniform(1.0, 5.7)  → equal representation per decade:
      10–100      ~25%   (IoT trickle)
      100–1k      ~25%   (moderate stream)
      1k–10k      ~25%   (high-throughput)
      10k–500k    ~25%   (hyper-scale)
  - noise_std = baseline * 0.1  — noise scales with signal, never dominates
  - plateau removed — adds nothing at burst detection; redistributed to wall/cliff
  - n=600 steps, 500 series — 4× training data
  - Same 6 remaining shapes: noise, ramp, wall, cliff, double_peak, sawtooth

The std floor in the per-window normaliser (mu * 0.05) still applies at
inference, so the model handles silent periods without divide-by-zero.
Lag generation (generate_series_with_lag2) uses the same capacity_ratio=1.2
model as data.py so the lag feature stays consistent across iterations.
"""

import numpy as np

SHAPES = ["noise", "ramp", "wall", "cliff", "double_peak", "sawtooth"]

# Plateau removed: probability mass redistributed to wall and cliff
_SHAPE_PROBS = [0.25, 0.15, 0.20, 0.15, 0.10, 0.15]


def _segment2(shape, baseline, noise_std, max_len):
    """Same shape vocabulary as data.py minus plateau."""
    peak = baseline * np.random.uniform(2, 10)

    if shape == "noise":
        n = np.random.randint(10, 60)
        seg = baseline + np.random.normal(0, noise_std, n)

    elif shape == "ramp":
        n = np.random.randint(15, 60)
        third = max(1, n // 3)
        tail  = n - 2 * third
        seg   = np.concatenate([
            np.linspace(baseline, peak, third),
            np.full(third, peak),
            np.linspace(peak, baseline, tail),
        ])
        seg += np.random.normal(0, noise_std, len(seg))

    elif shape == "wall":
        n    = np.random.randint(8, 25)
        hold = max(1, n // 3)
        seg  = np.concatenate([
            np.full(hold, peak),
            np.linspace(peak, baseline, n - hold),
        ])
        seg += np.random.normal(0, noise_std, len(seg))

    elif shape == "cliff":
        n       = np.random.randint(10, 30)
        drop_at = max(1, n // 3)
        seg     = np.concatenate([
            np.full(drop_at, peak),
            np.linspace(peak, baseline, n - drop_at),
        ])
        seg += np.random.normal(0, noise_std, len(seg))

    elif shape == "double_peak":
        half  = np.random.randint(12, 30)
        gap   = np.random.randint(3, 8)
        peak2 = peak * np.random.uniform(0.5, 0.9)
        h1    = half // 2
        s1    = np.concatenate([np.linspace(baseline, peak, h1),
                                 np.linspace(peak, baseline, half - h1)])
        h2    = half // 2
        s2    = np.concatenate([np.linspace(baseline, peak2, h2),
                                 np.linspace(peak2, baseline, half - h2)])
        seg   = np.concatenate([s1, np.full(gap, baseline), s2])
        seg  += np.random.normal(0, noise_std, len(seg))

    elif shape == "sawtooth":
        teeth    = np.random.randint(2, 5)
        tooth_len = np.random.randint(6, 18)
        tooth    = np.linspace(baseline, peak, tooth_len)
        seg      = np.tile(tooth, teeth).astype(float)
        seg     += np.random.normal(0, noise_std, len(seg))

    seg = np.clip(seg, 1, None)
    return seg[:max_len], shape


def generate_series2(n=600, seed=None):
    """
    Generate a single rate series with log-uniform scale.

    The baseline is drawn fresh each call from log-uniform [10, 500k], so
    every series has its own scale — the per-window normaliser handles it.

    Returns:
      values  : np.ndarray shape (n,)  — rate per second
      baseline: float                  — the series' baseline scale
      labels  : list of (start, end, shape_name) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    # log-uniform baseline: equal probability mass per decade
    baseline  = float(10 ** np.random.uniform(1.0, 5.7))
    noise_std = baseline * 0.1

    values = np.full(n, baseline)
    labels = []
    i = 0
    while i < n:
        shape = np.random.choice(SHAPES, p=_SHAPE_PROBS)
        seg, name = _segment2(shape, baseline, noise_std, n - i)
        length = len(seg)
        values[i:i + length] = seg
        labels.append((i, i + length, name))
        i += length

    return values[:n], baseline, labels


def generate_series_with_lag2(n=600, seed=None, capacity_ratio=1.2):
    """
    Log-uniform series + simulated consumer lag.

    Same lag model as data.py:
      capacity = baseline * capacity_ratio
      lag[t]   = max(0, lag[t-1] + rate[t] - capacity)

    Returns:
      values  : np.ndarray shape (n,)
      lag     : np.ndarray shape (n,)
      baseline: float
      labels  : list of (start, end, shape_name) tuples
    """
    values, baseline, labels = generate_series2(n=n, seed=seed)
    capacity = baseline * capacity_ratio
    lag = np.zeros(n, dtype=np.float32)
    for t in range(1, n):
        lag[t] = max(0.0, lag[t - 1] + float(values[t]) - capacity)
    return values, lag, baseline, labels


def shape_at_each_step2(labels, n):
    """Convert label list → per-timestep shape array."""
    out = ["noise"] * n
    for start, end, name in labels:
        for t in range(start, min(end, n)):
            out[t] = name
    return out
