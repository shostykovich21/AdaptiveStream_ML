"""
Synthetic rate series generator with diverse shape vocabulary.

Shapes (what the model needs to learn to recognise):
  noise        — jittery baseline, no event
  ramp         — gradual climb → hold → gradual drop  (original shape)
  wall         — sudden vertical spike, gradual recovery
  plateau      — sustained elevated traffic, low variance
  cliff        — elevated traffic that drops suddenly
  double_peak  — two bursts close together
  sawtooth     — repeated fast ramp → instant drop cycles

Each call to generate_series() returns (values, labels) where labels is a list
of (start_idx, end_idx, shape_name) so the evaluator knows which shape is active
at each timestep — enabling per-shape accuracy breakdown.
"""

import numpy as np

SHAPES = ["noise", "ramp", "wall", "plateau", "cliff", "double_peak", "sawtooth"]

# Probability of each shape per segment
_SHAPE_PROBS = [0.25, 0.15, 0.10, 0.10, 0.10, 0.10, 0.20]


def _segment(shape, baseline, noise_std, max_len):
    """Build one shape segment. Returns (array, shape_name)."""
    peak = baseline * np.random.uniform(2, 10)

    if shape == "noise":
        n = np.random.randint(10, 50)
        seg = baseline + np.random.normal(0, noise_std, n)

    elif shape == "ramp":
        n = np.random.randint(15, 45)
        third = max(1, n // 3)
        tail = n - 2 * third
        seg = np.concatenate([
            np.linspace(baseline, peak, third),
            np.full(third, peak),
            np.linspace(peak, baseline, tail),
        ])
        seg += np.random.normal(0, noise_std, len(seg))

    elif shape == "wall":
        n = np.random.randint(8, 22)
        hold = max(1, n // 3)
        seg = np.concatenate([
            np.full(hold, peak),
            np.linspace(peak, baseline, n - hold),
        ])
        seg += np.random.normal(0, noise_std, len(seg))

    elif shape == "plateau":
        n = np.random.randint(20, 60)
        level = baseline * np.random.uniform(2, 6)
        seg = np.full(n, level) + np.random.normal(0, noise_std * 0.3, n)

    elif shape == "cliff":
        n = np.random.randint(10, 25)
        drop_at = max(1, n // 3)
        seg = np.concatenate([
            np.full(drop_at, peak),
            np.linspace(peak, baseline, n - drop_at),
        ])
        seg += np.random.normal(0, noise_std, len(seg))

    elif shape == "double_peak":
        half = np.random.randint(12, 25)
        gap = np.random.randint(3, 8)
        peak2 = peak * np.random.uniform(0.5, 0.9)
        h1 = half // 2
        s1 = np.concatenate([np.linspace(baseline, peak, h1),
                              np.linspace(peak, baseline, half - h1)])
        h2 = half // 2
        s2 = np.concatenate([np.linspace(baseline, peak2, h2),
                              np.linspace(peak2, baseline, half - h2)])
        seg = np.concatenate([s1, np.full(gap, baseline), s2])
        seg += np.random.normal(0, noise_std, len(seg))

    elif shape == "sawtooth":
        teeth = np.random.randint(2, 5)
        tooth_len = np.random.randint(6, 15)
        tooth = np.linspace(baseline, peak, tooth_len)
        seg = np.tile(tooth, teeth).astype(float)
        seg += np.random.normal(0, noise_std, len(seg))

    seg = np.clip(seg, 1, None)
    return seg[:max_len], shape


def generate_series(n=300, baseline=100, noise_std=10, seed=None):
    """
    Generate a single rate time series with labelled shape segments.

    Returns:
      values : np.ndarray, shape (n,)  — rate per second
      labels : list of (start, end, shape_name) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    values = np.full(n, float(baseline))
    labels = []
    i = 0

    while i < n:
        shape = np.random.choice(SHAPES, p=_SHAPE_PROBS)
        seg, name = _segment(shape, baseline, noise_std, n - i)
        length = len(seg)
        values[i:i + length] = seg
        labels.append((i, i + length, name))
        i += length

    return values[:n], labels


def generate_series_with_lag(n=300, baseline=100, noise_std=10, seed=None,
                              capacity_ratio=1.2):
    """
    Generate a rate series and a simulated consumer lag alongside it.

    Lag model:
      capacity = baseline * capacity_ratio  (consumer throughput ceiling)
      lag(0)   = 0
      lag(t)   = max(0, lag(t-1) + rate(t) - capacity)

    capacity_ratio=1.2 means consumer handles 20% above baseline; any burst
    above that causes lag to grow. This ensures lag is non-zero during burst
    peaks so the model can learn it as a leading indicator.

    Returns:
      values   : np.ndarray shape (n,)  — rate per second
      lag      : np.ndarray shape (n,)  — consumer lag (accumulated events)
      labels   : list of (start, end, shape_name) tuples
    """
    values, labels = generate_series(n, baseline, noise_std, seed)
    capacity = baseline * capacity_ratio
    lag = np.zeros(n, dtype=np.float32)
    for t in range(1, n):
        lag[t] = max(0.0, lag[t - 1] + float(values[t]) - capacity)
    return values, lag, labels


def shape_at_each_step(labels, n):
    """Convert label list → per-timestep shape array (for the evaluator)."""
    out = ["noise"] * n
    for start, end, name in labels:
        for t in range(start, min(end, n)):
            out[t] = name
    return out
