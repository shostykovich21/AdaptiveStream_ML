"""
evaluate_stream.py — Synthetic Holdout Evaluation
==================================================

WHAT THIS EVAL DOES
-------------------
Replays held-out synthetic series (seeds 169–191, never seen during training)
through the exact same inference path used in production:
  - K=30 sliding window deque  (mirrors metrics_collector.py)
  - Per-window normalisation at each step  (mirrors predictor_server.py)
  - One observation at a time, strictly no look-ahead

Each model gets the same normalised window at each timestep and predicts
the next inputRowsPerSecond. Error is measured in raw events/s.

WHAT THE RESULTS TELL US
------------------------
This is a closed-world generalisation test. It answers:

  "Can a model, having learned from synthetic burst shapes, correctly predict
   the next step of a burst shape it has never seen before?"

It does NOT answer whether real production Spark traffic looks like these shapes.

The gap between neural DirAcc (72–77%) and EMA DirAcc (39–44% in iter3) is real
and meaningful: neural models learn the abstract structure of bursts (ramp-up,
plateau, cliff) well enough to correctly call the direction of the next step ~77%
of the time. EMA cannot — it has no structural knowledge, so it calls direction
wrong more than right when traffic is actively transitioning through a burst shape.

WHY DirAcc IS THE RIGHT CROSS-ITERATION METRIC
-----------------------------------------------
MAE is in raw events/s and scales with baseline. Iter 3 holdout baselines reach
500k events/s, so raw MAE of 6,696 (iter3) vs 28 (iter2) is not a regression —
it just reflects the scale change. DirAcc is scale-invariant: it only asks
"did the model correctly predict up or down?" and is directly comparable across
all three iterations.

LIMITATIONS (why this alone is insufficient)
--------------------------------------------
1. Closed-world: holdout is from the same distribution as training (same shapes,
   different random seeds). We cannot infer from this that real production traffic
   follows these shapes.

2. Favourable to neural models: EMA never trained on burst shapes, but burst shapes
   are exactly what this test uses. EMA's structural weakness is directly exposed.
   This is fair to the task (if production has bursts, EMA is genuinely limited),
   but it is not a balanced comparison if production traffic is random-walk.

3. No real Spark infrastructure: inference is simulated, not measured live. Latency,
   JVM integration, and TCP round-trip are not accounted for.

WHAT THIS POINTS TOWARD
-----------------------
A third evaluation (evaluate_stream3.py) is needed that either:
  (a) replays a real inputRowsPerSecond trace from a production Spark cluster — no
      assumptions about distribution, the only truly unbiased test; or
  (b) feeds synthetic burst-shaped traffic into a live Spark job so the infrastructure
      is real and the traffic structure reflects production burst behaviour.

Entries in evaluation table
----------------------------
  Neural (9):    lstm, gru, tcn, dlinear, mlp, attn, tide, fits, nbeats
  Baselines (2): sma, ema  (no training, operate in raw rate space)
  Ensembles (5): ens_mean, ens_wtd, ens_rnn, ens_top3, ens_diverse

Usage
-----
    python evaluate_stream.py
    python evaluate_stream.py --lag
    python evaluate_stream.py --lag --log-uniform
    python evaluate_stream.py --model lstm tcn
"""

import argparse
import time
import numpy as np
import torch
from collections import deque
from pathlib import Path

from models import MODELS
from data import generate_series, generate_series_with_lag, shape_at_each_step
from data2 import generate_series2, generate_series_with_lag2, shape_at_each_step2
from config import K, MODEL_DIR, VAL_SEEDS, HOLDOUT_SEEDS


# ── Parametric baselines (no training, raw-space predictions) ─────────────────

class SMAPredictor:
    """Predicts the mean of the last K values — equivalent to simple moving average."""
    is_baseline = True

    def predict_raw(self, raw_window):
        return float(np.mean(raw_window))


class EMAPredictor:
    """Exponential moving average. Alpha tuned on val split before evaluation."""
    is_baseline = True

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def predict_raw(self, raw_window):
        ema = raw_window[0]
        for v in raw_window[1:]:
            ema = self.alpha * v + (1 - self.alpha) * ema
        return float(ema)


# ── Ensemble wrappers (computed on-the-fly, no .pt file needed) ───────────────

class EnsemblePredictor:
    """Weighted average of a subset of neural model predictions."""

    def __init__(self, models, weights=None):
        self.models  = models   # dict name → model
        self.weights = weights  # dict name → float (normalised), or None for equal

    def __call__(self, x):
        preds, ws = [], []
        for name, model in self.models.items():
            with torch.no_grad():
                preds.append(model(x))
            ws.append(1.0 if self.weights is None else self.weights.get(name, 1.0))
        stacked = torch.stack(preds, dim=0)                  # [n, B, 1]
        w = torch.tensor(ws, dtype=torch.float32).view(-1, 1, 1)
        w = w / w.sum()
        return (stacked * w).sum(0)                          # [B, 1]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(only=None, input_size=1):
    loaded = {}
    for name, ModelClass in MODELS.items():
        if only and name not in only:
            continue
        path = MODEL_DIR / f"{name}_predictor.pt"
        if not path.exists():
            print(f"  [skip] {name} — no checkpoint. Run train.py first.")
            continue
        m = ModelClass(input_size=input_size)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        loaded[name] = m
        print(f"  [ok]   {name}")
    return loaded


# ── Val-split utilities for EMA tuning and ensemble weighting ─────────────────

def tune_ema_alpha(alphas=(0.1, 0.2, 0.3, 0.5, 0.7)):
    """Grid-search EMA alpha on val split. Returns the best alpha."""
    best_alpha, best_mae = 0.3, float("inf")
    for alpha in alphas:
        errors = []
        for seed in VAL_SEEDS:
            values, _ = generate_series(n=300, baseline=100, noise_std=10, seed=seed)
            window = deque(maxlen=K)
            for t in range(len(values) - 1):
                window.append(float(values[t]))
                if len(window) < K:
                    continue
                raw = list(window)
                ema = raw[0]
                for v in raw[1:]:
                    ema = alpha * v + (1 - alpha) * ema
                errors.append(abs(ema - float(values[t + 1])))
        mae = float(np.mean(errors))
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha
    return best_alpha


def _build_tensor(rate_window, lag_window=None):
    """Build normalised input tensor from sliding window deques."""
    h   = np.array(rate_window, dtype=np.float32)
    mu  = h.mean()
    std = max(float(h.std()), mu * 0.05) + 1e-8
    norm_r = (h - mu) / std
    if lag_window is not None:
        lag_arr = np.array(lag_window, dtype=np.float32)
        lag_std = float(lag_arr.std())
        norm_l  = np.zeros_like(lag_arr) if lag_std < 1e-6 \
                  else (lag_arr - lag_arr.mean()) / (lag_std + 1e-8)
        feat = np.stack([norm_r, norm_l], axis=-1)      # [K, 2]
        x = torch.FloatTensor(feat).unsqueeze(0)        # [1, K, 2]
    else:
        x = torch.FloatTensor(norm_r).unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
    return x, mu, std


def compute_val_maes(neural_models, use_lag=False):
    """
    Run each neural model through the val split and return per-model MAE.
    Used to compute weights for ens_wtd and to pick the best RNN for ens_diverse.
    """
    errors = {n: [] for n in neural_models}
    for seed in VAL_SEEDS:
        if use_lag:
            values, lag_vals, _ = generate_series_with_lag(
                n=300, baseline=100, noise_std=10, seed=seed)
            lag_win = deque(maxlen=K)
        else:
            values, _ = generate_series(n=300, baseline=100, noise_std=10, seed=seed)
            lag_win = None
        window = deque(maxlen=K)
        for t in range(len(values) - 1):
            window.append(float(values[t]))
            if use_lag:
                lag_win.append(float(lag_vals[t]))
            if len(window) < K:
                continue
            x, mu, std = _build_tensor(window, lag_win if use_lag else None)
            for name, model in neural_models.items():
                with torch.no_grad():
                    pred = max(model(x).item() * std + mu, 0.0)
                errors[name].append(abs(pred - float(values[t + 1])))
    return {n: float(np.mean(v)) for n, v in errors.items()}


def build_extended_models(neural_models, val_maes, ema_alpha):
    """
    Returns the full evaluation dict:
      neural models + sma + ema + 4 ensemble variants.
    Baselines and ensembles are appended after neural models.
    """
    extended = dict(neural_models)

    # parametric baselines
    extended["sma"] = SMAPredictor()
    extended["ema"] = EMAPredictor(alpha=ema_alpha)

    # ens_mean: equal-weight average of all available neural models
    extended["ens_mean"] = EnsemblePredictor(neural_models)

    # ens_wtd: weighted by 1/val_MAE (better models get more weight)
    inv   = {n: 1.0 / val_maes[n] for n in neural_models if n in val_maes}
    total = sum(inv.values())
    extended["ens_wtd"] = EnsemblePredictor(
        neural_models, {n: w / total for n, w in inv.items()}
    )

    # ens_rnn: lstm + gru only (recurrent family)
    rnn = {n: neural_models[n] for n in ("lstm", "gru") if n in neural_models}
    if len(rnn) >= 2:
        extended["ens_rnn"] = EnsemblePredictor(rnn)

    # ens_top3: top-3 by val_MAE — lstm + gru + tide (best non-recurrent)
    # diverse families, all validated as strong; recommended for deployment
    top3_names = ("lstm", "gru", "tide")
    top3 = {n: neural_models[n] for n in top3_names if n in neural_models}
    if len(top3) >= 2:
        extended["ens_top3"] = EnsemblePredictor(top3)

    # ens_diverse: one from each family — best RNN + tcn + tide + attn
    # (dlinear excluded: confirmed weak on non-linear burst shapes)
    best_rnn = min(
        [n for n in ("lstm", "gru") if n in neural_models],
        key=lambda n: val_maes.get(n, float("inf")),
        default=None,
    )
    diverse_names = [best_rnn, "tcn", "tide", "attn"]
    diverse = {n: neural_models[n] for n in diverse_names
               if n is not None and n in neural_models}
    if len(diverse) >= 2:
        extended["ens_diverse"] = EnsemblePredictor(diverse)

    return extended


# ── Evaluator ─────────────────────────────────────────────────────────────────

class StreamingEvaluator:
    """
    Shared sliding window across all models — every model sees the same
    normalised history at each timestep.
    Baseline models (is_baseline=True) receive the raw window instead.
    """

    def __init__(self, models, k=K, use_lag=False):
        self.models   = models
        self.k        = k
        self.use_lag  = use_lag
        self.window   = deque(maxlen=k)
        self.lag_win  = deque(maxlen=k) if use_lag else None
        self._abs_errors  = {n: [] for n in models}
        self._dir_correct = {n: [] for n in models}
        self._shape_errs  = {n: {} for n in models}

    def reset(self):
        # clears only the sliding window — accumulators persist across series
        # so that print_results() reflects the full source, not just the last stream
        self.window.clear()
        if self.lag_win is not None:
            self.lag_win.clear()

    def step(self, rate, lag=None):
        self.window.append(float(rate))
        if self.use_lag and lag is not None:
            self.lag_win.append(float(lag))
        if len(self.window) < self.k:
            return None
        x, mu, std = _build_tensor(
            self.window, self.lag_win if self.use_lag else None)
        raw = list(self.window)
        preds = {}
        for name, model in self.models.items():
            if getattr(model, "is_baseline", False):
                preds[name] = max(model.predict_raw(raw), 0.0)
            else:
                with torch.no_grad():
                    preds[name] = max(model(x).item() * std + mu, 0.0)
        return preds

    def record(self, preds, actual_next, shape="unknown"):
        if preds is None:
            return
        current    = list(self.window)[-1]
        dir_actual = 1 if actual_next > current else -1
        for name, predicted in preds.items():
            self._abs_errors[name].append(abs(predicted - actual_next))
            self._dir_correct[name].append(
                int((1 if predicted > current else -1) == dir_actual)
            )
            self._shape_errs[name].setdefault(shape, []).append(
                abs(predicted - actual_next)
            )

    def rolling(self, window=50):
        out = {}
        for n in self.models:
            errs = self._abs_errors[n][-window:]
            dirs = self._dir_correct[n][-window:]
            out[n] = dict(
                mae     = float(np.mean(errs)) if errs else 0.0,
                dir_acc = float(np.mean(dirs)) * 100 if dirs else 0.0,
            )
        return out

    def print_results(self, label=""):
        names   = list(self.models.keys())
        all_mae = {
            n: float(np.mean(self._abs_errors[n])) if self._abs_errors[n] else float("inf")
            for n in names
        }
        best = min(all_mae, key=all_mae.get)

        if label:
            print(f"\n  [{label}]")
        print(f"  {'Model':<14} {'MAE':>8} {'RMSE':>8} {'DirAcc':>9} {'n':>8}")
        print(f"  {'─'*52}")
        for n in sorted(names, key=lambda x: all_mae[x]):
            errs = np.array(self._abs_errors[n])
            rmse = float(np.sqrt(np.mean(errs ** 2))) if len(errs) else 0.0
            dacc = float(np.mean(self._dir_correct[n])) * 100 if self._dir_correct[n] else 0.0
            star = " ★" if n == best else ""
            print(f"  {n:<14} {all_mae[n]:>8.2f} {rmse:>8.2f}"
                  f" {dacc:>8.1f}% {len(errs):>8,}{star}")

        # per-shape breakdown — only for the top 6 neural models (keeps table width manageable)
        neural_names = [n for n in sorted(names, key=lambda x: all_mae[x])
                        if not getattr(self.models[n], "is_baseline", False)
                        and not isinstance(self.models[n], EnsemblePredictor)][:6]
        all_shapes = sorted({s for n in neural_names for s in self._shape_errs[n]})
        if all_shapes and neural_names:
            col = 10
            print(f"\n  Per-shape MAE (top-6 neural):")
            print(f"  {'Shape':<14}" + "".join(f"{n:>{col}}" for n in neural_names))
            print(f"  {'─'*(14 + col * len(neural_names))}")
            for shape in all_shapes:
                row = f"  {shape:<14}"
                for n in neural_names:
                    errs = self._shape_errs[n].get(shape, [])
                    row += f"{np.mean(errs):>{col}.2f}" if errs else f"{'—':>{col}}"
                print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(only=None, use_lag=False, log_uniform=False):
    input_size = 2 if use_lag else 1
    features   = "rate+lag" if use_lag else "rate only"
    dataset    = "log-uniform (data2)" if log_uniform else "fixed-baseline (data)"
    print(f"Loading models  (features: {features}, dataset: {dataset}) ...")
    neural_models = load_models(only, input_size=input_size)
    if not neural_models:
        print("No models found. Run train.py first.")
        return

    # tune EMA and compute val MAEs for ensemble weighting
    print("\nTuning EMA alpha on val split...")
    ema_alpha = tune_ema_alpha()
    print(f"  best alpha = {ema_alpha}")

    print("Computing val MAEs for ensemble weights...")
    val_maes = compute_val_maes(neural_models, use_lag=use_lag)
    for n, mae in sorted(val_maes.items(), key=lambda x: x[1]):
        print(f"  {n:<12} val_MAE={mae:.4f}")

    # build full model dict: neural + baselines + ensembles
    models = build_extended_models(neural_models, val_maes, ema_alpha)
    n_ensembles = len(models) - len(neural_models) - 2
    print(f"\n  {len(neural_models)} neural  +  2 baselines  +  "
          f"{n_ensembles} ensembles  =  {len(models)} total entries\n")

    # ── Synthetic hold-out ────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print("  Synthetic hold-out  (seeds 169–191, not seen during training)")
    streams, lag_streams, shape_maps = [], [], []
    for seed in HOLDOUT_SEEDS:
        if log_uniform:
            if use_lag:
                values, lag_vals, _, labels = generate_series_with_lag2(n=600, seed=seed)
                lag_streams.append(lag_vals)
            else:
                values, _, labels = generate_series2(n=600, seed=seed)
                lag_streams.append(None)
            shape_maps.append(shape_at_each_step2(labels, len(values)))
        elif use_lag:
            values, lag_vals, labels = generate_series_with_lag(
                n=300, baseline=100, noise_std=10, seed=seed)
            lag_streams.append(lag_vals)
            shape_maps.append(shape_at_each_step(labels, len(values)))
        else:
            values, labels = generate_series(n=300, baseline=100, noise_std=10, seed=seed)
            lag_streams.append(None)
            shape_maps.append(shape_at_each_step(labels, len(values)))
        streams.append(values)

    ev = StreamingEvaluator(models, use_lag=use_lag)
    total = 0
    for values, lag_vals, shapes in zip(streams, lag_streams, shape_maps):
        ev.reset()
        for t in range(len(values) - 1):
            lag_t = lag_vals[t] if lag_vals is not None else None
            preds = ev.step(values[t], lag=lag_t)
            # shapes[t+1]: tag by the regime the target value belongs to,
            # not the current regime — gives "error when predicting into shape X"
            ev.record(preds, actual_next=values[t + 1], shape=shapes[t + 1])
            total += 1
    print(f"  {len(streams)} series  {total:,} steps")
    ev.print_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=None)
    parser.add_argument("--lag", action="store_true",
                        help="Evaluate lag-aware models (input_size=2)")
    parser.add_argument("--log-uniform", action="store_true",
                        help="Use log-uniform holdout data (data2.py)")
    args = parser.parse_args()
    main(only=set(args.model) if args.model else None,
         use_lag=args.lag, log_uniform=args.log_uniform)
