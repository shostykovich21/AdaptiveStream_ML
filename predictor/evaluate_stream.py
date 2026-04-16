"""
Streaming evaluator — measures model accuracy in simulated live conditions.

Mimics exactly what happens in production:
  - Maintains K=30 sliding window (same as metrics_collector.py deque)
  - Feeds one rate observation at a time
  - At each step: normalize window → predict → denormalize → compare with actual
  - No Kafka, no Spark needed

Reports:
  - Rolling MAE + directional accuracy (printed every N steps)
  - Per-shape MAE breakdown (which shapes cause errors?)
  - Final comparison table across all loaded models

Usage:
    python evaluate_stream.py                   # default: 20 test series
    python evaluate_stream.py --series 50       # more test series
    python evaluate_stream.py --model lstm      # single model only
"""

import argparse
import torch
import numpy as np
from collections import deque
from pathlib import Path

from models import MODELS
from data import generate_series, shape_at_each_step

K         = 30
MODEL_DIR = Path(__file__).parent.parent / "models"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(only=None):
    loaded = {}
    for name, ModelClass in MODELS.items():
        if only and name not in only:
            continue
        path = MODEL_DIR / f"{name}_predictor.pt"
        if not path.exists():
            print(f"  [skip] {name} — no checkpoint at {path}. Run train_all.py first.")
            continue
        m = ModelClass()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        loaded[name] = m
        print(f"  [ok]   {name}")
    return loaded


# ── Evaluator ─────────────────────────────────────────────────────────────────

class StreamingEvaluator:
    """
    One sliding window per evaluator instance, shared across all models.
    At each timestep every model sees the same normalised history.
    """

    def __init__(self, models, k=K):
        self.models = models
        self.k      = k
        self.window = deque(maxlen=k)

        # Running stats per model
        self._abs_errors  = {n: [] for n in models}
        self._dir_correct = {n: [] for n in models}
        self._shape_errs  = {n: {} for n in models}

    def reset(self):
        self.window.clear()
        for n in self.models:
            self._abs_errors[n].clear()
            self._dir_correct[n].clear()
            # keep _shape_errs accumulating across series

    def step(self, rate):
        """
        Feed one observation. Returns dict {model_name: predicted_rate} if
        the window is full, else None.
        """
        self.window.append(float(rate))
        if len(self.window) < self.k:
            return None

        h    = np.array(self.window, dtype=np.float32)
        mu   = h.mean()
        std  = h.std() + 1e-8
        norm = (h - mu) / std

        x = torch.FloatTensor(norm).unsqueeze(0).unsqueeze(-1)  # [1, K, 1]

        preds = {}
        for name, model in self.models.items():
            with torch.no_grad():
                pred_norm = model(x).item()
            preds[name] = max(pred_norm * std + mu, 0.0)

        return preds

    def record(self, preds, actual_next, shape="unknown"):
        """Record prediction quality. Call after step() returns non-None."""
        if preds is None:
            return
        current = list(self.window)[-1]
        dir_actual = 1 if actual_next > current else -1

        for name, predicted in preds.items():
            err = abs(predicted - actual_next)
            self._abs_errors[name].append(err)

            dir_pred = 1 if predicted > current else -1
            self._dir_correct[name].append(int(dir_pred == dir_actual))

            self._shape_errs[name].setdefault(shape, []).append(err)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def rolling(self, window=50):
        out = {}
        for n in self.models:
            errs = self._abs_errors[n][-window:]
            dirs = self._dir_correct[n][-window:]
            out[n] = dict(
                mae     = float(np.mean(errs)) if errs else 0.0,
                dir_acc = float(np.mean(dirs)) * 100 if dirs else 0.0,
                n       = len(self._abs_errors[n]),
            )
        return out

    def print_final(self):
        print(f"\n{'='*72}")
        print("  STREAMING EVALUATION — FINAL RESULTS")
        print(f"{'='*72}")

        # ── Overall ───────────────────────────────────────────────────────────
        names = list(self.models.keys())
        all_mae = {n: float(np.mean(self._abs_errors[n])) for n in names}
        best    = min(all_mae, key=all_mae.get)

        print(f"\n  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'DirAcc':>9} {'Samples':>9}")
        print(f"  {'─'*50}")
        for n in sorted(names, key=lambda x: all_mae[x]):
            errs = np.array(self._abs_errors[n])
            rmse = float(np.sqrt(np.mean(errs ** 2)))
            dacc = float(np.mean(self._dir_correct[n])) * 100
            star = " ★" if n == best else ""
            print(f"  {n:<12} {all_mae[n]:>8.2f} {rmse:>8.2f} "
                  f"{dacc:>8.1f}% {len(errs):>9,}{star}")

        # ── Per-shape breakdown ────────────────────────────────────────────────
        all_shapes = sorted({s for n in names
                               for s in self._shape_errs[n]})
        if all_shapes:
            col = 10
            print(f"\n  Per-shape MAE:")
            header = f"  {'Shape':<14}" + "".join(f"{n:>{col}}" for n in names)
            print(header)
            print(f"  {'─'*(14 + col * len(names))}")
            for shape in all_shapes:
                row = f"  {shape:<14}"
                for n in names:
                    errs = self._shape_errs[n].get(shape, [])
                    row += f"{np.mean(errs):>{col}.2f}" if errs else f"{'—':>{col}}"
                print(row)

        # ── Improvement vs first model listed ─────────────────────────────────
        if len(names) > 1:
            ref_name = "lstm" if "lstm" in names else names[0]
            ref_mae  = all_mae[ref_name]
            print(f"\n  vs {ref_name.upper()} (reference):")
            for n in names:
                if n == ref_name:
                    continue
                delta = (ref_mae - all_mae[n]) / ref_mae * 100
                sign  = "better" if delta > 0 else "worse"
                print(f"    {n:<10}  {abs(delta):.1f}% {sign}")

        print(f"\n{'='*72}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n_series=20, series_len=300, print_every=100, only=None):
    print("Loading models...")
    models = load_models(only)
    if not models:
        print("No models loaded. Run train_all.py first.")
        return

    ev = StreamingEvaluator(models)
    total_steps = 0

    print(f"\nRunning on {n_series} test series ({series_len} steps each)...\n")
    header = f"{'step':>7}  " + "  ".join(
        f"{n}: MAE={'{mae:.1f}':>6} dir={'{dir_acc:.0f}':>3}%" for n in models
    )

    for series_idx in range(n_series):
        seed   = 5000 + series_idx          # well away from training seeds (42–191)
        values, labels = generate_series(n=series_len, baseline=100,
                                         noise_std=10, seed=seed)
        shapes = shape_at_each_step(labels, series_len)

        ev.reset()   # fresh window per series; shape_errs accumulate

        for t in range(series_len - 1):
            preds = ev.step(values[t])
            ev.record(preds, actual_next=values[t + 1], shape=shapes[t])
            total_steps += 1

            if total_steps % print_every == 0:
                stats = ev.rolling(window=print_every)
                parts = [f"step {total_steps:6d}"]
                for n, s in stats.items():
                    parts.append(f"{n}: MAE={s['mae']:5.1f} dir={s['dir_acc']:4.0f}%")
                print("  |  ".join(parts))

    ev.print_final()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series",      type=int,   default=20,
                        help="number of test series (default 20)")
    parser.add_argument("--len",         type=int,   default=300,
                        help="steps per series (default 300)")
    parser.add_argument("--print-every", type=int,   default=100,
                        help="rolling report interval (default 100)")
    parser.add_argument("--model",       type=str,   default=None,
                        nargs="+",
                        help="limit to specific models e.g. --model lstm tcn")
    args = parser.parse_args()

    run(
        n_series    = args.series,
        series_len  = args.len,
        print_every = args.print_every,
        only        = set(args.model) if args.model else None,
    )
