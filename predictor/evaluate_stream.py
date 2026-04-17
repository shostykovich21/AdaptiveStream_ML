"""
Streaming evaluator — tests trained models against held-out and real-world data.

Simulates production inference exactly:
  - K=30 sliding window deque (mirrors metrics_collector.py)
  - Per-window normalisation at each step (mirrors predictor_server.py)
  - One observation at a time, no look-ahead

Sources
-------
  1. Synthetic hold-out   — seeds 169-191 (train.py test split)
  2. Wikipedia pageviews  — hourly rates, 4 articles with known burst events
  3. GitHub Archive       — per-minute event counts (--github, downloads ~5 MB/hr)

Entries in evaluation table
----------------------------
  Neural (9):    lstm, gru, tcn, dlinear, mlp, attn, tide, fits, nbeats
  Baselines (2): sma, ema  (no training, operate in raw rate space)
  Ensembles (4): ens_mean, ens_wtd, ens_rnn, ens_diverse

Usage
-----
    python evaluate_stream.py
    python evaluate_stream.py --github
    python evaluate_stream.py --model lstm tcn
"""

import argparse
import gzip
import json
import time
import numpy as np
import requests
import torch
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path

from models import MODELS
from data import generate_series, shape_at_each_step
from config import K, MODEL_DIR, VAL_SEEDS, HOLDOUT_SEEDS

WIKI_ARTICLES = [
    ("2022_FIFA_World_Cup",  "20221101", "20221231", "World Cup matches + final"),
    ("ChatGPT",              "20221101", "20230228", "launch + hype peak"),
    ("Oppenheimer_(film)",   "20230601", "20230930", "Barbenheimer release"),
    ("2024_Summer_Olympics", "20240701", "20240831", "opening ceremony + events"),
]
WIKI_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
            "/en.wikipedia/all-access/all-agents/{article}/hourly/{start}/{end}")

GH_SLOTS = [
    ("2024-01-15", 14),
    ("2024-01-15", 20),
    ("2024-01-16",  4),
    ("2024-03-25", 14),
]
GH_URL = "https://data.gharchive.org/{date}-{hour}.json.gz"


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

def load_models(only=None):
    loaded = {}
    for name, ModelClass in MODELS.items():
        if only and name not in only:
            continue
        path = MODEL_DIR / f"{name}_predictor.pt"
        if not path.exists():
            print(f"  [skip] {name} — no checkpoint. Run train.py first.")
            continue
        m = ModelClass()
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


def compute_val_maes(neural_models):
    """
    Run each neural model through the val split and return per-model MAE.
    Used to compute weights for ens_wtd and to pick the best RNN for ens_diverse.
    """
    errors = {n: [] for n in neural_models}
    for seed in VAL_SEEDS:
        values, _ = generate_series(n=300, baseline=100, noise_std=10, seed=seed)
        window = deque(maxlen=K)
        for t in range(len(values) - 1):
            window.append(float(values[t]))
            if len(window) < K:
                continue
            h   = np.array(window, dtype=np.float32)
            mu  = h.mean()
            std = max(float(h.std()), mu * 0.05) + 1e-8
            x   = torch.FloatTensor((h - mu) / std).unsqueeze(0).unsqueeze(-1)
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

    # ens_diverse: one from each family — best RNN + tcn + dlinear + attn
    best_rnn = min(
        [n for n in ("lstm", "gru") if n in neural_models],
        key=lambda n: val_maes.get(n, float("inf")),
        default=None,
    )
    diverse_names = [best_rnn, "tcn", "dlinear", "attn"]
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

    def __init__(self, models, k=K):
        self.models = models
        self.k      = k
        self.window = deque(maxlen=k)
        self._abs_errors  = {n: [] for n in models}
        self._dir_correct = {n: [] for n in models}
        self._shape_errs  = {n: {} for n in models}

    def reset(self):
        # clears only the sliding window — accumulators persist across series
        # so that print_results() reflects the full source, not just the last stream
        self.window.clear()

    def step(self, rate):
        self.window.append(float(rate))
        if len(self.window) < self.k:
            return None
        h   = np.array(self.window, dtype=np.float32)
        mu  = h.mean()
        std = max(float(h.std()), mu * 0.05) + 1e-8
        x   = torch.FloatTensor((h - mu) / std).unsqueeze(0).unsqueeze(-1)
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


# previously run() ran synthetic-only evaluation using arbitrary seeds 5000+
# replaced by main() below which uses the actual held-out split and real sources
#
# def run(n_series=20, series_len=300, print_every=100, only=None):
#     models = load_models(only)
#     ev = StreamingEvaluator(models)
#     total_steps = 0
#     for series_idx in range(n_series):
#         seed = 5000 + series_idx
#         values, labels = generate_series(n=series_len, baseline=100,
#                                          noise_std=10, seed=seed)
#         shapes = shape_at_each_step(labels, series_len)
#         ev.reset()
#         for t in range(series_len - 1):
#             preds = ev.step(values[t])
#             ev.record(preds, actual_next=values[t + 1], shape=shapes[t])
#             total_steps += 1
#             if total_steps % print_every == 0:
#                 stats = ev.rolling(window=print_every)
#                 parts = [f"step {total_steps:6d}"]
#                 for n, s in stats.items():
#                     parts.append(f"{n}: MAE={s['mae']:5.1f} dir={s['dir_acc']:4.0f}%")
#                 print("  |  ".join(parts))
#     ev.print_final()


# ── Data fetchers ─────────────────────────────────────────────────────────────

def fetch_wikipedia(article, start, end):
    url = WIKI_API.format(article=article, start=start, end=end)
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "AdaptiveStream-research/1.0"})
        if resp.status_code != 200:
            print(f"    [wiki] {article}: HTTP {resp.status_code}")
            return None
        views = np.array([it["views"] for it in resp.json().get("items", [])],
                         dtype=np.float32)
        # keep zeros — they are real observations (no traffic that hour)
        # filtering them out compresses time and distorts burst structure
        if len(views) < K + 5:
            print(f"    [wiki] {article}: too few points ({len(views)})")
            return None
        return views
    except Exception as e:
        print(f"    [wiki] {article}: {e}")
        return None


def fetch_gh_hour(date, hour):
    url = GH_URL.format(date=date, hour=hour)
    try:
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code != 200:
            return None
        buf = BytesIO()
        for chunk in resp.iter_content(65536):
            buf.write(chunk)
        buf.seek(0)
        counts = defaultdict(int)
        with gzip.open(buf) as f:
            for line in f:
                try:
                    ts = json.loads(line).get("created_at", "")
                    if len(ts) >= 16:
                        counts[int(ts[14:16])] += 1
                except Exception:
                    pass
        result = [counts.get(m, 0) for m in range(60)]
        print(f"    [gh] {date}-{hour:02d}h  {buf.tell()//1024:,} KB  "
              f"events: {sum(result):,}")
        return result
    except Exception as e:
        print(f"    [gh] {date}-{hour:02d}: {e}")
        return None


# ── Per-source runner ─────────────────────────────────────────────────────────

def eval_source(models, streams, label, print_every=200):
    ev    = StreamingEvaluator(models)
    total = 0
    for stream in streams:
        ev.reset()
        for t in range(len(stream) - 1):
            preds = ev.step(stream[t])
            ev.record(preds, actual_next=stream[t + 1], shape=label)
            total += 1
            if total % print_every == 0:
                stats = ev.rolling(window=print_every)
                parts = [f"  step {total:6d}"]
                for n, s in stats.items():
                    parts.append(f"{n}: MAE={s['mae']:5.1f} dir={s['dir_acc']:4.0f}%")
                print("  |  ".join(parts))
    return ev, total


# ── Main ──────────────────────────────────────────────────────────────────────

def main(github=False, only=None):
    print("Loading models...")
    neural_models = load_models(only)
    if not neural_models:
        print("No models found. Run train.py first.")
        return

    # tune EMA and compute val MAEs for ensemble weighting
    print("\nTuning EMA alpha on val split...")
    ema_alpha = tune_ema_alpha()
    print(f"  best alpha = {ema_alpha}")

    print("Computing val MAEs for ensemble weights...")
    val_maes = compute_val_maes(neural_models)
    for n, mae in sorted(val_maes.items(), key=lambda x: x[1]):
        print(f"  {n:<12} val_MAE={mae:.4f}")

    # build full model dict: neural + baselines + ensembles
    models = build_extended_models(neural_models, val_maes, ema_alpha)
    print(f"\n  {len(neural_models)} neural  +  2 baselines  +  "
          f"{len(models) - len(neural_models) - 2} ensembles  =  {len(models)} total entries\n")

    results = {}

    # ── 1. Synthetic hold-out ─────────────────────────────────────────────────
    print(f"{'─'*60}")
    print("  Synthetic hold-out  (seeds 169–191, not seen during training)")
    streams, shape_maps = [], []
    for seed in HOLDOUT_SEEDS:
        values, labels = generate_series(n=300, baseline=100, noise_std=10, seed=seed)
        streams.append(values)
        shape_maps.append(shape_at_each_step(labels, len(values)))

    ev_syn = StreamingEvaluator(models)
    total  = 0
    for values, shapes in zip(streams, shape_maps):
        ev_syn.reset()
        for t in range(len(values) - 1):
            preds = ev_syn.step(values[t])
            # shapes[t+1]: tag by the regime the target value belongs to,
            # not the current regime — gives "error when predicting into shape X"
            ev_syn.record(preds, actual_next=values[t + 1], shape=shapes[t + 1])
            total += 1
    print(f"  {len(streams)} series  {total:,} steps")
    ev_syn.print_results()
    results["synthetic"] = ev_syn

    # ── 2. Wikipedia pageviews ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Wikipedia hourly pageviews")
    wiki_streams = []
    for article, start, end, note in WIKI_ARTICLES:
        print(f"  fetching {article}  ({note})")
        arr = fetch_wikipedia(article, start, end)
        if arr is not None:
            wiki_streams.append(arr)
            print(f"    → {len(arr)} points")
        time.sleep(0.3)

    if wiki_streams:
        ev_wiki, total = eval_source(models, wiki_streams, "wikipedia")
        print(f"  {len(wiki_streams)} articles  {total:,} steps")
        ev_wiki.print_results()
        results["wikipedia"] = ev_wiki
    else:
        print("  No data retrieved.")

    # ── 3. GitHub Archive (opt-in) ────────────────────────────────────────────
    if github:
        print(f"\n{'─'*60}")
        print("  GitHub Archive  (per-minute event counts)")
        gh_minutes = []
        for date, hour in GH_SLOTS:
            counts = fetch_gh_hour(date, hour)
            if counts:
                gh_minutes.extend(counts)
        if len(gh_minutes) >= K + 5:
            arr = np.array(gh_minutes, dtype=np.float32)
            ev_gh, total = eval_source(models, [arr], "github")
            print(f"  {total:,} steps")
            ev_gh.print_results()
            results["github"] = ev_gh
        else:
            print("  Not enough data.")

    # ── Aggregate table (models × sources) ───────────────────────────────────
    if len(results) > 1:
        sources = list(results.keys())
        col     = 14
        print(f"\n{'='*70}")
        print("  AGGREGATE  (MAE — rows=models, cols=sources)")
        print(f"  {'Model':<14}" +
              "".join(f"{s[:col-2]:>{col}}" for s in sources) +
              f"{'pooled':>{col}}")
        print(f"  {'─'*( 14 + col * (len(sources) + 1))}")

        pooled_all = {}
        for n in models:
            row = f"  {n:<14}"
            all_errs = []
            for src, ev in results.items():
                errs = ev._abs_errors[n]
                mae  = float(np.mean(errs)) if errs else float("inf")
                row += f"{mae:>{col}.2f}"
                all_errs.extend(errs)
            pooled = float(np.mean(all_errs)) if all_errs else float("inf")
            pooled_all[n] = pooled
            row += f"{pooled:>{col}.2f}"
            print(row)

        print(f"  {'─'*(14 + col * (len(sources) + 1))}")
        best = min(pooled_all, key=pooled_all.get)
        print(f"  best: {best}  (pooled MAE={pooled_all[best]:.2f})")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--github", action="store_true",
                        help="also fetch GitHub Archive (~5 MB per hour)")
    parser.add_argument("--model", nargs="+", default=None)
    args = parser.parse_args()
    main(github=args.github,
         only=set(args.model) if args.model else None)
