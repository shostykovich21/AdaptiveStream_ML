"""
Streaming evaluator — tests trained models against held-out and real-world data.

Simulates production inference exactly:
  - K=30 sliding window deque (mirrors metrics_collector.py)
  - Per-window normalisation at each step (mirrors predictor_server.py)
  - One observation at a time, no look-ahead

Sources
-------
  1. Synthetic hold-out   — seeds 162-191 (train.py held out series 120-149)
  2. Wikipedia pageviews  — hourly rates, 4 articles with known burst events
  3. GitHub Archive       — per-minute event counts (--github, downloads ~5 MB/hr)

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

K         = 30
MODEL_DIR = Path(__file__).parent.parent / "models"

# held-out series: train.py base_seed=42, 70/15/15 split
# train: indices 0-104 (seeds 42-146), val: 105-127 (seeds 147-169), test: 128-149 (seeds 170-191)
HOLDOUT_SEEDS = range(170, 192)

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


# ── Evaluator ─────────────────────────────────────────────────────────────────

class StreamingEvaluator:
    """
    Shared sliding window across all models — every model sees the same
    normalised history at each timestep.
    """

    def __init__(self, models, k=K):
        self.models = models
        self.k      = k
        self.window = deque(maxlen=k)
        self._abs_errors  = {n: [] for n in models}
        self._dir_correct = {n: [] for n in models}
        self._shape_errs  = {n: {} for n in models}

    def reset(self):
        self.window.clear()
        for n in self.models:
            self._abs_errors[n].clear()
            self._dir_correct[n].clear()

    def step(self, rate):
        self.window.append(float(rate))
        if len(self.window) < self.k:
            return None
        h   = np.array(self.window, dtype=np.float32)
        mu  = h.mean()
        std = max(float(h.std()), mu * 0.05) + 1e-8
        x   = torch.FloatTensor((h - mu) / std).unsqueeze(0).unsqueeze(-1)
        preds = {}
        for name, model in self.models.items():
            with torch.no_grad():
                preds[name] = max(model(x).item() * std + mu, 0.0)
        return preds

    def record(self, preds, actual_next, shape="unknown"):
        if preds is None:
            return
        current = list(self.window)[-1]
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
        all_mae = {n: float(np.mean(self._abs_errors[n])) for n in names}
        best    = min(all_mae, key=all_mae.get)

        if label:
            print(f"\n  [{label}]")
        print(f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'DirAcc':>9} {'n':>8}")
        print(f"  {'─'*48}")
        for n in sorted(names, key=lambda x: all_mae[x]):
            errs = np.array(self._abs_errors[n])
            rmse = float(np.sqrt(np.mean(errs ** 2)))
            dacc = float(np.mean(self._dir_correct[n])) * 100
            star = " ★" if n == best else ""
            print(f"  {n:<12} {all_mae[n]:>8.2f} {rmse:>8.2f}"
                  f" {dacc:>8.1f}% {len(errs):>8,}{star}")

        all_shapes = sorted({s for n in names for s in self._shape_errs[n]})
        if all_shapes:
            col = 10
            print(f"\n  Per-shape MAE:")
            print(f"  {'Shape':<14}" + "".join(f"{n:>{col}}" for n in names))
            print(f"  {'─'*(14 + col * len(names))}")
            for shape in all_shapes:
                row = f"  {shape:<14}"
                for n in names:
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
        views = views[views > 0]
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
    ev = StreamingEvaluator(models)
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
    models = load_models(only)
    if not models:
        print("No models found. Run train.py first.")
        return

    results = {}

    # ── 1. Synthetic hold-out ─────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Synthetic hold-out  (seeds 162–191, not seen during training)")
    streams, shape_maps = [], []
    for seed in HOLDOUT_SEEDS:
        values, labels = generate_series(n=300, baseline=100, noise_std=10, seed=seed)
        streams.append(values)
        shape_maps.append(shape_at_each_step(labels, len(values)))

    ev_syn = StreamingEvaluator(models)
    total = 0
    for values, shapes in zip(streams, shape_maps):
        ev_syn.reset()
        for t in range(len(values) - 1):
            preds = ev_syn.step(values[t])
            ev_syn.record(preds, actual_next=values[t + 1], shape=shapes[t])
            total += 1
    print(f"  {len(streams)} series  {total:,} steps")
    ev_syn.print_results()
    results["synthetic hold-out"] = ev_syn

    # ── 2. Wikipedia pageviews ────────────────────────────────────────────────
    print(f"\n{'─'*55}")
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
        print(f"\n{'─'*55}")
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
            results["github archive"] = ev_gh
        else:
            print("  Not enough data.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*65}")
        print("  AGGREGATE  (MAE per source)")
        print(f"  {'source':<24}" + "".join(f"{n:>12}" for n in models))
        print(f"  {'─'*60}")
        for src, ev in results.items():
            row = f"  {src:<24}"
            for n in models:
                mae = float(np.mean(ev._abs_errors[n])) if ev._abs_errors[n] else float("inf")
                row += f"{mae:>12.2f}"
            print(row)
        print(f"  {'─'*60}")
        print(f"  {'pooled':<24}", end="")
        for n in models:
            all_errs = [e for ev in results.values() for e in ev._abs_errors[n]]
            print(f"{float(np.mean(all_errs)):>12.2f}", end="")
        print(f"\n{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--github", action="store_true",
                        help="also fetch GitHub Archive (~5 MB per hour)")
    parser.add_argument("--model", nargs="+", default=None)
    args = parser.parse_args()
    main(github=args.github,
         only=set(args.model) if args.model else None)
