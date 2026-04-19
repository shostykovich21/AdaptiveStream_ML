"""
evaluate_stream3.py — Burst-Traffic Evaluation (Two Tables)
============================================================

Traffic source: data2.py burst shapes (holdout seeds 169–191) fed into a
live Spark Structured Streaming job via socket. Rates are shaped — ramps,
walls, sawteeth, cliffs — unlike evaluate_stream2.py's random-walk producer.

Table 1 (Option A — Replay):
  After the Spark run, the full list of Spark-measured inputRowsPerSecond
  values is replayed offline through the sliding-window evaluator. No live
  Spark needed; deterministic; fully reproducible.

Table 2 (Option B — Live):
  Real-time inference during the Spark run. At each 1-second tick: snapshot
  the K-window, predict, sleep, measure actual next batch rate.

The two tables use identical rates (same Spark run). Differences in the
numbers reflect real-time windowing effects vs. clean offline replay.
If they match closely, it validates that offline replay is a sufficient
substitute for live evaluation — important for reproducibility.

Usage:
  python evaluate_stream3.py
  python evaluate_stream3.py --lag
  python evaluate_stream3.py --duration 300
"""

import argparse
import csv
import logging
import sys
import socket
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql.streaming import StreamingQueryListener

sys.path.insert(0, str(Path(__file__).parent))
from models import MODELS
from config import K, MODEL_DIR, VAL_SEEDS, HOLDOUT_SEEDS
from data import generate_series, generate_series_with_lag
from data2 import generate_series2, generate_series_with_lag2

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
RUN_TS  = datetime.now().strftime("%Y%m%d_%H%M%S")

log = logging.getLogger("eval3")
log.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
_sh  = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_sh)

def add_file_handler(path):
    fh = logging.FileHandler(path)
    fh.setFormatter(_fmt)
    log.addHandler(fh)


# ── Burst Producer ─────────────────────────────────────────────────────────────

class BurstProducer:
    """
    Generates events following data2.py burst shapes (ramps, walls, sawteeth,
    cliffs, double-peaks). Cycles through HOLDOUT_SEEDS so the traffic is
    never from a series seen during training.

    Rates are capped at MAX_RATE to keep local Spark manageable. The shape
    structure (transitions, peaks, drop-offs) is preserved even when capped.
    """

    TICK     = 0.1        # seconds per tick
    MAX_RATE = 2000.0     # events/s ceiling for local Spark

    def __init__(self, seeds=None, use_lag=False, capacity_ratio=1.2):
        seeds = seeds or list(HOLDOUT_SEEDS)
        self._use_lag        = use_lag
        self._capacity_ratio = capacity_ratio

        # Pre-generate all series (rate + optional lag)
        self._series_list = []
        for seed in seeds:
            if use_lag:
                vals, lag, _, _ = generate_series_with_lag2(n=600, seed=seed)
            else:
                vals, _, _ = generate_series2(n=600, seed=seed)
                lag = None
            # cap without changing shape
            scale = min(1.0, self.MAX_RATE / max(float(vals.max()), 1.0))
            vals  = vals * scale
            if lag is not None:
                lag = lag * scale
            self._series_list.append((vals, lag))

        self._s_idx   = 0
        self._pos     = 0
        self._lag_val = 0.0
        self._queue   = deque()
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None
        self._rate_log = []

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        rng = np.random.default_rng()
        while self._running:
            t0 = time.time()
            vals, _ = self._series_list[self._s_idx]
            rate    = float(vals[self._pos])
            n       = int(rng.poisson(rate * self.TICK))
            with self._lock:
                self._queue.extend([b"x\n"] * n)
            self._rate_log.append((time.time(), rate))

            self._pos += 1
            if self._pos >= len(vals):
                self._pos  = 0
                self._s_idx = (self._s_idx + 1) % len(self._series_list)

            time.sleep(max(0.0, self.TICK - (time.time() - t0)))

    def drain(self):
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        return items

    def rate_summary(self):
        if not self._rate_log:
            return {}
        rates = [r for _, r in self._rate_log]
        return dict(min=min(rates), max=max(rates), mean=float(np.mean(rates)),
                    p25=float(np.percentile(rates, 25)),
                    p75=float(np.percentile(rates, 75)))


# ── Rate Collector (enhanced with full history for replay) ─────────────────────

class RateCollector(StreamingQueryListener):
    """
    StreamingQueryListener that captures inputRowsPerSecond.
    Maintains both a rolling K-window (for live inference) and a full
    history list (for post-hoc replay in Option A).
    """

    def __init__(self, window_size=K, simulate_lag=False, capacity_ratio=1.2):
        super().__init__()
        self._rates       = deque(maxlen=window_size)
        self._all_rates   = []          # full history for replay
        self._lock        = threading.Lock()
        self._batch_count = 0
        self._simulate_lag    = simulate_lag
        self._capacity_ratio  = capacity_ratio
        self._lag_val         = 0.0
        self._lags            = deque(maxlen=window_size) if simulate_lag else None
        self._all_lags        = [] if simulate_lag else None

    def onQueryStarted(self, event):
        log.info(f"[listener] Query started: {event.id}")

    def onQueryProgress(self, event):
        rate = event.progress.inputRowsPerSecond
        with self._lock:
            self._rates.append(float(rate))
            self._all_rates.append(float(rate))
            if self._simulate_lag:
                cap = (float(np.mean(self._rates)) * self._capacity_ratio
                       if self._rates else float(rate) * self._capacity_ratio)
                self._lag_val = max(0.0, self._lag_val + float(rate) - cap)
                self._lags.append(self._lag_val)
                self._all_lags.append(self._lag_val)
            self._batch_count += 1
        log.debug(f"[listener] batch={self._batch_count}  rate={rate:.1f}")

    def onQueryTerminated(self, event):
        log.info(f"[listener] Query terminated: {event.id}")

    def onQueryIdle(self, event):
        with self._lock:
            self._rates.append(0.0)
            self._all_rates.append(0.0)
            if self._simulate_lag:
                self._lags.append(self._lag_val)
                self._all_lags.append(self._lag_val)

    def get_window(self):
        with self._lock:
            return list(self._rates)

    def get_lag_window(self):
        with self._lock:
            return list(self._lags) if self._lags is not None else []

    def get_full_history(self):
        with self._lock:
            return list(self._all_rates)

    def get_full_lag_history(self):
        with self._lock:
            return list(self._all_lags) if self._all_lags is not None else []

    def is_ready(self):
        with self._lock:
            return len(self._rates) >= K

    @property
    def batch_count(self):
        return self._batch_count


# ── Socket Server (same as eval2) ──────────────────────────────────────────────

class SocketServer:
    def __init__(self, host, port, producer):
        self.host, self.port = host, port
        self.producer = producer
        self._srv     = None
        self._ready   = threading.Event()
        self._thread  = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("Socket server failed to start")

    def _serve(self):
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(1)
        self._ready.set()
        log.info(f"Socket server ready on {self.host}:{self.port}")
        try:
            self._srv.settimeout(30)
            conn, addr = self._srv.accept()
            log.info(f"Spark connected from {addr}")
            while True:
                items = self.producer.drain()
                if items:
                    try:
                        conn.sendall(b"".join(items))
                    except (BrokenPipeError, OSError):
                        break
                else:
                    time.sleep(0.01)
        except socket.timeout:
            log.error("Timed out waiting for Spark to connect")
        finally:
            try: self._srv.close()
            except Exception: pass

    def stop(self):
        try: self._srv.close()
        except Exception: pass


# ── Model utilities (shared with eval2) ───────────────────────────────────────

class SMABaseline:
    is_baseline = True
    def predict_raw(self, raw): return float(np.mean(raw))

class EMABaseline:
    is_baseline = True
    def __init__(self, alpha=0.3): self.alpha = alpha
    def predict_raw(self, raw):
        v = raw[0]
        for x in raw[1:]: v = self.alpha * x + (1 - self.alpha) * v
        return float(v)

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models, self.weights = models, weights
    def __call__(self, x):
        preds, ws = [], []
        for n, m in self.models.items():
            with torch.no_grad(): preds.append(m(x))
            ws.append(1.0 if self.weights is None else self.weights.get(n, 1.0))
        stacked = torch.stack(preds)
        w = torch.tensor(ws, dtype=torch.float32).view(-1, 1, 1)
        return (stacked * w / w.sum()).sum(0)


def _build_tensor(rate_window, lag_window=None):
    h    = np.array(rate_window, dtype=np.float32)
    mu   = h.mean()
    std  = max(float(h.std()), mu * 0.05) + 1e-8
    norm_r = (h - mu) / std
    if lag_window is not None and len(lag_window) == len(rate_window):
        lag_arr = np.array(lag_window, dtype=np.float32)
        lag_std = float(lag_arr.std())
        norm_l  = np.zeros_like(lag_arr) if lag_std < 1e-6 \
                  else (lag_arr - lag_arr.mean()) / (lag_std + 1e-8)
        x = torch.FloatTensor(np.stack([norm_r, norm_l], -1)).unsqueeze(0)
    else:
        x = torch.FloatTensor(norm_r).unsqueeze(0).unsqueeze(-1)
    return x, mu, std


def tune_ema():
    best_a, best_mae = 0.3, float("inf")
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7]:
        errs = []
        for seed in VAL_SEEDS:
            vals, _ = generate_series(n=300, baseline=100, noise_std=10, seed=seed)
            win = deque(maxlen=K)
            for t in range(len(vals) - 1):
                win.append(float(vals[t]))
                if len(win) < K: continue
                raw = list(win); v = raw[0]
                for x in raw[1:]: v = alpha * x + (1 - alpha) * v
                errs.append(abs(v - float(vals[t + 1])))
        mae = float(np.mean(errs))
        if mae < best_mae: best_mae, best_a = mae, alpha
    return best_a


def compute_val_maes(neural, use_lag=False):
    errs = {n: [] for n in neural}
    for seed in VAL_SEEDS:
        if use_lag:
            vals, lag_vals, _ = generate_series_with_lag(
                n=300, baseline=100, noise_std=10, seed=seed)
            lag_win = deque(maxlen=K)
        else:
            vals, _ = generate_series(n=300, baseline=100, noise_std=10, seed=seed)
            lag_win = None
        win = deque(maxlen=K)
        for t in range(len(vals) - 1):
            win.append(float(vals[t]))
            if use_lag: lag_win.append(float(lag_vals[t]))
            if len(win) < K: continue
            x, mu, std = _build_tensor(win, lag_win if use_lag else None)
            for n, m in neural.items():
                with torch.no_grad():
                    errs[n].append(abs(max(m(x).item() * std + mu, 0.0)
                                   - float(vals[t + 1])))
    return {n: float(np.mean(v)) for n, v in errs.items()}


def load_neural_models(input_size=1, model_dir=None):
    model_dir = Path(model_dir) if model_dir else MODEL_DIR
    neural = {}
    for name, Cls in MODELS.items():
        path = model_dir / f"{name}_predictor.pt"
        if not path.exists():
            log.warning(f"  [skip] {name} — no checkpoint in {model_dir}"); continue
        m = Cls(input_size=input_size)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        neural[name] = m
        log.info(f"  [ok]   {name}")
    return neural


def build_all_models(neural, use_lag=False):
    log.info("Tuning EMA alpha on val split...")
    alpha  = tune_ema()
    log.info(f"  best EMA alpha = {alpha}")
    log.info("Computing val MAEs for ensemble weights...")
    v_maes = compute_val_maes(neural, use_lag=use_lag)
    for n, mae in sorted(v_maes.items(), key=lambda x: x[1]):
        log.info(f"  {n:<12}  val_MAE={mae:.4f}")

    ext = dict(neural)
    ext["sma"] = SMABaseline()
    ext["ema"] = EMABaseline(alpha=alpha)
    ext["ens_mean"] = EnsembleModel(neural)

    inv = {n: 1.0 / v_maes[n] for n in neural if n in v_maes}
    tot = sum(inv.values())
    ext["ens_wtd"] = EnsembleModel(neural, {n: w / tot for n, w in inv.items()})

    rnn = {n: neural[n] for n in ("lstm", "gru") if n in neural}
    if len(rnn) >= 2: ext["ens_rnn"] = EnsembleModel(rnn)

    top3 = {n: neural[n] for n in ("lstm", "gru", "tide") if n in neural}
    if len(top3) >= 2: ext["ens_top3"] = EnsembleModel(top3)

    best_rnn = min([n for n in ("lstm", "gru") if n in neural],
                   key=lambda n: v_maes.get(n, float("inf")), default=None)
    diverse  = {n: neural[n] for n in [best_rnn, "tcn", "tide", "attn"]
                if n is not None and n in neural}
    if len(diverse) >= 2: ext["ens_diverse"] = EnsembleModel(diverse)

    return ext


# ── Table printing ─────────────────────────────────────────────────────────────

def _print_table(label, names, abs_errs, mape_errs, dir_corr, models):
    all_mae = {n: float(np.mean(abs_errs[n])) if abs_errs[n] else float("inf")
               for n in names}
    best    = min(all_mae, key=all_mae.get)

    print(f"\n{'─'*78}")
    print(f"  {label}")
    print(f"{'─'*78}")
    print(f"  {'Model':<14} {'MAE':>8} {'RMSE':>8} {'MAPE':>9} {'DirAcc':>9} {'n':>6}")
    print(f"  {'─'*62}")

    for n in sorted(names, key=lambda x: all_mae[x]):
        errs = np.array(abs_errs[n]) if abs_errs[n] else np.array([0.0])
        mae  = float(np.mean(errs))
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        mape = float(np.mean(mape_errs[n])) if mape_errs[n] else 0.0
        dacc = float(np.mean(dir_corr[n]))  * 100 if dir_corr[n] else 0.0
        star = " ★" if n == best else ""
        print(f"  {n:<14} {mae:>8.2f} {rmse:>8.2f} {mape:>8.1f}% {dacc:>8.1f}%"
              f" {len(errs):>6}{star}")
    print(f"{'─'*78}\n")


# ── Option B: Live evaluation during Spark run ────────────────────────────────

def run_live_eval(models, collector, duration, csv_path, use_lag):
    names     = list(models.keys())
    abs_errs  = {n: [] for n in names}
    mape_errs = {n: [] for n in names}
    dir_corr  = {n: [] for n in names}

    fields = (["timestamp", "actual"] +
              [f"pred_{n}" for n in names] +
              [f"abserr_{n}" for n in names])
    csv_f  = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fields)
    writer.writeheader()

    log.info(f"Waiting for window to fill (K={K} batches)...")
    wait_start = time.time()
    while not collector.is_ready():
        if time.time() - wait_start > 120:
            log.error("Window never filled after 120s.")
            csv_f.close(); return abs_errs, mape_errs, dir_corr
        time.sleep(3)

    log.info("Window full — live evaluation running (Option B)")
    eval_start = time.time()

    while time.time() - eval_start < duration:
        t0      = time.time()
        history = collector.get_window()
        if len(history) < K:
            time.sleep(1); continue

        lag_win = collector.get_lag_window() if use_lag else None
        x, mu, std = _build_tensor(history, lag_win)

        preds = {}
        for name, model in models.items():
            if getattr(model, "is_baseline", False):
                preds[name] = max(model.predict_raw(list(history)), 0.0)
            else:
                with torch.no_grad():
                    preds[name] = max(model(x).item() * std + mu, 0.0)

        time.sleep(max(0, 1.0 - (time.time() - t0)))

        new_hist = collector.get_window()
        if not new_hist: continue
        actual  = float(new_hist[-1])
        current = float(history[-1])
        dir_act = 1 if actual > current else -1

        row = {"timestamp": datetime.now().isoformat(), "actual": f"{actual:.2f}"}
        for name, pred in preds.items():
            err  = abs(pred - actual)
            mape = err / (actual + 1e-8) * 100
            abs_errs[name].append(err)
            mape_errs[name].append(mape)
            dir_corr[name].append(int((1 if pred > current else -1) == dir_act))
            row[f"pred_{name}"]   = f"{pred:.2f}"
            row[f"abserr_{name}"] = f"{err:.2f}"
        writer.writerow(row); csv_f.flush()

        elapsed = time.time() - eval_start
        log.debug(f"t={elapsed:.0f}s  actual={actual:.1f}  "
                  f"lstm={preds.get('lstm',0):.1f}  "
                  f"ens_top3={preds.get('ens_top3',0):.1f}")

    csv_f.close()
    return abs_errs, mape_errs, dir_corr


# ── Option A: Offline replay of captured Spark-measured rates ─────────────────

def run_replay_eval(models, full_rates, full_lags, use_lag):
    names     = list(models.keys())
    abs_errs  = {n: [] for n in names}
    mape_errs = {n: [] for n in names}
    dir_corr  = {n: [] for n in names}

    if len(full_rates) < K + 1:
        log.warning(f"Not enough captured rates for replay ({len(full_rates)} < {K+1})")
        return abs_errs, mape_errs, dir_corr

    log.info(f"Replay eval on {len(full_rates)} captured Spark rates (Option A)...")

    for t in range(K, len(full_rates) - 1):
        rate_win = full_rates[t - K:t]
        lag_win  = full_lags[t - K:t] if (use_lag and full_lags) else None
        x, mu, std = _build_tensor(rate_win, lag_win)

        actual  = float(full_rates[t])
        current = float(full_rates[t - 1])
        dir_act = 1 if actual > current else -1

        for name, model in models.items():
            if getattr(model, "is_baseline", False):
                pred = max(model.predict_raw(rate_win), 0.0)
            else:
                with torch.no_grad():
                    pred = max(model(x).item() * std + mu, 0.0)

            err  = abs(pred - actual)
            mape = err / (actual + 1e-8) * 100
            abs_errs[name].append(err)
            mape_errs[name].append(mape)
            dir_corr[name].append(int((1 if pred > current else -1) == dir_act))

    return abs_errs, mape_errs, dir_corr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--lag",      action="store_true")
    parser.add_argument("--log-dir",  default=str(LOG_DIR))
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing model checkpoints "
                             "(default: models/ at project root)")
    args = parser.parse_args()

    log_dir  = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"eval3_{RUN_TS}.log"
    csv_path = log_dir / f"eval3_{RUN_TS}.csv"
    add_file_handler(log_path)

    input_size = 2 if args.lag else 1
    log.info(f"evaluate_stream3  duration={args.duration}s  lag={args.lag}")
    log.info(f"Loading models (input_size={input_size})...")
    neural = load_neural_models(input_size=input_size, model_dir=args.model_dir)
    if not neural:
        log.error("No trained models found — run train.py first.")
        sys.exit(1)
    models = build_all_models(neural, use_lag=args.lag)
    log.info(f"{len(neural)} neural + 2 baselines + ensembles = {len(models)} total")

    spark = (SparkSession.builder
             .master("local[2]")
             .appName("AdaptiveStreamEval3")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    collector = RateCollector(window_size=K, simulate_lag=args.lag)
    spark.streams.addListener(collector)

    producer = BurstProducer(use_lag=args.lag)
    producer.start()

    host, port = "127.0.0.1", 9998
    server = SocketServer(host, port, producer)
    server.start()

    df = (spark.readStream.format("socket")
          .option("host", host).option("port", port).load())
    query = (df.writeStream
             .format("console")
             .option("truncate", False)
             .option("numRows", 1)
             .trigger(processingTime="1 second")
             .outputMode("append")
             .start())

    log.info("Spark streaming job started (burst traffic)")

    # ── Option B: live eval ───────────────────────────────────────────────────
    b_abs, b_mape, b_dir = run_live_eval(
        models, collector, args.duration, csv_path, args.lag)

    producer.stop()
    server.stop()
    query.stop()
    spark.stop()

    # ── Option A: replay of captured Spark rates ──────────────────────────────
    full_rates = collector.get_full_history()
    full_lags  = collector.get_full_lag_history() if args.lag else []
    log.info(f"Captured {len(full_rates)} Spark-measured rates for replay")

    a_abs, a_mape, a_dir = run_replay_eval(
        models, full_rates, full_lags, args.lag)

    names = list(models.keys())

    _print_table(
        "Option A — Replay: offline sliding-window on Spark-measured burst rates",
        names, a_abs, a_mape, a_dir, models)

    _print_table(
        "Option B — Live: real-time inference on Spark burst traffic",
        names, b_abs, b_mape, b_dir, models)

    summary = producer.rate_summary()
    if summary:
        log.info(
            f"Producer: min={summary['min']:.0f}  max={summary['max']:.0f}  "
            f"mean={summary['mean']:.0f}  p25={summary['p25']:.0f}  "
            f"p75={summary['p75']:.0f} events/s")
    log.info(f"CSV → {csv_path}")


if __name__ == "__main__":
    main()
