"""
evaluate_stream4.py — System-Level Trigger Policy Comparison

Runs a single Spark job with burst-shaped traffic (data2.py holdout seeds),
captures the 100ms-resolution rate trace from the producer, then simulates
multiple trigger policies offline on that trace.

This answers the question evaluate_stream3.py cannot: does adaptive triggering
actually improve system behaviour, or do better predictions not translate into
better scheduling decisions?

Policies compared:
  Fixed-100ms, Fixed-500ms, Fixed-1000ms  — fixed-interval baselines
  Adaptive-EMA    — intervalMs set from EMA prediction
  Adaptive-LSTM   — intervalMs set from LSTM prediction
  Adaptive-top3   — intervalMs set from ens_top3 prediction

Adaptive interval formula (same as JVM agent):
  intervalMs = clamp(TARGET_BATCH_EVENTS / max(predicted_ev_per_s, 0.1) * 1000,
                     MIN_INTERVAL_MS, MAX_INTERVAL_MS)
  TARGET_BATCH_EVENTS = 500
  MIN = 100ms, MAX = 5000ms

Metrics per policy:
  triggers      — total triggers fired in the eval window
  empty%        — fraction of triggers that processed 0 rows
  mean_batch    — mean rows per non-empty trigger
  throughput    — total_rows / total_triggers (efficiency)
  peak_backlog  — max events waiting at any trigger point
  clearance_s   — seconds to drain backlog below 10% of peak after the
                  largest burst (lower = faster response)

Usage:
  python evaluate_stream4.py
  python evaluate_stream4.py --lag
  python evaluate_stream4.py --duration 180
"""

import argparse
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

log = logging.getLogger("eval4")
log.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
_sh  = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_sh)

def add_file_handler(path):
    fh = logging.FileHandler(path)
    fh.setFormatter(_fmt)
    log.addHandler(fh)


# ── Adaptive trigger parameters ────────────────────────────────────────────────

TARGET_BATCH_EVENTS = 500   # desired rows per trigger for adaptive policy
MIN_INTERVAL_MS     = 100   # fastest trigger rate
MAX_INTERVAL_MS     = 5000  # slowest trigger rate
TICK_MS             = 100   # producer tick resolution (milliseconds)
TICK_S              = TICK_MS / 1000.0


# ── Burst Producer (same as eval3, but rate_log exposed at tick resolution) ────

class BurstProducer:
    TICK     = TICK_S
    MAX_RATE = 2000.0

    def __init__(self, seeds=None, use_lag=False):
        seeds = seeds or list(HOLDOUT_SEEDS)
        self._series_list = []
        for seed in seeds:
            if use_lag:
                vals, lag, _, _ = generate_series_with_lag2(n=600, seed=seed)
            else:
                vals, _, _ = generate_series2(n=600, seed=seed)
                lag = None
            scale = min(1.0, self.MAX_RATE / max(float(vals.max()), 1.0))
            vals  = vals * scale
            if lag is not None:
                lag = lag * scale
            self._series_list.append((vals, lag))

        self._s_idx   = 0
        self._pos     = 0
        self._queue   = deque()
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None
        self._tick_rates = []   # (timestamp_s, rate_ev_per_s) at every 100ms tick

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
            self._tick_rates.append((time.time(), rate))
            self._pos += 1
            if self._pos >= len(vals):
                self._pos   = 0
                self._s_idx = (self._s_idx + 1) % len(self._series_list)
            time.sleep(max(0.0, self.TICK - (time.time() - t0)))

    def drain(self):
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        return items

    def get_tick_rates(self):
        """Returns list of (timestamp_s, rate_ev_per_s) at 100ms resolution."""
        with self._lock:
            return list(self._tick_rates)

    def rate_summary(self):
        rates = [r for _, r in self._tick_rates]
        if not rates:
            return {}
        return dict(min=min(rates), max=max(rates), mean=float(np.mean(rates)),
                    p25=float(np.percentile(rates, 25)),
                    p75=float(np.percentile(rates, 75)))


# ── Spark listener (minimal — just keeps Spark alive) ─────────────────────────

class RateCollector(StreamingQueryListener):
    def __init__(self):
        super().__init__()
        self._batch_count = 0

    def onQueryStarted(self, event):
        log.info(f"[listener] Query started: {event.id}")

    def onQueryProgress(self, event):
        self._batch_count += 1
        log.debug(f"[listener] batch={self._batch_count}  "
                  f"rate={event.progress.inputRowsPerSecond:.1f}")

    def onQueryTerminated(self, event):
        log.info(f"[listener] Query terminated: {event.id}")

    def onQueryIdle(self, event):
        pass


# ── Socket server ──────────────────────────────────────────────────────────────

class SocketServer:
    def __init__(self, host, port, producer):
        self.host, self.port = host, port
        self.producer = producer
        self._srv    = None
        self._ready  = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

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
            log.error("Timed out waiting for Spark connection")
        finally:
            try: self._srv.close()
            except Exception: pass

    def stop(self):
        try: self._srv.close()
        except Exception: pass


# ── Model utilities ────────────────────────────────────────────────────────────

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
    h   = np.array(rate_window, dtype=np.float32)
    mu  = h.mean()
    std = max(float(h.std()), mu * 0.05) + 1e-8
    norm_r = (h - mu) / std
    if lag_window is not None and len(lag_window) == len(rate_window):
        lag_arr = np.array(lag_window, dtype=np.float32)
        lag_std = float(lag_arr.std())
        norm_l  = np.zeros_like(lag_arr) if lag_std < 1e-6 \
                  else (lag_arr - lag_arr.mean()) / (lag_std + 1e-8)
        feat = np.stack([norm_r, norm_l], axis=-1)
        x = torch.FloatTensor(feat).unsqueeze(0)
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
            if use_lag:
                lag_win.append(float(lag_vals[t]))
            if len(win) < K: continue
            x, mu, std = _build_tensor(win, lag_win if use_lag else None)
            for n, m in neural.items():
                with torch.no_grad():
                    errs[n].append(abs(max(m(x).item() * std + mu, 0.0) - float(vals[t + 1])))
    return {n: float(np.mean(v)) for n, v in errs.items()}


def _detect_input_size(state_dict):
    """
    Infer input_size from checkpoint weights.
    Strategy: find the smallest shape[1] across all weight tensors (ndim>=2).
    For all our architectures, the first-layer input_size (1 or 2) is the
    smallest value appearing in dim-1 of any weight tensor.
    """
    candidates = []
    for tensor in state_dict.values():
        if tensor.ndim >= 2:
            candidates.append(int(tensor.shape[1]))
    if not candidates:
        return 1
    return min(candidates)


def load_neural_models(model_dir=None):
    model_dir  = Path(model_dir) if model_dir else MODEL_DIR
    neural     = {}
    input_size = None
    for name, Cls in MODELS.items():
        path = model_dir / f"{name}_predictor.pt"
        if not path.exists():
            log.warning(f"  [skip] {name} — no checkpoint in {model_dir}"); continue
        sd = torch.load(path, map_location="cpu", weights_only=True)
        # try detected size, then fall back to 1 and 2
        hint = _detect_input_size(sd)
        loaded = False
        for inp in [hint] + [x for x in (1, 2) if x != hint]:
            try:
                m = Cls(input_size=inp)
                m.load_state_dict(sd)
                m.eval()
                neural[name] = m
                if input_size is None or (inp != 1):
                    input_size = inp
                log.info(f"  [ok]   {name}  (input_size={inp})")
                loaded = True
                break
            except RuntimeError:
                continue
        if not loaded:
            log.warning(f"  [skip] {name} — could not load checkpoint")
    return neural, (input_size or 1)


def build_predictors(neural, use_lag=False):
    alpha  = tune_ema()
    v_maes = compute_val_maes(neural, use_lag=use_lag)
    for n, mae in sorted(v_maes.items(), key=lambda x: x[1]):
        log.info(f"  {n:<12}  val_MAE={mae:.4f}")

    top3_names = {n: neural[n] for n in ("lstm", "gru", "tide") if n in neural}

    return {
        "ema":      EMABaseline(alpha=alpha),
        "lstm":     neural["lstm"] if "lstm" in neural else None,
        "ens_top3": EnsembleModel(top3_names) if len(top3_names) >= 2 else None,
    }, v_maes


# ── Adaptive interval formula ──────────────────────────────────────────────────

def predicted_interval_ms(predicted_rate_ev_per_s):
    """
    Convert a predicted input rate into a trigger interval.
    High rate → short interval (process frequently).
    Low rate  → long interval (save empty triggers).
    """
    ms = TARGET_BATCH_EVENTS / max(predicted_rate_ev_per_s, 0.1) * 1000.0
    return float(np.clip(ms, MIN_INTERVAL_MS, MAX_INTERVAL_MS))


# ── Policy simulation ──────────────────────────────────────────────────────────

def simulate_fixed_policy(tick_rates, interval_ms):
    """
    Simulate a fixed-interval trigger policy on a tick-resolution rate trace.
    Returns metrics dict.
    """
    ticks_per_interval = max(1, round(interval_ms / TICK_MS))
    backlog     = 0.0
    triggers    = []
    next_trig   = ticks_per_interval

    for i, (_, rate) in enumerate(tick_rates):
        backlog += rate * TICK_S         # events arriving this tick
        if i >= next_trig:
            triggers.append(backlog)
            backlog    = 0.0
            next_trig += ticks_per_interval

    return _metrics_from_triggers(triggers, tick_rates)


def simulate_adaptive_policy(tick_rates, predictor, use_lag=False, capacity_ratio=1.2):
    """
    Simulate adaptive-interval trigger policy.
    predictor is a callable (neural model) or has .predict_raw method.
    When use_lag=True, simulates a consumer lag channel alongside rate and feeds
    both into 2-feature models.
    Returns metrics dict.
    """
    backlog     = 0.0
    triggers    = []
    win         = deque(maxlen=K)
    lag_win     = deque(maxlen=K) if use_lag else None
    lag_val     = 0.0
    next_trig   = round(1000 / TICK_MS)
    is_baseline = getattr(predictor, "is_baseline", False)

    for i, (_, rate) in enumerate(tick_rates):
        backlog += rate * TICK_S
        win.append(rate)
        if use_lag:
            cap     = float(np.mean(win)) * capacity_ratio if win else rate * capacity_ratio
            lag_val = max(0.0, lag_val + rate - cap)
            lag_win.append(lag_val)

        if i >= next_trig:
            triggers.append(backlog)
            backlog = 0.0

            if len(win) >= K:
                raw = list(win)
                if is_baseline:
                    pred_rate = max(predictor.predict_raw(raw), 0.0)
                else:
                    lw = list(lag_win) if use_lag and len(lag_win) == K else None
                    x, mu, std = _build_tensor(raw, lw)
                    with torch.no_grad():
                        pred_rate = max(predictor(x).item() * std + mu, 0.0)
            else:
                pred_rate = rate

            interval_ms = predicted_interval_ms(pred_rate)
            next_trig   = i + max(1, round(interval_ms / TICK_MS))

    return _metrics_from_triggers(triggers, tick_rates)


def _metrics_from_triggers(triggers, tick_rates):
    if not triggers:
        return dict(triggers=0, empty_pct=0.0, mean_batch=0.0,
                    throughput=0.0, peak_backlog=0.0, clearance_s=float("inf"))

    rows = np.array(triggers)
    total_rows = float(rows.sum())
    n_triggers = len(rows)
    n_empty    = int((rows == 0).sum())

    non_empty = rows[rows > 0]
    mean_batch = float(non_empty.mean()) if len(non_empty) else 0.0

    peak_backlog = float(rows.max())

    # burst clearance: find the trigger with peak backlog, then count
    # triggers until backlog falls below 10% of peak
    peak_idx = int(np.argmax(rows))
    threshold = peak_backlog * 0.1
    clearance_s = float("inf")
    for j in range(peak_idx, len(rows)):
        if rows[j] <= threshold:
            clearance_s = (j - peak_idx) * (1000 / TICK_MS) * TICK_S
            break

    return dict(
        triggers     = n_triggers,
        empty_pct    = n_empty / n_triggers * 100,
        mean_batch   = mean_batch,
        throughput   = total_rows / n_triggers,
        peak_backlog = peak_backlog,
        clearance_s  = clearance_s,
    )


# ── Table printing ─────────────────────────────────────────────────────────────

def _print_table(results, duration_s, total_ticks):
    header = (f"  {'Policy':<22} {'Triggers':>8} {'Empty%':>8} "
              f"{'MeanBatch':>10} {'Throughput':>11} {'PeakBacklog':>12} {'Clearance':>10}")
    div = "─" * 90

    print(f"\n{div}")
    print(f"  evaluate_stream4.py — Trigger Policy Comparison  "
          f"(burst traffic, {duration_s:.0f}s, {total_ticks} ticks @ {TICK_MS}ms)")
    print(f"  Adaptive formula: intervalMs = "
          f"clamp({TARGET_BATCH_EVENTS}/pred_ev_s × 1000, "
          f"{MIN_INTERVAL_MS}, {MAX_INTERVAL_MS})")
    print(div)
    print(header)
    print(f"  {'─'*86}")

    # sort: fixed policies first (by interval), then adaptive
    fixed_keys   = [k for k in results if k.startswith("fixed")]
    adaptive_keys = [k for k in results if k.startswith("adaptive")]
    order = sorted(fixed_keys) + adaptive_keys

    fixed_1000 = results.get("fixed_1000ms", {})
    baseline_triggers = fixed_1000.get("triggers", 1) or 1

    for key in order:
        m = results[key]
        clr = f"{m['clearance_s']:.1f}s" if m['clearance_s'] != float("inf") else "∞"
        savings = ""
        if key.startswith("adaptive") and fixed_1000:
            saved = (1 - m["triggers"] / baseline_triggers) * 100
            savings = f"  ({saved:+.0f}% triggers vs fixed-1000ms)"
        print(f"  {key:<22} {m['triggers']:>8} {m['empty_pct']:>7.1f}%"
              f" {m['mean_batch']:>10.1f} {m['throughput']:>11.1f}"
              f" {m['peak_backlog']:>12.0f} {clr:>10}{savings}")
    print(f"{div}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=180,
                        help="Seconds to run the Spark + burst producer")
    parser.add_argument("--lag",      action="store_true",
                        help="Use lag-aware models (input_size=2)")
    parser.add_argument("--log-dir",  default=str(LOG_DIR))
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing model checkpoints "
                             "(default: models/ at project root)")
    args = parser.parse_args()

    log_dir  = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    add_file_handler(log_dir / f"eval4_{RUN_TS}.log")

    log.info(f"evaluate_stream4  duration={args.duration}s  lag={args.lag}")

    # ── Load models ───────────────────────────────────────────────────────────
    log.info("Loading neural models (auto-detecting input_size from checkpoints)...")
    neural, detected_input_size = load_neural_models(model_dir=args.model_dir)
    if not neural:
        log.error("No trained models found — run train.py first.")
        sys.exit(1)

    use_lag = args.lag or (detected_input_size == 2)
    log.info(f"  detected input_size={detected_input_size}  use_lag={use_lag}")

    predictors, v_maes = build_predictors(neural, use_lag=use_lag)
    predictors = {k: v for k, v in predictors.items() if v is not None}
    log.info(f"Predictors for adaptive policies: {list(predictors.keys())}")

    # ── Spark + BurstProducer ─────────────────────────────────────────────────
    spark = (SparkSession.builder
             .master("local[2]")
             .appName("AdaptiveStreamEval4")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    collector = RateCollector()
    spark.streams.addListener(collector)

    producer = BurstProducer(use_lag=False)   # tick rates always 1-feature
    producer.start()

    host, port = "127.0.0.1", 9997
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

    log.info(f"Spark running for {args.duration}s (burst traffic via BurstProducer)...")
    time.sleep(args.duration)

    producer.stop()
    server.stop()
    query.stop()
    spark.stop()

    # ── Retrieve tick-resolution rate trace ───────────────────────────────────
    tick_rates = producer.get_tick_rates()
    log.info(f"Captured {len(tick_rates)} ticks at {TICK_MS}ms resolution")
    if len(tick_rates) < K:
        log.error("Too few ticks captured — increase --duration")
        sys.exit(1)

    summary = producer.rate_summary()
    if summary:
        log.info(
            f"Producer rates: min={summary['min']:.0f}  max={summary['max']:.0f}  "
            f"mean={summary['mean']:.0f}  p25={summary['p25']:.0f}  "
            f"p75={summary['p75']:.0f} ev/s"
        )

    # ── Simulate all policies on the captured trace ───────────────────────────
    results = {}

    for ms in [100, 500, 1000, 2000]:
        key = f"fixed_{ms}ms"
        log.info(f"Simulating {key}...")
        results[key] = simulate_fixed_policy(tick_rates, interval_ms=ms)

    for pred_name, predictor in predictors.items():
        key = f"adaptive_{pred_name}"
        log.info(f"Simulating {key}...")
        results[key] = simulate_adaptive_policy(tick_rates, predictor, use_lag=use_lag)

    # ── Print results ─────────────────────────────────────────────────────────
    _print_table(results, duration_s=args.duration, total_ticks=len(tick_rates))

    # log individual model summary
    for key, m in results.items():
        log.info(
            f"  {key:<22}  triggers={m['triggers']}  empty={m['empty_pct']:.1f}%  "
            f"mean_batch={m['mean_batch']:.1f}  throughput={m['throughput']:.1f}  "
            f"clearance={m['clearance_s']:.1f}s"
        )


if __name__ == "__main__":
    main()
