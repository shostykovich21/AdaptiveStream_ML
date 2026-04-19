"""
evaluate_stream2.py — Real Spark Infrastructure, Synthetic Random-Walk Traffic
===============================================================================

WHAT THIS EVAL DOES
-------------------
Runs a genuine Spark Structured Streaming job in-process. A StreamingQueryListener
captures inputRowsPerSecond from every micro-batch — the exact same metric the JVM
AdaptiveStream agent reads in production. Models predict the next batch's rate; the
listener delivers the actual rate one second later.

The traffic source is a Poisson random-walk producer: log_rate drifts by Normal(0,0.12)
every 100ms, events sent via socket. Rate range: 1–10,000 events/s.

WHAT IS REAL vs. WHAT IS NOT
-----------------------------
  Real:   Spark Structured Streaming job (genuine micro-batches, real JVM)
          StreamingQueryListener capturing inputRowsPerSecond (production metric)
          TCP socket source (same path as production socket mode)
          Inference latency is real wall-clock time

  Not real: The traffic producer. It generates random-walk noise, NOT the burst
            patterns (ramps, walls, sawteeth) that real production Kafka topics
            exhibit. There is no diurnal cycle, no flash-sale spike, no retry storm.

WHAT THE RESULTS TELL US
------------------------
This eval tests: "Do models handle scale-variable random-walk traffic?"

It does NOT test: burst pattern recognition, which is what the models were trained for
and what distinguishes them from EMA.

For random-walk traffic, the optimal strategy is persistence — predict that the next
value is close to the current value. EMA exploits this directly. Neural models do not
(they look for structure that isn't there), which is why EMA dominated iter1 and iter2:

  iter1: EMA MAE=6.86, LSTM MAE=8.89  — producer mean=19 ev/s, tiny random walk
  iter2: EMA MAE=11.95, LSTM MAE=28.04 — producer mean=107 ev/s, still random walk

In iter3, LSTM finally beat EMA (229 vs 271 MAE) — but NOT because burst recognition
improved. The producer happened to reach mean=1,292 ev/s (max=10k). Iter1/2 models,
trained only at baseline=100, were out-of-distribution at that scale and over-predicted
hugely. Iter3 log-uniform models handle scale correctly, so their errors are smaller.
It's a scale coverage win, not a pattern recognition win.

THE HONEST LIMITATION
---------------------
This eval validates the infrastructure pipeline correctly. It does NOT tell us whether
training on synthetic burst shapes translates to any production benefit. A model that
does well here mainly has good scale coverage and doesn't over-predict on random walk.

EMA's consistently low MAPE (~30-48% across iterations) is the benchmark to beat on
any traffic that lacks burst structure. Neural models only meaningfully outperform EMA
when the traffic actually contains the burst transitions they were trained to recognise.

WHAT THIS POINTS TOWARD
-----------------------
evaluate_stream3.py (next): two modes:
  (a) Replay mode — feed a real historical inputRowsPerSecond trace (CSV) through
      the same sliding-window evaluator. No live Spark needed. No distribution
      assumption. This is the gold-standard eval: if a model wins here, it wins
      on actual production traffic.
  (b) Burst-producer mode — replace PoissonProducer with a generator that emits
      synthetic burst shapes (from data.py/data2.py) into the live Spark job. The
      infrastructure remains real; the traffic structure approximates production.
      Note: this favours neural models (same shapes as training, different seeds)
      but tests whether the pattern recognition benefit survives the real infrastructure.

Output:
  Console: comparison table (MAE, RMSE, MAPE, DirAcc — all models + ensembles)
  File:    logs/eval_real_{timestamp}.csv  (raw per-step predictions)
           logs/eval_real_{timestamp}.log  (structured run log)

Usage:
  python evaluate_stream2.py
  python evaluate_stream2.py --lag
  python evaluate_stream2.py --duration 300
  python evaluate_stream2.py --mode kafka --kafka-broker localhost:9092
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
from config import K, MODEL_DIR, VAL_SEEDS
from data import generate_series, generate_series_with_lag

# ── logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
RUN_TS  = datetime.now().strftime("%Y%m%d_%H%M%S")

log = logging.getLogger("eval2")
log.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
_sh  = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_sh)

def add_file_handler(path):
    fh = logging.FileHandler(path)
    fh.setFormatter(_fmt)
    log.addHandler(fh)


# ── StreamingQueryListener — captures real inputRowsPerSecond ─────────────────

class RateCollector(StreamingQueryListener):
    """
    Captures inputRowsPerSecond from every batch progress event.
    When simulate_lag=True, also maintains a simulated consumer lag alongside
    the rate stream using the same capacity model as training:
      lag[t] = max(0, lag[t-1] + rate[t] - capacity_ratio * rolling_mean(rates))
    Runs on Spark's listener thread — deque is thread-safe for append/popleft.
    """

    def __init__(self, window_size=K, simulate_lag=False, capacity_ratio=1.2):
        super().__init__()
        self._rates        = deque(maxlen=window_size)
        self._lock         = threading.Lock()
        self._batch_count  = 0
        self._simulate_lag = simulate_lag
        self._capacity_ratio = capacity_ratio
        self._lag_val      = 0.0
        self._lags         = deque(maxlen=window_size) if simulate_lag else None

    def onQueryStarted(self, event):
        log.info(f"[listener] Query started: {event.id}")

    def onQueryProgress(self, event):
        rate = event.progress.inputRowsPerSecond
        with self._lock:
            self._rates.append(float(rate))
            if self._simulate_lag:
                capacity = (float(np.mean(self._rates)) * self._capacity_ratio
                            if self._rates else float(rate) * self._capacity_ratio)
                self._lag_val = max(0.0, self._lag_val + float(rate) - capacity)
                self._lags.append(self._lag_val)
            self._batch_count += 1
        log.debug(f"[listener] batch={self._batch_count}  "
                  f"inputRowsPerSecond={rate:.1f}")

    def onQueryTerminated(self, event):
        log.info(f"[listener] Query terminated: {event.id}")

    def onQueryIdle(self, event):
        with self._lock:
            self._rates.append(0.0)
            if self._simulate_lag:
                self._lags.append(self._lag_val)

    def get_history(self):
        with self._lock:
            return list(self._rates)

    def get_lag_history(self):
        with self._lock:
            return list(self._lags) if self._lags is not None else []

    def is_ready(self):
        with self._lock:
            return len(self._rates) >= K

    @property
    def batch_count(self):
        return self._batch_count


# ── Poisson random-rate producer ──────────────────────────────────────────────

class PoissonProducer:
    """
    Generates variable-rate events via a random walk in log-rate space.
    Rate range: 1–10,000 events/s. New RNG state every instantiation.
    Does NOT use our synthetic shapes — genuinely unknown traffic.
    """

    TICK      = 0.1
    LOG_DRIFT = 0.12
    LOG_MIN   = np.log(1)
    LOG_MAX   = np.log(10_000)
    LOG_START = np.log(100)

    def __init__(self):
        self._rng      = np.random.default_rng()
        self._log_rate = self.LOG_START
        self._queue    = deque()
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None
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
        while self._running:
            t0   = time.time()
            rate = np.exp(self._log_rate)
            n    = int(self._rng.poisson(rate * self.TICK))
            with self._lock:
                self._queue.extend([b"x\n"] * n)
            self._rate_log.append((time.time(), rate))
            self._log_rate += self._rng.normal(0, self.LOG_DRIFT)
            self._log_rate  = float(np.clip(self._log_rate, self.LOG_MIN, self.LOG_MAX))
            time.sleep(max(0, self.TICK - (time.time() - t0)))

    def drain(self):
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        return items

    def rate_summary(self):
        if not self._rate_log:
            return {}
        rates = [r for _, r in self._rate_log]
        return dict(min=min(rates), max=max(rates), mean=np.mean(rates),
                    p25=np.percentile(rates, 25), p75=np.percentile(rates, 75))


# ── Socket server ──────────────────────────────────────────────────────────────

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
        log.info(f"Socket server ready on {self.host}:{self.port}")
        self._ready.set()
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
            log.error("Timed out waiting for Spark to connect to socket server")
        finally:
            try: self._srv.close()
            except Exception: pass

    def stop(self):
        try: self._srv.close()
        except Exception: pass


# ── Kafka producer thread ──────────────────────────────────────────────────────

def run_kafka_producer(broker, topic, producer, stop_event):
    try:
        from kafka import KafkaProducer
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError

        admin = KafkaAdminClient(bootstrap_servers=broker)
        try:
            admin.create_topics([NewTopic(topic, num_partitions=1,
                                          replication_factor=1)])
            log.info(f"Kafka topic '{topic}' created")
        except TopicAlreadyExistsError:
            log.info(f"Kafka topic '{topic}' exists")
        admin.close()

        kp = KafkaProducer(bootstrap_servers=broker)
        log.info(f"Kafka producer connected → {broker}")
        while not stop_event.is_set():
            for item in producer.drain():
                kp.send(topic, item)
            kp.flush()
            time.sleep(0.01)
        kp.close()
    except Exception as e:
        log.error(f"Kafka producer error: {e}")
        stop_event.set()


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


def _build_tensor2(rate_window, lag_window=None):
    """Normalise rate (and optional lag) windows into a model input tensor."""
    h   = np.array(rate_window, dtype=np.float32)
    mu  = h.mean()
    std = max(float(h.std()), mu * 0.05) + 1e-8
    norm_r = (h - mu) / std
    if lag_window is not None and len(lag_window) == len(rate_window):
        lag_arr = np.array(lag_window, dtype=np.float32)
        lag_std = float(lag_arr.std())
        norm_l  = np.zeros_like(lag_arr) if lag_std < 1e-6 \
                  else (lag_arr - lag_arr.mean()) / (lag_std + 1e-8)
        feat = np.stack([norm_r, norm_l], axis=-1)      # [K, 2]
        x = torch.FloatTensor(feat).unsqueeze(0)        # [1, K, 2]
    else:
        x = torch.FloatTensor(norm_r).unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
    return x, mu, std


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
            x, mu, std = _build_tensor2(win, lag_win if use_lag else None)
            for n, m in neural.items():
                with torch.no_grad():
                    errs[n].append(abs(max(m(x).item() * std + mu, 0.0) - float(vals[t + 1])))
    return {n: float(np.mean(v)) for n, v in errs.items()}


def load_neural_models(input_size=1):
    neural = {}
    for name, Cls in MODELS.items():
        path = MODEL_DIR / f"{name}_predictor.pt"
        if not path.exists():
            log.warning(f"  [skip] {name} — no checkpoint"); continue
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


# ── Evaluation loop ────────────────────────────────────────────────────────────

def run_evaluation(models, collector, duration, csv_path, use_lag=False):
    names     = list(models.keys())
    abs_errs  = {n: [] for n in names}
    mape_errs = {n: [] for n in names}
    dir_corr  = {n: [] for n in names}

    fields = (["timestamp", "window_len", "actual"] +
              [f"pred_{n}" for n in names] +
              [f"abserr_{n}" for n in names] +
              [f"mape_{n}" for n in names])

    log.info(f"Saving raw predictions → {csv_path}")
    csv_f  = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fields)
    writer.writeheader()

    # wait for window to fill
    log.info(f"Waiting for window to fill (K={K} batches)...")
    wait_start = time.time()
    while not collector.is_ready():
        if time.time() - wait_start > 120:
            log.error(
                "Window never filled after 120s.\n"
                "  Diagnostics:\n"
                f"  - history_len = {len(collector.get_history())}\n"
                "  - Is Spark producing data? Check if socket server connected.\n"
                "  - Is inputRowsPerSecond > 0? (0 on first batch is normal)\n"
                "  - Try increasing --duration or check Spark logs."
            )
            csv_f.close()
            return
        hist = collector.get_history()
        log.info(f"  history_len={len(hist)}/{K}" +
                 (f"  last={hist[-1]:.1f}" if hist else ""))
        time.sleep(3)

    log.info("Window full — evaluation running")
    eval_start = time.time()

    while time.time() - eval_start < duration:
        t0      = time.time()
        history = collector.get_history()
        if len(history) < K:
            time.sleep(1); continue

        lag_history = collector.get_lag_history() if use_lag else None
        x, mu, std  = _build_tensor2(history, lag_history)

        preds = {}
        for name, model in models.items():
            if getattr(model, "is_baseline", False):
                preds[name] = max(model.predict_raw(list(history)), 0.0)
            else:
                with torch.no_grad():
                    preds[name] = max(model(x).item() * std + mu, 0.0)

        time.sleep(max(0, 1.0 - (time.time() - t0)))

        new_hist = collector.get_history()
        if not new_hist: continue
        actual  = float(new_hist[-1])
        current = float(history[-1])
        dir_act = 1 if actual > current else -1

        row = {"timestamp": datetime.now().isoformat(),
               "window_len": len(history), "actual": f"{actual:.2f}"}
        for name, pred in preds.items():
            err  = abs(pred - actual)
            mape = err / (actual + 1e-8) * 100
            abs_errs[name].append(err)
            mape_errs[name].append(mape)
            dir_corr[name].append(
                int((1 if pred > current else -1) == dir_act)
            )
            row[f"pred_{name}"]   = f"{pred:.2f}"
            row[f"abserr_{name}"] = f"{err:.2f}"
            row[f"mape_{name}"]   = f"{mape:.2f}"
        writer.writerow(row); csv_f.flush()

        elapsed = time.time() - eval_start
        log.debug(f"t={elapsed:.0f}s  actual={actual:.1f}  "
                  f"lstm={preds.get('lstm',0):.1f}  "
                  f"ens_top3={preds.get('ens_top3',0):.1f}")

    csv_f.close()
    _print_table(names, abs_errs, mape_errs, dir_corr, models)


def _print_table(names, abs_errs, mape_errs, dir_corr, models):
    all_mae = {n: float(np.mean(abs_errs[n])) if abs_errs[n] else float("inf")
               for n in names}
    best    = min(all_mae, key=all_mae.get)

    print(f"\n{'─'*78}")
    print("  evaluate_stream2.py — Real Spark Job Results  (random Poisson traffic)")
    print(f"{'─'*78}")
    print(f"  {'Model':<14} {'MAE':>8} {'RMSE':>8} {'MAPE':>9} {'DirAcc':>9} {'n':>6}")
    print(f"  {'─'*62}")

    for n in sorted(names, key=lambda x: all_mae[x]):
        errs  = np.array(abs_errs[n]) if abs_errs[n] else np.array([0.0])
        mae   = float(np.mean(errs))
        rmse  = float(np.sqrt(np.mean(errs ** 2)))
        mape  = float(np.mean(mape_errs[n])) if mape_errs[n] else 0.0
        dacc  = float(np.mean(dir_corr[n])) * 100 if dir_corr[n] else 0.0
        star  = " ★" if n == best else ""
        print(f"  {n:<14} {mae:>8.2f} {rmse:>8.2f} {mape:>8.1f}% {dacc:>8.1f}%"
              f" {len(errs):>6}{star}")
    print(f"{'─'*78}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["socket", "kafka"], default="socket")
    parser.add_argument("--kafka-broker", default="localhost:9092")
    parser.add_argument("--kafka-topic",  default="adaptivestream-eval2")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--log-dir",  default=str(LOG_DIR))
    parser.add_argument("--lag", action="store_true",
                        help="Evaluate lag-aware models; simulates consumer lag from rate stream")
    args = parser.parse_args()

    log_dir  = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"eval_real_{RUN_TS}.log"
    csv_path = log_dir / f"eval_real_{RUN_TS}.csv"
    add_file_handler(log_path)

    log.info(f"Mode={args.mode}  Duration={args.duration}s  lag={args.lag}")
    log.info(f"Log: {log_path}")
    log.info(f"CSV: {csv_path}")

    # load models
    input_size = 2 if args.lag else 1
    log.info(f"Loading models (input_size={input_size})...")
    neural = load_neural_models(input_size=input_size)
    if not neural:
        log.error("No trained models found — run train.py first.")
        sys.exit(1)
    models = build_all_models(neural, use_lag=args.lag)
    log.info(f"{len(neural)} neural + 2 baselines + ensembles = {len(models)} total")

    # build Spark session
    spark_builder = (SparkSession.builder
                     .master("local[2]")
                     .appName("AdaptiveStreamEval2")
                     .config("spark.sql.shuffle.partitions", "2")
                     .config("spark.ui.enabled", "false"))
    if args.mode == "kafka":
        spark_builder = spark_builder.config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
        )
    spark = spark_builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # attach listener
    collector = RateCollector(window_size=K, simulate_lag=args.lag)
    spark.streams.addListener(collector)

    # producer + source
    producer   = PoissonProducer()
    stop_event = threading.Event()
    producer.start()

    if args.mode == "socket":
        host, port = "127.0.0.1", 9999
        server = SocketServer(host, port, producer)
        server.start()
        df = (spark.readStream.format("socket")
              .option("host", host).option("port", port).load())
    else:
        kt = threading.Thread(target=run_kafka_producer,
                              args=(args.kafka_broker, args.kafka_topic,
                                    producer, stop_event), daemon=True)
        kt.start()
        df = (spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", args.kafka_broker)
              .option("subscribe", args.kafka_topic)
              .option("startingOffsets", "latest").load())

    query = (df.writeStream
             .format("console")
             .option("truncate", False)
             .option("numRows", 1)
             .trigger(processingTime="1 second")
             .outputMode("append")
             .start())

    log.info("Spark streaming job started")

    try:
        run_evaluation(models, collector, args.duration, csv_path, use_lag=args.lag)
    finally:
        stop_event.set()
        producer.stop()
        if args.mode == "socket":
            server.stop()
        query.stop()
        spark.stop()

    summary = producer.rate_summary()
    if summary:
        log.info(
            f"Producer: min={summary['min']:.0f}  max={summary['max']:.0f}  "
            f"mean={summary['mean']:.0f}  p25={summary['p25']:.0f}  "
            f"p75={summary['p75']:.0f} events/s"
        )
    log.info(f"Raw data saved → {csv_path}")


if __name__ == "__main__":
    main()
