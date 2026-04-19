# AdaptiveStream ML — Strategy & Roadmap

---

## Current State (after Training Run 1)

Best model: `ens_top3` (LSTM + GRU + TiDE), val_MAE=28.49, DirAcc=72.4%  
Evaluation: synthetic holdout only (seeds 169–191, 6,210 steps)  
Deployment: Java agent + Python predictor server, tested via `demo_spark_job.py` (smoke test only — no burst variation, no accuracy measurement)

---

## The 9-Step Roadmap

| Step | What | Why |
|------|------|-----|
| 1 | `evaluate_stream2.py` — real Spark job evaluation | Ground truth: does the model work on real `inputRowsPerSecond`? |
| 2 | `run.py` — automate train → eval1 → eval2 | Reproducibility, one-command pipeline |
| 3 | Logging — structured logs + CSV for every run | Debug failures, validate results make sense |
| 4 | Run `evaluate_stream2.py` standalone | Establish real-world baseline before any model changes |
| 5 | Add Kafka lag as second input feature | Leading indicator — closes the distribution shift gap |
| 6 | Run `run.py` (train + both evals) | Full end-to-end test of lag feature |
| 7 | Fix issues via logs, validate consistency | Results must be consistent across eval1 and eval2 |
| 8 | `data2.py` — log-uniform baseline synthetic dataset | Scale invariance; fixes model generalising to any Spark deployment |
| 9 | Repeat step 6–7 with data2.py | Validate improvements hold on both synthetic and real evaluations |

---

## Step 1–3 Design

### evaluate_stream2.py — Real Spark Evaluation

**Source:** Spark Structured Streaming reading from a socket server (or Kafka with `--kafka-broker` flag).

**Why socket (default):** No external broker needed. Spark's socket source reports real `inputRowsPerSecond` via the REST API — same metric the agent uses in production. Kafka mode uses docker-compose.yml (Confluent 7.5.0, port 9092).

**Producer:** Genuinely random variable-rate traffic — not our synthetic shapes.
- Random walk in log-rate space: `log_rate += N(0, 0.15)` every 100ms
- Rate range: 1 – 10,000 events/second (log-uniform coverage)
- New random seed every run — never reproducible, never our training distribution
- Poisson process: inter-arrival counts sampled from `Poisson(rate * tick)` — models real traffic arrival statistics

**Evaluation loop:**
1. Fill 30-step window from real Spark metrics
2. At each second t: snapshot window → run all models → record predictions
3. At t+1: observe actual `inputRowsPerSecond` → compute errors
4. Metrics: MAE, RMSE, MAPE, DirAcc (same definitions as eval1)
5. Duration: configurable (default 120s, recommended 300s for statistical significance)

**Output:**
- `logs/eval_real_{timestamp}.csv` — raw (timestamp, actual, pred_lstm, pred_gru, ...) for every step
- Console table: all 9 models + 5 ensembles + 2 baselines, sorted by MAE
- Separate column: MAPE — the metric that makes cross-domain comparison valid

### run.py — Pipeline Orchestrator

```
python run.py [--skip-train] [--skip-eval1] [--skip-eval2] [--duration 120]
```

- Creates `logs/run_{timestamp}/` for each run
- Each step tees to both console and `{step}.log`
- Checks exit codes — stops and reports clearly on failure
- Final summary table across all steps

### Logging design

Every script writes:
```
logs/
  run_{timestamp}/
    train.log          — epoch logs, val_MAE at checkpoints, early stop events
    eval_synthetic.log — evaluate_stream.py full output
    eval_real.log      — evaluate_stream2.py full output
    eval_real_data.csv — raw step-by-step predictions for post-hoc analysis
```

If results don't make sense, check:
- `eval_real_data.csv`: are actuals varying? (if constant → socket server not sending variable rates)
- `eval_real.log`: does `history_len` grow? (if stuck at 0 → metrics_collector not polling)
- `train.log`: did early stopping fire too early? (check val_MAE at each checkpoint)

---

## Step 5 — Kafka Lag as Second Input Feature

From `results.md` Analysis section. Summary:

**What lag is:**
```
lag(t) = latestOffset(t) - consumerOffset(t)
```
Available from Spark REST API (`sources[].latestOffset`, `sources[].endOffset`) before the next batch starts. It is a present-state signal: it tells you whether the system is falling behind right now, regardless of workload domain.

**Why it helps:**
- `inputRowsPerSecond` is lagging — it tells you what Spark just processed
- Lag is a leading indicator — growing lag means producer is faster than consumer, next batch will be larger
- A model trained on (rate, lag) pairs generalises across Spark deployments because lag carries causal information that transcends the specific workload distribution

**No data leakage:**
```
Input: [(rate(t-29), lag(t-29)), ..., (rate(t), lag(t))]  →  predict rate(t+1)
```
`lag(t)` is derived only from `rate(0)...rate(t)`. Clean.

**Changes required:**
1. `data.py` — add `generate_series_with_lag()`: simulate producer-consumer dynamics alongside burst series
2. `models.py` — add `input_size` param to all architectures (default stays 1 for backwards compat); change `[K,1]` → `[K,2]` when `input_size=2`
3. `train.py` — build 2-feature dataset; normalise rate and lag independently
4. `metrics_collector.py` — parse `sources[].latestOffset` and `sources[].endOffset`; compute lag per partition, sum; lag=0 fallback for non-Kafka sources
5. `predictor_server.py` — build `[K,2]` tensor; lag=0 fallback when lag unavailable
6. `evaluate_stream.py` — simulate lag alongside rate in holdout evaluation
7. `evaluate_stream2.py` — collect real lag from Spark REST API (available when using Kafka source)

---

## Step 8 — Synthetic Dataset Improvements (data2.py)

### Problem: current dataset has wrong scale distribution

Every training series uses `baseline=100`. At inference time on a real Spark job running at 10 rows/s or 50,000 rows/s, the per-window normalisation is:
```python
std = max(h.std(), mu * 0.05) + 1e-8
```
The floor `mu * 0.05` is proportional to `mu`. If `noise_std` is not scaled with baseline, high-scale windows have their actual std dwarfed by the floor → every window looks artificially flat → model gives wrong predictions.

### Fix 1: Log-uniform baseline (scale invariance)

```python
rng = np.random.default_rng(seed)
baseline  = float(10 ** rng.uniform(1.0, 5.7))   # 10 → ~500k, log-uniform
noise_std = baseline * 0.10                        # keep relative noise at 10%
```

**Why log-uniform, not uniform:**  
`uniform(10, 500_000)` → 99.8% of samples have baseline > 1,000 → model almost never sees Spark-scale data (50–500 rows/s). Log-uniform gives equal representation at each order of magnitude: 10–100, 100–1k, 1k–10k, 10k–500k each get ~25% of series.

**Why noise_std must scale:**  
Without scaling, at baseline=300k with noise_std=10: `actual_std=10, floor=15,000` → divided by 15,000 → every window is perfectly flat. The model has never seen this at training time → wrong predictions at high-scale deployments.

**What this fixes:**
- Spark deployments at any scale (10 rows/s to 500k rows/s) — primary use case
- The std floor behaving consistently at inference time across all scales

**What this does NOT fix:**
- Wikipedia temporal dynamics (slow daily trends vs second-level bursts) — different problem
- The evaluation metric for Wikipedia: use MAPE, not raw MAE, for cross-domain comparison

### Fix 2: Drop `plateau` shape

After per-window normalisation, `plateau` (sustained level, σ=0.3×normal) is nearly identical to `noise` (jittery baseline): both look like near-flat lines after subtracting window mean. The 10% training probability is wasted. Redistribute to `wall` and `cliff` (hardest shapes in eval: MAE=65 and MAE=54 respectively, highest error regime).

```python
# data2.py
SHAPES       = ["noise", "ramp", "wall", "cliff", "double_peak", "sawtooth"]
_SHAPE_PROBS = [0.25,    0.15,   0.15,   0.15,    0.15,          0.15]
```

### Fix 3: Larger dataset

| | data.py (current) | data2.py (proposed) |
|--|-------------------|---------------------|
| N_SERIES | 150 | 500 |
| Steps/series | 300 | 600 |
| Total timesteps | 45,000 | 300,000 |
| Train windows | ~28,350 | ~189,000 |
| Scale range | 100 only | 10 – 500k (log-uniform) |
| Series/decade | ~30 | ~100 |

---

## Why Wikipedia Failed — and What "Fixing" It Actually Means

### What we tested vs what we should have tested

We pulled daily aggregated counts from the Wikipedia Pageviews API. A 30-step window covered 30 days. Wikipedia's day-to-day autocorrelation is ~0.95 → EMA naturally wins because today ≈ yesterday.

We should have tested on raw per-second event rates from a Spark job processing Wikipedia-style traffic. A viral article spike at the event level looks exactly like our synthetic `wall` or `cliff` shape — sudden burst completing in seconds, not a 30-day smooth trend.

### Two separate problems, two separate fixes

| Problem | Fix |
|---------|-----|
| Scale incomparability (50k MAE vs 50 MAE) | MAPE metric — percentage-based, comparable across domains |
| Temporal dynamics (daily cycles vs second-level bursts) | Add slow-trend and diurnal shapes to data.py — out of scope for now |

MAPE is worth adding for all evaluations regardless. The temporal dynamics gap is only relevant if we want the model to predict slowly-evolving workloads — not the current use case.

### The correct Wikipedia-equivalent benchmark

A Spark job reading raw Wikipedia edit events or access logs from Kafka at per-second granularity. The `inputRowsPerSecond` from that job would show burst patterns identical to our synthetic shapes. `evaluate_stream2.py` with a Kafka source feeding Wikipedia-style bursty traffic is the correct test.
