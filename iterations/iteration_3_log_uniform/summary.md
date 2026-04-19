# Iteration 3 — Log-Uniform Baseline

**Date:** 2026-04-19
**Features:** rate + simulated Kafka lag [K=30, 2 features]
**Dataset:** 500 series × 600 steps, baseline = 10^Uniform(1.0, 5.7) ∈ [10, 500k]
**Models:** 9 neural + 5 ensembles + 2 baselines
**Change from iter 2:** Switched to log-uniform baseline (data2.py)
  - Equal training mass per decade: 10–100, 100–1k, 1k–10k, 10k–500k each ~25%
  - noise_std = baseline × 0.1 (noise scales with signal, never drowns it)
  - Plateau shape removed (pure hold — no transition edges, easy for EMA too)
  - Wall/cliff probabilities increased

---

## Training Summary (val_MAE, best checkpoint @150s)

| Rank | Model   | val_MAE | val_DirAcc | Converged |
|------|---------|---------|------------|-----------|
| 1    | TiDE    | 0.435   | 77%        | ~79s early stop |
| 2    | N-BEATS | 0.437   | 77%        | ~71s early stop |
| 3    | LSTM    | 0.440   | 75%        | 150s (still improving) |
| 4    | GRU     | 0.450   | 75%        | 150s (still improving) |
| 5    | MLP     | 0.456   | 75%        | ~50s early stop |
| 6    | Attn    | 0.476   | 73%        | 150s |
| 7    | TCN     | 0.474   | 73%        | 150s |
| 8    | DLinear | 0.561   | 60%        | ~40s early stop |
| 9    | FITS    | 0.596   | 57%        | ~55s early stop |

---

## Synthetic Holdout Results (evaluate_stream.py, log-uniform holdout)

Seeds 169–191 · 23 series · 13,110 steps
⚠️ Raw MAE inflated by high-scale series (baselines up to 500k ev/s). DirAcc only.

| Model       |   MAE   |   RMSE   | DirAcc |
|-------------|---------|----------|--------|
| tide        |  6,696  | 47,854   | 77.5% ★ |
| nbeats      |  6,764  | 47,181   | 77.2% |
| ens_top3    |  6,850  | 48,327   | 77.2% |
| ens_diverse |  7,023  | 48,376   | 77.3% |
| ens_wtd     |  7,150  | 48,402   | 77.5% |
| ens_rnn     |  7,188  | 49,182   | 76.3% |
| lstm        |  7,221  | 48,786   | 75.6% |
| ens_mean    |  7,368  | 48,737   | 77.0% |
| mlp         |  7,460  | 49,113   | 76.0% |
| gru         |  7,609  | 50,885   | 75.1% |
| attn        |  7,907  | 50,866   | 74.6% |
| tcn         |  7,914  | 50,092   | 73.6% |
| dlinear     | 10,219  | 53,607   | 60.4% |
| fits        | 11,642  | 56,162   | 56.1% |
| ema         | 13,056  | 60,169   | 39.6% ← significantly below random |
| sma         | 43,220  | 149,548  | 51.4% |

---

## Real Spark Job Results (evaluate_stream2.py)

Socket mode · 120 steps · Random Poisson traffic
Producer: min=6, max=10,000, mean=1,292, p25=22, p75=1,228 events/s
Lag: simulated (capacity = rolling_mean × 1.2)

| Model       |  MAE  |  RMSE  |  MAPE  | DirAcc |
|-------------|-------|--------|--------|--------|
| lstm        |  229  |   553  | 30.8%  | 55.0% ★ |
| ens_rnn     |  238  |   559  | 30.8%  | 55.8% |
| ens_mean    |  239  |   580  | 35.4%  | 59.2% |
| dlinear     |  241  |   572  | 32.8%  | 50.8% |
| ens_wtd     |  242  |   581  | 36.0%  | 58.3% |
| ens_top3    |  244  |   569  | 33.5%  | 56.7% |
| ens_diverse |  247  |   586  | 36.3%  | 60.0% |
| fits        |  251  |   596  | 40.4%  | 49.2% |
| gru         |  252  |   583  | 31.9%  | 54.2% |
| mlp         |  255  |   587  | 41.7%  | 53.3% |
| nbeats      |  257  |   613  | 44.6%  | 50.8% |
| tide        |  260  |   597  | 42.0%  | 48.3% |
| tcn         |  265  |   632  | 36.3%  | 54.2% |
| attn        |  267  |   607  | 43.4%  | 55.0% |
| ema         |  271  |   608  | 30.4%  | 51.7% |
| sma         |  874  | 1,888  | 75.1%  | 44.2% |

---

## Burst Traffic Results (evaluate_stream3.py)

evaluate_stream3.py — real Spark job, burst-shaped traffic (data2.py holdout seeds 169–191).
Producer: min=13, max=2000, mean=393, p25=57, p75=531 events/s · 120s · 151 Spark batches

### Table 3 — Option A: Replay (offline sliding-window on Spark-measured burst rates)

| Model       |  MAE  |  RMSE  |  MAPE  | DirAcc |
|-------------|-------|--------|--------|--------|
| ema         | 179.1 |  324.9 |  84.8% | 60.0% ★ |
| dlinear     | 200.8 |  356.3 | 110.1% | 44.2% |
| ens_mean    | 208.3 |  347.7 | 148.2% | 53.3% |
| ens_top3    | 209.8 |  364.4 | 124.1% | 54.2% |
| ens_wtd     | 210.5 |  348.8 | 152.0% | 53.3% |
| ens_rnn     | 211.7 |  370.9 | 124.8% | 51.7% |
| lstm        | 212.7 |  389.6 | 111.8% | 52.5% |
| ens_diverse | 213.1 |  354.3 | 154.3% | 56.7% |
| fits        | 214.7 |  366.5 | 124.3% | 50.8% |
| gru         | 217.8 |  364.7 | 152.8% | 55.8% |
| attn        | 220.8 |  354.3 | 203.2% | 56.7% |
| tcn         | 223.1 |  346.0 | 198.7% | 55.0% |
| mlp         | 225.8 |  349.2 | 201.9% | 55.8% |
| tide        | 235.0 |  369.4 | 173.1% | 50.0% |
| nbeats      | 240.4 |  365.0 | 205.7% | 52.5% |
| sma         | 253.2 |  361.8 | 322.4% | 61.7% |

### Table 4 — Option B: Live (real-time inference during Spark run)

| Model       |  MAE  |  RMSE  |  MAPE  | DirAcc |
|-------------|-------|--------|--------|--------|
| ema         | 178.6 |  324.9 |  84.5% | 55.8% ★ |
| dlinear     | 199.4 |  356.2 | 114.0% | 45.0% |
| ens_mean    | 207.6 |  347.7 | 151.8% | 50.0% |
| ens_top3    | 208.5 |  364.3 | 127.5% | 51.7% |
| ens_wtd     | 209.7 |  348.8 | 155.3% | 50.0% |
| ens_rnn     | 210.1 |  370.8 | 128.0% | 51.7% |
| lstm        | 210.9 |  389.5 | 115.3% | 54.2% |
| ens_diverse | 212.7 |  354.4 | 157.4% | 52.5% |
| fits        | 215.0 |  367.2 | 138.2% | 51.7% |
| gru         | 216.7 |  364.6 | 154.9% | 53.3% |
| attn        | 221.0 |  354.4 | 207.0% | 52.5% |
| tcn         | 223.6 |  346.3 | 199.4% | 50.0% |
| mlp         | 226.5 |  349.5 | 204.9% | 51.7% |
| tide        | 234.7 |  369.3 | 175.9% | 45.8% |
| nbeats      | 239.1 |  365.0 | 206.3% | 50.0% |
| sma         | 256.6 |  363.1 | 328.5% | 56.7% |

---

## Trigger Policy Results (evaluate_stream4.py)

Simulated on 100ms-resolution tick trace from BurstProducer · 180s · 1803 ticks
Producer: min=13, max=2000, mean=441, p25=77, p75=618 ev/s
Formula: `intervalMs = clamp(500 / pred_ev_s × 1000, 100, 5000)`

### Table 5 — Fixed vs Adaptive Trigger Policy

| Policy           | Triggers | Empty% | MeanBatch | Throughput | PeakBacklog | ClearanceT |
|------------------|----------|--------|-----------|------------|-------------|------------|
| fixed_100ms      |    1802  |   0.0% |      44.1 |       44.1 |         200 |      61.0s |
| fixed_500ms      |     360  |   0.0% |     220.9 |      220.9 |         983 |      94.0s |
| fixed_1000ms     |     180  |   0.0% |     441.8 |      441.8 |        1964 |      47.0s |
| fixed_2000ms     |      90  |   0.0% |     883.6 |      883.6 |        2915 |          ∞ |
| adaptive_ema     |     131  |   0.0% |     606.4 |      606.4 |        2380 |       7.0s |
| adaptive_lstm    |     132  |   0.0% |     601.5 |      601.5 |        4248 |       1.0s ★ |
| adaptive_ens_top3|     158  |   0.0% |     503.4 |      503.4 |        2257 |      36.0s |

---

## Commands Used

```bash
# Training + synthetic eval (via run.py)
python run.py --lag --log-uniform --iteration 3 --iteration-name log_uniform --skip-eval2

# Real Spark eval (random-walk traffic)
python run.py --lag --log-uniform --iteration 3 --iteration-name log_uniform --skip-train --skip-eval1
# which internally runs:
# python predictor/evaluate_stream2.py --lag --duration 120 --log-dir iterations/iteration_3_log_uniform

# Burst-traffic eval (Tables 3 + 4)
python predictor/evaluate_stream3.py --lag --duration 120 --log-dir iterations/iteration_3_log_uniform

# Trigger policy comparison (Table 5)
python predictor/evaluate_stream4.py --duration 180 --log-dir iterations/iteration_3_log_uniform
```

---

## Learnings

- **Log-uniform training fixed scale coverage.** DirAcc jumped from 72% (iter1/2) to
  77.5% (iter3). Models now handle burst patterns across 10–500k ev/s, not just around
  100 ev/s. This was the highest-impact single change across all three iterations.

- **Raw MAE is not comparable across iterations.** Iter3 holdout baselines reach 500k
  ev/s — a 1% error at that scale is an absolute error of 5,000, versus ~1 at baseline=100.
  DirAcc is the only valid cross-iteration metric. All further comparison should use it.

- **EMA DirAcc dropped to 39.6% on burst traffic** (below random = 50%). This is the
  clearest evidence yet that neural models learn something EMA cannot: burst transition
  direction. EMA consistently guesses wrong at peak turnovers, cliff drops, and sawtooth
  resets — exactly the transitions that matter for adaptive trigger scheduling.

- **LSTM beat EMA on real Spark for the first time (229 vs 271 MAE) — but for the
  wrong reason.** The producer this run hit mean=1,292 ev/s. Iter1/2 models (trained at
  baseline=100) would produce predictions near 100 while actuals are near 1,000 — log-
  uniform models are in-distribution at that scale. This is a scale coverage win,
  not a burst recognition win. The producer still generates random-walk traffic.

- **TiDE: best in training (#1, val_MAE=0.435) but 12th on real eval.** Architectures
  that fit burst shapes tightly in training may not generalise well to random-walk
  traffic. LSTM's inductive bias (sequential state) adapts more smoothly to novel
  temporal patterns. Architecture rankings from synthetic eval don't transfer directly
  to real eval when the traffic distributions differ.

- **The unanswered question:** Do neural models beat EMA when real Spark traffic
  actually has burst structure? No eval has tested this yet. evaluate_stream.py uses
  burst traffic but fake infrastructure. evaluate_stream2.py uses real infrastructure
  but random-walk traffic. evaluate_stream3.py needs to do both:
  - **Option A (gold standard):** replay a real historical inputRowsPerSecond trace —
    no distribution assumptions, true production signal.
  - **Option B (burst producer):** feed data2.py burst shapes into the live Spark job
    (held-out seeds, never seen in training). Infrastructure is real; traffic structure
    approximates production. Not perfectly fair to EMA (neural models trained on same
    vocabulary) but tests whether pattern recognition survives real batch timing and
    inference latency.
