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

## Commands Used

```bash
# Training + synthetic eval (via run.py)
python run.py --lag --log-uniform --iteration 3 --iteration-name log_uniform --skip-eval2

# Real Spark eval (separate run)
python run.py --lag --log-uniform --iteration 3 --iteration-name log_uniform --skip-train --skip-eval1
# which internally runs:
# python predictor/evaluate_stream2.py --lag --duration 120 --log-dir iterations/iteration_3_log_uniform
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
