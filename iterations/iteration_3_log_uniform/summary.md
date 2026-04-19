# Iteration 3 — Log-Uniform Baseline

**Date:** 2026-04-19
**Features:** rate + simulated Kafka lag [K=30, 2 features]
**Dataset:** 500 series × 600 steps, baseline=10^Uniform(1.0,5.7) ∈ [10, 500k]
**Models:** 9 neural + 5 ensembles + 2 baselines
**Change from iter 2:** Switched from fixed baseline=100 to log-uniform baseline (data2.py)
  - Equal training mass per decade: 10–100, 100–1k, 1k–10k, 10k–500k each ~25%
  - noise_std = baseline × 0.1 (scales with signal)
  - Plateau shape removed (too easy, no edge transitions)
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
⚠️ Raw MAE not comparable to iter1/2 — holdout baselines up to 500k events/s.
DirAcc is the correct cross-iteration metric (scale-invariant).

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

**EMA DirAcc=39.6% proves neural models learn burst transitions; EMA fails them.**

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
| **ema**     | **271** | **608** | **30.4%** | **51.7%** |
| sma         |  874  | 1,888  | 75.1%  | 44.2% |

**LSTM beats EMA for the first time on real Spark traffic (229 vs 271 MAE).**

---

## Progression Across Iterations

| Metric | Iter 1 (rate-only) | Iter 2 (+lag) | Iter 3 (log-uniform) |
|--------|-------------------|---------------|----------------------|
| Synthetic DirAcc (best) | 72.4% | 72.1% | **77.5%** |
| Synthetic EMA DirAcc | 44.6% | 44.6% | **39.6%** (diverging) |
| Real EMA MAE | 6.86 | 11.95 | **271** |
| Real LSTM MAE | 8.89 | 28.04 | **229** (beats EMA) |
| Real producer mean rate | 19 ev/s | 107 ev/s | **1,292 ev/s** |

Note: Raw MAE is not directly comparable across iters — different producer runs and training scales.

---

## Key Observations

1. **Synthetic DirAcc**: Improved from 72.1% → 77.5% with log-uniform training. EMA's DirAcc dropped from 44.6% → 39.6% — neural models now learn burst transitions that EMA cannot handle.
2. **Real Spark**: First iteration where neural models beat EMA (LSTM 229 vs EMA 271 MAE). The producer this run reached max=10,000 events/s which is within the log-uniform training range.
3. **Why it worked**: Log-uniform training covers 10–500k events/s. When the Spark producer generates high-rate traffic (mean=1,292 this run), iter2 models (trained on baseline=100) were out-of-distribution; iter3 models handled it.
4. **What's next**: The real Spark eval still uses random-walk traffic. To make the distinction clearer, the producer should generate burst-shaped traffic (ramps, spikes). Also, TiDE which was best in training (#1) ranks 12th on real eval — training on log-uniform is right but TiDE may over-fit burst shapes vs LSTM's broader generalization.
