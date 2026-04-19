# Iteration 2 — Kafka Lag Feature

**Date:** 2026-04-19
**Features:** rate + simulated Kafka lag [K=30, 2 features]
**Dataset:** 150 series × 300 steps, baseline=100 (same as iter 1)
**Models:** 9 neural + 5 ensembles + 2 baselines
**Change from iter 1:** Added Kafka lag as 2nd input feature (input_size=2)
  - Lag model: `lag[t] = max(0, lag[t-1] + rate[t] - capacity)`, `capacity = baseline * 1.2`
  - Lag normalised per-window independently of rate

---

## Training Summary (val_MAE, best checkpoint)

| Rank | Model   | val_MAE | val_DirAcc | Converged |
|------|---------|---------|------------|-----------|
| 1    | LSTM    | 0.4044  | 73.5%      | 150s (still improving) |
| 2    | TiDE    | 0.4163  | 71.4%      | ~62s early stop |
| 3    | GRU     | 0.4171  | 70.9%      | 150s (still improving) |
| 4    | MLP     | 0.4228  | 71.9%      | ~46s early stop |
| 5    | Attn    | 0.4471  | 69.3%      | ~96s early stop |
| 6    | TCN     | 0.4527  | 70.3%      | ~68s early stop |
| 7    | N-BEATS | 0.4375  | 70.0%      | ~43s early stop |
| 8    | DLinear | 0.5493  | 57.6%      | ~40s early stop |
| 9    | FITS    | 0.5951  | 53.5%      | ~61s early stop |

---

## Synthetic Holdout Results (evaluate_stream.py)

Seeds 169–191 · 23 series · 6,210 steps

| Model       |  MAE  |  RMSE  | DirAcc |
|-------------|-------|--------|--------|
| ens_top3    | 27.64 | 80.78  | 72.1% ★ |
| ens_rnn     | 27.83 | 82.09  | 71.9% |
| lstm        | 28.06 | 82.08  | 71.6% |
| gru         | 29.83 | 84.32  | 70.7% |
| tide        | 29.97 | 81.52  | 70.5% |
| ens_diverse | 30.49 | 85.75  | 72.3% |
| ens_wtd     | 30.89 | 84.36  | 72.7% |
| nbeats      | 31.23 | 85.40  | 69.5% |
| ens_mean    | 32.33 | 86.13  | 72.3% |
| mlp         | 32.39 | 86.04  | 70.8% |
| attn        | 36.38 | 102.43 | 68.0% |
| tcn         | 37.08 | 101.27 | 70.0% |
| dlinear     | 47.04 | 107.31 | 57.5% |
| ema         | 55.30 | 112.87 | 44.6% |
| fits        | 56.01 | 112.38 | 51.1% |
| sma         | 151.00| 205.69 | 52.8% |

---

## Real Spark Job Results (evaluate_stream2.py)

Socket mode · 120 steps · Random Poisson traffic
Producer: min=1, max=1084, mean=107, p25=6, p75=126 events/s
Lag: **simulated** (capacity = rolling_mean × 1.2)

| Model       |  MAE  |  RMSE  |  MAPE  | DirAcc |
|-------------|-------|--------|--------|--------|
| ema         | 11.95 | 20.96  | 47.7%  | 46.7% ★ |
| dlinear     | 16.84 | 26.16  | 81.9%  | 47.5% |
| fits        | 18.23 | 30.37  | 88.7%  | 50.8% |
| ens_top3    | 20.03 | 33.35  | 97.2%  | 53.3% |
| gru         | 20.47 | 35.24  | 80.9%  | 47.5% |
| ens_rnn     | 22.76 | 37.54  | 89.5%  | 50.8% |
| ens_mean    | 23.57 | 39.81  | 123.7% | 49.2% |
| ens_wtd     | 23.81 | 40.78  | 125.8% | 50.0% |
| tide        | 24.84 | 43.70  | 140.9% | 48.3% |
| lstm        | 28.04 | 48.76  | 119.1% | 55.0% |
| mlp         | 29.37 | 51.00  | 154.6% | 55.8% |
| ens_diverse | 31.47 | 55.51  | 158.2% | 50.8% |
| nbeats      | 33.13 | 56.52  | 172.4% | 44.2% |
| tcn         | 36.01 | 65.57  | 175.7% | 49.2% |
| attn        | 43.58 | 77.24  | 209.8% | 49.2% |
| sma         | 66.13 | 120.07 | 269.8% | 50.8% |

---

## vs. Iteration 1 (rate-only baseline)

| Metric | iter1 (rate only) | iter2 (rate+lag) | Δ |
|--------|-------------------|-------------------|---|
| Synthetic ens_top3 MAE | 28.49 | 27.64 | **-0.85** |
| Synthetic lstm MAE | 28.64 | 28.06 | **-0.58** |
| Real EMA MAE | 6.86 | 11.95 | +5.09 (different run) |

Note: Real Spark MAE not directly comparable — different producer run each time.

---

## Key Observations

1. **Synthetic**: Lag feature gives a small but consistent improvement. ens_top3 improves from 28.49 → 27.64 MAE.
2. **Real Spark**: EMA still wins on random-walk Poisson traffic. The simulated lag in eval2 is derived from the same rate stream (no independent signal), so it provides no additional information to neural models.
3. **Root cause of real-eval gap**: Poisson random-walk traffic has no burst structure — neural models trained on ramps/walls/sawteeth don't generalise. EMA exploits temporal persistence.
4. **Fix**: Iteration 3 — log-uniform synthetic training (covers 10→500k events/s). Plus longer, more variable Spark producer.
