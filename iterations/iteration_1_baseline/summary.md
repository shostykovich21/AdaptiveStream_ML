# Iteration 1 — Baseline

**Date:** 2026-04-17 (training) / 2026-04-19 (real eval)
**Features:** rate only [K=30, 1 feature]
**Dataset:** 150 series × 300 steps, baseline=100 (fixed scale), 7 shapes
**Models:** 9 neural + 5 ensembles + 2 baselines

---

## Training Summary (val_MAE, best checkpoint)

| Rank | Model   | val_MAE | val_DirAcc | Converged |
|------|---------|---------|------------|-----------|
| 1    | LSTM    | 0.4040  | 72.5%      | 150s (still improving) |
| 2    | GRU     | 0.4139  | 70.5%      | 150s (still improving) |
| 3    | TiDE    | 0.4342  | 69.9%      | ~46s early stop |
| 4    | TCN     | 0.4472  | 70.3%      | ~69s early stop |
| 5    | MLP     | 0.4505  | 69.1%      | ~31s early stop |
| 6    | N-BEATS | 0.4626  | 68.4%      | ~31s early stop |
| 7    | Attn    | 0.4713  | 68.8%      | ~80s early stop |
| 8    | DLinear | 0.5453  | 56.3%      | ~50s early stop |
| 9    | FITS    | 0.6032  | 52.0%      | ~61s early stop |

---

## Synthetic Holdout Results (evaluate_stream.py)

Seeds 169–191 · 23 series · 6,210 steps

| Model       |  MAE  |  RMSE  | DirAcc |
|-------------|-------|--------|--------|
| ens_top3    | 28.49 | 81.36  | 72.4% ★ |
| ens_rnn     | 28.53 | 82.27  | 72.7% |
| lstm        | 28.64 | 82.82  | 72.2% |
| gru         | 30.90 | 84.06  | 70.4% |
| ens_diverse | 31.27 | 85.91  | 71.5% |
| tide        | 31.43 | 84.23  | 70.3% |
| ens_wtd     | 32.23 | 86.00  | 71.9% |
| ens_mean    | 33.49 | 87.63  | 71.7% |
| mlp         | 35.55 | 90.56  | 68.6% |
| tcn         | 35.99 | 98.99  | 70.8% |
| nbeats      | 36.65 | 90.19  | 68.9% |
| attn        | 38.23 | 98.94  | 67.3% |
| dlinear     | 46.46 | 108.31 | 55.3% |
| ema         | 55.30 | 112.87 | 44.6% |
| fits        | 56.80 | 115.82 | 51.2% |
| sma         | 151.00| 205.69 | 52.8% |

---

## Real Spark Job Results (evaluate_stream2.py)

Socket mode · 120 steps · Random Poisson traffic
Producer: min=1, max=151, mean=19, p25=4, p75=32 events/s

| Model       |  MAE  |  RMSE  |  MAPE  | DirAcc |
|-------------|-------|--------|--------|--------|
| ema         |  6.86 | 10.49  | 53.6%  | 61.7% ★ |
| dlinear     |  7.09 | 10.67  | 58.0%  | 51.7% |
| tide        |  7.20 | 10.80  | 65.8%  | 55.8% |
| attn        |  7.25 | 10.77  | 68.7%  | 56.7% |
| fits        |  7.27 | 10.76  | 68.7%  | 57.5% |
| ens_mean    |  7.31 | 10.72  | 69.0%  | 52.5% |
| mlp         |  7.34 | 11.28  | 67.5%  | 60.0% |
| ens_wtd     |  7.38 | 10.84  | 70.4%  | 55.0% |
| nbeats      |  7.39 | 11.14  | 74.8%  | 50.8% |
| tcn         |  7.39 | 10.95  | 73.4%  | 55.0% |
| ens_diverse |  7.47 | 10.88  | 72.9%  | 55.0% |
| ens_top3    |  7.85 | 11.60  | 78.1%  | 55.8% |
| gru         |  8.21 | 12.29  | 82.5%  | 48.3% |
| ens_rnn     |  8.36 | 12.39  | 87.2%  | 51.7% |
| lstm        |  8.89 | 13.15  | 95.3%  | 50.8% |
| sma         | 11.05 | 17.89  | 134.2% | 57.5% |

---

## Key Observations

1. **Synthetic**: Neural models win clearly. ens_top3 best (MAE=28.49), EMA worst (MAE=55.30).
2. **Real Spark**: EMA wins (MAE=6.86). Neural models lose because traffic was near-constant (mean=19 events/s, p75=32). No real bursts to differentiate models.
3. **Gap**: The real Spark producer generated slow random walk traffic, not burst patterns. Neural models were trained on burst shapes. EMA exploits persistence — it wins on any slow-moving signal.
4. **Fix**: Kafka lag feature should give neural models a leading indicator that EMA doesn't have. Log-uniform training broadens the distribution.
