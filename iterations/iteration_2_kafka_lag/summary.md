# Iteration 2 — Kafka Lag Feature

**Date:** 2026-04-19
**Features:** rate + simulated Kafka lag [K=30, 2 features]
**Dataset:** 150 series × 300 steps, fixed baseline=100
**Models:** 9 neural + 5 ensembles + 2 baselines
**Change from iter 1:** Added Kafka lag as 2nd input feature (input_size=2)
  - Lag model: `lag[t] = max(0, lag[t-1] + rate[t] - capacity)`, `capacity = baseline × 1.2`
  - Lag normalised per-window independently of rate

---

## What the Two Evaluations Tell Us

### evaluate_stream.py — Synthetic Holdout

Same interpretation as iter1: tests whether models generalise within the synthetic
burst distribution. The lag feature adds a second channel that reflects accumulated
backpressure during bursts — in theory, a leading indicator of burst magnitude.

**What the results show:** Small but consistent improvement. ens_top3 MAE improved
28.49 → 27.64, LSTM improved 28.64 → 28.06. DirAcc held at ~72%. The lag feature
helps but modestly — the rate channel already carries most of the structural signal.

**Why the improvement is small:** In synthetic data, the lag is derived from the same
rate series (it's simulated, not independently observed). The model is essentially
getting a filtered, integrated version of the rate signal as a second channel. Real
Kafka lag would be independently measured from broker offset tracking and would be a
genuinely leading indicator — it starts rising before inputRowsPerSecond does.

### evaluate_stream2.py — Real Spark, Random-Walk Traffic

The infrastructure is real (live Spark, StreamingQueryListener). The lag here is
also simulated — derived from the same rate stream inside the collector using the
capacity model. It provides no independent signal.

**What the results show:** EMA wins again (MAE=11.95 vs LSTM MAE=28.04). The producer
this run generated mean=107 ev/s — still random-walk, no burst structure.

**Why neural models got worse vs iter1 on this eval:** Iter1 models trained on
baseline=100 were approximately in-distribution for 107 ev/s traffic. The lag feature
added a second channel that gave no useful signal on random-walk data, so neural models
had more noise to contend with without any benefit. The result looks like regression
(iter1 LSTM=8.89, iter2 LSTM=28.04) but the two producer runs are not comparable —
iter2's producer reached higher rates (max=1,084 vs max=151 in iter1).

**The core issue remains:** This eval tests random-walk handling, not burst recognition.
The lag feature's theoretical advantage (leading indicator of burst onset) cannot be
demonstrated here because there are no bursts.

### Why the Lag Feature Is Still Worth Having

In real production, Kafka lag is measured independently — it's the difference between
the latest broker offset and the consumer's committed offset. It starts growing before
inputRowsPerSecond rises (because backlog accumulates before the rate metric reflects
it). This is a genuine leading indicator that EMA cannot use. The limitation here is
that we are simulating lag from the rate stream itself, not measuring it independently.

**This points to iter3:** The more urgent fix was scale — models trained at baseline=100
are out-of-distribution when Spark traffic is in the thousands or tens of thousands.
Log-uniform training (iter3) addresses this and is necessary before the lag feature's
real advantage can be measured.

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
Producer: min=1, max=1,084, mean=107, p25=6, p75=126 events/s
Lag: simulated (capacity = rolling_mean × 1.2)

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

## Key Observations

1. **Lag feature gives marginal but consistent synthetic improvement** (ens_top3
   28.49→27.64 MAE). Not a breakthrough because simulated lag is derived from the same
   rate signal — there is no independent leading information.

2. **Real eval: EMA wins again** (11.95 vs LSTM 28.04). Random-walk producer gives
   neural models nothing to work with. High MAPE (>100% for strong neural models)
   because models over-predict bursts that never come; when actual rate is very low
   (1–5 ev/s) but prediction is high, MAPE explodes.

3. **Apparent regression vs iter1 on real eval is a different producer run**, not a
   true regression. The iter2 producer reached higher rates (max=1,084 vs 151) which
   further exposes the scale mismatch: models trained at baseline=100 produce large
   absolute errors when the true rate is in the hundreds or thousands.

4. **The root cause is scale:** A model trained exclusively at 100 ev/s is
   out-of-distribution at 1,000 ev/s. It will produce predictions near 100 while
   actuals are near 1,000. This drives both high MAE and high MAPE. Fix: log-uniform
   training (iter3).
