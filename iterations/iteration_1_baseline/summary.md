# Iteration 1 — Baseline

**Date:** 2026-04-17 (training) / 2026-04-19 (real eval)
**Features:** rate only [K=30, 1 feature]
**Dataset:** 150 series × 300 steps, fixed baseline=100, 7 shapes
**Models:** 9 neural + 5 ensembles + 2 baselines

---

## What the Two Evaluations Tell Us

### evaluate_stream.py — Synthetic Holdout

This tests generalisation within the synthetic distribution. The holdout series
(seeds 169–191) use the same shape vocabulary as training (ramps, walls, sawteeth,
plateaus, cliffs, double-peaks, noise) but are entirely different random instances
never seen during training.

**What it answers:** "Can the model recognise and predict burst transitions it hasn't
seen before, given that production traffic structurally resembles these shapes?"

**What it doesn't answer:** Whether real Kafka/Spark traffic actually looks like these
shapes. This is a closed-world test.

**Why neural models win here:** EMA has no structural knowledge — it predicts "close to
recent average." On burst transitions (e.g. a ramp peaking and turning over), EMA
consistently calls direction wrong. Neural models correctly anticipate the transition.
DirAcc of 72.4% (ens_top3) vs 44.6% (EMA) is real signal, not artefact.

### evaluate_stream2.py — Real Spark, Random-Walk Traffic

This runs a genuine Spark Structured Streaming job and captures real
inputRowsPerSecond via StreamingQueryListener. The infrastructure is real.
The traffic is not: a Poisson random-walk generator produced mean=19 ev/s,
fluctuating slowly with no burst structure.

**What it answers:** "Which model handles slow random-walk traffic at low rate?"

**What it doesn't answer:** Whether burst pattern recognition matters in production.
Random-walk traffic has no burst structure, so models trained to recognise bursts
find no signal to exploit. EMA wins because persistence (predict ≈ recent value)
is the optimal strategy for any memoryless random walk.

**Why EMA wins here (and what it means):** EMA MAE=6.86 vs LSTM MAE=8.89. This is
not evidence that EMA is better than LSTM in production — it's evidence that the
producer generated traffic with no structure for neural models to leverage.
The models are not being tested on the problem they were trained for.

**The gap between these two evals is the core problem of this project.** Eval 1 shows
neural models clearly outperform EMA on burst traffic. Eval 2 shows EMA beats neural
models on random-walk traffic. Real production Kafka topics have both regimes. A fair
evaluation needs real traffic — or at least burst-structured synthetic traffic fed
into a live Spark job.

**This points to evaluate_stream3.py:** either replay a real inputRowsPerSecond trace
(gold standard, no distribution assumptions) or feed burst-shaped traffic into the
live Spark job (tests whether the pattern recognition benefit survives real infrastructure).

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

## Burst Traffic Results (evaluate_stream3.py) — Not Run

> Models for iteration 1 were overwritten when iteration 3 was trained.
> To reproduce: retrain iteration 1 (`python predictor/train.py`), then run:
> `python predictor/evaluate_stream3.py --duration 120 --log-dir iterations/iteration_1_baseline`

### Table 3 — Option A: Replay · Table 4 — Option B: Live

*Results not available — models not preserved.*

---

## Trigger Policy Results (evaluate_stream4.py) — Not Run

> Same dependency: iteration 1 models were overwritten.
> To reproduce: retrain, then run:
> `python predictor/evaluate_stream4.py --duration 180 --log-dir iterations/iteration_1_baseline`

### Table 5 — Fixed vs Adaptive Trigger Policy

*Results not available — models not preserved.*

---

## Commands Used

```bash
# Training
python predictor/train.py

# Synthetic holdout eval
python predictor/evaluate_stream.py

# Real Spark eval
python predictor/evaluate_stream2.py --duration 120
```

---

## Learnings

- **Neural models learn burst structure.** On synthetic holdout, ens_top3 DirAcc=72.4%
  vs EMA DirAcc=44.6%. Neural models correctly call the direction of burst transitions
  ~72% of the time; EMA calls it wrong more than right because it has no structural
  knowledge — it always predicts close to the recent average, which is precisely wrong
  at transition points (peak turning over, wall dropping, sawtooth resetting).

- **DirAcc is the metric that matters, not raw MAE.** MAE is in absolute events/s and
  scales with traffic magnitude. DirAcc is scale-invariant and directly comparable
  across iterations. All cross-iteration comparison should use DirAcc.

- **EMA wins on real Spark when traffic is random-walk** (mean=19 ev/s, no bursts).
  Persistence — predict close to the recent value — is the optimal strategy for a
  memoryless random walk. Neural models find no structure to exploit and lose to EMA.
  This is not a model failure; it is a distribution mismatch.

- **Fixed baseline=100 is a hard ceiling.** Models trained at one scale are
  out-of-distribution when traffic exceeds that scale. Any producer run reaching
  hundreds or thousands of events/s will expose this collapse. Two fixes are needed:
  log-uniform training (iter3) and an eval that actually contains burst structure.
