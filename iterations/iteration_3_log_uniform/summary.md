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

## What the Two Evaluations Tell Us

### evaluate_stream.py — Synthetic Holdout (Log-Uniform)

The holdout series now span the same log-uniform scale as training: baselines from
10 to 500k events/s. This is a stronger generalisation test than iter1/2 because the
model must handle pattern recognition at scales it has never seen as exact instances.

**What the results show:**
- DirAcc improved meaningfully: iter1=72.4%, iter2=72.1%, iter3=**77.5%** (TiDE)
- EMA DirAcc dropped from 44.6% (iter1/2) to **39.6%** — below random (50%)

**What the DirAcc numbers mean in plain terms:**
- Neural models correctly call "rate going up or down?" about 77 times out of 100
  on held-out burst series they've never seen.
- EMA calls it wrong 60 times out of 100. At burst transitions (ramp peaking, wall
  dropping, sawtooth resetting) EMA has no knowledge of the shape — it predicts
  "close to recent average" which is precisely wrong at the moment of transition.

**What the raw MAE numbers mean (and why they're misleading):**
- Iter3 best MAE ≈ 6,700 vs iter2 best MAE ≈ 27. This is NOT a regression.
- The holdout baselines now reach 500k ev/s, so a 1% directional error translates
  to an absolute error of 5,000 — orders of magnitude larger than at baseline=100.
- **DirAcc is the only valid cross-iteration metric. Raw MAE is not comparable.**

**Why this eval is still a closed-world test:**
The shapes (ramps, walls, sawteeth) are the same vocabulary as training. Different
seeds, different random instances, but the same structural distribution. We cannot
conclude from this that real production Kafka traffic follows these shapes. The eval
tells us models learned the abstract patterns well — it does not validate that those
patterns match reality.

### evaluate_stream2.py — Real Spark, Random-Walk Traffic

Same infrastructure as iter1/2 (genuine Spark, StreamingQueryListener). The producer
this run happened to generate much higher rates: mean=1,292 ev/s, max=10,000 ev/s.

**What the results show:** LSTM beats EMA for the first time (MAE=229 vs EMA=271).

**Why LSTM finally won — and what this actually proves:**
This is a scale coverage win, not a burst-recognition win. Here is the mechanism:

- Iter1/2 models trained at baseline=100. When true rate is 1,000–10,000 ev/s,
  the normalised input looks like a signal with enormous amplitude — the models
  have never seen anything like it. Their predictions cluster near 100 ev/s.
  Absolute error ≈ true_rate − 100, which grows linearly with traffic.

- Iter3 models trained at 10–500k ev/s. At 1,000–10,000 ev/s they are
  in-distribution. Their predictions are in the right ballpark. Errors are smaller
  not because they learned better patterns, but because they aren't confused by scale.

- EMA is scale-invariant by construction (it always predicts near recent values).
  It was already "in-distribution" in iter1/2. So EMA's score barely changed
  across iterations; what changed is that neural models caught up to EMA's scale
  coverage.

**What this does NOT prove:**
- That neural models' burst pattern recognition translates to production benefit.
- That neural models would beat EMA on the same producer settings as iter1 (mean=19
  ev/s). On low-rate random-walk traffic they still would not.

**The result is meaningful but incomplete:** It shows log-uniform training is necessary
for neural models not to collapse at high-rate traffic. It does not show they are
exploiting burst structure on real traffic — because the producer generates none.

### The Honest Conclusion from Three Iterations

| Question | Answered by | Answer |
|----------|-------------|--------|
| Do neural models learn burst transitions? | eval_stream (all iters) | Yes — DirAcc 72→77%, EMA DirAcc 44→39% |
| Is fixed-scale training sufficient? | eval_stream2 iter1/2 | No — models collapse above training scale |
| Does log-uniform fix scale coverage? | eval_stream2 iter3 | Yes — LSTM now beats EMA at 1k+ ev/s |
| Do neural models beat EMA on real production traffic? | **Not yet answered** | Needs evaluate_stream3.py |

**The gap that remains:** We have not evaluated on traffic that is simultaneously real
(from an actual Spark job) AND burst-structured (matching what models were trained for).
Every real-eval run so far used random-walk traffic. The synthetic holdout used burst
traffic but with fake infrastructure. No test has done both at once.

**evaluate_stream3.py (next)** addresses this with two modes:
  (a) **Replay** — feed a real historical inputRowsPerSecond trace through the sliding-
      window evaluator. No live Spark required. This is the gold standard: if a model
      wins here, it wins on actual production data with no assumptions.
  (b) **Burst producer** — replace PoissonProducer with burst-shaped traffic (data2.py
      shapes, held-out seeds) fed into the live Spark job. Infrastructure is real;
      traffic structure approximates production bursts. Note: this is not perfectly
      fair to EMA (neural models were trained on these shapes), but it does test
      whether the pattern recognition benefit survives real Spark infrastructure,
      latency, and batch timing — which synthetic replay does not.

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
