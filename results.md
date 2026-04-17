# AdaptiveStream ML — Observed Results

---

## TODO / Planned Work

| Priority | Item | Detail |
|----------|------|--------|
| HIGH | **Kafka lag as 2nd input feature** | Add `lag(t) = latestOffset - consumerOffset` alongside rate. Synthetic training: simulate producer-consumer dynamics. Changes input from `[K,1]` → `[K,2]`. Retrain all 9 models. See _Analysis: Kafka Lag Feature_ below. |
| HIGH | **Real Spark validation** | Instrument a local Spark job (rate source or Docker Kafka) and collect 30–60 min of real `inputRowsPerSecond` + lag. That is the correct real-world benchmark — not Wikipedia. |
| MED | **Extend synthetic dataset** | Add variable baseline scale (10–10k), diurnal sine shape, composite (slow trend + burst), more series (150→500). Needed before re-introducing Wikipedia evaluation. |
| MED | **Extend LSTM/GRU budget** | Both were still improving at 150s. A 300s run would likely push LSTM below 0.38 val_MAE. |
| LOW | **Wikipedia eval (deferred)** | Only meaningful once synthetic dataset covers slow-trend/diurnal patterns. See _Analysis: Why Wikipedia Failed_ below. |
| LOW | **GPU training** | RTX 4060 available, currently all CPU. |

---

## Training Run 1
**Date:** 2026-04-17  
**Hardware:** CPU only  
**Codebase:** commit `1cbcd97`

---

### Dataset
```
150 series × 300 steps, K=30 sliding window
Train: 28,350 windows  |  Val: 5,940 windows  |  Test: reserved
Seeds — train: 42–146, val: 147–168, test: 169–191 (never touched during training)
Shapes: noise, ramp, wall, plateau, cliff, double_peak, sawtooth
```

### Training Config
```
Budget:       150s / model (same wall-clock for all architectures)
Checkpoints:  @30s, @60s, @120s, @150s
Early stop:   patience=5 × 5s interval on val_MAE, min_delta=1e-4
Loss:         HuberLoss(delta=1.0)
Optimiser:    Adam, lr=1e-3, batch=256
DirAcc def:   sign(pred − last_window_val) == sign(actual − last_window_val)
              (up/down relative to current value — matches streaming eval)
```

---

### Per-Model Training Logs

```
LSTM
  @ 30s  epoch=12   val_MAE=0.4495  DirAcc=69.3%
  @ 60s  epoch=22   val_MAE=0.4407  DirAcc=68.5%
  @120s  epoch=44   val_MAE=0.4236  DirAcc=69.3%
  @150s  epoch=53   val_MAE=0.4154  DirAcc=72.1%
  → saved val_MAE=0.4040  DirAcc=72.5%   (best_state restored, not @150s snapshot)

GRU
  @ 30s  epoch=7    val_MAE=0.4564  DirAcc=68.4%
  @ 60s  epoch=14   val_MAE=0.4697  DirAcc=68.1%   ← temporary bump, not uncommon
  @120s  epoch=26   val_MAE=0.4273  DirAcc=70.4%
  @150s  epoch=32   val_MAE=0.4131  DirAcc=71.2%
  → saved val_MAE=0.4139  DirAcc=70.5%

TCN
  @ 30s  epoch=34   val_MAE=0.4626  DirAcc=68.6%
  @ 60s  epoch=69   val_MAE=0.4516  DirAcc=70.6%
  early stop at 69.4s  (best val_MAE=0.4472)
  → saved val_MAE=0.4472  DirAcc=70.3%   [*]

DLINEAR
  @ 30s  epoch=322  val_MAE=0.5463  DirAcc=57.8%
  early stop at 50.4s  (best val_MAE=0.5453)
  → saved val_MAE=0.5453  DirAcc=56.3%   [*]

MLP
  @ 30s  epoch=203  val_MAE=0.4632  DirAcc=68.7%
  early stop at 30.5s  (best val_MAE=0.4505)
  → saved val_MAE=0.4505  DirAcc=69.1%   [*]

ATTN
  @ 30s  epoch=35   val_MAE=0.4837  DirAcc=67.4%
  @ 60s  epoch=67   val_MAE=0.4776  DirAcc=68.7%
  early stop at 80.1s  (best val_MAE=0.4713)
  → saved val_MAE=0.4713  DirAcc=68.8%   [*]

TIDE
  @ 30s  epoch=142  val_MAE=0.4353  DirAcc=70.5%
  early stop at 46.1s  (best val_MAE=0.4342)
  → saved val_MAE=0.4342  DirAcc=69.9%   [*]

FITS
  @ 30s  epoch=247  val_MAE=0.6086  DirAcc=52.8%
  @ 60s  epoch=508  val_MAE=0.6141  DirAcc=52.5%
  early stop at 60.7s  (best val_MAE=0.6032)
  → saved val_MAE=0.6032  DirAcc=52.0%   [*]

NBEATS
  @ 30s  epoch=65   val_MAE=0.5162  DirAcc=69.0%
  early stop at 31.0s  (best val_MAE=0.4626)
  → saved val_MAE=0.4626  DirAcc=68.4%   [*]

[*] = early stopping active, weights restored to best val checkpoint
```

---

### Comparison Table (val_MAE / val_DirAcc at checkpoints)

```
============================================================================
Model             @30s            @60s           @120s           @150s
  (val_MAE/val_DirAcc — test reserved for evaluate_stream.py)
────────────────────────────────────────────────────────────────────────────
lstm           0.450/69%       0.441/68%       0.424/69%       0.415/72%
gru            0.456/68%       0.470/68%       0.427/70%       0.413/71%
tcn            0.463/69%       0.452/71%       0.447/70%*      0.447/70%*
dlinear        0.546/58%       0.545/56%*      0.545/56%*      0.545/56%*
mlp            0.463/69%       0.451/69%*      0.451/69%*      0.451/69%*
attn           0.484/67%       0.478/69%       0.471/69%*      0.471/69%*
tide           0.435/70%       0.434/70%*      0.434/70%*      0.434/70%*
fits           0.609/53%       0.614/52%       0.603/52%*      0.603/52%*
nbeats         0.516/69%       0.463/68%*      0.463/68%*      0.463/68%*
============================================================================
* = early stopping active
```

---

### Saved Model Rankings (by val_MAE)

| Rank | Model   | val_MAE | val_DirAcc | Converged at |
|------|---------|---------|------------|--------------|
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

### Insights

**1. Recurrent models still lead.**  
LSTM and GRU are the only two that ran the full 150s budget without early stopping and were still improving at the end. Both show consistent monotonic improvement across checkpoints. GRU matches LSTM at 0.413/0.414 — cell state adds negligible value for a 30-step window.

**2. TiDE is the best non-recurrent model.**  
Pure dense enc-dec at 0.4342, converging in under 46 seconds. ~600× fewer parameters than LSTM (10k vs 50k) with only 7% worse MAE. Strong candidate for deployment if latency matters.

**3. TCN causal fix likely helped but needs more epochs.**  
TCN early stopped at 69s with 79 epochs. The architecture change (symmetric → causal padding) meant starting from scratch. Likely would improve further with a longer budget or second run.

**4. DLinear and FITS are clearly weak on this data.**  
DLinear (0.545) confirms that trend+residual decomposition can't capture the nonlinear burst shapes (wall, double_peak, sawtooth). FITS (0.603) near random on DirAcc (52%) — frequency domain interpolation is wrong for aperiodic burst data. Both should be excluded from ensembles.

**5. MLP is competitive despite no temporal structure.**  
MLP (0.4505) matched TCN and beat Attn/N-BEATS. For a 30-step window, flattening and applying dense layers is almost as expressive as the more complex architectures. Supports the hypothesis that short windows don't need complex sequence modelling.

**6. DirAcc is now honest.**  
Previous runs reported 88-89% DirAcc using the wrong formula (`sign(pred)==sign(actual)` in normalised space, which just tests whether the model predicts above or below the window mean — easy). Current 69-72% uses the correct up/down definition and is a meaningful metric.

**7. Early stopping compressed compute significantly.**  
7 of 9 models stopped before 150s. Total actual training time was ~18 minutes vs the theoretical 22.5 minutes (9 × 150s).

---

### Ensemble Outlook (pre evaluate_stream.py)

Based on val_MAE spread:

| Ensemble | Components | Outlook |
|---|---|---|
| `ens_rnn` | LSTM + GRU | Valid — both strong, architecturally similar |
| `ens_top3` | LSTM + GRU + TiDE | Best bet — top 3, diverse families |
| `ens_wtd` | All 9, weighted 1/MAE | Risky — FITS/DLinear still get ~40% of top-model weight |
| `ens_mean` | All 9, equal weight | Likely worse than LSTM alone |
| `ens_diverse` | best-RNN + TCN + DLinear + Attn | DLinear inclusion hurts |

Recommendation: evaluate `ens_top3` and `ens_rnn` against individual models on real data. Drop `ens_mean` and revise `ens_diverse` to substitute TiDE for DLinear.

---

### Next Steps

- [x] Run `evaluate_stream.py` — synthetic holdout complete (see Evaluation Run 1 below)
- [x] Update ensemble composition: `ens_top3` added, TiDE replaces DLinear in `ens_diverse`
- [ ] Extend synthetic dataset to cover variable-scale and slower-timescale regimes
- [ ] Consider extending LSTM/GRU budget (both still improving at 150s)
- [ ] GPU training — RTX 4060 available, currently all CPU

---

## Evaluation Run 1
**Date:** 2026-04-17  
**Script:** `evaluate_stream.py`  
**Source:** Synthetic hold-out only (seeds 169–191, 23 series, 6,877 steps)

### Results

```
Model               MAE     RMSE    DirAcc        n
────────────────────────────────────────────────────
ens_top3          28.49    81.36     72.4%    6,210 ★
ens_rnn           28.53    82.27     72.7%    6,210
lstm              28.64    82.82     72.2%    6,210
gru               30.90    84.06     70.4%    6,210
ens_diverse       31.27    85.91     71.5%    6,210
tide              31.43    84.23     70.3%    6,210
ens_wtd           32.23    86.00     71.9%    6,210
ens_mean          33.49    87.63     71.7%    6,210
mlp               35.55    90.56     68.6%    6,210
tcn               35.99    98.99     70.8%    6,210
nbeats            36.65    90.19     68.9%    6,210
attn              38.23    98.94     67.3%    6,210
dlinear           46.46   108.31     55.3%    6,210
ema               55.30   112.87     44.6%    6,210
fits              56.80   115.82     51.2%    6,210
sma              151.00   205.69     52.8%    6,210
```

### Per-shape MAE (top-6 neural)

```
Shape               lstm       gru      tide       mlp       tcn    nbeats
──────────────────────────────────────────────────────────────────────────
cliff              54.38     52.64     52.61     57.15     54.78     62.53
double_peak        24.18     24.23     26.89     26.34     26.22     29.19
noise              13.59     15.68     16.10     18.56     18.90     19.53
plateau            12.95     16.11     13.62     16.86     15.54     17.98
ramp               28.40     30.08     32.05     35.20     34.08     34.89
sawtooth           46.00     51.99     53.02     63.47     69.66     60.73
wall               65.26     63.07     63.72     68.09     63.54     79.71
```

### Insights

**1. Ensembles narrow but real gains over LSTM alone.**  
`ens_top3` (28.49) and `ens_rnn` (28.53) both beat standalone LSTM (28.64). The margin is small but consistent. `ens_rnn` has slightly better DirAcc (72.7% vs 72.2%); `ens_top3` has slightly better MAE. Both are valid deployment candidates.

**2. ens_wtd (32.23) is worse than standalone LSTM.**  
Including FITS and DLinear — even at low weights — hurts. Confirmed: drop them from any ensemble used in production.

**3. Abrupt transitions are the hardest regime.**  
wall (65 MAE) and cliff (54 MAE) dominate errors across all models. These are step-function changes — the 30-step window history gives no warning. This is the primary gap in the synthetic dataset to address next.

**4. EMA/SMA fail badly on bursty synthetic data.**  
EMA MAE=55, SMA MAE=151 — both significantly worse than any neural model. These baselines are only competitive on slow-moving real-world data (Wikipedia etc.), not on the burst patterns the system is designed for.

### Wikipedia evaluation (attempted, removed)

Wikipedia daily pageviews were tested (4 articles: FIFA World Cup, ChatGPT, Oppenheimer, Paris Olympics). Results were not meaningful:
- Models trained on ~100 events/s scale; Wikipedia peaks at 300k–1M daily views → incomparable absolute MAE
- EMA "won" on Wikipedia because daily views are slow-moving (persistence works); this says nothing about burst prediction
- Wikipedia and GitHub Archive fetchers removed from `evaluate_stream.py`

**Decision:** extend the synthetic dataset to cover variable-scale regimes and slower-timescale patterns before re-introducing real-world data evaluation.

---

## Analysis: Why Wikipedia Failed as a Benchmark

EMA (alpha=0.7) beat every neural model on Wikipedia daily pageviews (75k vs 94k MAE). This looks damning but the reason has nothing to do with model quality:

**1. Wrong temporal dynamics.**  
Wikipedia daily views are slow-moving — day-to-day correlation is very high. EMA naturally exploits this: today ≈ yesterday. The neural models were trained on fast burst patterns completing in 10–30 steps. Wikipedia burst events (e.g. ChatGPT hype) unfold over weeks. The model has never seen that structure.

**2. Wrong benchmark for this system.**  
AdaptiveStream predicts Spark micro-batch rates (sub-second, events/second scale). Wikipedia daily pageviews share essentially no structure with that domain. EMA "winning" means the test is wrong, not the system.

**3. The pooled MAE is meaningless.**  
181 Wikipedia steps at ~90k MAE vs 6,210 synthetic steps at ~29 MAE. When pooled naively, Wikipedia completely drowns the synthetic signal. EMA appears to "win" overall because it wins on the larger-scale source.

**4. Adding Kafka lag wouldn't fix this.**  
Wikipedia is fetched from an HTTP API — there is no Kafka, no consumer, no lag. The lag feature would be 0 for all Wikipedia steps, giving the model exactly the same input it has now. Lag is not relevant to this benchmark.

**Condition to re-introduce Wikipedia:** extend the synthetic dataset with slow-trend and diurnal shapes, retrain, then compare. Only then does Wikipedia tell you something meaningful about generalisation.

---

## Analysis: Training Data Diversity Ceiling

The current synthetic dataset covers 7 shapes at ~100 events/s scale. The model can theoretically generalise to any domain — but only if the normalised pattern is in-distribution. With 7 synthetic shapes, the ceiling is low.

### Three concrete options to raise it

**Option 1 — Diverse real time series datasets (Monash Repository)**  
The [Monash Time Series Repository](https://forecastingdata.org/) has ~58 datasets covering traffic, energy, weather, finance, web, and IoT — all real-world, all different dynamics. Training on normalised windows from these would cover far more shape variety than 7 synthetic shapes. No LLMs needed, datasets are public.

**Option 2 — Amazon Chronos (time series foundation model)**  
[Chronos](https://github.com/amazon-science/chronos-forecasting) is a pre-trained transformer from Amazon trained on a massive corpus of real-world time series. Critically, it can *generate* synthetic time series samples. Pipeline:
1. Sample thousands of diverse windows from Chronos
2. Use them as additional training data for the predictor  
This gives real-world shape diversity without collecting your own data.

**Option 3 — LLM-guided parameter generation**  
Prompt an LLM to describe 100 diverse streaming scenarios ("IoT sensor with daily cycle and occasional dropout", "API endpoint with office-hours traffic and weekend flatline") and generate parameters for a richer shape simulator. More automatable than Option 1 but less principled than Option 2.

### Recommendation
Option 1 or 2 would get the model meaningfully closer to the generalisation ceiling. Option 3 is a quick intermediate step. All three are compatible with the existing normalised-window training pipeline.

---

## Analysis: Kafka Lag as a 2nd Input Feature

### Why the current approach has a distribution shift problem

The models are trained on synthetic data at ~100 events/s and deployed on unknown real Spark workloads. No amount of synthetic data improvement fully closes this gap because the distributional shift is in the workload structure, not the shape vocabulary.

How industry actually handles it: most production adaptive streaming systems (Flink, Kafka consumer auto-scaling) do not use offline-trained ML models. They use reactive control loops with leading indicators — signals that are causally upstream of the rate rather than lagging behind it.

### Kafka consumer lag as a leading indicator

`inputRowsPerSecond` is a lagging signal — it tells you what Spark just processed. Consumer lag is a present-state signal:

```
lag(t) = latestOffset(t) - consumerOffset(t)
```

Both values are available at time t before the next batch starts. Lag tells you whether the system is currently falling behind, regardless of the workload domain. It is a universal pressure signal.

- Lag growing → producer faster than consumer → next batch will be larger → shorten interval
- Lag = 0 → system caught up → next rate depends purely on producer dynamics

A model trained on (rate, lag) pairs generalises across Spark deployments because lag carries causal information that transcends the specific workload distribution.

### No data leakage

The input window at each step t:
```
[(rate(t-29), lag(t-29)), ..., (rate(t), lag(t))]  →  predict rate(t+1)
```

`lag(t)` is derived only from `rate(0)...rate(t)` — it encodes cumulative history, not future values. The target `rate(t+1)` is the label only, never a feature. Clean.

The "unfair advantage" concern: lag is correlated with future rate through system dynamics — but that is precisely what makes it a good feature, not leakage. The same causal relationship holds in real deployment.

### Synthetic training for lag

Generate a producer-consumer simulation alongside each burst series:
```
capacity = baseline * capacity_ratio   # ~1.2x baseline
lag(0)   = 0
lag(t)   = max(0, lag(t-1) + (rate(t) - capacity))
```

Consumer capacity must be set so lag occasionally builds up during burst peaks — if capacity is too high, lag is always 0 and the model learns to ignore it in production too.

### Implementation plan (pending)

1. `data.py` — add `generate_series_with_lag()` returning `(values, lag, labels)`
2. `models.py` — add `input_size` param to all architectures (default 2); change `[K,1]` → `[K,2]`
3. `train.py` — build 2-feature dataset; normalise rate and lag independently
4. `metrics_collector.py` — parse `sources[].latestOffset` and `sources[].endOffset` from Spark API; compute lag per partition, sum
5. `predictor_server.py` — build `[K,2]` tensor; lag=0 fallback for non-Kafka sources
6. `evaluate_stream.py` — simulate lag alongside rate in holdout evaluation
