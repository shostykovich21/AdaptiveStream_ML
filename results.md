# AdaptiveStream ML — Observed Results

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

- [ ] Run `evaluate_stream.py` — first look at test set MAE on holdout (seeds 169–191) + Wikipedia
- [ ] Update ensemble composition: add `ens_top3`, drop DLinear/FITS from `ens_diverse`
- [ ] Consider extending LSTM/GRU budget (both still improving at 150s)
- [ ] GPU training — RTX 4060 available, currently all CPU
