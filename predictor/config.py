"""
Shared constants for train.py, evaluate_stream.py, and predictor_server.py.

Change N_SERIES, BASE_SEED, TRAIN_RATIO, or VAL_RATIO here and all three
scripts stay in sync — seed ranges are derived, never hardcoded elsewhere.
"""
from pathlib import Path

K         = 30     # sliding window size — must match across train / serve / eval
N_SERIES  = 150    # total synthetic series
BASE_SEED = 42     # seed of the first series (series i gets seed BASE_SEED + i)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test ratio = 1 - TRAIN_RATIO - VAL_RATIO (implicit, ~0.15)

MODEL_DIR = Path(__file__).parent.parent / "models"

# ── Derived seed ranges ───────────────────────────────────────────────────────
_n_train = int(N_SERIES * TRAIN_RATIO)   # 105
_n_val   = int(N_SERIES * VAL_RATIO)     # 22

TRAIN_SEEDS   = range(BASE_SEED,                           BASE_SEED + _n_train)
VAL_SEEDS     = range(BASE_SEED + _n_train,                BASE_SEED + _n_train + _n_val)
HOLDOUT_SEEDS = range(BASE_SEED + _n_train + _n_val,       BASE_SEED + N_SERIES)
# TRAIN_SEEDS   = range(42, 147)   → indices 0-104
# VAL_SEEDS     = range(147, 169)  → indices 105-126
# HOLDOUT_SEEDS = range(169, 192)  → indices 127-149
