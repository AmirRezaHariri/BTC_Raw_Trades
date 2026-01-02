from pathlib import Path
import torch


# =========================
# PARAMS
# =========================

NOTEBOOK = False 
MODE = "train"  # "prepare" or "train"

SYMBOL = "BTCUSDT"
DATA_ROOT = Path("data")
TRADES_DIR = DATA_ROOT / "trades"
BOOK_DIR = DATA_ROOT / "bookDepth"
OUT_ROOT = Path("processed") / SYMBOL
CHECKPOINT_DIR = OUT_ROOT / "checkpoints"
RESUME = True

SEQ_LEN = 256
BASE_MS = 250

SCALES_MS = {
    "250ms": 250,
    "20s": 20_000,
    "5m": 300_000,
    "1h": 3_600_000,
}

SHARD_ROWS = {
    "250ms": 200_000,
    "20s": 120_000,
    "5m": 60_000,
    "1h": 30_000,
}

TRAIN_END_UTC = "2025-01-01 00:00:00"
VAL_END_UTC = "2025-07-01 00:00:00"

SEED = 13
EPS = 1e-8

# Environment
INIT_BALANCE = 100.0
LEVERAGE = 1

# Use only a fraction of equity for position sizing (reduces always-in-position + fee bleed).
POS_FRACTION = 1.0

# Transaction cost model beyond explicit fees.
# 1 bp = 0.0001. These are applied on execution (entry/exit), not just on close.
SPREAD_BPS = 1.0
SLIPPAGE_BPS = 0.5

FEE_OPEN = 0.0004
FEE_CLOSE = 0.0004
LOSS_THRESHOLD_FRAC = 0.666 # Minimum balance 
EPISODE_HOURS = 1.0
VOTE_N = 256

# Hard throttles to stop churn.
# BASE_MS=250ms => 4 steps/sec. 2400 steps                                     = 10 minutes.
MIN_HOLD_STEPS = 240
COOLDOWN_STEPS = 20

# Explicit churn discouragement in reward space.
TRADE_PENALTY_ENTRY = 0.0
TRADE_PENALTY_FLIP = 0.0

# Mild exposure penalty per step to avoid "always in position" when there is no edge.
POS_HOLD_PENALTY = 1e-8

# Action semantics (IMPORTANT):
# HOLD = keep current position
# LONG = target long
# SHORT = target short
# FLAT = target flat (close position)
ACTIONS = ["HOLD", "LONG", "SHORT", "FLAT"]
N_ACTIONS = len(ACTIONS)

USE_VOTE_FILTER_TRAIN = False
USE_VOTE_FILTER_EVAL = False

# Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
USE_TF32 = True
USE_AMP_BF16 = True
USE_COMPILE = False

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True

GAMMA = 0.999
TAU = 0.01
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
BATCH_SIZE = 128

ALPHA_INIT = 0.05
TARGET_ENTROPY = None 

REPLAY_CAP = 500_000
WARMUP_STEPS = 50_000
UPDATES_PER_STEP = 1
UPDATE_EVERY = 2
MAX_EPISODES = 300
CKPT_EVERY_EP = 10
EVAL_EVERY_EP = 10
EVAL_TOTAL_STEPS = 10_000

PBAR_EVERY = 32

# Model
HIDDEN = 128
ENC_HIDDEN = 256
DROPOUT = 0.1

CTX_BASE = 6  # pos, balance, unrealized, realized, hold_left, cooldown
CTX_LAST_ACT = N_ACTIONS
CTX_VOTE = VOTE_N * N_ACTIONS

CTX_DIM = CTX_BASE + CTX_LAST_ACT + CTX_VOTE