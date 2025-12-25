from pathlib import Path
import torch
import math


# =========================
# PARAMS
# =========================
MODE = "train"  # "prepare" or "train"

SYMBOL = "BTCUSDT"
DATA_ROOT = Path("data")
TRADES_DIR = DATA_ROOT / "trades"
BOOK_DIR = DATA_ROOT / "bookDepth"
OUT_ROOT = Path("processed") / SYMBOL

BASE_MS = 250
SCALES_MS = {
    "250ms": 250,
    "20s": 20_000,
    "5m": 300_000,
    "1h": 3_600_000,
}
SEQ_LEN = 512

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
FEE_OPEN = 0.0004
FEE_CLOSE = 0.0004
LOSS_THRESHOLD_FRAC = 0.666
EPISODE_HOURS = 6
VOTE_N = 20

ACTIONS = ["HOLD", "OPEN_LONG", "OPEN_SHORT", "CLOSE"]
N_ACTIONS = len(ACTIONS)

# Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

GAMMA = 0.997
TAU = 0.01
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
BATCH_SIZE = 1024
REPLAY_CAP = 800_000
WARMUP_STEPS = 50_000
UPDATES_PER_STEP = 1
UPDATE_EVERY = 1
MAX_EPISODES = 10_000

ALPHA_INIT = 0.2
TARGET_ENTROPY = math.log(N_ACTIONS) * 0.9

# Model
HIDDEN = 192
ENC_HIDDEN = 256
DROPOUT = 0.1