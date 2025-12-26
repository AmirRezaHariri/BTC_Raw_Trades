from pathlib import Path
import torch
import math


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
REPLAY_DIR = OUT_ROOT / "replay_memmap"
RESUME_PATH = ""  # "" or path to a .pt

SEQ_LEN = 128
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
FEE_OPEN = 0.0
FEE_CLOSE = 0.0
LOSS_THRESHOLD_FRAC = 0.666
EPISODE_HOURS = 0.5
VOTE_N = 5

ACTIONS = ["HOLD", "OPEN_LONG", "OPEN_SHORT", "CLOSE"]
N_ACTIONS = len(ACTIONS)

USE_VOTE_FILTER_TRAIN = False
USE_VOTE_FILTER_EVAL = False

# Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True

GAMMA = 0.997
TAU = 0.01
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
BATCH_SIZE = 128

ALPHA_INIT = 0.2
TARGET_ENTROPY = math.log(N_ACTIONS) * 0.9

REPLAY_CAP = 800_000
WARMUP_STEPS = 5_000
UPDATES_PER_STEP = 1
UPDATE_EVERY = 2
MAX_EPISODES = 300
CKPT_EVERY_EP = 25
EVAL_EVERY_EP = 25
EVAL_TOTAL_STEPS = 20_000

PBAR_EVERY = 32

# Model
HIDDEN = 64
ENC_HIDDEN = 128
DROPOUT = 0.1

USE_TF32 = True
USE_AMP_BF16 = True
USE_COMPILE = False