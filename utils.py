import os
import re
import csv
import random
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch

from config import *


# =========================
# HELPERS
# =========================
def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

# =========================
# REPRO
# =========================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# TIME UTILS
# =========================
def utc_str_to_ms(s: str) -> int:
    dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def book_ts_to_ms(s: str) -> int:
    dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ceil_to_grid_ms(t_ms: int, grid_ms: int) -> int:
    return ((t_ms + grid_ms - 1) // grid_ms) * grid_ms


def floor_to_grid_ms(t_ms: int, grid_ms: int) -> int:
    return (t_ms // grid_ms) * grid_ms


def parse_date_from_filename(fname: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    return m.group(1) if m else ""


# =========================
# FILE LISTING
# =========================
def list_sorted_files(folder: Path, kind: str):
    files = []
    for p in folder.glob(f"{SYMBOL}-{kind}-*.csv"):
        d = parse_date_from_filename(p.name)
        if d:
            files.append((d, p))
    files.sort(key=lambda x: x[0])
    return files


def filter_files_by_date(files_with_date, start_date: str, end_date: str):
    out = []
    for d, p in files_with_date:
        if start_date <= d <= end_date:
            out.append(p)
    return out


def read_first_data_row_csv(path: Path):
    with path.open("r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            k = row[0].strip().lower()
            if k in {"id", "timestamp"}:
                continue
            return row
    return None


def read_last_data_row_csv(path: Path, max_bytes: int = 4_000_000):
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        seek = max(0, size - max_bytes)
        f.seek(seek)
        chunk = f.read()
    lines = chunk.splitlines()
    for b in reversed(lines):
        s = b.decode("utf-8", errors="ignore").strip()
        if not s:
            continue
        k = s.split(",")[0].strip().lower()
        if k in {"id", "timestamp"}:
            continue
        return s.split(",")
    return None