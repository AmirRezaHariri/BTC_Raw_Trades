import numpy as np
from collections import deque
import json
import bisect

from config import *


# =========================
# MEMMAP STORES
# =========================
class ScaleStore:
    def __init__(self, scale_dir: Path):
        with (scale_dir / "meta.json").open("r") as f:
            self.meta = json.load(f)
        self.scale = self.meta["scale"]
        self.interval_ms = int(self.meta["interval_ms"])
        self.d = int(self.meta["d"])
        self.total_rows = int(self.meta["total_rows"])
        self.first_ts = int(self.meta["first_ts"])
        self.last_ts = int(self.meta["last_ts"])
        self.shards = self.meta["shards"]
        self.offsets = [int(s["offset"]) for s in self.shards] + [self.total_rows]

        self._cache = {}
        self._cache_order = deque()
        self._cache_cap = 4

    def _find_shard(self, idx: int):
        i = bisect.bisect_right(self.offsets, idx) - 1
        i = max(0, min(i, len(self.shards) - 1))
        shard = self.shards[i]
        base = int(shard["offset"])
        local = idx - base
        return i, local

    def _load_memmap(self, shard_i: int):
        if shard_i in self._cache:
            return self._cache[shard_i]
        shard = self.shards[shard_i]
        X = np.load(shard["x"], mmap_mode="r")
        T = np.load(shard["t"], mmap_mode="r")
        self._cache[shard_i] = (X, T)

        self._cache_order.append(shard_i)
        while len(self._cache_order) > self._cache_cap:
            old = self._cache_order.popleft()
            if old in self._cache:
                del self._cache[old]
        return X, T

    def get_row(self, idx: int):
        shard_i, local = self._find_shard(idx)
        X, T = self._load_memmap(shard_i)
        return int(T[local]), X[local]

    def get_seq_end(self, idx_end: int, seq_len: int):
        start = idx_end - (seq_len - 1)
        if start < 0:
            raise IndexError("seq start < 0")

        out = np.empty((seq_len, self.d), dtype=np.float32)
        cur = start
        filled = 0
        while cur <= idx_end:
            shard_i, local = self._find_shard(cur)
            X, _ = self._load_memmap(shard_i)
            take = min(X.shape[0] - local, (idx_end - cur + 1))
            out[filled:filled + take] = X[local:local + take]
            filled += take
            cur += take
        return out


def find_first_idx_ge(store: ScaleStore, target_ms: int):
    lo = 0
    hi = store.total_rows - 1
    ans = store.total_rows
    while lo <= hi:
        mid = (lo + hi) // 2
        ts, _ = store.get_row(mid)
        if ts >= target_ms:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


def map_base_ts_to_scale_idx(scale_store: ScaleStore, t_ms: int):
    if t_ms < scale_store.first_ts:
        return -1
    return int((int(t_ms) - int(scale_store.first_ts)) // int(scale_store.interval_ms))


# =========================
# ENVIRONMENT
# =========================
def vote_exec_action(vote_deque: deque, last_exec: int):
    counts = [0] * N_ACTIONS
    for a in vote_deque:
        counts[int(a)] += 1
    m = max(counts)
    cands = [i for i, v in enumerate(counts) if v == m]
    if len(cands) == 1:
        return cands[0]
    if int(last_exec) in cands:
        return int(last_exec)
    return 0


class TradingEnv:
    def __init__(self, stores, norm, close_idx_in_x250: int):
        self.stores = stores
        self.close_idx = int(close_idx_in_x250)
        self.episode_steps = int((EPISODE_HOURS * 3600 * 1000) // BASE_MS)

        self.mean = {k: np.array(norm[k]["mean"], dtype=np.float32) for k in norm}
        self.std = {k: np.array(norm[k]["std"], dtype=np.float32) for k in norm}

        self.i = 0
        self.step_i = 0
        self.balance = INIT_BALANCE
        self.pos = 0
        self.vote = deque(maxlen=VOTE_N)
        self.last_exec = 0
        self.done = False

        self.buf_250 = None
        self.buf_20 = None
        self.buf_5m = None
        self.buf_1h = None

        self.idx_20 = 0
        self.idx_5m = 0
        self.idx_1h = 0

    def _norm_scale(self, scale, x):
        return (x - self.mean[scale]) / self.std[scale]

    def _norm_row(self, scale, row):
        return (row - self.mean[scale]) / self.std[scale]

    def _get_price(self, x250_row):
        return float(x250_row[self.close_idx])

    def _fee_mult(self, fee_rate: float):
        m = 1.0 - float(fee_rate) * float(LEVERAGE)
        return max(0.0, m)

    def _ctx_vec(self):
        last = np.zeros(N_ACTIONS, dtype=np.float32)
        last[self.last_exec] = 1.0

        q = list(self.vote)
        if len(q) < VOTE_N:
            q = q + [0] * (VOTE_N - len(q))

        q_oh = np.zeros((VOTE_N, N_ACTIONS), dtype=np.float32)
        for i, a in enumerate(q):
            q_oh[i, int(a)] = 1.0

        ctx = np.concatenate([
            np.array([float(self.pos), float(self.balance / INIT_BALANCE)], dtype=np.float32),
            last,
            q_oh.reshape(-1).astype(np.float32)
        ]).astype(np.float32)
        return ctx

    def _obs(self):
        return {
            "250ms": self.buf_250,
            "20s": self.buf_20,
            "5m": self.buf_5m,
            "1h": self.buf_1h,
            "ctx": self._ctx_vec()
        }

    def reset(self, start_i):
        self.i = int(start_i)
        self.step_i = 0
        self.balance = float(INIT_BALANCE)
        self.pos = 0
        self.vote.clear()
        for _ in range(VOTE_N):
            self.vote.append(0)
        self.last_exec = 0
        self.done = False

        t_ms, _ = self.stores["250ms"].get_row(self.i)
        self.idx_20 = map_base_ts_to_scale_idx(self.stores["20s"], t_ms)
        self.idx_5m = map_base_ts_to_scale_idx(self.stores["5m"], t_ms)
        self.idx_1h = map_base_ts_to_scale_idx(self.stores["1h"], t_ms)

        x250 = self.stores["250ms"].get_seq_end(self.i, SEQ_LEN)
        x20 = self.stores["20s"].get_seq_end(self.idx_20, SEQ_LEN)
        x5m = self.stores["5m"].get_seq_end(self.idx_5m, SEQ_LEN)
        x1h = self.stores["1h"].get_seq_end(self.idx_1h, SEQ_LEN)

        self.buf_250 = self._norm_scale("250ms", x250)
        self.buf_20 = self._norm_scale("20s", x20)
        self.buf_5m = self._norm_scale("5m", x5m)
        self.buf_1h = self._norm_scale("1h", x1h)

        return self._obs()

    def step(self, action_proposed: int):
        if self.done:
            raise RuntimeError("step after done")

        action_proposed = int(action_proposed)
        if action_proposed < 0 or action_proposed >= N_ACTIONS:
            action_proposed = 0

        bal_start = float(self.balance)

        self.vote.append(action_proposed)
        exec_a = vote_exec_action(self.vote, self.last_exec)
        self.last_exec = int(exec_a)

        _, x_row = self.stores["250ms"].get_row(self.i)
        price_now = self._get_price(x_row)

        if exec_a == 1:
            if self.pos == 0:
                self.balance *= self._fee_mult(FEE_OPEN)
                self.pos = 1
            elif self.pos == -1:
                self.balance *= self._fee_mult(FEE_CLOSE)
                self.balance *= self._fee_mult(FEE_OPEN)
                self.pos = 1

        elif exec_a == 2:
            if self.pos == 0:
                self.balance *= self._fee_mult(FEE_OPEN)
                self.pos = -1
            elif self.pos == 1:
                self.balance *= self._fee_mult(FEE_CLOSE)
                self.balance *= self._fee_mult(FEE_OPEN)
                self.pos = -1

        elif exec_a == 3:
            if self.pos != 0:
                self.balance *= self._fee_mult(FEE_CLOSE)
                self.pos = 0

        next_i = self.i + 1
        if next_i >= self.stores["250ms"].total_rows:
            if self.pos != 0:
                self.balance *= self._fee_mult(FEE_CLOSE)
                self.pos = 0
            self.balance = max(0.0, float(self.balance))
            reward = float(self.balance - bal_start)
            self.i = self.stores["250ms"].total_rows - 1
            self.done = True
            info = {"balance": float(self.balance), "pos": int(self.pos), "exec_action": int(self.last_exec)}
            return self._obs(), reward, True, info

        _, x_next_row = self.stores["250ms"].get_row(next_i)
        price_next = self._get_price(x_next_row)

        ret = 0.0
        if price_now > 0:
            ret = (price_next - price_now) / price_now

        bal_mid = float(self.balance)
        if self.pos != 0:
            pnl = bal_mid * float(LEVERAGE) * float(self.pos) * float(ret)
            self.balance = bal_mid + pnl

        self.balance = max(0.0, float(self.balance))
        reward = float(self.balance - bal_start)

        self.i = next_i
        self.step_i += 1

        t_ms, _ = self.stores["250ms"].get_row(self.i)

        new_idx_20 = map_base_ts_to_scale_idx(self.stores["20s"], t_ms)
        new_idx_5m = map_base_ts_to_scale_idx(self.stores["5m"], t_ms)
        new_idx_1h = map_base_ts_to_scale_idx(self.stores["1h"], t_ms)

        row250 = self._norm_row("250ms", x_next_row)
        self.buf_250 = np.roll(self.buf_250, shift=-1, axis=0)
        self.buf_250[-1] = row250

        if new_idx_20 != self.idx_20:
            self.idx_20 = new_idx_20
            _, x20row = self.stores["20s"].get_row(self.idx_20)
            x20row = self._norm_row("20s", x20row)
            self.buf_20 = np.roll(self.buf_20, shift=-1, axis=0)
            self.buf_20[-1] = x20row

        if new_idx_5m != self.idx_5m:
            self.idx_5m = new_idx_5m
            _, x5mrow = self.stores["5m"].get_row(self.idx_5m)
            x5mrow = self._norm_row("5m", x5mrow)
            self.buf_5m = np.roll(self.buf_5m, shift=-1, axis=0)
            self.buf_5m[-1] = x5mrow

        if new_idx_1h != self.idx_1h:
            self.idx_1h = new_idx_1h
            _, x1hrow = self.stores["1h"].get_row(self.idx_1h)
            x1hrow = self._norm_row("1h", x1hrow)
            self.buf_1h = np.roll(self.buf_1h, shift=-1, axis=0)
            self.buf_1h[-1] = x1hrow

        thresh = float(LOSS_THRESHOLD_FRAC) * float(INIT_BALANCE)
        done = False
        if self.balance <= thresh:
            done = True
        if self.step_i >= self.episode_steps:
            done = True

        if done and self.pos != 0:
            bal_before = float(self.balance)
            self.balance *= self._fee_mult(FEE_CLOSE)
            self.pos = 0
            self.balance = max(0.0, float(self.balance))
            reward += float(self.balance - bal_before)

        self.done = done
        info = {"balance": float(self.balance), "pos": int(self.pos), "exec_action": int(self.last_exec)}
        return self._obs(), reward, done, info


# =========================
# REPLAY
# =========================
class ReplayBuffer:
    def __init__(self, cap: int, ctx_dim: int):
        self.cap = int(cap)
        self.ctx_dim = int(ctx_dim)
        self.ptr = 0
        self.size = 0

        self.i = np.zeros(self.cap, dtype=np.int64)
        self.a = np.zeros(self.cap, dtype=np.int64)
        self.r = np.zeros(self.cap, dtype=np.float32)
        self.d = np.zeros(self.cap, dtype=np.float32)
        self.ni = np.zeros(self.cap, dtype=np.int64)

        self.ctx = np.zeros((self.cap, self.ctx_dim), dtype=np.float32)
        self.nctx = np.zeros((self.cap, self.ctx_dim), dtype=np.float32)

    def add(self, i, a, r, d, ni, ctx, nctx):
        p = self.ptr
        self.i[p] = int(i)
        self.a[p] = int(a)
        self.r[p] = float(r)
        self.d[p] = float(d)
        self.ni[p] = int(ni)
        self.ctx[p] = ctx.astype(np.float32, copy=False)
        self.nctx[p] = nctx.astype(np.float32, copy=False)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=int(batch))
        return (
            self.i[idx],
            self.a[idx],
            self.r[idx],
            self.d[idx],
            self.ni[idx],
            self.ctx[idx],
            self.nctx[idx],
        )