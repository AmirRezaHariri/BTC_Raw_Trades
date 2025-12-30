import numpy as np
import math
from collections import deque
import torch

from config import *
from dataset import *


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
    # tie break: most recent among tied candidates (not HOLD by default)
    for a in reversed(vote_deque):
        aa = int(a)
        if aa in cands:
            return aa
    return int(last_exec)

class TradingEnv:
    def __init__(self, stores, norm, close_idx_in_x250: int, device: str, torch_obs: bool = True):
        self.stores = stores
        self.close_idx = int(close_idx_in_x250)
        self.device = device
        self.torch_obs = bool(torch_obs)

        self.episode_steps = int((EPISODE_HOURS * 3600 * 1000) // BASE_MS)

        self.mean = {k: np.array(norm[k]["mean"], dtype=np.float32) for k in norm}
        self.std = {k: np.array(norm[k]["std"], dtype=np.float32) for k in norm}

        self.i = 0
        self.step_i = 0
        self.balance = float(INIT_BALANCE)
        self.peak_balance = float(INIT_BALANCE)
        self.pos = 0
        self.entry_price = 0.0
        self.last_price = 0.0
        self.pos_notional = 0.0
        self.pos_qty = 0.0
        self.entry_step_i = 0
        self.cooldown_remaining = 0
        self.vote = deque(maxlen=VOTE_N)
        self.last_exec = 0
        self.done = False
        self.flat_hold_steps = 0

        self.buf_250 = None
        self.buf_20 = None
        self.buf_5m = None
        self.buf_1h = None

        self.tbuf_250 = None
        self.tbuf_20 = None
        self.tbuf_5m = None
        self.tbuf_1h = None

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
        return 1.0 - float(fee_rate)
    
    def _penalty_mult_entry(self):
        x = 1.0 - float(TRADE_PENALTY_ENTRY)
        if x < 0.0:
            return 0.0
        return x

    def _can_close(self):
        if self.pos == 0:
            return False
        held = int(self.step_i) - int(self.entry_step_i)
        return held >= int(MIN_HOLD_STEPS)
    
    def _ctx_vec(self):
        last = np.zeros(N_ACTIONS, dtype=np.float32)
        last[self.last_exec] = 1.0

        q = list(self.vote)
        if len(q) < VOTE_N:
            q = q + [0] * (VOTE_N - len(q))

        q_oh = np.zeros((VOTE_N, N_ACTIONS), dtype=np.float32)
        for i, a in enumerate(q):
            q_oh[i, int(a)] = 1.0

        unrealized = 0.0
        if self.pos != 0 and self.entry_price > 0:
            _, row = self.stores["250ms"].get_row(self.i)
            price_now = float(row[self.close_idx])
            unrealized = self.pos * (price_now - self.entry_price) / max(self.entry_price, 1e-8)

        realized = (self.balance - INIT_BALANCE) / INIT_BALANCE
        ctx = np.concatenate([
            np.array([
                float(self.pos),
                float(self.balance / INIT_BALANCE),
                float(unrealized),
                float(realized),
            ], dtype=np.float32),
            last,
            q_oh.reshape(-1).astype(np.float32)
        ]).astype(np.float32)

        return ctx

    def _obs(self):
        if self.torch_obs:
            return {"250ms": self.tbuf_250, "20s": self.tbuf_20, "5m": self.tbuf_5m, "1h": self.tbuf_1h, "ctx": self._ctx_vec()}
        return {"250ms": self.buf_250, "20s": self.buf_20, "5m": self.buf_5m, "1h": self.buf_1h, "ctx": self._ctx_vec()}

    def _sync_tbuf_full(self):
        self.tbuf_250 = torch.from_numpy(self.buf_250).unsqueeze(0).to(self.device, dtype=DTYPE)
        self.tbuf_20 = torch.from_numpy(self.buf_20).unsqueeze(0).to(self.device, dtype=DTYPE)
        self.tbuf_5m = torch.from_numpy(self.buf_5m).unsqueeze(0).to(self.device, dtype=DTYPE)
        self.tbuf_1h = torch.from_numpy(self.buf_1h).unsqueeze(0).to(self.device, dtype=DTYPE)

    def _shift_append_numpy(self, buf: np.ndarray, row: np.ndarray):
        buf[:-1] = buf[1:]
        buf[-1] = row

    def _shift_append_torch(self, tbuf: torch.Tensor, row_np: np.ndarray):
        tbuf[:, :-1].copy_(tbuf[:, 1:].clone())
        trow = torch.from_numpy(row_np).to(self.device, dtype=DTYPE).view(1, 1, -1)
        tbuf[:, -1:].copy_(trow)

    def reset(self, start_i):
        self.i = int(start_i)
        self.step_i = 0
        self.balance = float(INIT_BALANCE)
        self.peak_balance = float(INIT_BALANCE)
        self.pos = 0
        self.entry_price = 0.0
        self.last_price = 0.0
        self.pos_notional = 0.0
        self.pos_qty = 0.0
        self.entry_step_i = 0
        self.cooldown_remaining = 0
        self.vote.clear()
        for _ in range(VOTE_N):
            self.vote.append(0)
        self.last_exec = 0
        self.done = False
        self.flat_hold_steps = 0

        base_store = self.stores["250ms"]
        t_ms = base_store.ts_of_idx(self.i)

        self.idx_20 = map_base_ts_to_scale_idx(self.stores["20s"], t_ms)
        self.idx_5m = map_base_ts_to_scale_idx(self.stores["5m"], t_ms)
        self.idx_1h = map_base_ts_to_scale_idx(self.stores["1h"], t_ms)
        if self.idx_20 < (SEQ_LEN - 1) or self.idx_5m < (SEQ_LEN - 1) or self.idx_1h < (SEQ_LEN - 1):
            raise IndexError("reset start_i too early for one or more scales (need >= SEQ_LEN-1 history)")

        x250 = self.stores["250ms"].get_seq_end(self.i, SEQ_LEN)
        x20 = self.stores["20s"].get_seq_end(self.idx_20, SEQ_LEN)
        x5m = self.stores["5m"].get_seq_end(self.idx_5m, SEQ_LEN)
        x1h = self.stores["1h"].get_seq_end(self.idx_1h, SEQ_LEN)

        self.buf_250 = self._norm_scale("250ms", x250).astype(np.float32, copy=False)
        self.buf_20 = self._norm_scale("20s", x20).astype(np.float32, copy=False)
        self.buf_5m = self._norm_scale("5m", x5m).astype(np.float32, copy=False)
        self.buf_1h = self._norm_scale("1h", x1h).astype(np.float32, copy=False)

        if self.torch_obs:
            self._sync_tbuf_full()

        return self._obs()

    def step(self, action_proposed: int):
        if self.done:
            raise RuntimeError("step after done")

        action_proposed = int(action_proposed)
        if action_proposed < 0 or action_proposed >= N_ACTIONS:
            action_proposed = 0

        bal_start = float(self.balance)

        use_vote = USE_VOTE_FILTER_TRAIN if MODE == "train" else USE_VOTE_FILTER_EVAL

        self.vote.append(int(action_proposed))
        if use_vote:
            exec_a = vote_exec_action(self.vote, self.last_exec)
        else:
            exec_a = int(action_proposed)   

        _, x_row = self.stores["250ms"].get_row(self.i)
        price_now = self._get_price(x_row)

        bal_mid = float(self.balance)
        if self.pos != 0 and self.last_price > 0:
            pnl = self.pos_qty * self.pos * (price_now - self.last_price)
            self.balance = bal_mid + pnl
        self.last_price = price_now
        if int(self.cooldown_remaining) > 0 and exec_a in (1, 2):
            exec_a = 0

        if self.pos == 1 and exec_a == 2:
            exec_a = 3
        if self.pos == -1 and exec_a == 1:
            exec_a = 3

        if exec_a == 3 and self.pos != 0 and (not self._can_close()):
            exec_a = 0

        if self.pos == 0 and exec_a == 0:
            self.flat_hold_steps += 1
        else:
            self.flat_hold_steps = 0
        self.last_exec = int(exec_a)
        
        if exec_a == 1:
            if self.pos == 0:
                self.balance *= self._fee_mult(FEE_OPEN)
                self.balance *= self._penalty_mult_entry()
                self.pos = 1
                self.entry_price = price_now
                self.pos_notional = self.balance * float(LEVERAGE)
                self.pos_qty = self.pos_notional / max(price_now, EPS)
                self.entry_step_i = int(self.step_i)

        elif exec_a == 2:
            if self.pos == 0:
                self.balance *= self._fee_mult(FEE_OPEN)
                self.balance *= self._penalty_mult_entry()
                self.pos = -1
                self.entry_price = price_now
                self.pos_notional = self.balance * float(LEVERAGE)
                self.pos_qty = self.pos_notional / max(price_now, EPS)
                self.entry_step_i = int(self.step_i)

        elif exec_a == 3:
            if self.pos != 0:
                self.balance *= self._fee_mult(FEE_CLOSE)
                self.pos = 0
                self.entry_price = 0.0
                self.pos_notional = 0.0
                self.pos_qty = 0.0
                self.cooldown_remaining = int(COOLDOWN_STEPS)

        next_i = self.i + 1
        if next_i >= self.stores["250ms"].total_rows:
            done = True
            if self.pos != 0:
                self.balance *= self._fee_mult(FEE_CLOSE)
                self.pos = 0
                self.entry_price = 0.0
                self.pos_notional = 0.0
                self.pos_qty = 0.0
            self.balance = max(0.0, float(self.balance))
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            reward = float(math.log((self.balance + EPS) / (bal_start + EPS)))
            self.i = self.stores["250ms"].total_rows - 1
            self.done = True
            info = {"balance": float(self.balance), "pos": int(self.pos), 
                    "exec_action": int(exec_a)}
            return self._obs(), reward, True, info

        _, x_next_row = self.stores["250ms"].get_row(next_i)
        price_next = self._get_price(x_next_row)

        self.balance = max(0.0, float(self.balance))
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        reward = float(math.log((self.balance + EPS) / (bal_start + EPS)))

        self.i = next_i
        self.step_i += 1
        if int(self.cooldown_remaining) > 0:
            self.cooldown_remaining -= 1

        base_store = self.stores["250ms"]
        t_ms = base_store.ts_of_idx(self.i)

        new_idx_20 = map_base_ts_to_scale_idx(self.stores["20s"], t_ms)
        new_idx_5m = map_base_ts_to_scale_idx(self.stores["5m"], t_ms)
        new_idx_1h = map_base_ts_to_scale_idx(self.stores["1h"], t_ms)

        row250 = self._norm_row("250ms", x_next_row).astype(np.float32, copy=False)
        self._shift_append_numpy(self.buf_250, row250)
        if self.torch_obs:
            self._shift_append_torch(self.tbuf_250, row250)

        if new_idx_20 != self.idx_20:
            self.idx_20 = new_idx_20
            _, x20row = self.stores["20s"].get_row(self.idx_20)
            row20 = self._norm_row("20s", x20row).astype(np.float32, copy=False)
            self._shift_append_numpy(self.buf_20, row20)
            if self.torch_obs:
                self._shift_append_torch(self.tbuf_20, row20)

        if new_idx_5m != self.idx_5m:
            self.idx_5m = new_idx_5m
            _, x5mrow = self.stores["5m"].get_row(self.idx_5m)
            row5m = self._norm_row("5m", x5mrow).astype(np.float32, copy=False)
            self._shift_append_numpy(self.buf_5m, row5m)
            if self.torch_obs:
                self._shift_append_torch(self.tbuf_5m, row5m)

        if new_idx_1h != self.idx_1h:
            self.idx_1h = new_idx_1h
            _, x1hrow = self.stores["1h"].get_row(self.idx_1h)
            row1h = self._norm_row("1h", x1hrow).astype(np.float32, copy=False)
            self._shift_append_numpy(self.buf_1h, row1h)
            if self.torch_obs:
                self._shift_append_torch(self.tbuf_1h, row1h)

        thresh = float(LOSS_THRESHOLD_FRAC) * self.peak_balance
        done = False
        if self.balance <= thresh:
            done = True
        if self.step_i >= self.episode_steps:
            done = True

        if done and self.pos != 0:
            self.balance *= self._fee_mult(FEE_CLOSE)
            self.pos = 0
            self.entry_price = 0.0
            self.pos_notional = 0.0
            self.pos_qty = 0.0
            self.balance = max(0.0, float(self.balance))
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            reward = float(math.log((self.balance + EPS) / (bal_start + EPS)))

        self.done = done
        info = {"balance": float(self.balance), "pos": int(self.pos), 
                "exec_action": int(exec_a)}
        if (not done) and self.pos == 0 and exec_a == 0 and \
                                    self.flat_hold_steps > IDLE_PENALTY_AFTER_STEPS:
            excess = self.flat_hold_steps - IDLE_PENALTY_AFTER_STEPS
            reward -= IDLE_PENALTY_BASE * (excess / self.episode_steps)
        return self._obs(), reward, done, info
    
    
# =========================
# REPLAY
# =========================
class ReplayBuffer:
    def __init__(self, cap: int, ctx_dim: int, share_memory: bool = True):
        self.cap = int(cap)
        self.ctx_dim = int(ctx_dim)
        self.ptr = 0
        self.size = 0
        self.entry_price = 0.0

        # CPU tensors (optionally shared so DataLoader workers can read it on Windows)
        self.i = torch.empty((self.cap,), dtype=torch.int64)
        self.a = torch.empty((self.cap,), dtype=torch.int64)
        self.r = torch.empty((self.cap,), dtype=torch.float32)
        self.d = torch.empty((self.cap,), dtype=torch.float32)
        self.ctx = torch.empty((self.cap, self.ctx_dim), dtype=torch.float32)
        self.nctx = torch.empty((self.cap, self.ctx_dim), dtype=torch.float32)

        if share_memory:
            self.i.share_memory_()
            self.a.share_memory_()
            self.r.share_memory_()
            self.d.share_memory_()
            self.ctx.share_memory_()
            self.nctx.share_memory_()

    def add(self, i, a, r, d, ctx, nctx):
        p = self.ptr
        self.i[p] = int(i)
        self.a[p] = int(a)
        self.r[p] = float(r)
        self.d[p] = float(d)

        if isinstance(ctx, np.ndarray):
            self.ctx[p].copy_(torch.from_numpy(ctx).to(dtype=torch.float32))
        else:
            self.ctx[p].copy_(ctx.to(device="cpu", dtype=torch.float32))

        if isinstance(nctx, np.ndarray):
            self.nctx[p].copy_(torch.from_numpy(nctx).to(dtype=torch.float32))
        else:
            self.nctx[p].copy_(nctx.to(device="cpu", dtype=torch.float32))

        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def state_dict(self) -> dict:
        return {
            "cap": int(self.cap),
            "ctx_dim": int(self.ctx_dim),
            "ptr": int(self.ptr),
            "size": int(self.size),
            "i": self.i,
            "a": self.a,
            "r": self.r,
            "d": self.d,
            "ctx": self.ctx,
            "nctx": self.nctx,
        }

    def load_state_dict(self, sd: dict):
        if int(sd["cap"]) != int(self.cap) or int(sd["ctx_dim"]) != int(self.ctx_dim):
            raise RuntimeError("ReplayBuffer shape mismatch")

        self.ptr = int(sd["ptr"])
        self.size = int(sd["size"])

        self.i.copy_(sd["i"])
        self.a.copy_(sd["a"])
        self.r.copy_(sd["r"])
        self.d.copy_(sd["d"])
        self.ctx.copy_(sd["ctx"])
        self.nctx.copy_(sd["nctx"])
