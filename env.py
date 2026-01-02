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
    # tie break: most recent among tied candidates
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
        self.last_mid = 0.0
        self.pos_notional = 0.0
        self.pos_qty = 0.0
        self.entry_step_i = 0
        self.cooldown_remaining = 0
        self.vote = deque(maxlen=VOTE_N)
        self.last_exec = 0
        self.done = False

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
    
    def _entry_penalty(self):
        return float(TRADE_PENALTY_ENTRY)

    def _can_change_pos(self):
        if self.pos == 0:
            return True
        held = int(self.step_i) - int(self.entry_step_i)
        return held >= int(MIN_HOLD_STEPS)
    
    def _half_spread(self):
        return 0.5 * float(SPREAD_BPS) * 1e-4

    def _slip(self):
        return float(SLIPPAGE_BPS) * 1e-4

    def _exec_price(self, mid: float, side: int):
        # side: +1 buy, -1 sell
        hs = self._half_spread()
        sl = self._slip()
        if side >= 0:
            return float(mid) * (1.0 + hs + sl)
        return float(mid) * (1.0 - hs - sl)

    def _fee_cost(self, notional: float, fee_rate: float):
        return float(notional) * float(fee_rate)

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
        hold_left = max(0, MIN_HOLD_STEPS - (self.step_i - self.entry_step_i)) if self.pos != 0 else 0
        ctx = np.concatenate([
            np.array([
                float(self.pos),
                float(self.balance / INIT_BALANCE),
                float(unrealized),
                float(realized),
                float(hold_left / MIN_HOLD_STEPS),
                float(self.cooldown_remaining > 0),
            ], dtype=np.float32),
            last,
            q_oh.reshape(-1)
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
        self.last_mid = 0.0
        self.pos_notional = 0.0
        self.pos_qty = 0.0
        self.entry_step_i = 0
        self.cooldown_remaining = 0
        self.vote.clear()
        for _ in range(VOTE_N):
            self.vote.append(0)
        self.last_exec = 0
        self.done = False

        base_store = self.stores["250ms"]
        t_ms = base_store.ts_of_idx(self.i)

        self.idx_20 = map_base_ts_to_scale_idx_clamped(self.stores["20s"], t_ms)
        self.idx_5m = map_base_ts_to_scale_idx_clamped(self.stores["5m"], t_ms)
        self.idx_1h = map_base_ts_to_scale_idx_clamped(self.stores["1h"], t_ms)
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

        did_open = False
        did_close = False
        did_flip = False
        fee_paid = 0.0
        spread_paid = 0.0

        action_proposed = int(action_proposed)
        if action_proposed < 0 or action_proposed >= N_ACTIONS:
            action_proposed = 0

        bal_start = float(self.balance)
        reward = 0.0

        use_vote = USE_VOTE_FILTER_TRAIN if MODE == "train" else USE_VOTE_FILTER_EVAL

        self.vote.append(int(action_proposed))
        if use_vote:
            exec_a = vote_exec_action(self.vote, self.last_exec)
        else:
            exec_a = int(action_proposed)   

        _, x_row = self.stores["250ms"].get_row(self.i)
        mid_now = self._get_price(x_row)

        # Mark-to-market on mid (continuous PnL).
        if self.pos != 0 and self.last_mid > 0:
            pnl = self.pos_qty * self.pos * (mid_now - self.last_mid)
            self.balance = float(self.balance) + float(pnl)
        self.last_mid = float(mid_now)

        # Action semantics:
        # 0 HOLD: keep current position
        # 1 LONG: target long
        # 2 SHORT: target short
        # 3 FLAT: target flat
        desired_pos = int(self.pos)
        if exec_a == 1:
            desired_pos = 1
        elif exec_a == 2:
            desired_pos = -1
        elif exec_a == 3:
            desired_pos = 0

        # Cooldown blocks any position change.
        if int(self.cooldown_remaining) > 0 and desired_pos != int(self.pos):
            desired_pos = int(self.pos)
            exec_a = 0

        # Enforce minimum hold time before any change (close or flip) while in a position.
        if desired_pos != int(self.pos) and (not self._can_change_pos()):
            desired_pos = int(self.pos)
            exec_a = 0

        # Helpers for close/open with fees + spread/slippage.
        def do_close():
            nonlocal did_close, fee_paid, spread_paid
            if self.pos == 0:
                return
            # Close at executable price (sell long / buy short).
            side = -1 if self.pos > 0 else 1
            px_exec = self._exec_price(mid_now, side=side)
            # Spread/slippage cost relative to mid.
            spread_cost = abs(px_exec - float(mid_now)) * abs(float(self.pos_qty))
            self.balance = float(self.balance) - float(spread_cost)
            spread_paid += float(spread_cost)

            notional = abs(float(self.pos_qty) * float(px_exec))
            fee = self._fee_cost(notional, FEE_CLOSE)
            self.balance = float(self.balance) - float(fee)
            fee_paid += float(fee)

            self.pos = 0
            self.entry_price = 0.0
            self.pos_notional = 0.0
            self.pos_qty = 0.0
            self.entry_step_i = int(self.step_i)
            self.cooldown_remaining = int(COOLDOWN_STEPS)
            did_close = True

        def do_open(target_pos: int):
            nonlocal did_open, fee_paid, spread_paid
            if int(target_pos) == 0:
                return
            if self.pos != 0:
                return

            # Open at executable price (buy long / sell short).
            side = 1 if int(target_pos) > 0 else -1
            px_exec = self._exec_price(mid_now, side=side)

            notional = float(self.balance) * float(LEVERAGE) * float(POS_FRACTION)
            notional = max(0.0, float(notional))
            if notional <= 0.0:
                return

            fee = self._fee_cost(notional, FEE_OPEN)
            self.balance = float(self.balance) - float(fee)
            fee_paid += float(fee)

            qty = float(notional) / max(float(px_exec), EPS)

            # Immediate spread/slippage cost relative to mid.
            spread_cost = abs(px_exec - float(mid_now)) * abs(qty)
            self.balance = float(self.balance) - float(spread_cost)
            spread_paid += float(spread_cost)

            self.pos = int(target_pos)
            self.entry_price = float(px_exec)
            self.pos_notional = float(notional)
            self.pos_qty = float(qty)
            self.entry_step_i = int(self.step_i)
            did_open = True

        # Apply transitions.
        if desired_pos != int(self.pos):
            if self.pos != 0 and desired_pos != 0 and int(desired_pos) == -int(self.pos):
                # Flip: close then open opposite.
                do_close()
                do_open(desired_pos)
                did_flip = True
            elif desired_pos == 0:
                do_close()
            elif self.pos == 0 and desired_pos != 0:
                do_open(desired_pos)       

        self.last_exec = int(exec_a)

        next_i = self.i + 1
        if next_i >= self.stores["250ms"].total_rows:
            done = True
            prev_pos = int(self.pos)

            if self.pos != 0:
                # Force close on terminal.
                do_close()
            self.balance = max(0.0, float(self.balance))
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            reward = float(math.log((self.balance + EPS) / (bal_start + EPS)))
            if did_open:
                reward -= float(TRADE_PENALTY_ENTRY)
            if did_flip:
                reward -= float(TRADE_PENALTY_FLIP)
            reward -= float(POS_HOLD_PENALTY) * abs(float(prev_pos))
            self.i = self.stores["250ms"].total_rows - 1
            self.done = True
            info = {"balance": float(self.balance), "pos": int(self.pos), 
                    "exec_action": int(exec_a),
                    "trade_opened": bool(did_open),
                    "trade_closed": bool(did_close),
                    "trade_flipped": bool(did_flip),
                    "fee_paid": float(fee_paid),
                    "spread_paid": float(spread_paid)}
            return self._obs(), reward, True, info

        _, x_next_row = self.stores["250ms"].get_row(next_i)
        _ = self._get_price(x_next_row)

        self.balance = max(0.0, float(self.balance))
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        reward = float(math.log((self.balance + EPS) / (bal_start + EPS)))

        if did_open:
            reward -= float(TRADE_PENALTY_ENTRY)
        if did_flip:
            reward -= float(TRADE_PENALTY_FLIP)
        reward -= float(POS_HOLD_PENALTY) * abs(float(self.pos))

        self.i = next_i
        self.step_i += 1
        if int(self.cooldown_remaining) > 0:
            self.cooldown_remaining -= 1

        base_store = self.stores["250ms"]
        t_ms = base_store.ts_of_idx(self.i)

        new_idx_20 = map_base_ts_to_scale_idx_clamped(self.stores["20s"], t_ms)
        new_idx_5m = map_base_ts_to_scale_idx_clamped(self.stores["5m"], t_ms)
        new_idx_1h = map_base_ts_to_scale_idx_clamped(self.stores["1h"], t_ms)

        row250 = self._norm_row("250ms", x_next_row).astype(np.float32, copy=False)
        self._shift_append_numpy(self.buf_250, row250)
        if self.torch_obs:
            self._shift_append_torch(self.tbuf_250, row250)

        if new_idx_20 != self.idx_20 and new_idx_20 >= 0:
            self.idx_20 = new_idx_20
            _, x20row = self.stores["20s"].get_row(self.idx_20)
            row20 = self._norm_row("20s", x20row).astype(np.float32, copy=False)
            self._shift_append_numpy(self.buf_20, row20)
            if self.torch_obs:
                self._shift_append_torch(self.tbuf_20, row20)

        if new_idx_5m != self.idx_5m and new_idx_5m >= 0:
            self.idx_5m = new_idx_5m
            _, x5mrow = self.stores["5m"].get_row(self.idx_5m)
            row5m = self._norm_row("5m", x5mrow).astype(np.float32, copy=False)
            self._shift_append_numpy(self.buf_5m, row5m)
            if self.torch_obs:
                self._shift_append_torch(self.tbuf_5m, row5m)

        if new_idx_1h != self.idx_1h and new_idx_1h >= 0:
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
        prev_pos = int(self.pos)

        if self.step_i >= self.episode_steps:
            done = True

        if done and self.pos != 0:
            do_close()
            self.balance = max(0.0, float(self.balance))
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            reward = float(math.log((self.balance + EPS) / (bal_start + EPS)))
            if did_open:
                reward -= float(TRADE_PENALTY_ENTRY)
            if did_flip:
                reward -= float(TRADE_PENALTY_FLIP)
            reward -= float(POS_HOLD_PENALTY) * abs(float(prev_pos))

        self.done = done
        info = {"balance": float(self.balance), "pos": int(self.pos), 
                "exec_action": int(exec_a),
                "trade_opened": bool(did_open),
                "trade_closed": bool(did_close),
                "trade_flipped": bool(did_flip),
                "fee_paid": float(fee_paid),
                "spread_paid": float(spread_paid)}
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
