import numpy as np
from collections import deque
import json

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
        self.pos = 0
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
        self.pos = 0
        self.vote.clear()
        for _ in range(VOTE_N):
            self.vote.append(0)
        self.last_exec = 0
        self.done = False

        base_store = self.stores["250ms"]
        t_ms = base_store.ts_of_idx(self.i)

        self.idx_20 = map_base_ts_to_scale_idx(self.stores["20s"], t_ms)
        self.idx_5m = map_base_ts_to_scale_idx(self.stores["5m"], t_ms)
        self.idx_1h = map_base_ts_to_scale_idx(self.stores["1h"], t_ms)

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

        if use_vote:
            exec_a = vote_exec_action(self.vote, self.last_exec)  
        else:
            exec_a = int(action_proposed)                         

        self.last_exec = int(exec_a)
        self.vote.append(int(action_proposed))                    

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
            reward = float((self.balance - bal_start) / (bal_start + EPS))
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
        reward = float((self.balance - bal_start) / (bal_start + EPS))

        self.i = next_i
        self.step_i += 1

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
    def __init__(self, cap: int, ctx_dim: int, replay_dir: Path):
        self.cap = int(cap)
        self.ctx_dim = int(ctx_dim)
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.replay_dir / "replay_meta.json"

        def mm(path: Path, dtype, shape, mode):
            return np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)

        def wipe_dir():
            for name in ["i.dat", "a.dat", "r.dat", "d.dat", "ctx.dat", "nctx.dat", "replay_meta.json"]:
                p = self.replay_dir / name
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass

        mode = "w+"
        self.ptr = 0
        self.size = 0

        if self.meta_path.exists():
            try:
                with self.meta_path.open("r") as f:
                    meta = json.load(f)
                cap_ok = int(meta.get("cap", -1)) == self.cap
                ctx_ok = int(meta.get("ctx_dim", -1)) == self.ctx_dim
                if cap_ok and ctx_ok:
                    self.ptr = int(meta.get("ptr", 0))
                    self.size = int(meta.get("size", 0))
                    mode = "r+"
                else:
                    wipe_dir()
            except Exception:
                wipe_dir()

        self.i = mm(self.replay_dir / "i.dat", np.int64, (self.cap,), mode)
        self.a = mm(self.replay_dir / "a.dat", np.int64, (self.cap,), mode)
        self.r = mm(self.replay_dir / "r.dat", np.float32, (self.cap,), mode)
        self.d = mm(self.replay_dir / "d.dat", np.float32, (self.cap,), mode)
        self.ctx = mm(self.replay_dir / "ctx.dat", np.float32, (self.cap, self.ctx_dim), mode)
        self.nctx = mm(self.replay_dir / "nctx.dat", np.float32, (self.cap, self.ctx_dim), mode)

        self._save_meta()

    def _save_meta(self):
        meta = {"cap": int(self.cap), "ctx_dim": int(self.ctx_dim), "ptr": int(self.ptr), "size": int(self.size)}
        with self.meta_path.open("w") as f:
            json.dump(meta, f)

    def flush(self):
        self.i.flush()
        self.a.flush()
        self.r.flush()
        self.d.flush()
        self.ctx.flush()
        self.nctx.flush()
        self._save_meta()

    def add(self, i, a, r, d, ctx, nctx):
        p = self.ptr
        self.i[p] = int(i)
        self.a[p] = int(a)
        self.r[p] = float(r)
        self.d[p] = float(d)
        self.ctx[p] = ctx.astype(np.float32, copy=False)
        self.nctx[p] = nctx.astype(np.float32, copy=False)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=int(batch))
        return (
            self.i[idx].copy(),
            self.a[idx].copy(),
            self.r[idx].copy(),
            self.d[idx].copy(),
            self.ctx[idx].copy(),
            self.nctx[idx].copy(),
        )

    def set_state(self, ptr: int, size: int):
        self.ptr = int(ptr)
        self.size = int(size)
        self._save_meta()
