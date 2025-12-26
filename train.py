import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import multiprocessing as mp

from config import *
from utils import *
from model import *
from dataset import *
from env import *

if NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# =========================
# TRAIN
# =========================
@torch.no_grad()
def select_action(actor: ActorNet, obs, device: str, deterministic: bool = False):
    s250 = obs["250ms"]
    s20 = obs["20s"]
    s5m = obs["5m"]
    s1h = obs["1h"]

    if not torch.is_tensor(s250):
        s250 = torch.from_numpy(s250).unsqueeze(0).to(device, dtype=DTYPE)
        s20 = torch.from_numpy(s20).unsqueeze(0).to(device, dtype=DTYPE)
        s5m = torch.from_numpy(s5m).unsqueeze(0).to(device, dtype=DTYPE)
        s1h = torch.from_numpy(s1h).unsqueeze(0).to(device, dtype=DTYPE)

    ctx = torch.from_numpy(obs["ctx"]).unsqueeze(0).to(device, dtype=DTYPE)

    logits = actor(s250, s20, s5m, s1h, ctx)
    if deterministic:
        return int(torch.argmax(logits, dim=-1).item())
    dist = torch.distributions.Categorical(logits=logits)
    return int(dist.sample().item())


def load_batch_s_and_ns(stores, i_arr, seq_len):
    i_arr = np.asarray(i_arr, dtype=np.int64)
    b = len(i_arr)

    base = stores["250ms"]
    s20s = stores["20s"]
    s5s = stores["5m"]
    s1s = stores["1h"]

    base_first = int(base.first_ts)
    base_dt = int(base.interval_ms)

    t = base_first + i_arr * base_dt
    t_next = t + base_dt

    i20 = ((t - int(s20s.first_ts)) // int(s20s.interval_ms)).astype(np.int64)
    i5  = ((t - int(s5s.first_ts))  // int(s5s.interval_ms)).astype(np.int64)
    i1  = ((t - int(s1s.first_ts))  // int(s1s.interval_ms)).astype(np.int64)

    i20n = ((t_next - int(s20s.first_ts)) // int(s20s.interval_ms)).astype(np.int64)
    i5n  = ((t_next - int(s5s.first_ts))  // int(s5s.interval_ms)).astype(np.int64)
    i1n  = ((t_next - int(s1s.first_ts))  // int(s1s.interval_ms)).astype(np.int64)

    s250 = base.get_seqs_end_batch(i_arr, seq_len)
    s20  = s20s.get_seqs_end_batch(i20, seq_len)
    s5m  = s5s.get_seqs_end_batch(i5, seq_len)
    s1h  = s1s.get_seqs_end_batch(i1, seq_len)

    ns250 = np.empty_like(s250)
    ns20  = np.empty_like(s20)
    ns5m  = np.empty_like(s5m)
    ns1h  = np.empty_like(s1h)

    ns250[:, :-1] = s250[:, 1:]
    ns250[:, -1] = base.get_rows_batch(i_arr + 1)

    ns20[:] = s20
    ch = (i20n != i20)
    if ch.any():
        ns20[ch, :-1] = s20[ch, 1:]
        ns20[ch, -1] = s20s.get_rows_batch(i20n[ch])

    ns5m[:] = s5m
    ch = (i5n != i5)
    if ch.any():
        ns5m[ch, :-1] = s5m[ch, 1:]
        ns5m[ch, -1] = s5s.get_rows_batch(i5n[ch])

    ns1h[:] = s1h
    ch = (i1n != i1)
    if ch.any():
        ns1h[ch, :-1] = s1h[ch, 1:]
        ns1h[ch, -1] = s1s.get_rows_batch(i1n[ch])

    return s250, s20, s5m, s1h, ns250, ns20, ns5m, ns1h


@torch.no_grad()
def evaluate_policy(actor: nn.Module, stores, norm, start_lo: int, start_hi: int, total_steps: int, device: str):
    actor.eval()
    env = TradingEnv(stores, norm, close_idx_in_x250=8, device=device, torch_obs=True)

    steps = 0
    ep = 0
    sum_r = 0.0
    sum_bal = 0.0

    while steps < total_steps:
        start_i = random.randint(start_lo, start_hi)
        obs = env.reset(start_i)

        done = False
        while (not done) and (steps < total_steps):
            a = select_action(actor, obs, device=device, deterministic=True)
            obs, r, done, info = env.step(a)
            sum_r += float(r)
            steps += 1

        sum_bal += float(env.balance)
        ep += 1

    actor.train()
    avg_r_step = sum_r / max(1, steps)
    avg_bal = sum_bal / max(1, ep)
    return {"steps": int(steps), "episodes": int(ep), "avg_reward_per_step": float(avg_r_step), "avg_final_balance": float(avg_bal)}


class ReplayBatchIterable(IterableDataset):
    def __init__(self, out_root: Path, replay_dir: Path, ctx_dim: int, batch_size: int, seq_len: int, rb_size_value):
        super().__init__()
        self.out_root = Path(out_root)
        self.replay_dir = Path(replay_dir)
        self.ctx_dim = int(ctx_dim)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.rb_size_value = rb_size_value

        self._inited = False
        self.stores = None
        self.cap = None

        self.i = None
        self.a = None
        self.r = None
        self.d = None
        self.ctx = None
        self.nctx = None

    def _init_once(self):
        if self._inited:
            return

        with (self.replay_dir / "replay_meta.json").open("r") as f:
            meta = json.load(f)
        self.cap = int(meta["cap"])

        def mm(path: Path, dtype, shape):
            return np.memmap(str(path), dtype=dtype, mode="r", shape=shape)

        self.i = mm(self.replay_dir / "i.dat", np.int64, (self.cap,))
        self.a = mm(self.replay_dir / "a.dat", np.int64, (self.cap,))
        self.r = mm(self.replay_dir / "r.dat", np.float32, (self.cap,))
        self.d = mm(self.replay_dir / "d.dat", np.float32, (self.cap,))
        self.ctx = mm(self.replay_dir / "ctx.dat", np.float32, (self.cap, self.ctx_dim))
        self.nctx = mm(self.replay_dir / "nctx.dat", np.float32, (self.cap, self.ctx_dim))

        self.stores = {
            "250ms": ScaleStore(self.out_root / "250ms"),
            "20s": ScaleStore(self.out_root / "20s"),
            "5m": ScaleStore(self.out_root / "5m"),
            "1h": ScaleStore(self.out_root / "1h"),
        }

        self._inited = True

    def __iter__(self):
        self._init_once()
        wi = get_worker_info()
        wid = 0 if wi is None else int(wi.id)
        rng = np.random.default_rng(SEED + 10007 * wid)

        while True:
            size = int(self.rb_size_value.value)
            if size < self.batch_size:
                time.sleep(0.01)
                continue

            ridx = rng.integers(0, size, size=self.batch_size, dtype=np.int64)

            i_end = self.i[ridx].astype(np.int64, copy=False)
            a = self.a[ridx].astype(np.int64, copy=False)
            r = self.r[ridx].astype(np.float32, copy=False)
            d = self.d[ridx].astype(np.float32, copy=False)
            ctx = self.ctx[ridx].astype(np.float32, copy=False)
            nctx = self.nctx[ridx].astype(np.float32, copy=False)

            s250, s20, s5m, s1h, ns250, ns20, ns5m, ns1h = load_batch_s_and_ns(self.stores, i_end, self.seq_len)

            yield (
                torch.from_numpy(s250),
                torch.from_numpy(s20),
                torch.from_numpy(s5m),
                torch.from_numpy(s1h),
                torch.from_numpy(ns250),
                torch.from_numpy(ns20),
                torch.from_numpy(ns5m),
                torch.from_numpy(ns1h),
                torch.from_numpy(a),
                torch.from_numpy(r),
                torch.from_numpy(d),
                torch.from_numpy(ctx),
                torch.from_numpy(nctx),
            )


def train():
    seed_all(SEED)

    if DEVICE == "cuda":
        if USE_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    with (OUT_ROOT / "dataset_meta.json").open("r") as f:
        ds_meta = json.load(f)

    norm = ds_meta["norm"]
    train_end_ms = int(ds_meta["train_end_ms"])
    val_end_ms = int(ds_meta["val_end_ms"])

    stores = {
        "250ms": ScaleStore(OUT_ROOT / "250ms"),
        "20s": ScaleStore(OUT_ROOT / "20s"),
        "5m": ScaleStore(OUT_ROOT / "5m"),
        "1h": ScaleStore(OUT_ROOT / "1h"),
    }

    close_idx = 8
    ctx_dim = 2 + N_ACTIONS + (VOTE_N * N_ACTIONS)

    rb = ReplayBuffer(REPLAY_CAP, ctx_dim, REPLAY_DIR)
    rb_size_value = mp.Value("i", 0)
    rb_size_value.value = int(rb.size)

    def make_loader():
        ds = ReplayBatchIterable(OUT_ROOT, REPLAY_DIR, ctx_dim, BATCH_SIZE, SEQ_LEN, rb_size_value)
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        )

    batch_loader = make_loader()
    batch_iter = iter(batch_loader)

    actor = ActorNet(stores["250ms"].d, stores["20s"].d, stores["5m"].d, stores["1h"].d, ctx_dim).to(DEVICE)
    critic = CriticNet(stores["250ms"].d, stores["20s"].d, stores["5m"].d, stores["1h"].d, ctx_dim).to(DEVICE)
    critic_tgt = CriticNet(stores["250ms"].d, stores["20s"].d, stores["5m"].d, stores["1h"].d, ctx_dim).to(DEVICE)
    critic_tgt.load_state_dict(critic.state_dict())

    if USE_COMPILE and hasattr(torch, "compile"):
        actor = torch.compile(actor)
        critic = torch.compile(critic)
        critic_tgt = torch.compile(critic_tgt)

    print(f"actor params:", f"{count_params(actor):,}")
    print(f"critic params:", f"{count_params(critic):,}")

    opt_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    log_alpha = torch.tensor(math.log(ALPHA_INIT), device=DEVICE, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=LR_ALPHA)

    mean = {k: torch.tensor(norm[k]["mean"], device=DEVICE, dtype=torch.float32) for k in norm}
    std = {k: torch.tensor(norm[k]["std"], device=DEVICE, dtype=torch.float32) for k in norm}

    def norm_t(scale, x):
        return (x - mean[scale]) / std[scale]

    base_store = stores["250ms"]
    base_train_end = first_idx_ge_regular(base_store.first_ts, base_store.interval_ms, base_store.total_rows, train_end_ms)
    base_val_end = first_idx_ge_regular(base_store.first_ts, base_store.interval_ms, base_store.total_rows, val_end_ms)

    req_t = max(
        stores["250ms"].first_ts + (SEQ_LEN - 1) * stores["250ms"].interval_ms,
        stores["20s"].first_ts + (SEQ_LEN - 1) * stores["20s"].interval_ms,
        stores["5m"].first_ts + (SEQ_LEN - 1) * stores["5m"].interval_ms,
        stores["1h"].first_ts + (SEQ_LEN - 1) * stores["1h"].interval_ms,
    )
    warmup_min_i = first_idx_ge_regular(base_store.first_ts, base_store.interval_ms, base_store.total_rows, req_t)

    env = TradingEnv(stores, norm, close_idx_in_x250=close_idx, device=DEVICE, torch_obs=True)
    episode_steps = env.episode_steps

    def sample_start(lo: int, hi: int):
        while True:
            i = random.randint(lo, hi)
            t_ms = base_store.ts_of_idx(i)
            i20 = map_base_ts_to_scale_idx(stores["20s"], t_ms)
            i5m = map_base_ts_to_scale_idx(stores["5m"], t_ms)
            i1h = map_base_ts_to_scale_idx(stores["1h"], t_ms)
            if i20 < (SEQ_LEN - 1) or i5m < (SEQ_LEN - 1) or i1h < (SEQ_LEN - 1):
                continue
            return i

    def soft_update(tgt: nn.Module, src: nn.Module, tau: float):
        with torch.no_grad():
            for p_t, p in zip(tgt.parameters(), src.parameters()):
                p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def update_step():
        batch = next(batch_iter)
        (s250, s20, s5m, s1h, ns250, ns20, ns5m, ns1h, a, r, d, ctx, nctx) = batch

        s250 = s250.to(DEVICE, dtype=torch.float32, non_blocking=True)
        s20  = s20.to(DEVICE, dtype=torch.float32, non_blocking=True)
        s5m  = s5m.to(DEVICE, dtype=torch.float32, non_blocking=True)
        s1h  = s1h.to(DEVICE, dtype=torch.float32, non_blocking=True)

        ns250 = ns250.to(DEVICE, dtype=torch.float32, non_blocking=True)
        ns20  = ns20.to(DEVICE, dtype=torch.float32, non_blocking=True)
        ns5m  = ns5m.to(DEVICE, dtype=torch.float32, non_blocking=True)
        ns1h  = ns1h.to(DEVICE, dtype=torch.float32, non_blocking=True)

        a   = a.to(DEVICE, dtype=torch.long, non_blocking=True)
        r   = r.to(DEVICE, dtype=torch.float32, non_blocking=True)
        d   = d.to(DEVICE, dtype=torch.float32, non_blocking=True)
        ctx = ctx.to(DEVICE, dtype=torch.float32, non_blocking=True)
        nctx = nctx.to(DEVICE, dtype=torch.float32, non_blocking=True)

        s250 = norm_t("250ms", s250)
        s20  = norm_t("20s", s20)
        s5m  = norm_t("5m", s5m)
        s1h  = norm_t("1h", s1h)

        ns250 = norm_t("250ms", ns250)
        ns20  = norm_t("20s", ns20)
        ns5m  = norm_t("5m", ns5m)
        ns1h  = norm_t("1h", ns1h)

        alpha_det = log_alpha.exp().detach()

        amp_on = (DEVICE == "cuda") and USE_AMP_BF16
        amp_dtype = torch.bfloat16

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_on):
            with torch.no_grad():
                logits_n = actor(ns250, ns20, ns5m, ns1h, nctx)
                logp_n = F.log_softmax(logits_n, dim=-1)
                p_n = torch.exp(logp_n)

                tq1, tq2 = critic_tgt(ns250, ns20, ns5m, ns1h, nctx)
                tq = torch.min(tq1, tq2).float()
                v = (p_n.float() * (tq - alpha_det * logp_n.float())).sum(dim=-1)
                y = r + (1.0 - d) * float(GAMMA) * v

            q1, q2 = critic(s250, s20, s5m, s1h, ctx)
            q1 = q1.float()
            q2 = q2.float()
            q1_a = q1.gather(1, a.view(-1, 1)).squeeze(1)
            q2_a = q2.gather(1, a.view(-1, 1)).squeeze(1)
            critic_loss = F.mse_loss(q1_a, y) + F.mse_loss(q2_a, y)

        opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        opt_critic.step()

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_on):
            logits = actor(s250, s20, s5m, s1h, ctx)
            logp = F.log_softmax(logits, dim=-1)
            p = torch.exp(logp)

            with torch.no_grad():
                q1p, q2p = critic(s250, s20, s5m, s1h, ctx)
                qmin = torch.min(q1p, q2p).float()

            actor_loss = (p.float() * (alpha_det * logp.float() - qmin)).sum(dim=-1).mean()

        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()

        ent = -(p.float() * logp.float()).sum(dim=-1).mean()

        alpha = log_alpha.exp()
        alpha_loss = alpha * (ent.detach() - float(TARGET_ENTROPY))

        opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        opt_alpha.step()

        soft_update(critic_tgt, critic, TAU)

        return float(critic_loss.item()), float(actor_loss.item()), float(ent.item()), float(log_alpha.exp().item())

    def make_ckpt_state(ep: int, global_step: int):
        rng = {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        state = {
            "ep": int(ep),
            "global_step": int(global_step),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "critic_tgt": critic_tgt.state_dict(),
            "opt_actor": opt_actor.state_dict(),
            "opt_critic": opt_critic.state_dict(),
            "opt_alpha": opt_alpha.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "rb": {"ptr": int(rb.ptr), "size": int(rb.size), "cap": int(rb.cap), "ctx_dim": int(rb.ctx_dim), "replay_dir": str(rb.replay_dir.as_posix())},
            "rng": rng,
        }
        return state

    global_step = 0
    start_ep = 0

    if RESUME_PATH:
        ckpt = load_checkpoint(Path(RESUME_PATH), DEVICE)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        critic_tgt.load_state_dict(ckpt["critic_tgt"])
        opt_actor.load_state_dict(ckpt["opt_actor"])
        opt_critic.load_state_dict(ckpt["opt_critic"])
        opt_alpha.load_state_dict(ckpt["opt_alpha"])
        log_alpha.data.copy_(ckpt["log_alpha"].to(DEVICE))

        rb.set_state(ckpt["rb"]["ptr"], ckpt["rb"]["size"])

        random.setstate(ckpt["rng"]["py"])
        np.random.set_state(ckpt["rng"]["np"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])

        global_step = int(ckpt["global_step"])
        start_ep = int(ckpt["ep"]) + 1
        print("resumed:", RESUME_PATH, "ep", start_ep, "global_step", global_step)

    train_hi = max(warmup_min_i + 1, base_train_end - episode_steps - 2)
    val_lo = base_train_end
    val_hi = max(val_lo + 1, base_val_end - episode_steps - 2)

    for ep in range(start_ep, MAX_EPISODES):
        start_i = sample_start(warmup_min_i, train_hi)
        obs = env.reset(start_i)

        pbar = tqdm(total=env.episode_steps, desc=f"ep {ep}", dynamic_ncols=True, mininterval=0.2)

        ep_reward = 0.0
        ep_c = 0.0
        ep_a = 0.0
        ep_ent = 0.0
        ep_alpha = 0.0
        ep_updates = 0

        chunk = 0
        steps_in_ep = 0

        while True:
            if global_step < WARMUP_STEPS:
                a_prop = random.randrange(N_ACTIONS)
            else:
                a_prop = select_action(actor, obs, DEVICE, deterministic=False)

            ctx = obs["ctx"].copy()
            cur_i = env.i

            next_obs, rwd, done, info = env.step(a_prop)

            nctx = next_obs["ctx"].copy()

            rb.add(cur_i, int(a_prop), float(rwd), float(done), ctx, nctx)

            ep_reward += float(rwd)
            global_step += 1
            steps_in_ep += 1

            if rb.size >= BATCH_SIZE and global_step >= WARMUP_STEPS and (global_step % UPDATE_EVERY) == 0:
                rb_size_value.value = int(rb.size)
                for _ in range(UPDATES_PER_STEP):
                    cl, al, ent, alp = update_step()
                    ep_c += cl
                    ep_a += al
                    ep_ent += ent
                    ep_alpha += alp
                    ep_updates += 1

            chunk += 1
            if chunk >= PBAR_EVERY:
                pbar.update(chunk)
                chunk = 0
                pbar.set_postfix({
                    "bal": f"{info['balance']:6.2f}",
                    "R": f"{ep_reward:10.6f}",
                    "rb": f"{rb.size:08d}",
                })

            obs = next_obs
            if done:
                break

        if chunk > 0:
            pbar.update(chunk)
            pbar.set_postfix({
                "bal": f"{info['balance']:6.2f}",
                "R": f"{ep_reward:10.6f}",
                "rb": f"{rb.size:08d}",
            })

        pbar.close()

        if ep_updates > 0:
            ep_c /= ep_updates
            ep_a /= ep_updates
            ep_ent /= ep_updates
            ep_alpha /= ep_updates

        print(
            f"episode {ep} steps={steps_in_ep} reward={ep_reward:.4f} bal={env.balance:.2f} "
            f"critic={ep_c:.4f} actor={ep_a:.4f} ent={ep_ent:.4f} alpha={ep_alpha:.4f}"
        )

        if (CKPT_EVERY_EP > 0) and ((ep + 1) % CKPT_EVERY_EP == 0):
            rb.flush()
            ckpt_path = CHECKPOINT_DIR / f"ckpt_ep{ep:06d}.pt"
            save_checkpoint(ckpt_path, make_ckpt_state(ep, global_step))
            print("saved ckpt:", ckpt_path.as_posix())

        if (EVAL_EVERY_EP > 0) and ((ep + 1) % EVAL_EVERY_EP == 0):
            ev = evaluate_policy(actor, stores, norm, val_lo, val_hi, EVAL_TOTAL_STEPS, DEVICE)
            print("eval:", ev)