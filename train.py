import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from utils import *
from model import *
from dataset import *
from env import *


# =========================
# TRAIN
# =========================
@torch.no_grad()
def select_action(actor: ActorNet, obs, device: str):
    s250 = torch.tensor(obs["250ms"][None], device=device, dtype=DTYPE)
    s20 = torch.tensor(obs["20s"][None], device=device, dtype=DTYPE)
    s5m = torch.tensor(obs["5m"][None], device=device, dtype=DTYPE)
    s1h = torch.tensor(obs["1h"][None], device=device, dtype=DTYPE)
    ctx = torch.tensor(obs["ctx"][None], device=device, dtype=DTYPE)
    logits = actor(s250, s20, s5m, s1h, ctx)
    dist = torch.distributions.Categorical(logits=logits)
    return int(dist.sample().item())


def load_batch_seqs(stores, idx_end_arr, seq_len):
    b = len(idx_end_arr)

    s250 = np.empty((b, seq_len, stores["250ms"].d), dtype=np.float32)
    s20 = np.empty((b, seq_len, stores["20s"].d), dtype=np.float32)
    s5m = np.empty((b, seq_len, stores["5m"].d), dtype=np.float32)
    s1h = np.empty((b, seq_len, stores["1h"].d), dtype=np.float32)

    for k in range(b):
        i250 = int(idx_end_arr[k])
        t_ms, _ = stores["250ms"].get_row(i250)

        i20 = map_base_ts_to_scale_idx(stores["20s"], t_ms)
        i5m = map_base_ts_to_scale_idx(stores["5m"], t_ms)
        i1h = map_base_ts_to_scale_idx(stores["1h"], t_ms)

        s250[k] = stores["250ms"].get_seq_end(i250, seq_len)
        s20[k] = stores["20s"].get_seq_end(i20, seq_len)
        s5m[k] = stores["5m"].get_seq_end(i5m, seq_len)
        s1h[k] = stores["1h"].get_seq_end(i1h, seq_len)

    return s250, s20, s5m, s1h


def train():
    seed_all(SEED)

    with (OUT_ROOT / "dataset_meta.json").open("r") as f:
        ds_meta = json.load(f)

    norm = ds_meta["norm"]
    train_end_ms = int(ds_meta["train_end_ms"])

    stores = {
        "250ms": ScaleStore(OUT_ROOT / "250ms"),
        "20s": ScaleStore(OUT_ROOT / "20s"),
        "5m": ScaleStore(OUT_ROOT / "5m"),
        "1h": ScaleStore(OUT_ROOT / "1h"),
    }

    close_idx = 8
    ctx_dim = 2 + N_ACTIONS + (VOTE_N * N_ACTIONS)

    rb = ReplayBuffer(REPLAY_CAP, ctx_dim)

    actor = ActorNet(stores["250ms"].d, stores["20s"].d, stores["5m"].d, stores["1h"].d, ctx_dim).to(DEVICE)
    critic = CriticNet(stores["250ms"].d, stores["20s"].d, stores["5m"].d, stores["1h"].d, ctx_dim).to(DEVICE)
    critic_tgt = CriticNet(stores["250ms"].d, stores["20s"].d, stores["5m"].d, stores["1h"].d, ctx_dim).to(DEVICE)
    critic_tgt.load_state_dict(critic.state_dict())

    print(f"actor params:", f"{count_params(actor):,}")
    print(f"critic params:", f"{count_params(critic):,}")

    opt_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    log_alpha = torch.tensor(math.log(ALPHA_INIT), device=DEVICE, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=LR_ALPHA)

    mean = {k: torch.tensor(norm[k]["mean"], device=DEVICE, dtype=DTYPE) for k in norm}
    std = {k: torch.tensor(norm[k]["std"], device=DEVICE, dtype=DTYPE) for k in norm}

    def norm_t(scale, x):
        return (x - mean[scale]) / std[scale]

    base_train_end = find_first_idx_ge(stores["250ms"], train_end_ms)

    req_t = max(
        stores["250ms"].first_ts + (SEQ_LEN - 1) * stores["250ms"].interval_ms,
        stores["20s"].first_ts + (SEQ_LEN - 1) * stores["20s"].interval_ms,
        stores["5m"].first_ts + (SEQ_LEN - 1) * stores["5m"].interval_ms,
        stores["1h"].first_ts + (SEQ_LEN - 1) * stores["1h"].interval_ms,
    )
    warmup_min_i = find_first_idx_ge(stores["250ms"], req_t)

    env = TradingEnv(stores, norm, close_idx_in_x250=close_idx)
    episode_steps = env.episode_steps

    def sample_start():
        hi = max(warmup_min_i + 1, base_train_end - episode_steps - 2)
        while True:
            i = random.randint(warmup_min_i, hi)
            t_ms, _ = stores["250ms"].get_row(i)
            i20 = map_base_ts_to_scale_idx(stores["20s"], t_ms)
            i5m = map_base_ts_to_scale_idx(stores["5m"], t_ms)
            i1h = map_base_ts_to_scale_idx(stores["1h"], t_ms)
            if i < (SEQ_LEN - 1):
                continue
            if i20 < (SEQ_LEN - 1) or i5m < (SEQ_LEN - 1) or i1h < (SEQ_LEN - 1):
                continue
            if i + episode_steps + 2 >= base_train_end:
                continue
            return i

    def soft_update(tgt: nn.Module, src: nn.Module, tau: float):
        with torch.no_grad():
            for p_t, p in zip(tgt.parameters(), src.parameters()):
                p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def update_step():
        i, a, r, d, ni, ctx, nctx = rb.sample(BATCH_SIZE)

        s250, s20, s5m, s1h = load_batch_seqs(stores, i, SEQ_LEN)
        ns250, ns20, ns5m, ns1h = load_batch_seqs(stores, ni, SEQ_LEN)

        s250 = torch.tensor(s250, device=DEVICE, dtype=DTYPE)
        s20 = torch.tensor(s20, device=DEVICE, dtype=DTYPE)
        s5m = torch.tensor(s5m, device=DEVICE, dtype=DTYPE)
        s1h = torch.tensor(s1h, device=DEVICE, dtype=DTYPE)

        ns250 = torch.tensor(ns250, device=DEVICE, dtype=DTYPE)
        ns20 = torch.tensor(ns20, device=DEVICE, dtype=DTYPE)
        ns5m = torch.tensor(ns5m, device=DEVICE, dtype=DTYPE)
        ns1h = torch.tensor(ns1h, device=DEVICE, dtype=DTYPE)

        ctx = torch.tensor(ctx, device=DEVICE, dtype=DTYPE)
        nctx = torch.tensor(nctx, device=DEVICE, dtype=DTYPE)

        s250 = norm_t("250ms", s250)
        s20 = norm_t("20s", s20)
        s5m = norm_t("5m", s5m)
        s1h = norm_t("1h", s1h)

        ns250 = norm_t("250ms", ns250)
        ns20 = norm_t("20s", ns20)
        ns5m = norm_t("5m", ns5m)
        ns1h = norm_t("1h", ns1h)

        a = torch.tensor(a, device=DEVICE, dtype=torch.long)
        r = torch.tensor(r, device=DEVICE, dtype=DTYPE)
        d = torch.tensor(d, device=DEVICE, dtype=DTYPE)

        alpha_det = log_alpha.exp().detach()

        with torch.no_grad():
            logits_n = actor(ns250, ns20, ns5m, ns1h, nctx)
            logp_n = F.log_softmax(logits_n, dim=-1)
            p_n = torch.exp(logp_n)

            tq1, tq2 = critic_tgt(ns250, ns20, ns5m, ns1h, nctx)
            tq = torch.min(tq1, tq2)
            v = (p_n * (tq - alpha_det * logp_n)).sum(dim=-1)
            y = r + (1.0 - d) * float(GAMMA) * v

        q1, q2 = critic(s250, s20, s5m, s1h, ctx)
        q1_a = q1.gather(1, a.view(-1, 1)).squeeze(1)
        q2_a = q2.gather(1, a.view(-1, 1)).squeeze(1)
        critic_loss = F.mse_loss(q1_a, y) + F.mse_loss(q2_a, y)

        opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        opt_critic.step()

        logits = actor(s250, s20, s5m, s1h, ctx)
        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)

        with torch.no_grad():
            q1p, q2p = critic(s250, s20, s5m, s1h, ctx)
            qmin = torch.min(q1p, q2p)

        actor_loss = (p * (alpha_det * logp - qmin)).sum(dim=-1).mean()

        opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()

        ent = -(p * logp).sum(dim=-1).mean()

        alpha = log_alpha.exp()
        alpha_loss = alpha * (ent.detach() - float(TARGET_ENTROPY))

        opt_alpha.zero_grad()
        alpha_loss.backward()
        opt_alpha.step()

        soft_update(critic_tgt, critic, TAU)

        return float(critic_loss.item()), float(actor_loss.item()), float(ent.item()), float(log_alpha.exp().item())

    global_step = 0
    for ep in range(MAX_EPISODES):
        start_i = sample_start()
        obs = env.reset(start_i)

        pbar = tqdm(total=env.episode_steps, desc=f"ep {ep}", dynamic_ncols=True)
        ep_reward = 0.0
        ep_c = 0.0
        ep_a = 0.0
        ep_ent = 0.0
        ep_alpha = 0.0
        ep_updates = 0

        while True:
            if global_step < WARMUP_STEPS:
                a_prop = random.randrange(N_ACTIONS)
            else:
                a_prop = select_action(actor, obs, DEVICE)

            ctx = obs["ctx"].copy()
            cur_i = env.i

            next_obs, rwd, done, info = env.step(a_prop)

            nctx = next_obs["ctx"].copy()
            next_i = env.i

            rb.add(cur_i, int(a_prop), float(rwd), float(done), next_i, ctx, nctx)

            ep_reward += float(rwd)
            global_step += 1

            if rb.size >= BATCH_SIZE and global_step >= WARMUP_STEPS and (global_step % UPDATE_EVERY) == 0:
                for _ in range(UPDATES_PER_STEP):
                    cl, al, ent, alp = update_step()
                    ep_c += cl
                    ep_a += al
                    ep_ent += ent
                    ep_alpha += alp
                    ep_updates += 1

            pbar.set_postfix({
                "bal": f"{info['balance']:6.2f}",
                # "pos": info["pos"],
                "r": f"{ep_reward:10.6f}",
                # "exec": ACTIONS[int(info["exec_action"])],
                "rb": f"{rb.size:08d}",
            })
            pbar.update(1)

            obs = next_obs
            if done:
                break

        pbar.close()
        if ep_updates > 0:
            ep_c /= ep_updates
            ep_a /= ep_updates
            ep_ent /= ep_updates
            ep_alpha /= ep_updates

        print(
            f"episode {ep} reward={ep_reward:.4f} bal={env.balance:.2f} "
            f"critic={ep_c:.4f} actor={ep_a:.4f} ent={ep_ent:.4f} alpha={ep_alpha:.4f}"
        )