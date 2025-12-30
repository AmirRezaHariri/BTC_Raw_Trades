import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from utils import *


# =========================
# MODEL
# =========================
class ScaleEncoder(nn.Module):
    def __init__(self, d_in: int, hidden: int, seq_len: int):
        super().__init__()
        self.d_in = int(d_in)
        self.hidden = int(hidden)
        self.seq_len = int(seq_len)

        self.conv1 = nn.Conv1d(self.d_in, self.hidden, kernel_size=8, padding="same")
        self.conv2 = nn.Conv1d(self.hidden, self.hidden // 2, kernel_size=4, padding="same")
        self.pool1 = nn.AvgPool1d(kernel_size=4)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.drop = nn.Dropout(DROPOUT)

        self.flat_t = self.seq_len // (self.pool1.kernel_size[0] * self.pool2.kernel_size[0])
        self.flat_dim = (self.hidden // 2) * self.flat_t
        
        self.ln = nn.LayerNorm(self.flat_dim)
        self.emb1 = nn.Linear(self.flat_dim, ENC_HIDDEN)
        self.emb2 = nn.Linear(ENC_HIDDEN, HIDDEN)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, T]
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = self.pool2(x)

        x = x.transpose(1, 2)  # [B, T, H]
        x = x.reshape(x.shape[0], -1)  # [B, T*H]  (FLATTEN instead of mean)

        x = self.ln(x)
        x = self.emb1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.emb2(x)
        x = F.relu(x)
        return x


class EncStack(nn.Module):
    def __init__(self, d250, d20, d5m, d1h, ctx_dim):
        super().__init__()
        self.ctx_in_dim = int(ctx_dim)

        self.e250 = ScaleEncoder(d250, HIDDEN, SEQ_LEN)
        self.e20  = ScaleEncoder(d20,  HIDDEN, SEQ_LEN)
        self.e5m  = ScaleEncoder(d5m,  HIDDEN, SEQ_LEN)
        self.e1h  = ScaleEncoder(d1h,  HIDDEN, SEQ_LEN)

        self.ctx = nn.Sequential(
            nn.Linear(self.ctx_in_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )

        self.z_dim = HIDDEN * 5

    def forward(self, s250, s20, s5m, s1h, ctx):
        return torch.cat([
            self.e250(s250),
            self.e20(s20),
            self.e5m(s5m),
            self.e1h(s1h),
            self.ctx(ctx),
        ], dim=-1)


class ActorNet(nn.Module):
    def __init__(self, d250, d20, d5m, d1h, ctx_dim):
        super().__init__()
        self.enc = EncStack(d250, d20, d5m, d1h, ctx_dim)
        self.head = nn.Sequential(
            nn.Linear(self.enc.z_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )

    def forward(self, s250, s20, s5m, s1h, ctx):
        z = self.enc(s250, s20, s5m, s1h, ctx)
        return self.head(z)


class CriticNet(nn.Module):
    def __init__(self, d250, d20, d5m, d1h, ctx_dim):
        super().__init__()
        self.enc = EncStack(d250, d20, d5m, d1h, ctx_dim)
        self.q1 = nn.Sequential(
            nn.Linear(self.enc.z_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )
        self.q2 = nn.Sequential(
            nn.Linear(self.enc.z_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )

    def forward(self, s250, s20, s5m, s1h, ctx):
        z = self.enc(s250, s20, s5m, s1h, ctx)
        return self.q1(z), self.q2(z)
