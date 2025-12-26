import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from utils import *


# =========================
# MODEL
# =========================
class ScaleEncoder(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d_in, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(DROPOUT)
        self.proj = nn.Linear(hidden, ENC_HIDDEN)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x = x.mean(dim=1)
        x = self.ln(x)
        x = self.proj(x)
        x = F.relu(x)
        return x


class EncStack(nn.Module):
    def __init__(self, d250, d20, d5m, d1h, ctx_dim):
        super().__init__()
        self.e250 = ScaleEncoder(d250, HIDDEN)
        self.e20 = ScaleEncoder(d20, HIDDEN)
        self.e5m = ScaleEncoder(d5m, HIDDEN)
        self.e1h = ScaleEncoder(d1h, HIDDEN)
        self.ctx = nn.Sequential(
            nn.Linear(ctx_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, ENC_HIDDEN),
            nn.ReLU(),
        )
        self.z_dim = ENC_HIDDEN * 5

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
