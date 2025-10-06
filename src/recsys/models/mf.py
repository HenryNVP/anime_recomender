from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class MF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 32, user_bias: bool=True, item_bias: bool=True):
        super().__init__()
        self.P = nn.Embedding(n_users, dim)
        self.Q = nn.Embedding(n_items, dim)
        self.ub = nn.Embedding(n_users, 1) if user_bias else None
        self.ib = nn.Embedding(n_items, 1) if item_bias else None
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        if self.ub is not None: nn.init.zeros_(self.ub.weight)
        if self.ib is not None: nn.init.zeros_(self.ib.weight)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        p = self.P(u)         # (B, d)
        q = self.Q(i)         # (B, d)
        s = (p * q).sum(-1)   # (B,)
        if self.ub is not None: s = s + self.ub(u).squeeze(-1)
        if self.ib is not None: s = s + self.ib(i).squeeze(-1)
        return s              # regression on rating scale
