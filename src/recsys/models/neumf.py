# src/recsys/models/neumf.py
from __future__ import annotations
import torch
import torch.nn as nn

class NeuMF(nn.Module):
    """Hybrid MF/MLP recommender following NeuMF without auxiliary towers."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        mf_dim: int = 32,
        mlp_layers=(128, 64),
        dropout: float = 0.1,
        user_bias: bool = True,
        item_bias: bool = True,
    ):
        super().__init__()
        assert n_users > 0 and n_items > 0

        # ----- GMF -----
        self.P = nn.Embedding(n_users, mf_dim)
        self.Q = nn.Embedding(n_items, mf_dim)

        # ----- MLP (ID embeddings → MLP) -----
        mlp0 = int(mlp_layers[0]) if len(mlp_layers) else 64
        mlp_emb_dim = mlp0 // 2
        self.Pm = nn.Embedding(n_users, mlp_emb_dim)
        self.Qm = nn.Embedding(n_items, mlp_emb_dim)

        layers = []
        in_d = mlp_emb_dim * 2
        for d in mlp_layers:
            layers += [nn.Linear(in_d, d), nn.ReLU(), nn.Dropout(dropout)]
            in_d = d
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        mlp_out_dim = in_d if layers else in_d  # identity keeps input dim

        # ----- Head -----
        head_in = mlp_out_dim + mf_dim
        self.out = nn.Linear(head_in, 1)

        # biases
        self.ub = nn.Embedding(n_users, 1) if user_bias else None
        self.ib = nn.Embedding(n_items, 1) if item_bias else None

        # init
        for emb in [self.P, self.Q, self.Pm, self.Qm]:
            nn.init.normal_(emb.weight, std=0.01)
        if self.ub is not None: nn.init.zeros_(self.ub.weight)
        if self.ib is not None: nn.init.zeros_(self.ib.weight)

    # --------- Forward ----------
    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        # GMF vector (B, mf_dim)
        gmf_vec = self.P(u) * self.Q(i)
        # ID-MLP vector (B, mlp_out)
        mlp_in = torch.cat([self.Pm(u), self.Qm(i)], dim=-1)
        mlp_vec = self.mlp(mlp_in)

        h = torch.cat([mlp_vec, gmf_vec], dim=-1)
        s = self.out(h).squeeze(-1)
        if self.ub is not None: s = s + self.ub(u).squeeze(-1)
        if self.ib is not None: s = s + self.ib(i).squeeze(-1)
        return s
