from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, layers: tuple[int, ...], dropout: float) -> tuple[nn.Module, int]:
    """Helper to build an MLP tower; returns (module, out_dim)."""
    if not layers:
        return nn.Identity(), in_dim

    modules: list[nn.Module] = []
    prev = in_dim
    for hidden in layers:
        modules.append(nn.Linear(prev, int(hidden)))
        modules.append(nn.ReLU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        prev = int(hidden)

    return nn.Sequential(*modules), prev


class TwoTower(nn.Module):
    """Two-tower recommender with ID embeddings passed through per-side MLPs."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        user_layers: tuple[int, ...] = (),
        item_layers: tuple[int, ...] = (),
        dropout: float = 0.0,
        user_bias: bool = True,
        item_bias: bool = True,
    ) -> None:
        super().__init__()
        if n_users <= 0 or n_items <= 0:
            raise ValueError("n_users and n_items must be > 0 for TwoTower")

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        self.user_tower, u_dim = _build_mlp(embed_dim, tuple(user_layers), dropout)
        self.item_tower, i_dim = _build_mlp(embed_dim, tuple(item_layers), dropout)
        if u_dim != i_dim:
            raise ValueError(
                f"TwoTower user/item tower output dims must match (got {u_dim} vs {i_dim})."
            )
        self.latent_dim = u_dim

        self.ub = nn.Embedding(n_users, 1) if user_bias else None
        self.ib = nn.Embedding(n_items, 1) if item_bias else None

        # initialise embeddings similar to MF
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        if self.ub is not None:
            nn.init.zeros_(self.ub.weight)
        if self.ib is not None:
            nn.init.zeros_(self.ib.weight)

    def encode_users(self, users: torch.Tensor) -> torch.Tensor:
        z = self.user_emb(users)
        return self.user_tower(z)

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        z = self.item_emb(items)
        return self.item_tower(z)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u_vec = self.encode_users(users)
        i_vec = self.encode_items(items)
        scores = torch.sum(u_vec * i_vec, dim=-1)
        if self.ub is not None:
            scores = scores + self.ub(users).squeeze(-1)
        if self.ib is not None:
            scores = scores + self.ib(items).squeeze(-1)
        return scores

