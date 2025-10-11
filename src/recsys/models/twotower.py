from __future__ import annotations

import torch
import torch.nn as nn

COMBINE_MODES = {"concat", "add"}


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
        *,
        item_features: torch.Tensor | None = None,
        item_feature_layers: tuple[int, ...] = (),
        item_feature_dropout: float = 0.0,
        item_feature_combine: str = "concat",
    ) -> None:
        super().__init__()
        if n_users <= 0 or n_items <= 0:
            raise ValueError("n_users and n_items must be > 0 for TwoTower")

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        self.user_tower, u_dim = _build_mlp(embed_dim, tuple(user_layers), dropout)

        combine_mode = str(item_feature_combine).lower()
        if combine_mode not in COMBINE_MODES:
            raise ValueError(
                f"Unsupported item_feature_combine='{item_feature_combine}'. "
                f"Expected one of {sorted(COMBINE_MODES)}."
            )
        self.item_feature_combine = combine_mode if item_features is not None else None

        feat_out_dim = 0
        if item_features is not None:
            if not torch.is_tensor(item_features):
                item_features = torch.as_tensor(item_features, dtype=torch.float32)
            if item_features.dim() != 2:
                raise ValueError(
                    f"item_features must be rank-2 tensor [n_items, feat_dim], got shape {item_features.shape}"
                )
            item_features = item_features.float()
            if item_features.size(0) < n_items:
                raise ValueError(
                    f"item_features rows ({item_features.size(0)}) < n_items ({n_items})."
                )
            if item_features.size(0) > n_items:
                item_features = item_features[:n_items]
            self.register_buffer("item_features", item_features)
            self.item_feat_net, feat_out_dim = _build_mlp(
                item_features.size(1),
                tuple(item_feature_layers),
                item_feature_dropout,
            )
        else:
            self.item_features = None  # type: ignore[assignment]
            self.item_feat_net = None

        if self.item_feature_combine == "concat":
            item_tower_in = embed_dim + feat_out_dim
        elif self.item_feature_combine == "add":
            if feat_out_dim != embed_dim:
                raise ValueError(
                    "item_feature_combine='add' requires item_feature_layers to project "
                    f"to embed_dim={embed_dim}, but got {feat_out_dim}."
                )
            item_tower_in = embed_dim
        else:
            item_tower_in = embed_dim

        self.item_tower, i_dim = _build_mlp(item_tower_in, tuple(item_layers), dropout)
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
        if self.item_features is not None:
            feats = self.item_features.index_select(0, items)
            feats = self.item_feat_net(feats)
            if self.item_feature_combine == "concat":
                z = torch.cat([z, feats], dim=-1)
            elif self.item_feature_combine == "add":
                z = z + feats
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
