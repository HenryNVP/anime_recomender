# src/recsys/models/neumf.py
from __future__ import annotations
import torch
import torch.nn as nn

class NeuMF(nn.Module):
    """
    NeuMF with optional item-feature tower.
    If item features are attached, they are transformed by a small MLP and
    concatenated with [MLP(u,i) ⊕ GMF(u,i)] before the final head.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        mf_dim: int = 32,
        mlp_layers=(128, 64),
        dropout: float = 0.1,
        user_bias: bool = True,
        item_bias: bool = True,
        # feature tower config (optional)
        item_feat_layers: tuple[int, ...] | None = None,
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

        # ----- Feature tower (set up later when features attached) -----
        self.item_feats_dim: int | None = None
        self.item_feats: torch.Tensor | None = None   # registered buffer later
        self.feat_tower: nn.Module | None = None
        self._feat_layers_cfg = tuple(item_feat_layers) if item_feat_layers else tuple()

        # ----- Head -----
        head_in = mlp_out_dim + mf_dim  # + (feat_dim_out) added dynamically when attached
        self.out = nn.Linear(head_in, 1)

        # biases
        self.ub = nn.Embedding(n_users, 1) if user_bias else None
        self.ib = nn.Embedding(n_items, 1) if item_bias else None

        # init
        for emb in [self.P, self.Q, self.Pm, self.Qm]:
            nn.init.normal_(emb.weight, std=0.01)
        if self.ub is not None: nn.init.zeros_(self.ub.weight)
        if self.ib is not None: nn.init.zeros_(self.ib.weight)

        # keep for head resize if features are later attached
        self._mlp_out_dim = mlp_out_dim
        self._mf_dim = mf_dim

    # --------- Feature attachment API ----------
    def attach_item_features(self, feats: torch.Tensor, freeze: bool = True) -> None:
        """
        feats: (n_items, F) float32 tensor (same item index space).
        Registers as a non-trainable buffer, builds a small MLP (feat_tower),
        and expands the output head to accept the extra feature representation.
        """
        assert feats.ndim == 2 and feats.shape[0] == self.Q.num_embeddings
        F = feats.shape[1]
        self.item_feats_dim = F
        # register as buffer (optionally detach to be safe)
        self.register_buffer("item_feats", feats.detach() if freeze else feats, persistent=False)

        # build feature tower if requested
        feat_in = F
        feat_out = 0
        if self._feat_layers_cfg:
            blocks = []
            in_d = feat_in
            for d in self._feat_layers_cfg:
                blocks += [nn.Linear(in_d, d), nn.ReLU()]
                in_d = d
            self.feat_tower = nn.Sequential(*blocks)
            feat_out = in_d
        else:
            self.feat_tower = None
            feat_out = 0

        # replace head with expanded input (mlp + gmf [+ feat])
        new_in = self._mlp_out_dim + self._mf_dim + feat_out
        old = self.out
        self.out = nn.Linear(new_in, 1)
        # (optional) copy old weights for the common dims
        with torch.no_grad():
            k = min(old.weight.shape[1], self.out.weight.shape[1])
            self.out.weight[:, :k].copy_(old.weight[:, :k])
            self.out.bias.copy_(old.bias)

    # --------- Forward ----------
    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        # GMF vector (B, mf_dim)
        gmf_vec = self.P(u) * self.Q(i)
        # ID-MLP vector (B, mlp_out)
        mlp_in = torch.cat([self.Pm(u), self.Qm(i)], dim=-1)
        mlp_vec = self.mlp(mlp_in)

        parts = [mlp_vec, gmf_vec]

        # feature branch
        if self.item_feats is not None:
            f = self.item_feats.index_select(0, i)
            if self.feat_tower is not None:
                f = self.feat_tower(f)
            parts.append(f)

        h = torch.cat(parts, dim=-1)
        s = self.out(h).squeeze(-1)
        if self.ub is not None: s = s + self.ub(u).squeeze(-1)
        if self.ib is not None: s = s + self.ib(i).squeeze(-1)
        return s
