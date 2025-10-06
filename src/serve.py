from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.eval import load_cfg, resolve_ckpt_path, load_model_from_ckpt
from src.recsys.models.mf import MF
from src.recsys.models.twotower import TwoTower
from src.utils import get_device


def _load_anime_metadata(processed_dir: str) -> Dict[int, str]:
    path = os.path.join(processed_dir, "anime.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, usecols=["anime_id", "name"]).dropna()
    return {int(r.anime_id): str(r.name) for r in df.itertuples(index=False)}


def _user_history(processed_dir: str, user_id: int) -> pd.DataFrame:
    ratings_path = os.path.join(processed_dir, "ratings.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Missing ratings file: {ratings_path}")
    ratings = pd.read_csv(ratings_path, usecols=["user_id", "anime_id", "rating"])
    return ratings[ratings["user_id"] == user_id].copy()


def _score_user(
    model: torch.nn.Module,
    user_id: int,
    n_items: int,
    device: str,
    exclude: List[int],
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        if isinstance(model, MF):
            user_vec = model.P.weight[user_id].to(device)
            item_vecs = model.Q.weight.to(device)
            scores = (item_vecs @ user_vec).clone()
            if model.ub is not None:
                scores += model.ub.weight[user_id, 0].to(device)
            if model.ib is not None:
                scores += model.ib.weight[:, 0].to(device)
        else:
            idx = torch.arange(n_items, device=device)
            batches: List[torch.Tensor] = []
            B = 8192
            for t in range(0, n_items, B):
                it = idx[t : t + B]
                ut = torch.full_like(it, fill_value=user_id, dtype=torch.long)
                batches.append(model(ut, it))
            scores = torch.cat(batches, dim=0)

    scores_np = scores.detach().cpu().numpy()
    if exclude:
        scores_np[np.asarray(exclude, dtype=np.int64)] = -np.inf
    return scores_np


def _format_items(anime_map: Dict[int, str], items: List[int]) -> List[str]:
    formatted = []
    for idx, item in enumerate(items, start=1):
        name = anime_map.get(item, f"anime_id={item}")
        formatted.append(f"{idx:>2d}. {name} (id={item})")
    return formatted


def serve_recommendations(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)
    ckpt_path = resolve_ckpt_path(args.ckpt, cfg)
    device = get_device(args.device)

    model, cfg_ck, n_users, n_items = load_model_from_ckpt(ckpt_path, device)

    if args.user < 0 or args.user >= n_users:
        raise ValueError(f"user id {args.user} out of range [0, {n_users - 1}]")

    cfg_data = cfg_ck.get("recsys", {}) if cfg_ck else {}
    processed_dir = cfg_data.get("processed_dir") or cfg["recsys"]["processed_dir"]

    anime_map = _load_anime_metadata(processed_dir)
    history_df = _user_history(processed_dir, args.user)

    if history_df.empty:
        print(f"User {args.user} has no recorded history. Recommending top {args.k} globally.")
        seen_items: List[int] = []
    else:
        seen_items = history_df["anime_id"].astype(int).tolist()

    scores = _score_user(model, args.user, n_items, device, exclude=seen_items)
    top_idx = np.argpartition(-scores, range(min(args.k, len(scores))))[: args.k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    recommendations = top_idx.tolist()

    # Print history (sorted by rating desc)
    if not history_df.empty:
        history_df = history_df.sort_values("rating", ascending=False)
        watched = history_df.head(args.history or 20)
        print(f"\nUser {args.user} history (top {len(watched)} by rating):")
        for t, row in enumerate(watched.itertuples(index=False), start=1):
            name = anime_map.get(int(row.anime_id), f"anime_id={int(row.anime_id)}")
            print(f"{t:>2d}. {name} (id={int(row.anime_id)}, rating={float(row.rating):.2f})")

    print(f"\nRecommended {args.k} items for user {args.user}:")
    for line in _format_items(anime_map, recommendations):
        print(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve top-N recommendations for a user")
    p.add_argument("--config", default="configs/config.yaml", help="Config used during training")
    p.add_argument("--ckpt", default=None, help="Path to checkpoint (defaults to resolved best)")
    p.add_argument("--user", type=int, required=True, help="Reindexed user id")
    p.add_argument("--k", type=int, default=10, help="Number of recommendations to show")
    p.add_argument("--history", type=int, default=10, help="Number of watched items to display")
    p.add_argument("--device", default=None, help="Override device (cuda|cpu|mps)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    serve_recommendations(args)


if __name__ == "__main__":
    main()
