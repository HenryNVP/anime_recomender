# scripts/make_demo_data.py
from __future__ import annotations
import argparse
import os
from typing import Tuple, Set

import numpy as np
import pandas as pd


def _clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    need = {"user_id", "anime_id", "rating"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"rating.csv missing columns: {miss}")
    df = df.dropna(subset=["user_id", "anime_id", "rating"])
    df = df[df["rating"] != -1]              # -1 means "watched but not rated"
    df = df.drop_duplicates(subset=["user_id", "anime_id"])
    df["user_id"] = df["user_id"].astype(int)
    df["anime_id"] = df["anime_id"].astype(int)
    df["rating"]   = df["rating"].astype(float)
    return df


def _clean_anime(df: pd.DataFrame) -> pd.DataFrame:
    need = {"anime_id", "name", "genre", "type", "episodes", "rating", "members"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"anime.csv missing columns: {miss}")
    df = df.dropna(subset=["anime_id", "name"]).drop_duplicates(subset=["anime_id"])
    # keep other columns as-is (preprocess will impute/convert)
    df["anime_id"] = df["anime_id"].astype(int)
    return df


def _filter_fixed_point(r: pd.DataFrame, min_user_ratings: int, min_item_ratings: int) -> pd.DataFrame:
    """Iteratively drop users/items below min counts until stable."""
    prev_len = -1
    cur = r.copy()
    while len(cur) != prev_len:
        prev_len = len(cur)
        uc = cur["user_id"].value_counts()
        ic = cur["anime_id"].value_counts()
        cur = cur[cur["user_id"].isin(uc[uc >= min_user_ratings].index)]
        cur = cur[cur["anime_id"].isin(ic[ic >= min_item_ratings].index)]
    return cur


def _choose_sets(
    ratings: pd.DataFrame,
    n_users: int,
    n_items: int,
    strategy: str,
    seed: int,
    min_user_ratings: int,
    min_item_ratings: int,
) -> Tuple[Set[int], Set[int]]:
    rng = np.random.default_rng(seed)

    # Start from users/items that are at least minimally viable
    base = _filter_fixed_point(ratings, min_user_ratings, min_item_ratings)
    if base.empty:
        raise ValueError("After minimal filtering, no ratings remain. Lower min_* thresholds.")

    # Item selection
    item_counts = base["anime_id"].value_counts()
    viable_items = item_counts[item_counts >= min_item_ratings].index.to_numpy()
    if n_items <= 0 or n_items >= viable_items.size:
        chosen_items = viable_items
    else:
        if strategy == "top":
            chosen_items = item_counts.sort_values(ascending=False).index.to_numpy()[:n_items]
        elif strategy == "random":
            chosen_items = rng.choice(viable_items, size=n_items, replace=False)
        else:  # stratified by popularity buckets
            # simple 3-bucket stratified sample: head/mid/tail
            sorted_items = item_counts.sort_values(ascending=False).index.to_numpy()
            thirds = np.array_split(sorted_items, 3)
            take = [max(1, int(round(n_items * w))) for w in (0.34, 0.33, 0.33)]
            chosen_items = np.concatenate([rng.choice(arr, size=min(len(arr), t), replace=False) for arr, t in zip(thirds, take)])

    # Filter to chosen items then select users
    stage = base[base["anime_id"].isin(chosen_items)]
    user_counts = stage["user_id"].value_counts()
    viable_users = user_counts[user_counts >= min_user_ratings].index.to_numpy()
    if n_users <= 0 or n_users >= viable_users.size:
        chosen_users = viable_users
    else:
        if strategy == "top":
            chosen_users = user_counts.sort_values(ascending=False).index.to_numpy()[:n_users]
        elif strategy == "random":
            chosen_users = rng.choice(viable_users, size=n_users, replace=False)
        else:
            sorted_users = user_counts.sort_values(ascending=False).index.to_numpy()
            thirds = np.array_split(sorted_users, 3)
            take = [max(1, int(round(n_users * w))) for w in (0.34, 0.33, 0.33)]
            chosen_users = np.concatenate([rng.choice(arr, size=min(len(arr), t), replace=False) for arr, t in zip(thirds, take)])

    return set(chosen_users.tolist()), set(chosen_items.tolist())


def make_mini(
    src_dir: str,
    out_dir: str = "data/demo",
    n_users: int = 400,
    n_items: int = 600,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
    strategy: str = "top",
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    # Load full raw
    anime = pd.read_csv(os.path.join(src_dir, "anime.csv"))
    ratings = pd.read_csv(os.path.join(src_dir, "rating.csv"))

    # Clean
    anime = _clean_anime(anime)
    ratings = _clean_ratings(ratings)

    print(f"[full] users={ratings['user_id'].nunique()} items={ratings['anime_id'].nunique()} ratings={len(ratings)}")

    # Choose subsets
    users_keep, items_keep = _choose_sets(
        ratings, n_users, n_items, strategy, seed, min_user_ratings, min_item_ratings
    )

    mini = ratings[ratings["user_id"].isin(users_keep) & ratings["anime_id"].isin(items_keep)].copy()

    # Enforce fixed-point counts one more time
    mini = _filter_fixed_point(mini, min_user_ratings, min_item_ratings)

    # Final keep sets after pruning
    users_keep = set(mini["user_id"].unique().tolist())
    items_keep = set(mini["anime_id"].unique().tolist())

    mini_anime = anime[anime["anime_id"].isin(items_keep)].copy()

    # Save mini raw (IDs are original; your preprocess will reindex)
    mini_anime.to_csv(os.path.join(out_dir, "anime.csv"), index=False)
    mini.sort_values(["user_id", "anime_id"]).to_csv(os.path.join(out_dir, "rating.csv"), index=False)

    print(f"[mini] users={mini['user_id'].nunique()} items={mini['anime_id'].nunique()} ratings={len(mini)}")
    print(f"Saved mini dataset to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Create a smaller subset of the anime dataset.")
    p.add_argument("--src_dir", required=True, help="Folder containing original anime.csv and rating.csv")
    p.add_argument("--out_dir", default="data_mini", help="Output folder for mini dataset")
    p.add_argument("--n_users", type=int, default=5000, help="Target number of users (<=0 means auto)")
    p.add_argument("--n_items", type=int, default=3000, help="Target number of items (<=0 means auto)")
    p.add_argument("--min_user_ratings", type=int, default=5, help="Min ratings per user")
    p.add_argument("--min_item_ratings", type=int, default=5, help="Min ratings per item")
    p.add_argument("--strategy", choices=["top", "random", "stratified"], default="top",
                   help="Sampling strategy for users/items")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_mini(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        n_users=args.n_users,
        n_items=args.n_items,
        min_user_ratings=args.min_user_ratings,
        min_item_ratings=args.min_item_ratings,
        strategy=args.strategy,
        seed=args.seed,
    )
