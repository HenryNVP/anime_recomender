# scripts/preprocess_ratings.py
from __future__ import annotations

import os, argparse, pickle, json
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# ===============================
# Cleaning helpers
# ===============================

def _clean_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for ratings: drop NAs, remove -1, dedupe, cast to float."""
    need = {"user_id", "anime_id", "rating"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"rating.csv missing columns: {missing}")

    df = df.dropna(subset=["user_id", "anime_id", "rating"])
    df = df[df["rating"] != -1]  # -1 means not watched
    df = df.drop_duplicates(subset=["user_id", "anime_id"])

    df["user_id"] = df["user_id"].astype(int)
    df["anime_id"] = df["anime_id"].astype(int)
    df["rating"]   = df["rating"].astype(float)
    return df


def _clean_anime(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for anime table: drop NAs on id/name, dedupe id."""
    need = {"anime_id", "name", "genre", "type", "episodes", "rating", "members"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"anime.csv missing columns: {missing}")

    df = df.dropna(subset=["anime_id", "name"])
    df = df.drop_duplicates(subset=["anime_id"])

    # episodes/members often non-numeric; coerce to numeric (Unknown->NaN)
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    df["members"]  = pd.to_numeric(df["members"], errors="coerce")

    df["anime_id"] = df["anime_id"].astype(int)
    return df


def _filter_sparsity(
    df: pd.DataFrame,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5
) -> pd.DataFrame:
    """Filter users/items with too-few ratings. Re-run iteratively until fixed-point."""
    prev_len = -1
    cur = df.copy()
    while len(cur) != prev_len:
        prev_len = len(cur)
        user_counts = cur["user_id"].value_counts()
        item_counts = cur["anime_id"].value_counts()
        cur = cur[cur["user_id"].isin(user_counts[user_counts >= min_user_ratings].index)]
        cur = cur[cur["anime_id"].isin(item_counts[item_counts >= min_item_ratings].index)]
    return cur


def _reindex_ids(
    ratings: pd.DataFrame,
    anime: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[int, int], Dict[int, int]]:
    """
    Reindex user_id and anime_id in ratings to consecutive ints starting at 0.
    If anime is provided, keep only rated items and apply the same mapping.
    """
    uid_map: Dict[int, int] = {u: i for i, u in enumerate(ratings["user_id"].unique())}
    iid_map: Dict[int, int] = {i: j for j, i in enumerate(ratings["anime_id"].unique())}

    ratings = ratings.copy()
    ratings["user_id"]  = ratings["user_id"].map(uid_map).astype(int)
    ratings["anime_id"] = ratings["anime_id"].map(iid_map).astype(int)

    anime_keep = None
    if anime is not None:
        anime_keep = anime[anime["anime_id"].isin(iid_map.keys())].copy()
        anime_keep["anime_id"] = anime_keep["anime_id"].map(iid_map).astype(int)

    return ratings, anime_keep, uid_map, iid_map


def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing anime metadata:
      - fill 'type' and 'genre' with 'Unknown'
      - fill 'rating' using the average of median genre ratings (fallback: global median)
    """
    df = df.copy()
    df["type"]  = df["type"].fillna("Unknown")
    df["genre"] = df["genre"].fillna("Unknown")

    # split genres
    df["genre_split"] = df["genre"].fillna("").apply(
        lambda x: [g.strip() for g in x.split(",")] if isinstance(x, str) and x else []
    )

    exploded = df.explode("genre_split")
    genre_medians = exploded.groupby("genre_split")["rating"].median()
    global_median = float(df["rating"].median())

    def fill_rating(row: pd.Series) -> float:
        if pd.isna(row["rating"]):
            genres = row["genre_split"]
            if genres:
                medians = [genre_medians.get(g, np.nan) for g in genres]
                medians = [m for m in medians if not pd.isna(m)]
                if medians:
                    return float(np.mean(medians))
            return global_median
        return float(row["rating"])

    df["rating"] = df.apply(fill_rating, axis=1)

    # missingness flags for feature building
    df["miss_genre"]  = (df["genre"] == "Unknown").astype(float)
    df["miss_type"]   = (df["type"]  == "Unknown").astype(float)
    df["miss_rating"] = df["rating"].isna().astype(float)  # after fill this should be 0.0

    # ensure numeric
    for col in ["episodes", "rating", "members"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop(columns=["genre_split"])


# ===============================
# Splitting
# ===============================

def _split_userwise(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """User-wise split: for each user, shuffle their indices and slice."""
    rng = np.random.default_rng(seed)
    parts: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

    for _, user_group in df.groupby("user_id", sort=False):
        n = len(user_group)
        if n < 2:
            parts.append((user_group, user_group.iloc[0:0], user_group.iloc[0:0]))
            continue

        indices = np.arange(n)
        rng.shuffle(indices)

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        # ensure at least 1 example goes to test if possible
        n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(n_train, 1)
        n_val   = min(max(n_val,   1), n - n_train - 1) if n - n_train >= 2 else 0

        train_idx = indices[:n_train]
        val_idx   = indices[n_train:n_train + n_val]
        test_idx  = indices[n_train + n_val:]

        parts.append((
            user_group.iloc[train_idx].copy(),
            user_group.iloc[val_idx].copy(),
            user_group.iloc[test_idx].copy(),
        ))

    train = pd.concat([p[0] for p in parts], ignore_index=True)
    val   = pd.concat([p[1] for p in parts], ignore_index=True)
    test  = pd.concat([p[2] for p in parts], ignore_index=True)
    return train, val, test


# ===============================
# Item feature builder (optional)
# ===============================

def _build_item_features(anime_df: pd.DataFrame, out_dir: str) -> None:
    """
    Build a feature matrix:
      [ multi-hot(genre_vocab) | one-hot(type_vocab) | scaled(episodes,rating,members) | miss_* flags ]
    Saves:
      - {out_dir}/outputs/item_feats.npy
      - {out_dir}/outputs/item_feats_meta.json
    """
    os.makedirs(os.path.join(out_dir, "outputs"), exist_ok=True)

    # ----- Genre vocab -----
    all_genres = set()
    for xs in anime_df["genre"].fillna("Unknown"):
        if isinstance(xs, str) and xs.strip():
            toks = [g.strip() for g in xs.split(",") if g.strip()]
            all_genres.update(toks if toks else ["Unknown"])
        else:
            all_genres.add("Unknown")
    all_genres.add("Unknown")
    genre_vocab = sorted(all_genres)

    # ----- Type vocab -----
    types_clean = (
        anime_df["type"]
        .astype(str)
        .apply(lambda s: s.strip() if s and s.strip() else "Unknown")
        .tolist()
    )
    type_vocab = sorted(set(types_clean))
    if "Unknown" not in type_vocab:
        type_vocab = ["Unknown"] + type_vocab

    # ----- Encodings -----
    genre_idx = {g: i for i, g in enumerate(genre_vocab)}
    G = np.zeros((len(anime_df), len(genre_vocab)), dtype=np.float32)
    for r, xs in enumerate(anime_df["genre"].fillna("Unknown")):
        tokens = [t.strip() for t in xs.split(",")] if isinstance(xs, str) else ["Unknown"]
        tokens = [t for t in tokens if t] or ["Unknown"]
        for t in tokens:
            G[r, genre_idx.get(t, genre_idx["Unknown"])] = 1.0

    type_idx = {t: i for i, t in enumerate(type_vocab)}
    T = np.zeros((len(anime_df), len(type_vocab)), dtype=np.float32)
    for r, t in enumerate(types_clean):
        T[r, type_idx.get(t if t else "Unknown", type_idx["Unknown"])] = 1.0

    # Numeric (min-max; NaN -> median)
    num_cols = ["episodes", "rating", "members"]
    nums = anime_df[num_cols].copy()
    for c in num_cols:
        med = float(nums[c].median()) if not np.isnan(nums[c].median()) else 0.0
        nums[c] = nums[c].fillna(med)

    mins = nums.min(axis=0).to_dict()
    maxs = nums.max(axis=0).to_dict()

    def minmax(col: str, arr: np.ndarray) -> np.ndarray:
        mn, mx = mins[col], maxs[col]
        if mx == mn:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mn) / (mx - mn)).astype(np.float32)

    E = minmax("episodes", nums["episodes"].to_numpy(np.float32))
    R = minmax("rating",   nums["rating"].to_numpy(np.float32))
    M = minmax("members",  nums["members"].to_numpy(np.float32))
    NUM = np.stack([E, R, M], axis=1)

    # Missingness flags
    MISS = anime_df[["miss_genre", "miss_type", "miss_rating"]].copy()
    for c in ["miss_genre", "miss_type", "miss_rating"]:
        if c not in MISS.columns:
            MISS[c] = 0.0
    MISS = MISS.fillna(0.0).to_numpy(np.float32)

    FEATS = np.concatenate([G, T, NUM, MISS], axis=1).astype(np.float32)

    meta = {
        "genre_vocab": genre_vocab,
        "type_vocab": type_vocab,
        "num_cols": num_cols,
        "num_mins": {k: float(v) for k, v in mins.items()},
        "num_maxs": {k: float(v) for k, v in maxs.items()},
        "feature_blocks": {
            "genre": [0, len(genre_vocab)],
            "type": [len(genre_vocab), len(genre_vocab) + len(type_vocab)],
            "numeric": [
                len(genre_vocab) + len(type_vocab),
                len(genre_vocab) + len(type_vocab) + len(num_cols),
            ],
            "missingness": [
                len(genre_vocab) + len(type_vocab) + len(num_cols),
                FEATS.shape[1],
            ],
        },
    }

    np.save(os.path.join(out_dir, "outputs", "item_feats.npy"), FEATS)
    with open(os.path.join(out_dir, "outputs", "item_feats_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ===============================
# Diagnostics (optional CLI flag)
# ===============================

def check_features(clean_dir="data_clean", outputs="outputs"):
    anime = pd.read_csv(os.path.join(clean_dir, "anime.csv"))
    with open(os.path.join(outputs, "item_feats_meta.json")) as f:
        meta = json.load(f)
    feats = np.load(os.path.join(outputs, "item_feats.npy"))

    print("=== Anime metadata (cleaned) ===")
    print(f"Total items: {len(anime)}")
    print(f"Feature matrix shape: {feats.shape}")

    # Missingness stats
    miss_genre  = anime["genre"].isna().sum()
    miss_type   = anime["type"].isna().sum()
    miss_rating = anime["rating"].isna().sum()
    print("\n=== Missing values in anime.csv ===")
    print(f"Missing genres: {miss_genre} ({miss_genre/len(anime):.2%})")
    print(f"Missing types: {miss_type} ({miss_type/len(anime):.2%})")
    print(f"Missing ratings: {miss_rating} ({miss_rating/len(anime):.2%})")

    # Numeric summaries
    print("\n=== Numeric fields summary (before scaling) ===")
    for col in ["episodes", "rating", "members"]:
        desc = anime[col].describe(percentiles=[.25, .5, .75])
        print(f"\n{col}:")
        print(desc)

    # Sanity checks
    feat_cols = (
        [f"genre_{g}" for g in meta["genre_vocab"]] +
        [f"type_{t}"  for t in meta["type_vocab"]]  +
        ["episodes", "rating", "members", "miss_genre", "miss_type", "miss_rating"]
    )
    df_feats = pd.DataFrame(feats, columns=feat_cols)

    print("\n=== Sanity check: missingness flags ===")
    for c in ["miss_genre", "miss_type", "miss_rating"]:
        print(f"{c}: {int(df_feats[c].sum())} items flagged ({df_feats[c].mean():.2%})")

    print("\n=== Sanity check: numeric feature ranges (scaled) ===")
    for c in ["episodes", "rating", "members"]:
        print(f"{c}: min={df_feats[c].min():.3f}, max={df_feats[c].max():.3f}, mean={df_feats[c].mean():.3f})")


# ===============================
# Main pipeline
# ===============================

def main():
    p = argparse.ArgumentParser(description="Preprocess ratings (and optional anime metadata)")
    # Ratings path (required). Also accept legacy --input for backward compat.
    p.add_argument("--ratings_csv", "--input", dest="ratings_csv", default=None,
                   help="Path to ratings CSV (cols: user_id, anime_id, rating)")
    # Anime path (optional)
    p.add_argument("--anime_csv", default=None, help="Path to anime metadata CSV (optional)")
    # Alternatively, a folder with both files named anime.csv/rating.csv
    p.add_argument("--data_dir", default=None, help="Folder containing anime.csv and rating.csv")
    p.add_argument("--out_dir",  default="data_clean", help="Output folder for cleaned/split data")

    p.add_argument("--min_user_ratings", type=int,   default=5)
    p.add_argument("--min_item_ratings", type=int,   default=5)
    p.add_argument("--train_ratio",      type=float, default=0.70)
    p.add_argument("--val_ratio",        type=float, default=0.15)
    p.add_argument("--seed",             type=int,   default=42)

    p.add_argument("--build_item_features", action="store_true",
                   help="If anime_csv is provided, build item feature matrix")
    p.add_argument("--check", action="store_true", help="Run feature checks after build (requires features)")

    a = p.parse_args()

    # Resolve inputs
    ratings_path = a.ratings_csv
    anime_path   = a.anime_csv
    if a.data_dir:
        if ratings_path is None:
            ratings_path = os.path.join(a.data_dir, "rating.csv")
        if anime_path is None:
            apath = os.path.join(a.data_dir, "anime.csv")
            anime_path = apath if os.path.exists(apath) else None

    if ratings_path is None or not os.path.exists(ratings_path):
        raise FileNotFoundError("Provide --ratings_csv or --data_dir with rating.csv")

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "splits"),   exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "mappings"), exist_ok=True)

    # 1) load raw
    ratings_raw = pd.read_csv(ratings_path)
    anime_raw   = pd.read_csv(anime_path) if (anime_path and os.path.exists(anime_path)) else None

    # 2) clean
    ratings = _clean_rating(ratings_raw)
    anime   = _clean_anime(anime_raw) if anime_raw is not None else None

    # 3) filter sparsity (ratings only)
    ratings = _filter_sparsity(
        ratings,
        min_user_ratings=a.min_user_ratings,
        min_item_ratings=a.min_item_ratings,
    )

    # 4) reindex & (if anime) keep only rated items
    ratings, anime_keep, uid_map, iid_map = _reindex_ids(ratings, anime)

    def _summary_stats(df: pd.DataFrame, *, columns: list[str] | None = None) -> dict:
        stats = {"count": int(len(df))}
        if len(df) > 0:
            if "user_id" in df.columns:
                stats["n_users"] = int(df["user_id"].nunique())
            if "anime_id" in df.columns:
                stats["n_items"] = int(df["anime_id"].nunique())
        if columns and len(df) > 0:
            for col in columns:
                if col in df.columns:
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(series) == 0:
                        continue
                    stats[col] = {
                        "mean": float(series.mean()),
                        "std": float(series.std(ddof=0)),
                        "min": float(series.min()),
                        "max": float(series.max()),
                    }
        return stats

    # 5) (optional) impute anime metadata + build features
    if anime_keep is not None:
        anime_keep = _impute_missing_values(anime_keep)

    # 6) save cleaned tables
    ratings.sort_values(["user_id", "anime_id"]).to_csv(os.path.join(a.out_dir, "ratings.csv"), index=False)
    if anime_keep is not None:
        anime_keep.sort_values("anime_id").to_csv(os.path.join(a.out_dir, "anime.csv"), index=False)

    # save mappings
    with open(os.path.join(a.out_dir, "mappings", "uid_map.pkl"), "wb") as f:
        pickle.dump(uid_map, f)
    with open(os.path.join(a.out_dir, "mappings", "iid_map.pkl"), "wb") as f:
        pickle.dump(iid_map, f)

    # 7) (optional) build item features (requires anime)
    if a.build_item_features and anime_keep is not None:
        _build_item_features(anime_keep, a.out_dir)

    # 8) user-wise split
    train, val, test = _split_userwise(
        ratings,
        train_ratio=a.train_ratio,
        val_ratio=a.val_ratio,
        seed=a.seed
    )
    train.to_csv(os.path.join(a.out_dir, "splits", "train.csv"), index=False)
    val.to_csv(  os.path.join(a.out_dir, "splits", "val.csv"),   index=False)
    test.to_csv( os.path.join(a.out_dir, "splits", "test.csv"),  index=False)

    # dataset summaries
    outputs_dir = os.path.join(a.out_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    meta_general = {
        "ratings": _summary_stats(ratings, columns=["rating"]),
        "splits": {
            "train": _summary_stats(train, columns=["rating"]),
            "val": _summary_stats(val, columns=["rating"]),
            "test": _summary_stats(test, columns=["rating"]),
        },
    }
    if anime_keep is not None:
        meta_general["anime"] = _summary_stats(anime_keep, columns=["rating", "episodes", "members"])

    with open(os.path.join(outputs_dir, "dataset_meta.json"), "w") as f:
        json.dump(meta_general, f, indent=2)

    # 9) quick stats
    n_users = ratings["user_id"].nunique()
    n_items = ratings["anime_id"].nunique()
    print(f"Users: {n_users} | Items: {n_items} | Ratings: {len(ratings)}")
    print(f"Splits -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"Saved cleaned data to: {a.out_dir}")

    # 10) optional checks
    if a.check:
        if not a.build_item_features or anime_keep is None:
            print("[warn] --check requested but features not built; skipping diagnostics.")
        else:
            check_features(clean_dir=a.out_dir, outputs=os.path.join(a.out_dir, "outputs"))


if __name__ == "__main__":
    main()
