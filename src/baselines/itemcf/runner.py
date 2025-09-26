from __future__ import annotations
import os, json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .model import ItemCF
from src.baselines.itemcf.metrics import precision_recall_at_k, hit_rate_at_k, average_precision_at_k, ndcg_at_k

# ---------- helpers ----------
def _infer_shape(splits_dir: str) -> Tuple[int, int]:
    max_u = max_i = -1
    for name in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{name}.csv")
        if not os.path.exists(p): continue
        df = pd.read_csv(p, usecols=["user_id", "anime_id"])
        if not df.empty:
            max_u = max(max_u, int(df["user_id"].max()))
            max_i = max(max_i, int(df["anime_id"].max()))
    if max_u < 0 or max_i < 0:
        raise FileNotFoundError("No split files found to infer shapes.")
    return max_u + 1, max_i + 1

def _load_seen_sets(splits_dir: str, also_exclude_val: bool) -> Dict[int, set[int]]:
    seen: Dict[int, set[int]] = {}
    for fname in ("train.csv", "val.csv" if also_exclude_val else None):
        if not fname: continue
        path = os.path.join(splits_dir, fname)
        if not os.path.exists(path): continue
        df = pd.read_csv(path, usecols=["user_id", "anime_id"])
        for u, i in df.itertuples(index=False):
            seen.setdefault(int(u), set()).add(int(i))
    return seen

def _ground_truth_pos(df_split: pd.DataFrame, pos_threshold: float) -> Dict[int, List[int]]:
    pos: Dict[int, List[int]] = {}
    mask = df_split["rating"] >= pos_threshold
    for u, i in df_split.loc[mask, ["user_id", "anime_id"]].itertuples(index=False):
        pos.setdefault(int(u), []).append(int(i))
    return pos

def _item_popularity_from_train(splits_dir: str, n_items: int) -> np.ndarray:
    tr = pd.read_csv(os.path.join(splits_dir, "train.csv"), usecols=["anime_id"])
    vc = tr["anime_id"].value_counts().sort_index()
    pop = np.zeros(n_items, dtype=np.int64)
    pop[vc.index.to_numpy()] = vc.to_numpy()
    return pop

# ---------- API: training ----------
def train_from_splits(
    data_dir: str = "data_clean",
    splits_dir: Optional[str] = None,
    out_prefix: str = "outputs/itemcf",
    k: int = 100,
    shrink: float = 0.0,
    clip_min: float = 1.0,
    clip_max: float = 10.0,
    eval_on: str = "val",   # none|val|test
) -> Dict[str, float]:
    splits = splits_dir or os.path.join(data_dir, "splits")
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    train_path = os.path.join(splits, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}")
    train = pd.read_csv(train_path, usecols=["user_id", "anime_id", "rating"])

    n_users, n_items = _infer_shape(splits)
    model = ItemCF(k=k, shrink=shrink, clip=(clip_min, clip_max))
    model.fit_from_df(train, n_users=n_users, n_items=n_items)

    metrics: Dict[str, float] = {}
    if eval_on in ("val", "test"):
        split_path = os.path.join(splits, f"{eval_on}.csv")
        if os.path.exists(split_path):
            df = pd.read_csv(split_path, usecols=["user_id", "anime_id", "rating"])
            y_true = df["rating"].to_numpy(np.float32)
            y_pred = model.predict_batch(df["user_id"].to_numpy(), df["anime_id"].to_numpy())
            metrics["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            metrics["mae"]  = float(np.mean(np.abs(y_true - y_pred)))
        else:
            print(f"[warn] {eval_on} split not found; skipping rating eval")

    model.save(out_prefix)
    meta = {"algo":"itemcf", "k":k, "shrink":shrink, "clip":[clip_min, clip_max],
            "n_users":int(n_users), "n_items":int(n_items)}
    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return metrics

# ---------- API: evaluation ----------
def eval_from_splits(
    model_prefix: str,
    splits_dir: str,
    split: str = "test",
    k_list: List[int] | None = None,
    pos_threshold: float = 8.0,
    exclude_seen: bool = True,
    also_exclude_val: bool = False,
    max_users: int = 0,
) -> Dict[str, Dict]:
    model = ItemCF().load(model_prefix)
    results: Dict[str, Dict] = {}

    # ---------- Rating ----------
    split_path = os.path.join(splits_dir, f"{split}.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing split {split_path}")
    df = pd.read_csv(split_path, usecols=["user_id", "anime_id", "rating"])

    y = df["rating"].to_numpy(np.float32)
    p = model.predict_batch(df["user_id"].to_numpy(), df["anime_id"].to_numpy())
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    results["rating"] = {"rmse": rmse, "mae": mae}

    # ---------- Ranking ----------
    n_items = model.n_items
    Ks = sorted(set(k_list or [10, 20]))
    absS = model.S.copy()
    absS.data = np.abs(absS.data, dtype=np.float32)
    denom = np.ravel(np.asarray(absS.sum(axis=1))).astype(np.float32) + 1e-12
    seen = _load_seen_sets(splits_dir, also_exclude_val) if exclude_seen else {}
    gt_pos = _ground_truth_pos(df, pos_threshold)
    pop = _item_popularity_from_train(splits_dir, n_items)

    users = sorted(gt_pos.keys())
    if max_users and max_users > 0:
        users = users[:max_users]

    ranked, truths = [], []
    for u in users:
        urow = model.R.getrow(u)
        base = model._user_base(u)
        if urow.nnz == 0:
            scores = pop.astype(np.float32)
        else:
            r_c = np.zeros(n_items, dtype=np.float32)
            r_c[urow.indices] = urow.data
            numer = model.S.dot(r_c).astype(np.float32)
            scores = base + numer / denom
        if exclude_seen and u in seen:
            scores[list(seen[u])] = -np.inf
        order = np.argsort(-scores, kind="mergesort")
        ranked.append(order.astype(int).tolist())
        truths.append(gt_pos.get(u, []))

    results["ranking"] = {}
    for K in Ks:
        P, R = precision_recall_at_k(truths, ranked, K)
        hr = hit_rate_at_k(truths, ranked, K)
        MAP = average_precision_at_k(truths, ranked, K)
        nd = ndcg_at_k(truths, ranked, K)
        results["ranking"][K] = {
            "HR": hr,
            "NDCG": nd,
            "P": float(P),
            "R": float(R),
            "MAP": float(MAP),
        }

    return results
