# src/eval.py
from __future__ import annotations
"""
Unified evaluation script for explicit MF/NeuMF:
- rating metrics: RMSE / MAE
- ranking metrics: HR@K / NDCG@K with exclude-seen (train/optional val)
- robust device selection (safe CPU fallback if CUDA not usable)
- flexible checkpoint resolution (file, dir with best/last, or runs/<exp>/latest/best.ckpt)
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.recsys.models.mf import MF
from src.recsys.models.neumf import NeuMF
from src.recsys.metrics import hit_rate_at_k, ndcg_at_k, precision_recall_at_k, average_precision_at_k
from src.utils import get_device


# -----------------------------
# Config / checkpoint helpers
# -----------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_ckpt_path(ckpt_arg: str | None, cfg: dict) -> str:
    """Return an absolute path to a checkpoint file."""
    if ckpt_arg:
        if os.path.isdir(ckpt_arg):
            # prefer best.ckpt -> last.ckpt
            best = os.path.join(ckpt_arg, "best.ckpt")
            last = os.path.join(ckpt_arg, "last.ckpt")
            if os.path.exists(best): return best
            if os.path.exists(last): return last
            raise FileNotFoundError(f"No best.ckpt/last.ckpt in {ckpt_arg}")
        if os.path.isfile(ckpt_arg):
            return ckpt_arg
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")

    # Default: runs/<exp_name>/latest/best.ckpt
    base = cfg["log"]["dir"]
    exp = cfg.get("exp_name", "exp")
    best = os.path.join(base, exp, "latest", "best.ckpt")
    if os.path.exists(best):
        return best
    raise FileNotFoundError(
        f"Could not resolve checkpoint. Pass --ckpt or ensure {best} exists."
    )


def load_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[torch.nn.Module, dict, int, int]:
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg_ck = ck.get("cfg", {})
    n_users = int(ck["n_users"])
    n_items = int(ck["n_items"])
    name = str(cfg_ck.get("model", {}).get("name", "neumf")).lower()

    if name == "mf":
        model = MF(
            n_users, n_items,
            dim=int(cfg_ck["model"].get("mf_dim", 64)),
            user_bias=bool(cfg_ck["model"].get("user_bias", True)),
            item_bias=bool(cfg_ck["model"].get("item_bias", True)),
        )
    else:
        model = NeuMF(
            n_users, n_items,
            mf_dim=int(cfg_ck["model"].get("mf_dim", 32)),
            mlp_layers=tuple(cfg_ck["model"].get("mlp_layers", [128, 64])),
            dropout=float(cfg_ck["model"].get("dropout", 0.1)),
            user_bias=bool(cfg_ck["model"].get("user_bias", True)),
            item_bias=bool(cfg_ck["model"].get("item_bias", True)),
        )

    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, cfg_ck, n_users, n_items


# -----------------------------
# Data helpers
# -----------------------------
def load_split(processed_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(processed_dir, f"splits/{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path, usecols=["user_id", "anime_id", "rating"])


def build_seen_sets(processed_dir: str, also_val: bool) -> Dict[int, set[int]]:
    seen: Dict[int, set[int]] = {}
    for fname in ["train.csv"] + (["val.csv"] if also_val else []):
        path = os.path.join(processed_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["user_id", "anime_id"])
        for u, i in df.itertuples(index=False):
            s = seen.get(u)
            if s is None:
                s = set()
                seen[u] = s
            s.add(int(i))
    return seen


def positives_from_split(df: pd.DataFrame, threshold: float) -> Dict[int, List[int]]:
    pos: Dict[int, List[int]] = {}
    mask = df["rating"] >= threshold
    for u, i in df.loc[mask, ["user_id", "anime_id"]].itertuples(index=False):
        pos.setdefault(int(u), []).append(int(i))
    return pos


# -----------------------------
# Metrics (rating)
# -----------------------------
def eval_rating(model: torch.nn.Module, df: pd.DataFrame, device: str) -> Tuple[float, float]:
    """Compute RMSE/MAE on the given split."""
    u = torch.as_tensor(df["user_id"].to_numpy(), dtype=torch.long, device=device)
    i = torch.as_tensor(df["anime_id"].to_numpy(), dtype=torch.long, device=device)
    r = torch.as_tensor(df["rating"].to_numpy(np.float32), dtype=torch.float32, device=device)

    with torch.no_grad():
        # Chunk to avoid OOM on huge splits
        B = 65536
        preds = []
        for t in range(0, u.shape[0], B):
            preds.append(model(u[t:t+B], i[t:t+B]).detach())
        y_pred = torch.cat(preds, dim=0)
    y_true = r
    rmse = float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item())
    mae = float(torch.mean(torch.abs(y_true - y_pred)).item())
    return rmse, mae


# -----------------------------
# Ranking (top-K) eval
# -----------------------------
def eval_ranking(
    model: torch.nn.Module,
    processed_dir: str,
    df_split: pd.DataFrame,
    n_items: int,
    k_list: List[int],
    pos_threshold: float,
    exclude_seen: bool,
    also_exclude_val: bool,
    device: str,
    limit_users: int | None = None,
) -> Dict[int, Dict[str, float]]:
    # Positives and users to evaluate
    pos = positives_from_split(df_split, pos_threshold)
    users = sorted(pos.keys())
    if limit_users and limit_users > 0:
        users = users[:limit_users]

    # Seen sets
    seen = build_seen_sets(processed_dir, also_exclude_val) if exclude_seen else {}

    is_mf = isinstance(model, MF)
    ranked_lists: List[List[int]] = []
    true_lists: List[List[int]] = []

    with torch.no_grad():
        if is_mf:
            Q = model.Q.weight.to(device)  # (n_items, d)
            ib = model.ib.weight[:, 0].to(device) if model.ib is not None else None

        for u in users:
            if is_mf:
                p = model.P.weight[u:u+1].to(device)  # (1, d)
                scores = (p * Q).sum(-1)
                if model.ub is not None:
                    scores += model.ub.weight[u, 0].to(device)
                if ib is not None:
                    scores += ib
            else:
                B = 8192
                idx = torch.arange(n_items, device=device)
                chunks = []
                for t in range(0, n_items, B):
                    it = idx[t:t+B]
                    ut = torch.full_like(it, fill_value=u, dtype=torch.long)
                    chunks.append(model(ut, it))
                scores = torch.cat(chunks, dim=0)

            scores_np = scores.detach().cpu().numpy()

            # Mask seen items
            if exclude_seen and u in seen:
                seen_idxs = list(seen[u])
                scores_np[seen_idxs] = -np.inf

            order = np.argsort(-scores_np, kind="mergesort")
            ranked_lists.append(order.tolist())
            true_lists.append(pos.get(u, []))

    # --- Compute metrics per K ---
    results: Dict[int, Dict[str, float]] = {}
    for K in k_list:
        hr_list = []
        ndcg_list = []
        for truth, ranked in zip(true_lists, ranked_lists):
            hr_list.append(hit_rate_at_k(truth, ranked, K))
            ndcg_list.append(ndcg_at_k(truth, ranked, K))
        hr = float(np.mean(hr_list))
        ndcg = float(np.mean(ndcg_list))
        P, R = precision_recall_at_k(true_lists, ranked_lists, K)
        MAP = average_precision_at_k(true_lists, ranked_lists, K)
        results[K] = {"HR": hr, "NDCG": ndcg, "P": float(P), "R": float(R), "MAP": float(MAP)}
        print(f"[ranking] K={K:>3} | HR={hr:.4f} NDCG={ndcg:.4f} P={P:.4f} R={R:.4f} MAP={MAP:.4f}")

    return results


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified evaluation (rating and/or ranking)")
    p.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p.add_argument("--ckpt", default=None, help="Path to .ckpt file or a run dir (best/last). If omitted, uses runs/<exp>/latest/best.ckpt")
    p.add_argument("--split", choices=["val", "test"], default="test")

    # ranking params
    p.add_argument("--k", default="10,20", help="Comma-separated Ks, e.g. '10,20,50'")
    p.add_argument("--pos_threshold", type=float, default=8.0, help="rating >= threshold counts as positive")
    p.add_argument("--exclude_seen", type=int, default=1, help="exclude TRAIN items from rec lists (1/0)")
    p.add_argument("--also_exclude_val", type=int, default=0, help="also exclude VAL items (1/0)")
    p.add_argument("--max_users", type=int, default=0, help="evaluate only first N users with positives; 0=all")

    # device & output
    p.add_argument("--device", default=None, help="Force device: cuda|cpu|mps (default: auto)")
    p.add_argument("--out", default=None, help="Optional path to write metrics JSON")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    a = parse_args()
    cfg = load_cfg(a.config)

    device = get_device(a.device)
    processed_dir = cfg["recsys"]["processed_dir"]

    # Resolve checkpoint and load model
    ckpt_path = resolve_ckpt_path(a.ckpt, cfg)
    model, cfg_ck, n_users, n_items = load_model_from_ckpt(ckpt_path, device)

    # Load split
    df = load_split(processed_dir, a.split)

    all_metrics = {}

    rmse, mae = eval_rating(model, df, device)
    all_metrics["rating"] = {"RMSE": rmse, "MAE": mae}
    print(f"[rating] {a.split} | RMSE={rmse:.4f} MAE={mae:.4f}")

    K_list = sorted({int(x) for x in a.k.split(",") if x.strip()})
    rank_metrics = eval_ranking(
        model=model,
        processed_dir=processed_dir,
        df_split=df,
        n_items=n_items,
        k_list=K_list,
        pos_threshold=a.pos_threshold,
        exclude_seen=bool(a.exclude_seen),
        also_exclude_val=bool(a.also_exclude_val),
        device=device,
        limit_users=(a.max_users if a.max_users > 0 else None),
    )
    all_metrics["ranking"] = rank_metrics

    # Optional JSON dump
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"[saved] metrics -> {a.out}")


if __name__ == "__main__":
    main()
