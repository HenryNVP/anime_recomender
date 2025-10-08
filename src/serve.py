from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.eval import load_cfg, resolve_ckpt_path, load_model_from_ckpt
from src.recsys.models.mf import MF
from src.utils import get_device


def _load_anime_metadata(processed_dir: str) -> Dict[int, str]:
    path = os.path.join(processed_dir, "anime.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, usecols=["anime_id", "name"]).dropna()
    return {int(r.anime_id): str(r.name) for r in df.itertuples(index=False)}


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


class UserNotFoundError(Exception):
    """Raised when a raw user id cannot be mapped to a training index."""


def _load_uid_maps(processed_dir: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    path = os.path.join(processed_dir, "mappings", "uid_map.pkl")
    if not os.path.exists(path):
        return {}, {}
    with open(path, "rb") as f:
        uid_map = pickle.load(f)
    uid_map = {int(k): int(v) for k, v in uid_map.items()}
    uid_map_inv = {v: k for k, v in uid_map.items()}
    return uid_map, uid_map_inv


def _load_iid_maps(processed_dir: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    path = os.path.join(processed_dir, "mappings", "iid_map.pkl")
    if not os.path.exists(path):
        return {}, {}
    with open(path, "rb") as f:
        iid_map = pickle.load(f)
    iid_map = {int(k): int(v) for k, v in iid_map.items()}
    iid_map_inv = {v: k for k, v in iid_map.items()}
    return iid_map, iid_map_inv


def _load_user_interactions(processed_dir: str) -> Tuple[Dict[int, np.ndarray], Dict[int, List[Tuple[int, float]]]]:
    """
    Load per-user watched item ids and rating history from the reindexed ratings.csv file.
    Returns:
      - seen dict[user_idx] -> np.ndarray[item_idx]
      - history dict[user_idx] -> List[(item_idx, rating)] sorted by rating desc
    """
    path = os.path.join(processed_dir, "ratings.csv")
    if not os.path.exists(path):
        return {}, {}

    df = pd.read_csv(path, usecols=["user_id", "anime_id", "rating"])
    seen: Dict[int, List[int]] = {}
    history: Dict[int, List[Tuple[int, float]]] = {}
    for row in df.itertuples(index=False):
        u = int(row.user_id)
        i = int(row.anime_id)
        r = float(row.rating)
        seen.setdefault(u, []).append(i)
        history.setdefault(u, []).append((i, r))

    # sort history per user by rating desc
    for u, entries in history.items():
        entries.sort(key=lambda x: x[1], reverse=True)

    seen_np = {u: np.asarray(items, dtype=np.int64) for u, items in seen.items()}
    return seen_np, history


@dataclass
class RecommendationModel:
    alias: str
    config_path: str
    ckpt_path: str
    model: torch.nn.Module
    device: str
    n_users: int
    n_items: int
    processed_dir: str
    anime_map: Dict[int, str]
    uid_map: Dict[int, int]
    uid_map_inv: Dict[int, int]
    iid_map_inv: Dict[int, int]
    seen_items: Dict[int, np.ndarray]
    history: Dict[int, List[Tuple[int, float]]]

    def map_user(self, raw_user_id: int) -> int:
        if self.uid_map:
            mapped = self.uid_map.get(raw_user_id)
            if mapped is None:
                # allow callers to pass reindexed ids even when a uid_map exists
                if raw_user_id in self.uid_map_inv:
                    return raw_user_id
                if 0 <= raw_user_id < self.n_users:
                    return raw_user_id
                raise UserNotFoundError(f"User {raw_user_id} not found in uid_map")
            return mapped
        if raw_user_id < 0 or raw_user_id >= self.n_users:
            raise UserNotFoundError(f"User {raw_user_id} outside trained range 0-{self.n_users - 1}")
        return raw_user_id

    def _original_item_id(self, item_idx: int) -> int:
        if self.iid_map_inv:
            return int(self.iid_map_inv.get(item_idx, item_idx))
        return int(item_idx)

    def recommend(
        self,
        raw_user_id: int,
        k: int,
        include_history: bool = False,
        history_k: int = 10,
    ) -> Dict[str, object]:
        user_idx = self.map_user(int(raw_user_id))
        seen_arr = self.seen_items.get(user_idx)
        exclude = seen_arr.tolist() if seen_arr is not None else []

        scores = _score_user(self.model, user_idx, self.n_items, self.device, exclude=exclude)

        k = max(min(int(k), self.n_items), 0)
        ranked: List[int] = []
        if k > 0 and self.n_items > 0:
            if k >= len(scores):
                ranked_idx = np.argsort(-scores, kind="mergesort")
            else:
                top_idx = np.argpartition(-scores, k - 1)[:k]
                ranked_idx = top_idx[np.argsort(-scores[top_idx])]
            ranked = [int(i) for i in ranked_idx if np.isfinite(scores[i])]

        recommendations = []
        for rank, item_idx in enumerate(ranked[:k], start=1):
            score = float(scores[item_idx])
            orig_id = self._original_item_id(item_idx)
            recommendations.append(
                {
                    "rank": rank,
                    "item_id": int(item_idx),
                    "original_anime_id": orig_id,
                    "title": self.anime_map.get(item_idx),
                    "score": score,
                }
            )

        payload: Dict[str, object] = {
            "model": self.alias,
            "config_path": self.config_path,
            "ckpt_path": self.ckpt_path,
            "user_id": int(raw_user_id),
            "user_index": int(user_idx),
            "k": int(k),
            "returned": len(recommendations),
            "recommendations": recommendations,
            "excluded_items": int(len(exclude)),
            "available_items": int(self.n_items),
        }

        if include_history:
            history_entries = self.history.get(user_idx, [])
            hist_payload = []
            for item_idx, rating in history_entries[: max(history_k, 0)]:
                hist_payload.append(
                    {
                        "item_id": int(item_idx),
                        "original_anime_id": self._original_item_id(item_idx),
                        "title": self.anime_map.get(item_idx),
                        "rating": float(rating),
                    }
                )
            payload["history"] = hist_payload
            payload["history_total"] = len(history_entries)

        return payload


def _resolve_processed_dir(cfg: dict, cfg_ck: Optional[dict]) -> str:
    processed_dir = None
    if cfg_ck:
        processed_dir = cfg_ck.get("recsys", {}).get("processed_dir")
    if not processed_dir:
        processed_dir = cfg.get("recsys", {}).get("processed_dir")
    if not processed_dir:
        raise ValueError("processed_dir must be specified in config or checkpoint recsys section")
    return os.path.abspath(os.path.expanduser(processed_dir))


def load_recommendation_model(
    alias: Optional[str],
    config_path: str,
    ckpt_path: Optional[str],
    device: str,
) -> RecommendationModel:
    cfg = load_cfg(config_path)
    resolved_ckpt = resolve_ckpt_path(ckpt_path, cfg)
    model, cfg_ck, n_users, n_items = load_model_from_ckpt(resolved_ckpt, device)

    processed_dir = _resolve_processed_dir(cfg, cfg_ck)
    anime_map = _load_anime_metadata(processed_dir)
    uid_map, uid_map_inv = _load_uid_maps(processed_dir)
    _, iid_map_inv = _load_iid_maps(processed_dir)
    seen, history = _load_user_interactions(processed_dir)

    if not alias:
        alias = cfg.get("exp_name") or os.path.splitext(os.path.basename(config_path))[0]

    return RecommendationModel(
        alias=alias,
        config_path=os.path.abspath(config_path),
        ckpt_path=os.path.abspath(resolved_ckpt),
        model=model,
        device=device,
        n_users=n_users,
        n_items=n_items,
        processed_dir=processed_dir,
        anime_map=anime_map,
        uid_map=uid_map,
        uid_map_inv=uid_map_inv,
        iid_map_inv=iid_map_inv,
        seen_items=seen,
        history=history,
    )


def _parse_model_spec(spec: str) -> Tuple[Optional[str], str, Optional[str]]:
    raw = spec.strip()
    if not raw:
        raise ValueError("Empty model spec provided")
    alias = None
    rest = raw
    if "=" in raw:
        alias_part, rest = raw.split("=", 1)
        alias = alias_part.strip() or None
    if ":" in rest:
        cfg, ckpt = rest.split(":", 1)
        cfg = cfg.strip()
        ckpt = ckpt.strip() or None
    else:
        cfg, ckpt = rest.strip(), None
    if not cfg:
        raise ValueError(f"Invalid model spec '{spec}': missing config path")
    return alias, cfg, ckpt


def _parse_model_specs(args: argparse.Namespace) -> List[Tuple[Optional[str], str, Optional[str]]]:
    specs: List[Tuple[Optional[str], str, Optional[str]]] = []
    for spec in getattr(args, "model", []) or []:
        specs.append(_parse_model_spec(spec))
    if not specs:
        specs.append((None, args.config, args.ckpt))
    return specs


def build_model_registry(
    specs: List[Tuple[Optional[str], str, Optional[str]]],
    device: str,
) -> Tuple[Dict[str, RecommendationModel], str]:
    registry: Dict[str, RecommendationModel] = {}
    default_alias: Optional[str] = None
    for alias, cfg_path, ckpt_path in specs:
        model = load_recommendation_model(alias, cfg_path, ckpt_path, device)
        alias_key = model.alias
        if alias_key in registry:
            base = alias_key
            suffix = 2
            while f"{base}_{suffix}" in registry:
                suffix += 1
            alias_key = f"{base}_{suffix}"
            model.alias = alias_key
        registry[alias_key] = model
        if default_alias is None:
            default_alias = alias_key
    if not registry:
        raise ValueError("No models could be loaded for serving.")
    assert default_alias is not None

    preferred = next(
        (name for name in registry if name.lower().replace("-", "_") in {"twotower", "two_tower"}),
        None,
    )
    if preferred:
        default_alias = preferred
    return registry, default_alias


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="Original user id")
    k: int = Field(10, ge=0, description="Number of recommendations to return")
    model: Optional[str] = Field(
        default=None, description="Model alias to use (defaults to first loaded model)"
    )
    include_history: bool = Field(False, description="Include top rated history in response")
    history_k: Optional[int] = Field(
        default=None, ge=0, description="History entries to include when requested"
    )


class RecommendationItem(BaseModel):
    rank: int
    item_id: int
    original_anime_id: Optional[int] = None
    title: Optional[str] = None
    score: float


class HistoryItem(BaseModel):
    item_id: int
    original_anime_id: Optional[int] = None
    title: Optional[str] = None
    rating: float


class RecommendationPayload(BaseModel):
    model: str
    config_path: str
    ckpt_path: str
    user_id: int
    user_index: int
    k: int
    returned: int
    recommendations: List[RecommendationItem]
    excluded_items: int
    available_items: int
    history: Optional[List[HistoryItem]] = None
    history_total: Optional[int] = None


def create_app(models: Dict[str, RecommendationModel], default_alias: str) -> FastAPI:
    app = FastAPI(title="Anime Recommendation Service", version="1.0.0")

    def _serve(
        alias: Optional[str],
        user_id: int,
        k: int,
        include_history: bool,
        history_k: Optional[int],
    ) -> Dict[str, object]:
        target_alias = alias or default_alias
        model = models.get(target_alias)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Unknown model alias '{target_alias}'")
        hist_k = history_k if history_k is not None else 10
        try:
            payload = model.recommend(
                raw_user_id=user_id,
                k=k,
                include_history=include_history,
                history_k=hist_k,
            )
        except UserNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RecommendationPayload(**payload).dict()

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {"status": "ok", "models": list(models.keys())}

    @app.get("/v1/models")
    def list_models() -> Dict[str, List[Dict[str, object]]]:
        payload = []
        for alias, model in models.items():
            payload.append(
                {
                    "alias": alias,
                    "config_path": model.config_path,
                    "ckpt_path": model.ckpt_path,
                    "n_users": model.n_users,
                    "n_items": model.n_items,
                    "processed_dir": model.processed_dir,
                    "has_uid_map": bool(model.uid_map),
                    "has_iid_map": bool(model.iid_map_inv),
                }
            )
        return {"models": payload}

    @app.post("/v1/recommendations")
    def recommend(req: RecommendRequest) -> Dict[str, object]:
        return _serve(req.model, req.user_id, req.k, req.include_history, req.history_k)

    @app.get("/v1/recommendations")
    def recommend_query(
        user_id: int = Query(..., description="Original user id"),
        k: int = Query(10, ge=0),
        model: Optional[str] = Query(None, description="Model alias to use"),
        include_history: bool = Query(False),
        history_k: Optional[int] = Query(None, ge=0),
    ) -> Dict[str, object]:
        return _serve(model, user_id, k, include_history, history_k)

    return app


def serve_cli(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    specs = _parse_model_specs(args)
    if len(specs) > 1:
        raise ValueError("CLI mode supports a single model. Use --http for multi-model serving.")

    alias, config_path, ckpt_path = specs[0]
    model = load_recommendation_model(alias, config_path, ckpt_path, device)

    try:
        result = model.recommend(
            raw_user_id=args.user,
            k=args.k,
            include_history=True,
            history_k=args.history,
        )
    except UserNotFoundError as exc:
        raise SystemExit(str(exc))

    history = result.get("history") or []
    user_id = result["user_id"]
    user_idx = result["user_index"]

    if history:
        print(f"\nUser {user_id} (train index {user_idx}) history (top {len(history)} by rating):")
        for rank, item in enumerate(history, start=1):
            title = item.get("title") or f"anime_id={item['item_id']}"
            original = item.get("original_anime_id")
            rating = item.get("rating", 0.0)
            print(
                f"{rank:>2d}. {title} "
                f"(id={item['item_id']}, original_id={original}, rating={rating:.2f})"
            )
    else:
        print(f"\nUser {user_id} (train index {user_idx}) has no recorded history.")

    recs = result.get("recommendations", [])
    if not recs:
        print(f"\nNo recommendations available for user {user_id}.")
        return

    print(f"\nRecommended top-{result['k']} items for user {user_id}:")
    for item in recs:
        title = item.get("title") or f"anime_id={item['item_id']}"
        original = item.get("original_anime_id")
        score = item.get("score", 0.0)
        print(
            f"{item['rank']:>2d}. {title} "
            f"(id={item['item_id']}, original_id={original}, score={score:.4f})"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve top-N recommendations for a user")
    p.add_argument("--config", default="configs/config.yaml", help="Config used during training")
    p.add_argument("--ckpt", default=None, help="Path to checkpoint (defaults to resolved best)")
    p.add_argument("--user", type=int, help="Original user id to query (CLI mode)")
    p.add_argument("--k", type=int, default=10, help="Number of recommendations to show")
    p.add_argument("--history", type=int, default=10, help="Number of watched items to display")
    p.add_argument("--device", default=None, help="Override device (cuda|cpu|mps)")
    p.add_argument(
        "--http",
        action="store_true",
        help="Start FastAPI HTTP service instead of printing recommendations",
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec alias=config.yaml[:ckpt_path] (repeatable, HTTP mode)",
    )
    p.add_argument("--host", default="0.0.0.0", help="Host interface for HTTP mode")
    p.add_argument("--port", type=int, default=8000, help="Port for HTTP mode")
    p.add_argument("--log-level", default="info", help="Uvicorn log level for HTTP mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.http:
        device = get_device(args.device)
        specs = _parse_model_specs(args)
        models, default_alias = build_model_registry(specs, device)
        app = create_app(models, default_alias)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    else:
        if args.user is None:
            raise SystemExit("--user is required in CLI mode (omit --http)")
        serve_cli(args)


if __name__ == "__main__":
    main()
