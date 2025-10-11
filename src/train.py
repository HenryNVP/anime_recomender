# src/train_recsys.py
from __future__ import annotations

"""
Train script for explicit-rating MF/NeuMF with:
- robust device selection (CPU fallback if CUDA not usable)
- mixed precision on CUDA (optional)
- timestamped run directory + config snapshot
- save both last.ckpt and best.ckpt
- optional resume from last.ckpt
"""

import argparse
import math
import os
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.recsys.ratingds import RatingsDS
from src.recsys.models.mf import MF
from src.recsys.models.neumf import NeuMF
from src.recsys.models.twotower import TwoTower
from src.utils import seed_all, Logger, get_device
from src.recsys import metrics
from src.recsys.losses import approx_ndcg_loss, bpr_loss


# -----------------------------
# Helpers
# -----------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(base_dir: str, exp_name: str) -> str:
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, exp_name, run_id)
    os.makedirs(run_dir, exist_ok=True)
    # Maintain a 'latest' symlink if possible
    latest = os.path.join(base_dir, exp_name, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            try:
                os.remove(latest)
            except IsADirectoryError:
                pass
        os.symlink(run_dir, latest)
    except Exception:
        pass
    return run_dir


def save_cfg_snapshot(cfg: dict, run_dir: str) -> None:
    try:
        with open(os.path.join(run_dir, "config.snapshot.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception:
        pass


def build_model(cfg: dict, n_users: int, n_items: int) -> torch.nn.Module:
    m = cfg["model"]
    name = str(m.get("name", "neumf")).lower()
    if name == "mf":
        return MF(
            n_users=n_users,
            n_items=n_items,
            dim=int(m.get("mf_dim", 32)),
            user_bias=bool(m.get("user_bias", True)),
            item_bias=bool(m.get("item_bias", True)),
        )
    elif name == "neumf":
        return NeuMF(
            n_users=n_users,
            n_items=n_items,
            mf_dim=int(m.get("mf_dim", 32)),
            mlp_layers=tuple(m.get("mlp_layers", [128, 64])),
            dropout=float(m.get("dropout", 0.1)),
            user_bias=bool(m.get("user_bias", True)),
            item_bias=bool(m.get("item_bias", True)),
        )
    elif name in {"two_tower", "twotower"}:
        item_features = None
        feat_path = m.get("item_feature_path") or m.get("item_features_path")
        if feat_path:
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"Configured item_feature_path not found: {feat_path}")
            feats_np = np.load(feat_path)
            if not isinstance(feats_np, np.ndarray) or feats_np.ndim != 2:
                raise ValueError(
                    f"item_feature_path must load a 2D array, got shape {getattr(feats_np, 'shape', None)}"
                )
            item_features = torch.from_numpy(feats_np.astype(np.float32, copy=False))

        return TwoTower(
            n_users=n_users,
            n_items=n_items,
            embed_dim=int(m.get("embed_dim", m.get("mf_dim", 64))),
            user_layers=tuple(m.get("user_layers", [])),
            item_layers=tuple(m.get("item_layers", [])),
            dropout=float(m.get("dropout", 0.0)),
            user_bias=bool(m.get("user_bias", True)),
            item_bias=bool(m.get("item_bias", True)),
            item_features=item_features,
            item_feature_layers=tuple(m.get("item_feature_layers", [])),
            item_feature_dropout=float(m.get("item_feature_dropout", m.get("dropout", 0.0))),
            item_feature_combine=str(m.get("item_feature_combine", "concat")),
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def save_ckpt(
    path: str,
    model: torch.nn.Module,
    opt: Optional[torch.optim.Optimizer],
    epoch: int,
    best_val: float,
    n_users: int,
    n_items: int,
    cfg: dict,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": (opt.state_dict() if opt is not None else None),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "n_users": int(n_users),
            "n_items": int(n_items),
            "cfg": cfg,
        },
        path,
    )


def load_ckpt_for_resume(path: str, model: torch.nn.Module, opt: Optional[torch.optim.Optimizer]):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["state_dict"])
    if opt is not None and ck.get("optimizer") is not None:
        opt.load_state_dict(ck["optimizer"])
    start_epoch = int(ck.get("epoch", 0)) + 1
    best_val = float(ck.get("best_val", float("inf")))
    n_users_ck = int(ck.get("n_users", 0))
    n_items_ck = int(ck.get("n_items", 0))
    return start_epoch, best_val, n_users_ck, n_items_ck, ck.get("cfg", None)


def prepare_validation_context(
    processed_dir: str,
    *,
    pos_threshold: float = 8.0,
    max_users: int = 300,
) -> dict:
    train_path = os.path.join(processed_dir, "splits/train.csv")
    val_path = os.path.join(processed_dir, "splits/val.csv")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Missing train/val splits for validation.")

    df_tr = pd.read_csv(train_path, usecols=["user_id", "anime_id"])
    df_va = pd.read_csv(val_path, usecols=["user_id", "anime_id", "rating"])

    seen: dict[int, set[int]] = {}
    for u, i in df_tr.itertuples(index=False):
        seen.setdefault(int(u), set()).add(int(i))

    pos_va = df_va[df_va["rating"] >= pos_threshold]
    truths_all = {
        int(u): grp["anime_id"].astype(int).tolist()
        for u, grp in pos_va.groupby("user_id")
    }
    users = sorted(truths_all.keys())
    if max_users and max_users > 0:
        users = users[:max_users]
    truths = {u: truths_all[u] for u in users}

    return {
        "users": users,
        "truths": truths,
        "seen": seen,
    }


@torch.no_grad()
def validate_epoch(
    model,
    device,
    dl_val: DataLoader,
    *,
    n_items: int,
    epoch: int,
    val_ctx: dict,
):
    model.eval()

    # ------- RMSE on val loader -------
    se = 0.0; n = 0
    for u, i, r in dl_val:
        u, i, r = u.to(device), i.to(device), r.to(device)
        p = model(u, i)
        se += torch.sum((p - r) ** 2).item()
        n  += r.numel()
    val_rmse = float(np.sqrt(se / max(n, 1)))

    # ------- Ranking on a subset of users -------
    users = val_ctx["users"]
    truths = val_ctx["truths"]
    seen = val_ctx["seen"]

    hr_list, ndcg_list = [], []
    B = 8192  # score items in chunks to reduce memory
    items = torch.arange(n_items, device=device)
    for u in users:
        truth = truths.get(u)
        if not truth:
            continue

        scores = []
        uu = torch.full_like(items, u, dtype=torch.long)
        for t in range(0, n_items, B):
            it = items[t:t+B]; ut = uu[t:t+B]
            scores.append(model(ut, it))
        s = torch.cat(scores)  # (n_items,)
        s = s.cpu().numpy()

        # exclude seen (train)
        if u in seen and seen[u]:
            s[list(seen[u])] = -np.inf

        order = np.argsort(-s)  # best first
        ranked = order.tolist()

        hr_list.append(metrics.hit_rate_at_k(truth, ranked, k=10))
        ndcg_list.append(metrics.ndcg_at_k(truth, ranked, k=10))

    val_hr10    = float(np.mean(hr_list)) if hr_list else 0.0
    val_ndcg10  = float(np.mean(ndcg_list)) if ndcg_list else 0.0

    return {
        "rmse": val_rmse,
        "hr@10": val_hr10,
        "ndcg@10": val_ndcg10,
    }



# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train MF/NeuMF on explicit ratings")
    ap.add_argument("--config", default="configs/config.yaml", help="YAML config path")
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume from (use last.ckpt)")
    ap.add_argument("--run_dir", default=None, help="Override run directory (otherwise timestamped dir is created)")
    ap.add_argument("--device", default=None, help="Force device: cuda|cpu|mps (default: auto)")
    return ap.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # Repro
    seed_all(cfg.get("seed", 42))

    optim_cfg = cfg.get("optim", {})
    loss_cfg = str(optim_cfg.get("loss", "mse")).lower()
    if loss_cfg == "rank":
        loss_cfg = "approx_ndcg"
    if loss_cfg not in {"mse", "approx_ndcg", "bpr"}:
        raise ValueError(f"Unsupported loss type: {loss_cfg}")
    loss_type = loss_cfg

    metric_default = "ndcg@10" if loss_type in {"approx_ndcg", "bpr"} else "rmse"
    metric_cfg = optim_cfg.get("early_stopping_metric", "auto")
    metric_alias = str(metric_cfg).lower()
    if metric_alias == "auto":
        metric_alias = metric_default
    if metric_alias in {"rmse"}:
        early_stop_metric = "rmse"
    elif metric_alias in {"ndcg@10", "ndcg10", "ndcg"}:
        early_stop_metric = "ndcg@10"
    elif metric_alias in {"hr@10", "hr10", "hr"}:
        early_stop_metric = "hr@10"
    else:
        raise ValueError(f"Unsupported early_stopping_metric: {metric_alias}")
    metric_display = {"rmse": "RMSE", "hr@10": "HR@10", "ndcg@10": "NDCG@10"}[early_stop_metric]
    maximize_metric = early_stop_metric != "rmse"

    # Paths / run dir
    base_runs = cfg["log"]["dir"]
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "neumf")).lower()
    exp_name = cfg.get("exp_name") or model_name
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        os.makedirs(base_runs, exist_ok=True)
        run_dir = make_run_dir(base_runs, exp_name)
    save_cfg_snapshot(cfg, run_dir)
    logger = Logger(run_dir, use_tb=cfg["log"].get("tensorboard", False))
    print(f"[run_dir] {run_dir}")

    # Data
    proc = cfg["recsys"]["processed_dir"]
    train_ds = RatingsDS(os.path.join(proc, "splits/train.csv"))
    val_ds = RatingsDS(os.path.join(proc, "splits/val.csv"))

    n_users = max(train_ds.n_users, val_ds.n_users)
    n_items = max(train_ds.n_items, val_ds.n_items)
    if n_users == 0 or n_items == 0:
        raise RuntimeError("Empty dataset: n_users or n_items is zero.")

    user_pos: list[set[int]] = []
    if loss_type in {"approx_ndcg", "bpr"}:
        user_pos = [set() for _ in range(n_users)]
        for u_id, i_id in zip(train_ds.u.tolist(), train_ds.i.tolist()):
            user_pos[int(u_id)].add(int(i_id))

    val_ctx = prepare_validation_context(proc)

    # Model
    model = build_model(cfg, n_users, n_items)

    # Device (safe)
    device = get_device(args.device)
    model = model.to(device)

    # CUDA dry-run guard (if on cuda, verify kernels actually run)
    use_amp = bool(cfg["optim"].get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device == "cuda":
        try:
            u0 = torch.tensor([0], dtype=torch.long, device=device)
            i0 = torch.tensor([0], dtype=torch.long, device=device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    _ = model(u0, i0)
        except Exception as e:
            print(f"[warn] CUDA failed on dry run ({e}); falling back to CPU.")
            device = "cpu"
            model = model.to(device)
            use_amp = False
            scaler = torch.amp.GradScaler("cuda", enabled=False)

    # Optimizer / Loss
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss() if loss_type == "mse" else None
    approx_ndcg_negatives = max(1, int(optim_cfg.get("approx_ndcg_negatives", 50)))
    approx_ndcg_temperature = float(optim_cfg.get("approx_ndcg_temperature", 1.0))
    bpr_negatives = max(1, int(optim_cfg.get("bpr_negatives", 1)))

    # Dataloaders
    dl_train = DataLoader(
        train_ds,
        batch_size=int(cfg["optim"]["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )
    dl_val = DataLoader(
        val_ds,
        batch_size=4096,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )

    # Resume if requested
    start_epoch = 1
    best_metric_val = math.inf if not maximize_metric else -math.inf
    if args.resume and os.path.exists(args.resume):
        try:
            se, bv, n_users_ck, n_items_ck, _cfg_ck = load_ckpt_for_resume(args.resume, model, opt)
            assert n_users_ck == n_users and n_items_ck == n_items, (
                f"Checkpoint shape mismatch: ckpt users/items {n_users_ck}/{n_items_ck} "
                f"vs data {n_users}/{n_items}"
            )
            start_epoch = se
            ckpt_optim = (_cfg_ck or {}).get("optim", {}) if isinstance(_cfg_ck, dict) else {}
            ckpt_loss = str(ckpt_optim.get("loss", "mse")).lower()
            if ckpt_loss == "rank":
                ckpt_loss = "approx_ndcg"
            ckpt_metric_alias = str(ckpt_optim.get("early_stopping_metric", "auto")).lower()
            if ckpt_metric_alias == "auto":
                ckpt_metric_alias = "ndcg@10" if ckpt_loss == "approx_ndcg" else "rmse"
            elif ckpt_metric_alias in {"hr@10", "hr10", "hr"}:
                ckpt_metric_alias = "hr@10"
            elif ckpt_metric_alias in {"ndcg@10", "ndcg10", "ndcg"}:
                ckpt_metric_alias = "ndcg@10"
            else:
                ckpt_metric_alias = "rmse"

            metric_matches = (ckpt_metric_alias == early_stop_metric)
            if metric_matches:
                best_metric_val = bv
                print(f"[resume] {args.resume} -> epoch {start_epoch} ({metric_display} best={best_metric_val:.6f})")
            else:
                best_metric_val = math.inf if not maximize_metric else -math.inf
                print(
                    f"[resume] {args.resume} -> epoch {start_epoch} "
                    f"(resetting early-stop metric: ckpt tracked {ckpt_metric_alias.upper()}={bv:.4f})"
                )
        except Exception as e:
            print(f"[warn] Failed to resume from {args.resume}: {e}")

    # Training loop
    epochs_no_improve = 0
    patience = int(optim_cfg.get("early_stopping_patience", 3))

    def sample_negatives(users: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        if loss_type not in {"approx_ndcg", "bpr"}:
            raise RuntimeError("sample_negatives called while loss does not require negatives")
        users_cpu = users.detach().cpu().tolist()
        shape = (len(users_cpu), n_samples)
        neg = torch.randint(0, n_items, shape, device=device)
        for idx, u in enumerate(users_cpu):
            if not user_pos[u]:
                continue
            for col in range(n_samples):
                attempts = 0
                while int(neg[idx, col].item()) in user_pos[u]:
                    neg[idx, col] = torch.randint(0, n_items, (), device=device)
                    attempts += 1
                    if attempts > 10:
                        break
        if n_samples == 1:
            return neg.view(-1)
        return neg

    if loss_type == "approx_ndcg":
        train_metric_label = "loss (ApproxNDCG)"
    elif loss_type == "bpr":
        train_metric_label = "loss (BPR)"
    else:
        train_metric_label = "loss (RMSE)"

    for epoch in range(1, cfg["optim"]["epochs"] + 1):
        model.train()
        train_loss = 0.0; steps = 0
        for u, i, r in dl_train:
            u, i, r = u.to(device), i.to(device), r.to(device)
            opt.zero_grad(set_to_none=True)
            if loss_type == "mse":
                pred = model(u, i)
                loss = loss_fn(pred, r)
            elif loss_type == "approx_ndcg":
                neg_items = sample_negatives(u, approx_ndcg_negatives)
                candidates = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                labels = torch.zeros_like(candidates, dtype=torch.float32)
                labels[:, 0] = r
                perm = torch.rand(u.size(0), candidates.size(1), device=device).argsort(dim=1)
                candidates = torch.gather(candidates, 1, perm)
                labels = torch.gather(labels, 1, perm)
                users_expanded = u.unsqueeze(1).expand_as(candidates)
                scores = model(users_expanded.reshape(-1), candidates.reshape(-1)).view_as(candidates)
                loss = approx_ndcg_loss(
                    scores,
                    labels,
                    temperature=approx_ndcg_temperature,
                )
            else:  # bpr
                neg_items = sample_negatives(u, bpr_negatives)
                pos_scores = model(u, i)
                if neg_items.dim() == 1:
                    neg_scores = model(u, neg_items)
                    loss = bpr_loss(pos_scores, neg_scores)
                else:
                    users_expanded = u.unsqueeze(1).expand_as(neg_items)
                    neg_scores = model(users_expanded.reshape(-1), neg_items.reshape(-1)).view_as(neg_items)
                    pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)
                    loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            opt.step()
            train_loss += loss.item(); steps += 1

        train_loss /= max(steps, 1)
        logger.log_scalar(epoch, "train", "loss", train_loss)

        # --- validate ---
        val_metrics = validate_epoch(
            model,
            device,
            dl_val,
            n_items=n_items,
            epoch=epoch,
            val_ctx=val_ctx,
        )
        logger.log_dict(epoch, "val", {
            "rmse": val_metrics["rmse"],
            "hr@10": val_metrics["hr@10"],
            "ndcg@10": val_metrics["ndcg@10"],
        })

        # --- checkpointing ---
        cur_rmse = val_metrics["rmse"]
        cur_hr10 = val_metrics["hr@10"]
        cur_ndcg10 = val_metrics["ndcg@10"]
        cur_metric = {
            "rmse": cur_rmse,
            "hr@10": cur_hr10,
            "ndcg@10": cur_ndcg10,
        }[early_stop_metric]
        if maximize_metric:
            improved = cur_metric > best_metric_val + 1e-6
        else:
            improved = cur_metric < best_metric_val - 1e-6

        if improved:
            best_metric_val = cur_metric
            epochs_no_improve = 0
            save_ckpt(
                path=os.path.join(run_dir, "best.ckpt"),
                model=model,
                opt=opt,
                epoch=epoch,
                best_val=best_metric_val,
                n_users=n_users,
                n_items=n_items,
                cfg=cfg,
            )

        else:
            epochs_no_improve += 1

        # always save last
        save_ckpt(
            path=os.path.join(run_dir, "last.ckpt"),
            model=model,
            opt=opt,
            epoch=epoch,
            best_val=best_metric_val,
            n_users=n_users,
            n_items=n_items,
            cfg=cfg,
        )


        print(
            f"epoch {epoch:02d} | train {train_metric_label} {train_loss:.4f} | "
            f"val RMSE {cur_rmse:.4f} | HR@10 {cur_hr10:.4f} | "
            f"NDCG@10 {cur_ndcg10:.4f} | "
            f"early-stop {metric_display} best {best_metric_val:.4f}"
        )

        # --- early stopping ---
        if patience > 0 and epochs_no_improve >= patience:
            print(f"[early-stop] no {metric_display} improvement for {patience} epochs")
            break

if __name__ == "__main__":
    main()
