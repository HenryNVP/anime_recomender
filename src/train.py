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
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.recsys.ratingds import RatingsDS
from src.recsys.models.mf import MF
from src.recsys.models.neumf import NeuMF
from src.utils import seed_all, Logger, get_device
from src.recsys import metrics


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
            dim=int(m.get("mf_dim", 64)),
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

import os, pandas as pd, torch, numpy as np
from torch.utils.data import DataLoader

@torch.no_grad()
def validate_epoch(model, device, cfg, dl_val: DataLoader, *, n_items: int, epoch: int):
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
    proc = cfg["recsys"]["processed_dir"]
    df_tr = pd.read_csv(os.path.join(proc, "splits/train.csv"))
    df_va = pd.read_csv(os.path.join(proc, "splits/val.csv"))

    # choose up to 300 users that have at least one positive in val (>= 8)
    pos_va = df_va[df_va["rating"] >= 8.0]
    users = sorted(pos_va["user_id"].unique().tolist())[:300]

    # build seen masks from train (exclude-seen)
    seen = {}
    for u, i in df_tr[["user_id","anime_id"]].itertuples(index=False):
        seen.setdefault(int(u), set()).add(int(i))

    hr_list, ndcg_list = [], []
    B = 8192  # score items in chunks to reduce memory
    for u in users:
        # positives in val
        truth = pos_va[pos_va["user_id"] == u]["anime_id"].astype(int).tolist()
        if not truth:
            continue

        scores = []
        items = torch.arange(n_items, device=device)
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

    # Paths / run dir
    base_runs = cfg["log"]["dir"]
    exp_name = cfg.get("neumf_name", "neumf")
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

    # Model
    model = build_model(cfg, n_users, n_items)
    # ---- attach item features if configured ----
    feat_path = cfg["recsys"].get("item_features_path")
    if feat_path and os.path.exists(feat_path) and isinstance(model, NeuMF):
        feats_np = np.load(feat_path).astype("float32", copy=False)   # (n_items, F)
        # safety if features are for a superset: slice rows we have
        if feats_np.shape[0] != n_items:
            feats_np = feats_np[:n_items]
        model.attach_item_features(torch.from_numpy(feats_np), freeze=bool(cfg["model"].get("freeze_item_feats", True)))
        print(f"[features] attached item features: shape={tuple(feats_np.shape)}")

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
    loss_fn = nn.MSELoss()

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
    best_val = math.inf
    if args.resume and os.path.exists(args.resume):
        try:
            se, bv, n_users_ck, n_items_ck, _cfg_ck = load_ckpt_for_resume(args.resume, model, opt)
            assert n_users_ck == n_users and n_items_ck == n_items, (
                f"Checkpoint shape mismatch: ckpt users/items {n_users_ck}/{n_items_ck} "
                f"vs data {n_users}/{n_items}"
            )
            start_epoch = se
            best_val = bv
            print(f"[resume] {args.resume} -> epoch {start_epoch} (best_val={best_val:.6f})")
        except Exception as e:
            print(f"[warn] Failed to resume from {args.resume}: {e}")

    # Training loop
    best_rmse = -1.0
    epochs_no_improve = 0
    patience = int(cfg["optim"].get("early_stopping_patience", 3))

    for epoch in range(1, cfg["optim"]["epochs"] + 1):
        model.train()
        train_loss = 0.0; steps = 0
        for u, i, r in dl_train:
            u, i, r = u.to(device), i.to(device), r.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(u, i)
            loss = loss_fn(pred, r)
            loss.backward()
            opt.step()
            train_loss += loss.item(); steps += 1

        train_loss /= max(steps, 1)
        logger.log_scalar(epoch, "train", "loss", train_loss)

        # --- validate ---
        val_metrics = validate_epoch(model, device, cfg, dl_val, n_items=n_items, epoch=epoch)
        logger.log_dict(epoch, "val", {
            "rmse": val_metrics["rmse"],
            "hr@10": val_metrics["hr@10"],
            "ndcg@10": val_metrics["ndcg@10"],
        })

        # --- checkpointing by RMSE ---
        cur = val_metrics["rmse"]
        is_best = cur < best_val - 1e-6
        if is_best:
            best_rmse = cur
            epochs_no_improve = 0
            save_ckpt(
                path=os.path.join(run_dir, "best.ckpt"),
                model=model,
                opt=opt,
                epoch=epoch,
                best_val=best_rmse,
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
            best_val=best_rmse,
            n_users=n_users,
            n_items=n_items,
            cfg=cfg,
        )


        print(f"epoch {epoch:02d} | train loss {train_loss:.4f} | "
            f"val RMSE {val_metrics['rmse']:.4f} | HR@10 {val_metrics['hr@10']:.4f} | "
            f"NDCG@10 {val_metrics['ndcg@10']:.4f} | best RMSE {best_rmse:.4f}")

        # --- early stopping on RMSE ---
        if patience > 0 and epochs_no_improve >= patience:
            print(f"[early-stop] no RMSE improvement for {patience} epochs")
            break

if __name__ == "__main__":
    main()
