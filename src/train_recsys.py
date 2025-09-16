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

from src.recsys.data_recsys import RatingsDS
from src.recsys.models.mf import MF
from src.recsys.models.neumf import NeuMF
from src.recsys.models.item_cf import ItemCF
from src.utils import seed_all, Logger, get_device


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
    exp_name = cfg.get("exp_name", "exp")
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
    patience = int(cfg["optim"].get("early_stopping_patience", 3))
    bad = 0
    epochs = int(cfg["optim"]["epochs"])

    for epoch in range(start_epoch, epochs + 1):
        # --- Train ---
        model.train()
        tr_loss = 0.0
        for u, i, r in tqdm(dl_train, desc=f"epoch {epoch}"):
            u = u.to(device)
            i = i.to(device)
            r = r.to(device)

            opt.zero_grad(set_to_none=True)
            if device == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    pred = model(u, i)
                    loss = loss_fn(pred, r)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(u, i)
                loss = loss_fn(pred, r)
                loss.backward()
                opt.step()

            tr_loss += float(loss.item()) * u.shape[0]

        tr_loss /= max(1, len(train_ds))

        # --- Val ---
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for u, i, r in dl_val:
                u = u.to(device)
                i = i.to(device)
                r = r.to(device)
                pred = model(u, i)
                va_loss += float(loss_fn(pred, r).item()) * u.shape[0]
        va_loss /= max(1, len(val_ds))

        # Log
        logger.log(epoch, "train", tr_loss, tr_loss)
        logger.log(epoch, "val", va_loss, va_loss)
        print(f"[epoch {epoch}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        # Save last
        save_ckpt(os.path.join(run_dir, "last.ckpt"), model, opt, epoch, best_val, n_users, n_items, cfg)

        # Early stopping / save best
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            bad = 0
            save_ckpt(os.path.join(run_dir, "best.ckpt"), model, opt, epoch, best_val, n_users, n_items, cfg)
        else:
            bad += 1
            if bad >= patience:
                print(f"[early-stop] no improvement for {patience} epochs. Stopping at epoch {epoch}.")
                break

    logger.close()
    print(f"[done] best_val={best_val:.6f} | artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
