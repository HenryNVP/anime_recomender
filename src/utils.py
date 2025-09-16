# src/utils.py
from __future__ import annotations

"""
Lightweight utilities used across training/eval scripts.
- seed_all(seed): set all RNGs for reproducibility
- get_device(): pick 'cuda' | 'mps' | 'cpu'
- Logger: CSV + (optional) TensorBoard logging
- timer(): context manager to time code blocks
"""

import os
import random
import time
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch

# TensorBoard is optional; guard the import.
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc,assignment]


# ------------------------
# Repro & device helpers
# ------------------------
def seed_all(seed: int = 42) -> None:
    """Set random seeds for reproducibility (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    # Matmul precision (helps consistency across devices)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")  # type: ignore[attr-defined]


# src/utils.py (replace get_device with this)
def get_device(prefer: Optional[str] = None) -> str:
    """
    Choose compute device. Safely refuse GPUs with too-old compute capability.
    """
    # explicit preference
    if prefer == "cpu":
        return "cpu"
    if prefer in ("cuda", "gpu") and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            # Modern PyTorch wheels ship >= sm_70. If lower, refuse.
            if (major, minor) >= (7, 0):
                return "cuda"
            else:
                print(f"[warn] Detected CUDA compute capability {major}.{minor} "
                      f"which this PyTorch build does not support. Using CPU.")
        except Exception as e:
            print(f"[warn] CUDA not usable: {e}. Using CPU.")
        return "cpu"

    # auto
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) >= (7, 0):
                return "cuda"
        except Exception:
            pass
    # Apple MPS if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"



# ------------------------
# Simple logger
# ------------------------
class Logger:
    """
    Minimal experiment logger.
    - Writes a CSV with columns: epoch,split,loss,metric
    - Optionally writes TensorBoard scalars (loss, metric, and any custom scalars)
    """

    def __init__(self, outdir: str, use_tb: bool = True, filename: str = "log.csv") -> None:
        os.makedirs(outdir, exist_ok=True)
        self.csv_path = os.path.join(outdir, filename)
        # initialize CSV with header
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,split,loss,metric\n")

        self.tb: Optional[Any] = None
        if use_tb and SummaryWriter is not None:
            try:
                self.tb = SummaryWriter(outdir)  # type: ignore[operator]
            except Exception:
                self.tb = None  # fail gracefully if TB backend missing

    def log(self, epoch: int, split: str, loss: float, metric: float) -> None:
        """Log a standard line (epoch, split, loss, metric)."""
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{int(epoch)},{split},{float(loss):.6f},{float(metric):.6f}\n")

        if self.tb is not None:
            # Scalar names mirror CSV for easy comparison
            self.tb.add_scalar(f"{split}/loss", float(loss), epoch)    # type: ignore[union-attr]
            self.tb.add_scalar(f"{split}/metric", float(metric), epoch)  # type: ignore[union-attr]

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log any additional scalar to TensorBoard (no-op if TB disabled)."""
        if self.tb is not None:
            self.tb.add_scalar(tag, float(value), step)  # type: ignore[union-attr]

    def log_dict(self, step: int, **scalars: float) -> None:
        """Log a dict of scalars to TensorBoard under their own tags."""
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, float(v), step)  # type: ignore[union-attr]

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()  # type: ignore[union-attr]


# ------------------------
# Timing helper
# ------------------------
@contextmanager
def timer(name: str = "block"):
    """Context manager to time a code block."""
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[timer] {name}: {dt:.3f}s")


# ------------------------
# Misc small helpers
# ------------------------
def count_trainable_params(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def set_num_threads(n: int) -> None:
    """Limit PyTorch intra-op threads (useful on CPU to avoid oversubscription)."""
    try:
        torch.set_num_threads(int(n))
    except Exception:
        pass
