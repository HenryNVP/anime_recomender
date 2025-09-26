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
    def __init__(self, run_dir: str, use_tb: bool = True):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.tb = SummaryWriter(run_dir) if use_tb else None
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,split,metric,value\n")

    def log_scalar(self, epoch: int, split: str, metric: str, value: float):
        v = float(value)
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{split},{metric},{v}\n")
        if self.tb:
            self.tb.add_scalar(f"{split}/{metric}", v, epoch)
            self.tb.flush()

    def log_dict(self, epoch: int, split: str, metrics: dict[str, float]):
        for k, v in metrics.items():
            self.log_scalar(epoch, split, k, v)

    def close(self):
        if self.tb:
            self.tb.close()


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
