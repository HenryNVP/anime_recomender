"""Shared helpers for the train/evaluate variant pipelines.

Both `scripts.train_variants` and `scripts.evaluate_all` walk the same set of
configs and the same per-model variants, so the config rewriting, checkpoint
lookup and subprocess plumbing live here.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

DEFAULT_CONFIGS = [
    "configs/config_twotower.yaml",
    "configs/config_neumf.yaml",
    "configs/config_mf.yaml",
]

VALID_LOSSES = {"mse", "bpr", "approx_ndcg"}


@dataclass(frozen=True)
class Variant:
    """One training/evaluation run of a model: a run sub-dir and its loss."""

    name: str  # sub-directory name, e.g. "mse" or "mse_to_bpr"
    loss: str
    base_config: Path


# -----------------------------
# CLI
# -----------------------------
def parse_cli(argv: Sequence[str], roots: Sequence[str]) -> Tuple[List[str], List[Path]]:
    """Split `[root...] [config.yaml...]` argv, falling back to the defaults.

    Leading non-YAML arguments override `roots` positionally; the rest (or
    DEFAULT_CONFIGS when empty) are the config paths.
    """
    rest = list(argv)
    out_roots = list(roots)
    for i in range(len(out_roots)):
        if not rest or rest[0].endswith(".yaml"):
            break
        out_roots[i] = rest.pop(0)
    return out_roots, [Path(p) for p in (rest or DEFAULT_CONFIGS)]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def info(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr, flush=True)


# -----------------------------
# Variants & configs
# -----------------------------
def variants_for(config: Path) -> List[Variant]:
    """MSE for every model; Two-Tower additionally gets a BPR fine-tune."""
    variants = [Variant("mse", "mse", config)]
    if "twotower" in config.stem:
        bpr_base = config.with_name(f"{config.stem}_bpr.yaml")
        variants.append(
            Variant("mse_to_bpr", "bpr", bpr_base if bpr_base.is_file() else config)
        )
    return variants


def write_config(variant: Variant, dest: Path) -> Path:
    """Copy the variant's base config to `dest` with `optim.loss` overridden."""
    if variant.loss not in VALID_LOSSES:
        raise ValueError(f"unsupported loss alias: {variant.loss}")
    cfg = yaml.safe_load(variant.base_config.read_text(encoding="utf-8")) or {}
    optim = cfg.setdefault("optim", {})
    optim["loss"] = variant.loss
    optim.setdefault("early_stopping_metric", "auto")
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dest


# -----------------------------
# Checkpoints
# -----------------------------
def find_checkpoint(run_dir: Path) -> Optional[Path]:
    """Prefer best.ckpt, fall back to last.ckpt."""
    for name in ("best.ckpt", "last.ckpt"):
        ckpt = run_dir / name
        if ckpt.is_file():
            return ckpt
    return None


def latest_run_dir(base_dir: Path) -> Optional[Path]:
    """Most recent timestamped run directory under `base_dir`."""
    if not base_dir.is_dir():
        return None
    runs = sorted(p for p in base_dir.iterdir() if p.is_dir())
    return runs[-1] if runs else None


# -----------------------------
# Subprocess
# -----------------------------
def run_module(module: str, args: Sequence[str], log_path: Optional[Path] = None) -> bool:
    """Run `python -m <module> <args>`, mirroring output to `log_path` if given."""
    cmd = [sys.executable, "-m", module, *(str(a) for a in args)]
    if log_path is None:
        return subprocess.run(cmd).returncode == 0

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc, log_path.open("w", encoding="utf-8") as log:
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        sys.stdout.flush()
    return proc.returncode == 0
