#!/usr/bin/env python3
"""Train the default variants:

  - Two-Tower (MSE) + BPR fine-tune
  - NeuMF (MSE)
  - MF (MSE)

Usage:
    python -m scripts.train_variants [runs_root] [config.yaml ...]

If config paths are provided they override the defaults. The first non-YAML
argument is treated as the runs root (default: runs/variants).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from scripts import pipeline as P


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    (runs_root,), configs = P.parse_cli(argv, ["runs/variants"])
    stamp = P.timestamp()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for config in configs:
            if not config.is_file():
                P.warn(f"Skipping missing config {config}")
                continue

            model_id = config.stem
            root_dir = Path(runs_root) / model_id / stamp
            root_dir.mkdir(parents=True, exist_ok=True)

            for variant in P.variants_for(config):
                run_dir = root_dir / variant.name
                cfg_path = P.write_config(variant, tmp_dir / f"{model_id}_{variant.loss}.yaml")
                args = ["--config", cfg_path, "--run_dir", run_dir]

                if variant.name == "mse":
                    P.info(f"[train] ({model_id}) MSE run -> {run_dir}")
                else:
                    resume = P.find_checkpoint(root_dir / "mse")
                    if resume is None:
                        P.warn(
                            f"({model_id}) no checkpoint found in {root_dir / 'mse'}; "
                            f"skipping {variant.loss.upper()} fine-tune"
                        )
                        continue
                    args += ["--resume", resume]
                    P.info(
                        f"[train] ({model_id}) {variant.loss.upper()} fine-tune "
                        f"from {resume} -> {run_dir}"
                    )

                if not P.run_module("src.train", args):
                    P.warn(f"({model_id}) training failed for variant {variant.name}")
                    return 1

            P.info(f"[done] ({model_id}) Outputs under {root_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
