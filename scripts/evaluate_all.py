#!/usr/bin/env python3
"""Evaluate the default recommender variants and write a summary table:

  - Two-Tower (MSE checkpoint + BPR fine-tune)
  - NeuMF (MSE)
  - MF (MSE)

Usage:
    python -m scripts.evaluate_all [out_root] [variants_root] [config.yaml ...]

Leading non-YAML arguments override the output directory (default
runs/evaluation) and the variants directory (default runs/variants).

The ItemCF baseline is evaluated too; it is configured through the ITEMCF_*
environment variables (set ITEMCF_ENABLED=0 to skip it).
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from scripts import pipeline as P


def itemcf_args() -> Optional[tuple]:
    """ItemCF baseline (variant name, CLI args) from the environment, or None."""
    if os.environ.get("ITEMCF_ENABLED", "1") == "0":
        return None
    variant = os.environ.get("ITEMCF_VARIANT", "baseline")
    args = [
        "--model_prefix", os.environ.get("ITEMCF_MODEL_PREFIX", "runs/itemcf"),
        "--splits_dir", os.environ.get("ITEMCF_SPLITS_DIR", "data/processed/splits"),
        "--split", os.environ.get("ITEMCF_SPLIT", "test"),
        "--k", os.environ.get("ITEMCF_K_LIST", "10,20"),
    ]
    args += shlex.split(os.environ.get("ITEMCF_EVAL_ARGS_STR", ""))
    return variant, args


def build_summary(metric_files: Sequence[Path]) -> str:
    """Render the per-model metrics JSONs as one aligned table."""
    records, ks = [], set()
    for path in metric_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        model, _, variant = path.stem.partition("__")
        rating = data.get("rating", {})
        rec = {
            "Model": model,
            "Variant": variant or "-",
            "RMSE": rating.get("RMSE"),
            "MAE": rating.get("MAE"),
        }
        for k_str, metrics in data.get("ranking", {}).items():
            if not k_str.isdigit():
                continue
            k = int(k_str)
            ks.add(k)
            rec[f"HR@{k}"] = metrics.get("HR")
            rec[f"NDCG@{k}"] = metrics.get("NDCG")
        records.append(rec)

    headers = ["Model", "Variant", "RMSE", "MAE"]
    for k in sorted(ks):
        headers += [f"HR@{k}", f"NDCG@{k}"]

    def fmt(value) -> str:
        if value is None:
            return "-"
        return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)

    rows = [
        [fmt(rec.get(h)) for h in headers]
        for rec in sorted(records, key=lambda r: (r["Model"], r["Variant"]))
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def line(values):
        return " | ".join(cell.ljust(w) for cell, w in zip(values, widths))

    return "\n".join(
        [line(headers), "-+-".join("-" * w for w in widths)] + [line(r) for r in rows]
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    (out_root, variants_root), configs = P.parse_cli(
        argv, ["runs/evaluation", "runs/variants"]
    )

    dest_dir = Path(out_root) / P.timestamp()
    dest_dir.mkdir(parents=True, exist_ok=True)
    metric_files: List[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for config in configs:
            if not config.is_file():
                P.warn(f"skipping missing config {config}")
                continue

            model_id = config.stem
            for variant in P.variants_for(config):
                latest = P.latest_run_dir(Path(variants_root) / model_id)
                ckpt = P.find_checkpoint(latest / variant.name) if latest else None
                if ckpt is None:
                    P.warn(f"no checkpoint found for {model_id} ({variant.name})")
                    continue

                cfg_path = P.write_config(
                    variant, tmp_dir / f"{model_id}_{variant.name}.yaml"
                )
                metrics_file = dest_dir / f"{model_id}__{variant.name}.json"
                log_file = dest_dir / f"{model_id}__{variant.name}.log"

                P.info(f"[eval] {model_id} ({variant.name}) -> {metrics_file}")
                ok = P.run_module(
                    "src.eval",
                    ["--config", cfg_path, "--ckpt", ckpt, "--out", metrics_file],
                    log_path=log_file,
                )
                if ok:
                    metric_files.append(metrics_file)
                else:
                    P.warn(
                        f"[error] evaluation failed for {config} ({variant.name}) "
                        f"(see {log_file})"
                    )

    baseline = itemcf_args()
    if baseline is not None:
        variant, args = baseline
        metrics_file = dest_dir / f"itemcf__{variant}.json"
        log_file = dest_dir / f"itemcf__{variant}.log"
        P.info(f"[eval] itemcf ({variant}) -> {metrics_file}")
        if P.run_module(
            "src.baselines.itemcf.eval",
            args + ["--out_json", metrics_file],
            log_path=log_file,
        ):
            metric_files.append(metrics_file)
        else:
            P.warn(f"itemcf evaluation failed (see {log_file})")
            metrics_file.unlink(missing_ok=True)

    if not metric_files:
        P.warn("no evaluation metrics produced")
        return 1

    summary_path = dest_dir / "summary.tsv"
    table = build_summary(metric_files)
    summary_path.write_text(table + "\n", encoding="utf-8")

    print(table)
    print(f"\n[summary] metrics table -> {summary_path}")
    P.info(f"[done] evaluations stored in {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
