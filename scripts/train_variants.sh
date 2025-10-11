#!/usr/bin/env bash
set -euo pipefail

# Runs the default variants:
#   - Two-Tower (MSE) + BPR fine-tune
#   - NeuMF (MSE)
#   - MF (MSE)
#
# Usage:
#   scripts/train_variants.sh [runs_root] [opt_config_paths...]
# If config paths are provided they override the defaults. The first non-YAML
# argument is treated as the runs root.

RUNS_ROOT="runs/variants"
if (($# > 0)) && [[ $1 != *.yaml ]]; then
  RUNS_ROOT=$1
  shift
fi

if (($# == 0)); then
  CONFIGS=(
    configs/config_twotower.yaml
    configs/config_neumf.yaml
    configs/config_mf.yaml
  )
else
  CONFIGS=("$@")
fi

timestamp=$(date +%Y%m%d_%H%M%S)

tmp_dir=$(mktemp -d)
cleanup() { rm -rf "$tmp_dir"; }
trap cleanup EXIT

make_cfg() {
  local base_cfg=$1
  local out_cfg=$2
  local loss_name=$3
  python3 - "$base_cfg" "$out_cfg" "$loss_name" <<'PY'
import sys, yaml
base, out, loss = sys.argv[1:4]
with open(base, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
optim = cfg.setdefault("optim", {})
loss = loss.lower()
if loss not in {"mse", "bpr"}:
    raise ValueError(f"unsupported loss alias: {loss}")
optim["loss"] = loss
if optim.get("early_stopping_metric", "auto") == "auto":
    optim["early_stopping_metric"] = "auto"
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

for CONFIG in "${CONFIGS[@]}"; do
  if [[ ! -f "$CONFIG" ]]; then
    echo "[warn] Skipping missing config $CONFIG" >&2
    continue
  fi

  cfg_name=$(basename "$CONFIG")
  model_id=${cfg_name%.yaml}
  root_dir="${RUNS_ROOT}/${model_id}/${timestamp}"
  mkdir -p "$root_dir"

  cfg_mse="${tmp_dir}/${model_id}_mse.yaml"
  make_cfg "$CONFIG" "$cfg_mse" "mse"

  echo "[train] ($model_id) MSE run -> $root_dir/mse"
  python3 -m src.train --config "$cfg_mse" --run_dir "$root_dir/mse"

  if [[ "$model_id" == *twotower* ]]; then
    resume_ckpt=""
    if [[ -f "$root_dir/mse/best.ckpt" ]]; then
      resume_ckpt="$root_dir/mse/best.ckpt"
    elif [[ -f "$root_dir/mse/last.ckpt" ]]; then
      resume_ckpt="$root_dir/mse/last.ckpt"
    fi

    bpr_base="$CONFIG"
    candidate_bpr_cfg="${CONFIG%.yaml}_bpr.yaml"
    if [[ -f "$candidate_bpr_cfg" ]]; then
      bpr_base="$candidate_bpr_cfg"
    fi

    cfg_bpr="${tmp_dir}/${model_id}_bpr.yaml"
    make_cfg "$bpr_base" "$cfg_bpr" "bpr"
    finetune_dir="$root_dir/mse_to_bpr"

    if [[ -n "$resume_ckpt" ]]; then
      echo "[train] ($model_id) BPR fine-tune from $resume_ckpt -> $finetune_dir"
      python3 -m src.train --config "$cfg_bpr" --run_dir "$finetune_dir" --resume "$resume_ckpt"
    else
      echo "[warn] ($model_id) no checkpoint found in $root_dir/mse; skipping BPR fine-tune" >&2
    fi
  fi

  echo "[done] ($model_id) Outputs under $root_dir"
done
