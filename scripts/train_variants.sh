#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="runs/variants"
if (($# > 0)) && [[ $1 != *.yaml ]]; then
  RUNS_ROOT=$1
  shift
fi

if (($# == 0)); then
  CONFIGS=(
    configs/config_mf.yaml
    configs/config_neumf.yaml
    configs/config_twotower.yaml
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
optim["loss"] = loss
if optim.get("early_stopping_metric", "auto") == "auto":
    if loss == "bpr":
        optim["early_stopping_metric"] = "auto"  # resolves to HR@10 later
    else:
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
  cfg_bpr="${tmp_dir}/${model_id}_bpr.yaml"
  make_cfg "$CONFIG" "$cfg_mse" "mse"
  make_cfg "$CONFIG" "$cfg_bpr" "bpr"

  echo "[train] ($model_id) MSE run -> $root_dir/mse"
  python3 -m src.train --config "$cfg_mse" --run_dir "$root_dir/mse"

  echo "[train] ($model_id) BPR run -> $root_dir/bpr"
  python3 -m src.train --config "$cfg_bpr" --run_dir "$root_dir/bpr"

  pretrain_dir="${tmp_dir}/${model_id}_mse_to_bpr_pretrain"
  finetune_dir="$root_dir/mse_to_bpr"

  echo "[train] ($model_id) Fine-tune stage 1 (MSE) -> $pretrain_dir"
  python3 -m src.train --config "$cfg_mse" --run_dir "$pretrain_dir"

  resume_ckpt="$pretrain_dir/best.ckpt"
  if [[ ! -f "$resume_ckpt" ]]; then
    echo "[error] Expected checkpoint $resume_ckpt not found; skipping fine-tune BPR" >&2
    continue
  fi

  echo "[train] ($model_id) Fine-tune stage 2 (BPR from $resume_ckpt) -> $finetune_dir"
  python3 -m src.train --config "$cfg_bpr" --run_dir "$finetune_dir" --resume "$resume_ckpt"

  echo "[done] ($model_id) Outputs under $root_dir"
done
