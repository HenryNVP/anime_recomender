#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="runs/variants"
if (($# > 0)) && [[ $1 != *.yaml ]]; then
  RUNS_ROOT=$1
  shift
fi

ITEMCF_ENABLED=${ITEMCF_ENABLED:-1}
ITEMCF_DATA_DIR=${ITEMCF_DATA_DIR:-data/processed}
ITEMCF_SPLITS_DIR=${ITEMCF_SPLITS_DIR:-data/processed/splits}
ITEMCF_OUT_PREFIX=${ITEMCF_OUT_PREFIX:-runs/itemcf/anime}
ITEMCF_K=${ITEMCF_K:-100}
ITEMCF_SHRINK=${ITEMCF_SHRINK:-50}
ITEMCF_CLIP_MIN=${ITEMCF_CLIP_MIN:-1.0}
ITEMCF_CLIP_MAX=${ITEMCF_CLIP_MAX:-10.0}
ITEMCF_EVAL_ON=${ITEMCF_EVAL_ON:-val}
ITEMCF_TRAIN_ARGS_STR=${ITEMCF_TRAIN_ARGS_STR:-}
ITEMCF_TRAIN_ARGS=()
if [[ -n "$ITEMCF_TRAIN_ARGS_STR" ]]; then
  # shellcheck disable=SC2206
  ITEMCF_TRAIN_ARGS=($ITEMCF_TRAIN_ARGS_STR)
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
loss = loss.lower()
if loss not in {"mse", "approx_ndcg", "rank"}:
    raise ValueError(f"unsupported loss alias: {loss}")
loss_value = "approx_ndcg" if loss == "rank" else loss
optim["loss"] = loss_value
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

  should_finetune=0
  cfg_rank=""
  if [[ "$model_id" == *twotower* ]]; then
    should_finetune=1
    cfg_rank="${tmp_dir}/${model_id}_rank.yaml"
    make_cfg "$CONFIG" "$cfg_rank" "rank"
  fi

  echo "[train] ($model_id) MSE run -> $root_dir/mse"
  python3 -m src.train --config "$cfg_mse" --run_dir "$root_dir/mse"

  if (( should_finetune )); then
    finetune_dir="$root_dir/mse_to_rank"
    resume_ckpt=""
    if [[ -f "$root_dir/mse/best.ckpt" ]]; then
      resume_ckpt="$root_dir/mse/best.ckpt"
    elif [[ -f "$root_dir/mse/last.ckpt" ]]; then
      resume_ckpt="$root_dir/mse/last.ckpt"
    fi

    if [[ -n "$resume_ckpt" ]]; then
      echo "[train] ($model_id) ApproxNDCG fine-tune from $resume_ckpt -> $finetune_dir"
      python3 -m src.train --config "$cfg_rank" --run_dir "$finetune_dir" --resume "$resume_ckpt"
    else
      echo "[warn] ($model_id) no checkpoint found in $root_dir/mse; skipping ApproxNDCG fine-tune" >&2
    fi
  else
    echo "[info] ($model_id) skipping ApproxNDCG fine-tune (Two-Tower only)"
  fi

  echo "[done] ($model_id) Outputs under $root_dir"
done

if (( ITEMCF_ENABLED )); then
  echo "[train] (itemcf) -> $ITEMCF_OUT_PREFIX"
  itemcf_cmd=(
    python3 -m src.baselines.itemcf.train
    --data_dir "$ITEMCF_DATA_DIR"
    --out_prefix "$ITEMCF_OUT_PREFIX"
    --k "$ITEMCF_K"
    --shrink "$ITEMCF_SHRINK"
    --clip_min "$ITEMCF_CLIP_MIN"
    --clip_max "$ITEMCF_CLIP_MAX"
    --eval_on "$ITEMCF_EVAL_ON"
  )
  if [[ -n "$ITEMCF_SPLITS_DIR" ]]; then
    itemcf_cmd+=(--splits_dir "$ITEMCF_SPLITS_DIR")
  fi
  itemcf_cmd+=("${ITEMCF_TRAIN_ARGS[@]}")
  "${itemcf_cmd[@]}"
else
  echo "[skip] itemcf training disabled"
fi
