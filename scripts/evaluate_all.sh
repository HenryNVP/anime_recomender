#!/usr/bin/env bash
set -euo pipefail

# Evaluate the default recommender variants:
#   - Two-Tower (MSE checkpoint + BPR fine-tune)
#   - NeuMF (MSE)
#   - MF (MSE)
#
# Usage:
#   scripts/evaluate_all.sh [out_root] [variants_root] [opt_config_paths...]
# Non-YAML arguments override the output directory and the variants directory.

OUT_ROOT="runs/evaluation"
VARIANTS_ROOT="runs/variants"
ITEMCF_ENABLED=${ITEMCF_ENABLED:-1}
ITEMCF_MODEL_PREFIX=${ITEMCF_MODEL_PREFIX:-runs/itemcf/anime}
ITEMCF_SPLITS_DIR=${ITEMCF_SPLITS_DIR:-data/processed/splits}
ITEMCF_SPLIT=${ITEMCF_SPLIT:-test}
ITEMCF_K_LIST=${ITEMCF_K_LIST:-10,20}
ITEMCF_VARIANT=${ITEMCF_VARIANT:-baseline}
ITEMCF_EVAL_ARGS_STR=${ITEMCF_EVAL_ARGS_STR:-}
ITEMCF_EVAL_ARGS=()
if [[ -n "$ITEMCF_EVAL_ARGS_STR" ]]; then
  # shellcheck disable=SC2206
  ITEMCF_EVAL_ARGS=($ITEMCF_EVAL_ARGS_STR)
fi

if (($# > 0)) && [[ $1 != *.yaml ]]; then
  OUT_ROOT=$1
  shift
fi
if (($# > 0)) && [[ $1 != *.yaml ]]; then
  VARIANTS_ROOT=$1
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
dest_dir="${OUT_ROOT}/${timestamp}"
mkdir -p "$dest_dir"

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
optim["loss"] = loss.lower()
if optim.get("early_stopping_metric", "auto") == "auto":
    optim["early_stopping_metric"] = "auto"
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

find_latest_variant_dir() {
  local base_dir=$1
  [[ -d "$base_dir" ]] || return 1
  local latest
  latest=$(ls -1 "$base_dir" | sort | tail -n 1)
  [[ -n "$latest" ]] || return 1
  echo "$base_dir/$latest"
}

find_variant_ckpt() {
  local model_id=$1
  local variant=$2
  local base_dir="${VARIANTS_ROOT}/${model_id}"
  if latest_dir=$(find_latest_variant_dir "$base_dir" 2>/dev/null); then
    local run_dir="${latest_dir}/${variant}"
    if [[ -d "$run_dir" ]]; then
      if [[ -f "$run_dir/best.ckpt" ]]; then
        echo "$run_dir/best.ckpt"
        return 0
      elif [[ -f "$run_dir/last.ckpt" ]]; then
        echo "$run_dir/last.ckpt"
        return 0
      fi
    fi
  fi
  return 1
}

declare -a METRIC_FILES=()

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[warn] skipping missing config $cfg" >&2
    continue
  fi

  cfg_name=$(basename "$cfg")
  model_id=${cfg_name%.yaml}

  variants=("mse")
  variant_losses=("mse")
  variant_cfg_bases=("$cfg")

  if [[ "$model_id" == *twotower* ]]; then
    variants+=("mse_to_bpr")
    variant_losses+=("bpr")
    bpr_base="$cfg"
    candidate_bpr_cfg="${cfg%.yaml}_bpr.yaml"
    if [[ -f "$candidate_bpr_cfg" ]]; then
      bpr_base="$candidate_bpr_cfg"
    fi
    variant_cfg_bases+=("$bpr_base")
  fi

  for idx in "${!variants[@]}"; do
    variant="${variants[$idx]}"
    loss="${variant_losses[$idx]}"
    base_cfg="${variant_cfg_bases[$idx]}"

    ckpt_path=$(find_variant_ckpt "$model_id" "$variant" || true)
    if [[ -z "$ckpt_path" ]]; then
      echo "[warn] no checkpoint found for $model_id ($variant)" >&2
      continue
    fi

    eval_cfg="${tmp_dir}/${model_id}_${variant}.yaml"
    make_cfg "$base_cfg" "$eval_cfg" "$loss"

    metrics_file="${dest_dir}/${model_id}__${variant}.json"
    log_file="${dest_dir}/${model_id}__${variant}.log"

    echo "[eval] $model_id ($variant) -> $metrics_file"
    if python3 -m src.eval --config "$eval_cfg" --ckpt "$ckpt_path" --out "$metrics_file" 2>&1 | tee "$log_file"; then
      METRIC_FILES+=("$metrics_file")
    else
      echo "[error] evaluation failed for $cfg ($variant) (see $log_file)" >&2
    fi
  done
done

if (( ITEMCF_ENABLED )); then
  metrics_file="${dest_dir}/itemcf__${ITEMCF_VARIANT}.json"
  log_file="${dest_dir}/itemcf__${ITEMCF_VARIANT}.log"
  echo "[eval] itemcf (${ITEMCF_VARIANT}) -> $metrics_file"
  if python3 -m src.baselines.itemcf.eval \
      --model_prefix "$ITEMCF_MODEL_PREFIX" \
      --splits_dir "$ITEMCF_SPLITS_DIR" \
      --split "$ITEMCF_SPLIT" \
      --k "$ITEMCF_K_LIST" \
      --out_json "$metrics_file" \
      "${ITEMCF_EVAL_ARGS[@]}" 2>&1 | tee "$log_file"; then
    METRIC_FILES+=("$metrics_file")
  else
    echo "[warn] itemcf evaluation failed (see $log_file)" >&2
    rm -f "$metrics_file"
  fi
fi

if ((${#METRIC_FILES[@]} == 0)); then
  echo "[warn] no evaluation metrics produced" >&2
  exit 1
fi

summary_path="${dest_dir}/summary.tsv"

python3 - "$summary_path" "${METRIC_FILES[@]}" <<'PY'
import json
import os
import sys

summary = sys.argv[1]
metric_files = sys.argv[2:]

records = []
ks = set()
for path in metric_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model_variant = os.path.splitext(os.path.basename(path))[0]
    if "__" in model_variant:
        model, variant = model_variant.split("__", 1)
    else:
        model, variant = model_variant, "-"
    rating = data.get("rating", {})
    ranking = data.get("ranking", {})
    rec = {
        "Model": model,
        "Variant": variant,
        "RMSE": rating.get("RMSE"),
        "MAE": rating.get("MAE"),
    }
    for k_str, metrics in ranking.items():
        try:
            k = int(k_str)
        except ValueError:
            continue
        ks.add(k)
        rec[f"HR@{k}"] = metrics.get("HR")
        rec[f"NDCG@{k}"] = metrics.get("NDCG")
    records.append(rec)

ks = sorted(ks)
headers = ["Model", "Variant", "RMSE", "MAE"]
for k in ks:
    headers.extend([f"HR@{k}", f"NDCG@{k}"])

def fmt(value):
    if value is None:
        return "-"
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)

rows = []
for rec in sorted(records, key=lambda r: (r["Model"], r["Variant"])):
    row = [fmt(rec.get(h)) for h in headers]
    rows.append(row)

widths = [len(h) for h in headers]
for row in rows:
    widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

def format_row(values):
    return " | ".join(cell.ljust(width) for cell, width in zip(values, widths))

table_lines = [format_row(headers)]
table_lines.append("-+-".join("-" * w for w in widths))
for row in rows:
    table_lines.append(format_row(row))

table = "\n".join(table_lines)

with open(summary, "w", encoding="utf-8") as f:
    f.write(table + "\n")

print(table)
print(f"\n[summary] metrics table -> {summary}")
PY

echo "[done] evaluations stored in $dest_dir"
