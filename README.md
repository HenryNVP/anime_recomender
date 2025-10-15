# Anime Recommender
Item-based Collaborative Filtering | Matrix Factorization | Neural Matrix Factorization (NeuMF) | Two-Tower

✨ Highlights

- Models: Matrix Factorization (MF), NeuMF, Two-Tower, and ItemCF baseline.

- Objectives: MSE (explicit ratings), ApproxNDCG (implicit ranking), and MSE→ApproxNDCG fine-tune.

- Reproducible runs: auto-timestamped run dirs, TensorBoard logs, best/last checkpoints.

- Batch evaluation: generate a single summary table (RMSE/MAE + HR@K/NDCG@K) across all variants.

- Feature support: plug item features (item_feats.npy) directly into NeuMF’s MLP tower.


# 1) Preprocess data
python scripts/preprocess_data.py --data_dir data/demo --out_dir data/processed --build_item_features --check

# 2) Baseline Item CF
# Train baseline
python -m src.baselines.itemcf.train --data_dir data/processed --out_prefix runs/itemcf/anime --k 50 --shrink 25

# Evaluate baseline
python -m src.baselines.itemcf.eval \
  --model_prefix runs/itemcf/anime \
  --splits_dir data/processed/splits \
  --split test --k 10,20

# 4) MF
# Train MF
python -m src.train --config configs/config_mf.yaml

# Evaluate MF
python -m src.eval --ckpt runs/neumf/[latest_run]/best.ckpt --config configs/config_mf.yaml

# 3) NeuMF
# Train NeuMF
python -m src.train --config configs/config_neumf.yaml

# Evaluate NeuMF
python -m src.eval --ckpt runs/neumf/[latest_run]/best.ckpt --config configs/config_neumf.yaml

# 5) Two-Tower
# Train Two-Tower
python -m src.train --config configs/config_twotower.yaml

# Evaluate Two-Tower
python -m src.eval --ckpt runs/two_tower/[latest_run]/best.ckpt --config configs/config_twotower.yaml

# 6) Serve Recommendations
## CLI (one-off; original user id)
python -m src.serve --config configs/config_twotower.yaml --ckpt runs/two_tower/[latest_run]/best.ckpt --user 654321 --k 10

## FastAPI service
python -m src.serve --http \
  --model twotower=configs/config_twotower.yaml:runs/two_tower/[latest_run]/best.ckpt \
  --port 8080

# Request example (JSON)
curl -X POST http://localhost:8080/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id":1, "k":10, "include_history":true}'

# List loaded models
curl http://localhost:8080/v1/models

# Optional: ApproxNDCG Loss
# Set optim.loss: approx_ndcg in any config (or use configs/config_*_rank.yaml) to optimise the differentiable NDCG surrogate.

# Batch Variants Helper
# Run all variants for MF, NeuMF, and Two-Tower (writes to runs/variants/...)
./scripts/train_variants.sh

# Or target specific configs / custom output root
./scripts/train_variants.sh custom_runs configs/config_mf.yaml configs/config_twotower.yaml

# Evaluate All Models
# Writes per-model metrics + summary table under runs/evaluation/<timestamp>
./scripts/evaluate_all.sh
# Includes ItemCF baseline automatically (set ITEMCF_ENABLED=0 to skip).
# Customise destinations or configs (eval_root, variants_root, configs...)
./scripts/evaluate_all.sh custom_eval_root runs/variants configs/config_mf.yaml
