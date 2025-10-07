# 1) Preprocess data
python scripts/preprocess_data.py --data_dir data/demo --out_dir data/processed --build_item_features --check

# 2) Baseline Item CF
# Train baseline
python -m src.baselines.itemcf.train --data_dir data/processed --out_prefix runs/itemcf/anime --k 100 --shrink 50

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
python -m src.serve --config configs/config_twotower.yaml --ckpt runs/two_tower/[latest_run]/best.ckpt --user 123 --k 10

# Optional: BPR Loss
# Set optim.loss: bpr in any config (or use configs/config_*_bpr.yaml) to optimise Bayesian Personalized Ranking.

# Batch Variants Helper
# Run all variants for MF, NeuMF, and Two-Tower (writes to runs/variants/...)
./scripts/train_variants.sh

# Or target specific configs / custom output root
./scripts/train_variants.sh custom_runs configs/config_mf.yaml configs/config_twotower.yaml

# Evaluate All Models
# Writes per-model metrics + summary table under runs/evaluation/<timestamp>
./scripts/evaluate_all.sh
# Customise destinations or configs (eval_root, variants_root, configs...)
./scripts/evaluate_all.sh custom_eval_root runs/variants configs/config_mf.yaml
