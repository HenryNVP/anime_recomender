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

# 3) NeuMF
# Train NeuMF
python -m src.train --config configs/config_neumf.yaml

# Evaluate NeuMF
python -m src.eval --ckpt runs/neumf/[latest_run]/best.ckpt --config configs/config_neumf.yaml

# 4) (Optional) MF
# Train MF
python -m src.train --config configs/config_mf.yaml

# Evaluate MF
python -m src.eval --ckpt runs/neumf/[latest_run]/best.ckpt --config configs/config_mf.yaml
