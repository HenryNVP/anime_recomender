# 0) demo data (tiny)
python scripts/make_demo_data.py   --src_dir data/raw   --out_dir data/demo   --n_users 500   --n_items 100   --min_user_ratings 5   --min_item_ratings 5   --strategy top   --seed 42
# 1) preprocess
python scripts/preprocess_data.py --data_dir data/demo --out_dir data/processed --build_item_features --check

# 2) train (MF or NeuMF per config)
python -m src.train --config configs/config.yaml

# 3) ranking eval
python -m src.eval --ckpt runs/neumf/[latest_run]/best.ckpt --config configs/config.yaml

# Baseline
# Train baseline
python -m src.baselines.itemcf.train --data_dir data/processed --out_prefix runs/itemcf/anime --k 100 --shrink 50

# Evaluate baseline
python -m src.baselines.itemcf.eval \
  --model_prefix runs/itemcf/anime \
  --splits_dir data/processed/splits \
  --split test --mode both --k 10,20
