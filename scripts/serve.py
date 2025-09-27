from __future__ import annotations
from fastapi import FastAPI
import torch, os
from src.recsys.models.mf import MF
from src.recsys.models.neumf import NeuMF
from src.utils import get_device  # Import your utility function

app = FastAPI(title="Anime RecSys")

# Load on startup (adjust path)
CKPT_PATH = os.environ.get("CKPT_PATH", "runs/neumf/2025-09-26_15-30-42/best.ckpt")
_model = None
_device = get_device()  # Use your utility to pick device safely

@app.on_event("startup")
def load_model():
    global _model
    if not CKPT_PATH or not os.path.exists(CKPT_PATH):
        print("[warn] Checkpoint path does not exist or is not specified.")
        return
    
    ck = torch.load(CKPT_PATH, map_location="cpu")

    # Try to get n_users and n_items from checkpoint or config
    n_users = ck.get("n_users") or ck.get("cfg", {}).get("n_users")
    n_items = ck.get("n_items") or ck.get("cfg", {}).get("n_items")

    if n_users is None or n_items is None:
        raise RuntimeError("Must provide n_users and n_items to load model")

    model_cfg = ck.get("cfg", {}).get("model", {})
    name = model_cfg.get("name", "").lower()
    
    if name == "mf":
        mf_dim = model_cfg.get("mf_dim", 64)
        _model = MF(n_users, n_items, dim=mf_dim)
    elif name == "neumf":
        mf_dim = model_cfg.get("mf_dim", 32)
        mlp_layers = tuple(model_cfg.get("mlp_layers", [128, 64]))
        dropout = model_cfg.get("dropout", 0.1)
        _model = NeuMF(n_users, n_items, mf_dim=mf_dim, mlp_layers=mlp_layers, dropout=dropout)
    else:
        raise RuntimeError(f"Unknown model name '{name}' in checkpoint config")

    _model.load_state_dict(ck["state_dict"])
    _model.to(_device).eval()
    print(f"[info] Loaded model '{name}' with {n_users} users, {n_items} items on {_device}")

@app.get("/score")
def score(user_id: int, item_id: int):
    if _model is None:
        return {"error": "model not loaded"}
    with torch.no_grad():
        u = torch.tensor([user_id], dtype=torch.long, device=_device)
        i = torch.tensor([item_id], dtype=torch.long, device=_device)
        s = float(_model(u, i).item())
    return {"user_id": user_id, "item_id": item_id, "score": s}
