from __future__ import annotations
from fastapi import FastAPI
import torch, os
from src.recsys.models.mf import MF
from src.recsys.models.neumf import NeuMF

app = FastAPI(title="Anime RecSys")

# Load on startup (adjust path)
CKPT_PATH = os.environ.get("CKPT_PATH", "runs/anime_exp1/best.ckpt")
_model = None; _device = "cuda" if torch.cuda.is_available() else "cpu"
@app.on_event("startup")
def load_model():
    global _model
    if not CKPT_PATH or not os.path.exists(CKPT_PATH): return
    ck = torch.load(CKPT_PATH, map_location="cpu")
    n_users, n_items = ck["n_users"], ck["n_items"]
    name = ck["cfg"]["model"]["name"].lower()
    _model = MF(n_users,n_items,dim=ck["cfg"]["model"].get("mf_dim",64)) if name=="mf" \
        else NeuMF(n_users,n_items)
    _model.load_state_dict(ck["state_dict"]); _model.to(_device).eval()

@app.get("/score")
def score(user_id: int, item_id: int):
    if _model is None: return {"error":"model not loaded"}
    with torch.no_grad():
        u = torch.tensor([user_id], dtype=torch.long, device=_device)
        i = torch.tensor([item_id], dtype=torch.long, device=_device)
        s = float(_model(u,i).item())
    return {"user_id":user_id,"item_id":item_id,"score":s}
