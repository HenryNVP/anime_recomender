from __future__ import annotations
import pandas as pd, torch
from torch.utils.data import Dataset

class RatingsDS(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, usecols=["user_id","anime_id","rating"])
        self.u = torch.as_tensor(df["user_id"].to_numpy(), dtype=torch.long)
        self.i = torch.as_tensor(df["anime_id"].to_numpy(), dtype=torch.long)
        self.r = torch.as_tensor(df["rating"].to_numpy(),   dtype=torch.float32)
        self.n_users = int(self.u.max().item())+1 if len(self.u)>0 else 0
        self.n_items = int(self.i.max().item())+1 if len(self.i)>0 else 0
    def __len__(self): return self.u.shape[0]
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]
