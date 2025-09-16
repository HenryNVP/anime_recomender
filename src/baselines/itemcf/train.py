from __future__ import annotations
import argparse, os
from .runner import train_from_splits

def parse_args():
    p = argparse.ArgumentParser(description="Train ItemCF baseline")
    p.add_argument("--data_dir", default="data_clean")
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--out_prefix", default="outputs/itemcf")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--shrink", type=float, default=0.0)
    p.add_argument("--clip_min", type=float, default=1.0)
    p.add_argument("--clip_max", type=float, default=10.0)
    p.add_argument("--eval_on", choices=["none","val","test"], default="val")
    return p.parse_args()

def main():
    a = parse_args()
    metrics = train_from_splits(
        data_dir=a.data_dir,
        splits_dir=a.splits_dir,
        out_prefix=a.out_prefix,
        k=a.k,
        shrink=a.shrink,
        clip_min=a.clip_min,
        clip_max=a.clip_max,
        eval_on=a.eval_on,
    )
    if metrics:
        print("[val]" if a.eval_on!="none" else "[info]", metrics)

if __name__ == "__main__":
    main()
