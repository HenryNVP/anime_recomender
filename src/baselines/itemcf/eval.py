from __future__ import annotations
import argparse, os, json
from .runner import eval_from_splits

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ItemCF baseline")
    p.add_argument("--model_prefix", default="outputs/itemcf")
    p.add_argument("--splits_dir", default="data_clean/splits")
    p.add_argument("--split", choices=["val","test"], default="test")
    p.add_argument("--mode", choices=["rating","ranking","both"], default="both")
    p.add_argument("--k", default="10,20")
    p.add_argument("--pos_threshold", type=float, default=8.0)
    p.add_argument("--exclude_seen", type=int, default=1)
    p.add_argument("--also_exclude_val", type=int, default=0)
    p.add_argument("--max_users", type=int, default=0)
    p.add_argument("--out_json", default="")
    return p.parse_args()

def main():
    a = parse_args()
    Ks = [int(x) for x in a.k.split(",") if x.strip()]
    res = eval_from_splits(
        model_prefix=a.model_prefix,
        splits_dir=a.splits_dir,
        split=a.split,
        mode=a.mode,
        k_list=Ks,
        pos_threshold=a.pos_threshold,
        exclude_seen=bool(a.exclude_seen),
        also_exclude_val=bool(a.also_exclude_val),
        max_users=a.max_users,
    )
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[saved] {a.out_json}")
    else:
        print(res)

if __name__ == "__main__":
    main()
