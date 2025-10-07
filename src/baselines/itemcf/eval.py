from __future__ import annotations
import argparse, os, json
from .runner import eval_from_splits


def _normalize_metrics(metrics: dict) -> dict:
    """Convert ItemCF metrics to the aggregate JSON schema (uppercase keys, str Ks)."""
    rating_norm = {}
    rating = metrics.get("rating") or {}
    for key_src, key_dst in (("rmse", "RMSE"), ("mae", "MAE")):
        if key_src in rating and rating[key_src] is not None:
            rating_norm[key_dst] = float(rating[key_src])

    ranking_norm = {}
    for k, values in (metrics.get("ranking") or {}).items():
        key = str(k)
        ranking_norm[key] = {}
        for metric_key in ("HR", "NDCG", "P", "R", "MAP"):
            val = values.get(metric_key)
            if val is not None:
                ranking_norm[key][metric_key] = float(val)
    return {"rating": rating_norm, "ranking": ranking_norm}

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ItemCF baseline")
    p.add_argument("--model_prefix", default="outputs/itemcf")
    p.add_argument("--splits_dir", default="data/processed/splits")
    p.add_argument("--split", choices=["val","test"], default="test")
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
        k_list=Ks,
        pos_threshold=a.pos_threshold,
        exclude_seen=bool(a.exclude_seen),
        also_exclude_val=bool(a.also_exclude_val),
        max_users=a.max_users,
    )

    # Save metrics
    if a.out_json:
        out_dir = os.path.dirname(os.path.abspath(a.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(_normalize_metrics(res), f, indent=2)
        print(f"[saved] {a.out_json}")

    # Print RMSE / MAE
    print(f"[rating] {a.split} | RMSE={res['rating']['rmse']:.4f} MAE={res['rating']['mae']:.4f}")

    # Print Ranking metrics like NeuMF
    for K in Ks:
        r = res["ranking"].get(K)
        if r:
            print(
                f"[ranking] K={K:>3} | HR={r['HR']:.4f} "
                f"NDCG={r['NDCG']:.4f} P={r['P']:.4f} "
                f"R={r['R']:.4f} MAP={r['MAP']:.4f}"
            )


if __name__ == "__main__":
    main()
