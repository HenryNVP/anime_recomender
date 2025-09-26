# recsys/utils_metrics.py
from __future__ import annotations
from typing import List, Set, Sequence, Tuple, Optional
import numpy as np

# -------- Helpers --------
def _as_sets(true_items: Sequence[Sequence[int]]) -> List[Set[int]]:
    return [set(x) for x in true_items]

def _clip_k(ranked: Sequence[Sequence[int]], k: int) -> List[List[int]]:
    return [list(r[:k]) for r in ranked]

# -------- Core ranking metrics --------
def precision_recall_at_k(
    true_items: Sequence[Sequence[int]],
    ranked_items: Sequence[Sequence[int]],
    k: int,
) -> Tuple[float, float]:
    """Macro Precision@K and Recall@K."""
    T = _as_sets(true_items)
    R = _clip_k(ranked_items, k)
    precisions, recalls = [], []
    for t, r in zip(T, R):
        if k == 0: 
            continue
        hit = len(t.intersection(r))
        precisions.append(hit / max(len(r), 1))
        recalls.append(hit / max(len(t), 1) if len(t) > 0 else 0.0)
    return float(np.mean(precisions)), float(np.mean(recalls))

def hit_rate_at_k(
    true_items: Sequence[Sequence[int]],
    ranked_items: Sequence[Sequence[int]],
    k: int,
) -> float:
    """Fraction of users with at least one hit in top-K."""
    T = _as_sets(true_items)
    R = _clip_k(ranked_items, k)
    hits = [(len(set(r).intersection(t)) > 0) for t, r in zip(T, R)]
    return float(np.mean(hits))

def average_precision_at_k(
    true_items: Sequence[Sequence[int]],
    ranked_items: Sequence[Sequence[int]],
    k: int,
) -> float:
    """MAP@K (macro)."""
    T = _as_sets(true_items)
    R = _clip_k(ranked_items, k)
    ap_vals = []
    for t, r in zip(T, R):
        if len(t) == 0:
            ap_vals.append(0.0); continue
        hit = 0
        prec_sum = 0.0
        for j, item in enumerate(r, start=1):
            if item in t:
                hit += 1
                prec_sum += hit / j
        ap_vals.append(prec_sum / min(len(t), k))
    return float(np.mean(ap_vals))

def ndcg_at_k(
    true_items: Sequence[Sequence[int]],
    ranked_items: Sequence[Sequence[int]],
    k: int,
) -> float:
    """Binary relevance NDCG@K (macro)."""
    T = _as_sets(true_items)
    R = _clip_k(ranked_items, k)

    def dcg(rel: np.ndarray) -> float:
        if rel.size == 0: return 0.0
        denom = np.log2(np.arange(2, rel.size + 2))
        return float(np.sum(rel / denom))

    ndcgs = []
    for t, r in zip(T, R):
        rel = np.array([1.0 if x in t else 0.0 for x in r], dtype=np.float32)
        idcg = dcg(np.sort(rel)[::-1])
        ndcgs.append(dcg(rel) / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))

