from __future__ import annotations
from typing import List, Sequence, Set, Tuple
import numpy as np

def _as_sets(true_items: Sequence[Sequence[int]]) -> List[Set[int]]:
    return [set(x) for x in true_items]

def _clip_k(ranked: Sequence[Sequence[int]], k: int) -> List[List[int]]:
    return [list(r[:k]) for r in ranked]

def hitrate_at_k(true_items, ranked_items, k: int) -> float:
    T = _as_sets(true_items); R = _clip_k(ranked_items, k)
    return float(np.mean([len(t.intersection(r))>0 for t,r in zip(T,R)]))

def ndcg_at_k(true_items, ranked_items, k: int) -> float:
    T = _as_sets(true_items); R = _clip_k(ranked_items, k)
    def dcg(rel: np.ndarray) -> float:
        if rel.size==0: return 0.0
        denom = np.log2(np.arange(2, rel.size+2))
        return float((rel/denom).sum())
    vals = []
    for t, r in zip(T,R):
        rel = np.array([1.0 if x in t else 0.0 for x in r], dtype=np.float32)
        idcg = dcg(np.sort(rel)[::-1]); vals.append(dcg(rel)/idcg if idcg>0 else 0.0)
    return float(np.mean(vals))
