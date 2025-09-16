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

def mrr_at_k(
    true_items: Sequence[Sequence[int]],
    ranked_items: Sequence[Sequence[int]],
    k: int,
) -> float:
    """Mean Reciprocal Rank@K."""
    T = _as_sets(true_items)
    R = _clip_k(ranked_items, k)
    rr = []
    for t, r in zip(T, R):
        recip = 0.0
        for j, item in enumerate(r, start=1):
            if item in t:
                recip = 1.0 / j
                break
        rr.append(recip)
    return float(np.mean(rr))

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

# -------- Coverage / bias / novelty / diversity --------
def catalog_coverage(ranked_items: Sequence[Sequence[int]], n_items: int, k: Optional[int] = None) -> float:
    """Fraction of catalog recommended at least once."""
    R = _clip_k(ranked_items, k) if k else ranked_items
    seen = set()
    for r in R: seen.update(r)
    return float(len(seen) / max(n_items, 1))

def user_coverage(ranked_items: Sequence[Sequence[int]], k: int) -> float:
    """Fraction of users who receive at least one recommendation (non-empty top-K)."""
    R = _clip_k(ranked_items, k)
    return float(np.mean([len(r) > 0 for r in R]))

def avg_popularity(
    ranked_items: Sequence[Sequence[int]],
    item_popularity: np.ndarray,
    k: int,
) -> float:
    """Mean popularity of recommended items (lower can mean more novel)."""
    R = _clip_k(ranked_items, k)
    vals = []
    for r in R:
        if not r: continue
        vals.extend(item_popularity[np.array(r, dtype=int)].tolist())
    return float(np.mean(vals)) if vals else 0.0

def novelty_at_k(
    ranked_items: Sequence[Sequence[int]],
    item_popularity: np.ndarray,
    k: int,
    log_base: float = 2.0,
) -> float:
    """
    Novelty@K via -log(popularity fraction). 
    item_popularity should be counts; we'll convert to probabilities.
    """
    R = _clip_k(ranked_items, k)
    pop = item_popularity.astype(np.float64)
    p = pop / max(pop.sum(), 1.0)
    eps = 1e-12
    logs = []
    for r in R:
        if not r: continue
        logs.extend([-np.log(p[i] + eps) / np.log(log_base) for i in r])
    return float(np.mean(logs)) if logs else 0.0

def intra_list_diversity_at_k(
    ranked_items: Sequence[Sequence[int]],
    item_embeddings: np.ndarray,
    k: int,
) -> float:
    """
    Mean pairwise cosine distance within each user's top-K; averaged over users.
    item_embeddings: [n_items, d], assumed L2-normalized.
    """
    R = _clip_k(ranked_items, k)
    dists = []
    for r in R:
        if len(r) < 2: 
            continue
        M = item_embeddings[np.array(r, dtype=int)]  # (K, d)
        S = M @ M.T  # cosine similarity matrix
        # take upper triangle (i<j)
        iu = np.triu_indices(len(r), k=1)
        sims = S[iu]
        d = 1.0 - sims  # cosine distance
        dists.append(float(np.mean(d)))
    return float(np.mean(dists)) if dists else 0.0

