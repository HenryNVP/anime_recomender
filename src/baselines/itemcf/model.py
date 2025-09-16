from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp

class ItemCF:
    """
    Item-based CF with ADJUSTED COSINE (user-mean centering) + optional shrink + Top-K.
    Prediction:
      r̂(u,i) = mean_u + sum_j s(i,j) * (r(u,j) - mean_u) / sum_j |s(i,j)|
    """
    def __init__(
        self,
        k: int = 100,
        shrink: float = 0.0,
        clip: Optional[Tuple[float, float]] = (1.0, 10.0),
        dtype: np.dtype = np.float32,
    ):
        self.k = int(k)
        self.shrink = float(shrink)
        self.clip = clip
        self.dtype = dtype
        # learned state
        self.n_users = 0
        self.n_items = 0
        self.R: Optional[sp.csr_matrix] = None         # centered ratings
        self.S: Optional[sp.csr_matrix] = None         # item-item similarity
        self.user_means: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

    # -------- fit --------
    def fit_from_df(self, df, n_users: int, n_items: int) -> None:
        rows = df["user_id"].to_numpy(np.int64)
        cols = df["anime_id"].to_numpy(np.int64)
        vals = df["rating"].to_numpy(self.dtype)
        R = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=self.dtype)
        self.fit(R)

    def fit(self, R_raw: sp.csr_matrix) -> None:
        R_raw = sp.csr_matrix(R_raw, dtype=self.dtype)
        self.n_users, self.n_items = R_raw.shape

        # user means (over nonzeros)
        row_sums = np.asarray(R_raw.sum(axis=1)).ravel().astype(self.dtype)
        row_nnz = np.diff(R_raw.indptr).astype(self.dtype)
        means = np.divide(row_sums, row_nnz, out=np.zeros_like(row_sums), where=row_nnz > 0)
        self.user_means = means
        finite = means[np.isfinite(means)]
        self.global_mean = float(finite.mean()) if finite.size else 0.0

        # center by user means
        coo = R_raw.tocoo()
        data = (coo.data - means[coo.row]).astype(self.dtype)
        Rc = sp.csr_matrix((data, (coo.row, coo.col)), shape=R_raw.shape, dtype=self.dtype)
        self.R = Rc

        # cosine prep: column-normalize
        col_sq = np.asarray(Rc.multiply(Rc).sum(axis=0)).ravel().astype(np.float64)
        inv = (1.0 / (np.sqrt(col_sq) + 1e-12)).astype(self.dtype)
        Rn = (Rc @ sp.diags(inv, 0, shape=(self.n_items, self.n_items), dtype=self.dtype)).tocsr()

        # S = Rn^T Rn
        S = (Rn.T @ Rn).tocsr()
        S.setdiag(0.0); S.eliminate_zeros()

        # shrinkage by co-rating counts
        if self.shrink > 0.0:
            Rb = Rc.copy().tocsr(); Rb.data[:] = 1.0
            C = (Rb.T @ Rb).tocsr(); C.setdiag(0.0); C.eliminate_zeros()
            W = C.copy()
            W.data = (W.data / (W.data + self.shrink)).astype(self.dtype, copy=False)
            S = S.multiply(W).tocsr()

        # Top-K per row
        if self.k and 0 < self.k < self.n_items:
            S = self._topk_rows_csr(S, self.k)

        self.S = S.tocsr()

    # -------- predict --------
    def _user_base(self, u: int) -> float:
        if self.user_means is not None and 0 <= u < self.user_means.shape[0]:
            x = float(self.user_means[u])
            if np.isfinite(x): return x
        return self.global_mean

    def predict(self, u: int, i: int) -> float:
        if self.R is None or self.S is None:
            raise RuntimeError("Model not fitted.")
        nbr = self.S.getrow(i)
        if nbr.nnz == 0:
            return self._clip(self._user_base(u))
        urow = self.R.getrow(u)
        if urow.nnz == 0:
            return self._clip(self._user_base(u))
        u_map = dict(zip(urow.indices.tolist(), urow.data.tolist()))
        r_c = np.array([u_map.get(int(j), 0.0) for j in nbr.indices], dtype=self.dtype)
        num = float((nbr.data * r_c).sum())
        den = float(np.abs(nbr.data).sum()) + 1e-12
        return self._clip(self._user_base(u) + num / den)

    def predict_batch(self, users, items) -> np.ndarray:
        users = np.asarray(users, np.int64)
        items = np.asarray(items, np.int64)
        out = np.empty(users.shape[0], dtype=self.dtype)
        for t, (u, i) in enumerate(zip(users, items)):
            out[t] = self.predict(int(u), int(i))
        return out

    # -------- utils --------
    def _clip(self, x: float) -> float:
        if self.clip is None: return float(x)
        lo, hi = self.clip; return float(np.clip(x, lo, hi))

    @staticmethod
    def _topk_rows_csr(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
        M = M.tocsr(copy=True)
        indptr, indices, data = M.indptr, M.indices, M.data
        new_indptr = [0]; new_indices = []; new_data = []
        for r in range(M.shape[0]):
            s, e = indptr[r], indptr[r+1]
            if s == e: new_indptr.append(len(new_indices)); continue
            cols = indices[s:e]; vals = data[s:e]
            if vals.size > k:
                topk = np.argpartition(np.abs(vals), -k)[-k:]
                order = np.argsort(-np.abs(vals[topk])); sel = topk[order]
                cols = cols[sel]; vals = vals[sel]
            new_indices.extend(cols.tolist()); new_data.extend(vals.tolist())
            new_indptr.append(len(new_indices))
        out = sp.csr_matrix(
            (np.asarray(new_data, dtype=data.dtype),
             np.asarray(new_indices, dtype=indices.dtype),
             np.asarray(new_indptr, dtype=indptr.dtype)),
            shape=M.shape,
        )
        out.eliminate_zeros()
        return out

    # -------- io --------
    def save(self, prefix: str) -> None:
        if self.R is None or self.S is None or self.user_means is None:
            raise RuntimeError("Nothing to save.")
        sp.save_npz(prefix + ".R_centered.npz", self.R.tocsr())
        sp.save_npz(prefix + ".S_topk.npz", self.S.tocsr())
        np.save(prefix + ".user_means.npy", self.user_means)
        meta = np.array([self.n_users, self.n_items, self.k, self.shrink, self.global_mean], np.float64)
        np.save(prefix + ".meta.npy", meta)
        if self.clip is not None:
            np.save(prefix + ".clip.npy", np.asarray(self.clip, np.float32))

    def load(self, prefix: str) -> "ItemCF":
        self.R = sp.load_npz(prefix + ".R_centered.npz").tocsr()
        self.S = sp.load_npz(prefix + ".S_topk.npz").tocsr()
        self.user_means = np.load(prefix + ".user_means.npy")
        meta = np.load(prefix + ".meta.npy").astype(np.float64)
        self.n_users, self.n_items = int(meta[0]), int(meta[1])
        self.k, self.shrink, self.global_mean = int(meta[2]), float(meta[3]), float(meta[4])
        clip_path = prefix + ".clip.npy"
        try:
            clip = np.load(clip_path); self.clip = (float(clip[0]), float(clip[1]))
        except FileNotFoundError:
            pass
        return self
