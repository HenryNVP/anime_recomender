# recsys/models/item_cf.py
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp


class ItemCF:
    """
    Item-based Collaborative Filtering with ADJUSTED COSINE similarity.

    Key choices:
      - User-mean centering (removes user bias).
      - Cosine similarity on centered matrix.
      - Optional shrinkage by co-rating counts.
      - Per-row Top-K pruning of similarity matrix.
      - CSR everywhere; Pylance-safe types.

    Prediction:
        r_hat(u,i) = mean_u + sum_j s(i,j) * (r(u,j) - mean_u) / sum_j |s(i,j)|

    Parameters
    ----------
    k : int
        Keep top-k neighbors per item (0 or None disables pruning).
    shrink : float
        (Optional) shrinkage strength; if > 0, reweights similarities by
        n_ij/(n_ij + shrink) using co-rating counts.
    clip : tuple[float, float] | None
        If provided, clip predictions to [min, max].
    dtype : numpy dtype
        Internal float dtype (default float32).
    """

    def __init__(
        self,
        k: int = 100,
        shrink: float = 0.0,
        clip: Optional[tuple[float, float]] = (1.0, 10.0),
        dtype: np.dtype = np.float32,
    ):
        self.k: int = int(k)
        self.shrink: float = float(shrink)
        self.clip = clip
        self.dtype = dtype

        # Learned state
        self.n_users: int = 0
        self.n_items: int = 0
        self.R: Optional[sp.csr_matrix] = None          # user x item (centered)
        self.S: Optional[sp.csr_matrix] = None          # item x item similarity (Top-K pruned)
        self.user_means: Optional[np.ndarray] = None     # shape [n_users]
        self.global_mean: float = 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit_from_df(self, df, n_users: int, n_items: int) -> None:
        """
        Build ratings CSR from a DataFrame with columns: user_id, anime_id, rating.
        IDs must already be 0..n-1 (your preprocess ensures this).
        """
        rows = df["user_id"].to_numpy(dtype=np.int64)
        cols = df["anime_id"].to_numpy(dtype=np.int64)
        vals = df["rating"].to_numpy(dtype=self.dtype)
        R_raw = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=self.dtype)
        self.fit(R_raw)

    def fit(self, R_raw: sp.csr_matrix) -> None:
        """
        Train the model from a CSR ratings matrix (user x item).
        Steps:
          1) user-mean centering
          2) column-normalize (cosine prep)
          3) S = (R_n)^T (R_n)
          4) (optional) shrinkage by co-rating counts
          5) per-row Top-K pruning
        """
        R_raw = sp.csr_matrix(R_raw, dtype=self.dtype)
        self.n_users, self.n_items = R_raw.shape

        # ---- user means over non-zeros ----
        row_sums = np.asarray(R_raw.sum(axis=1)).ravel().astype(self.dtype)
        row_nnz = np.diff(R_raw.indptr).astype(self.dtype)
        user_means = np.divide(
            row_sums, row_nnz, out=np.zeros_like(row_sums, dtype=self.dtype), where=row_nnz > 0
        )
        self.user_means = user_means
        finite = user_means[np.isfinite(user_means)]
        self.global_mean = float(finite.mean()) if finite.size > 0 else 0.0

        # ---- center by user means (sparse-safe via COO writeback) ----
        Rcoo = R_raw.tocoo()
        centered = (Rcoo.data - user_means[Rcoo.row]).astype(self.dtype)
        Rc = sp.csr_matrix((centered, (Rcoo.row, Rcoo.col)), shape=R_raw.shape, dtype=self.dtype)
        self.R = Rc  # store centered ratings

        # ---- column normalization for cosine ----
        col_sq = np.asarray(Rc.multiply(Rc).sum(axis=0)).ravel().astype(np.float64)
        inv_norm = (1.0 / (np.sqrt(col_sq) + 1e-12)).astype(self.dtype)
        Dinv = sp.diags(inv_norm, offsets=0, shape=(self.n_items, self.n_items), dtype=self.dtype)
        Rn = (Rc @ Dinv).tocsr()

        # ---- similarity S = Rn^T @ Rn ----
        S = (Rn.T @ Rn).tocsr()
        S.setdiag(0.0)
        S.eliminate_zeros()

        # ---- optional shrinkage by co-rating counts ----
        if self.shrink > 0.0:
            Rbin = Rc.copy().tocsr()
            Rbin.data[:] = 1.0
            C = (Rbin.T @ Rbin).tocsr()
            C.setdiag(0.0)
            C.eliminate_zeros()
            # weight = n_ij / (n_ij + shrink)
            Cw = C.copy()
            Cw.data = (Cw.data / (Cw.data + self.shrink)).astype(self.dtype, copy=False)
            S = S.multiply(Cw).tocsr()

        # ---- Top-K pruning per row ----
        if self.k and 0 < self.k < self.n_items:
            S = self._topk_rows_csr(S, self.k)

        self.S = S.tocsr()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _user_base(self, u: int) -> float:
        if self.user_means is not None and 0 <= u < self.user_means.shape[0]:
            x = float(self.user_means[u])
            if np.isfinite(x):
                return x
        return self.global_mean

    def _neighbors(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if self.S is None:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=self.dtype)
        row = self.S.getrow(i)
        return row.indices, row.data

    def predict(self, u: int, i: int) -> float:
        if self.R is None or self.S is None:
            raise RuntimeError("Model not trained. Call fit()/fit_from_df first.")

        nbr_idx, nbr_sim = self._neighbors(i)
        if nbr_idx.size == 0:
            return self._clip(self._user_base(u))

        urow = self.R.getrow(u)  # centered ratings for user u
        if urow.nnz == 0:
            return self._clip(self._user_base(u))

        # Map user-rated items to centered scores
        u_map = dict(zip(urow.indices.tolist(), urow.data.tolist()))
        r_c = np.array([u_map.get(int(j), 0.0) for j in nbr_idx], dtype=self.dtype)

        num = float((nbr_sim * r_c).sum())
        den = float(np.abs(nbr_sim).sum()) + 1e-12
        pred = self._user_base(u) + (num / den)
        return self._clip(pred)

    def predict_batch(self, users, items) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        out = np.empty(users.shape[0], dtype=self.dtype)
        for t in range(users.shape[0]):
            out[t] = self.predict(int(users[t]), int(items[t]))
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _clip(self, x: float) -> float:
        if self.clip is None:
            return float(x)
        lo, hi = self.clip
        return float(np.clip(x, lo, hi))

    @staticmethod
    def _topk_rows_csr(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
        M = M.tocsr(copy=True)
        indptr, indices, data = M.indptr, M.indices, M.data

        new_indptr = [0]
        new_indices: list[int] = []
        new_data: list[float] = []

        for r in range(M.shape[0]):
            s, e = indptr[r], indptr[r + 1]
            if s == e:
                new_indptr.append(len(new_indices))
                continue
            cols = indices[s:e]
            vals = data[s:e]
            if vals.size > k:
                topk_idx = np.argpartition(np.abs(vals), -k)[-k:]
                order = np.argsort(-np.abs(vals[topk_idx]))
                sel = topk_idx[order]
                cols = cols[sel]
                vals = vals[sel]
            new_indices.extend(cols.tolist())
            new_data.extend(vals.tolist())
            new_indptr.append(len(new_indices))

        out = sp.csr_matrix(
            (np.asarray(new_data, dtype=data.dtype),
             np.asarray(new_indices, dtype=indices.dtype),
             np.asarray(new_indptr, dtype=indptr.dtype)),
            shape=M.shape,
        )
        out.eliminate_zeros()
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path_prefix: str) -> None:
        if self.R is None or self.S is None or self.user_means is None:
            raise RuntimeError("Nothing to save: fit the model first.")
        sp.save_npz(path_prefix + ".R_centered.npz", self.R.tocsr())
        sp.save_npz(path_prefix + ".S_topk.npz", self.S.tocsr())
        np.save(path_prefix + ".user_means.npy", self.user_means)
        meta = np.array(
            [self.n_users, self.n_items, self.k, self.shrink, self.global_mean], dtype=np.float64
        )
        np.save(path_prefix + ".meta.npy", meta)
        if self.clip is not None:
            np.save(path_prefix + ".clip.npy", np.asarray(self.clip, dtype=np.float32))

    def load(self, path_prefix: str) -> "ItemCF":
        self.R = sp.load_npz(path_prefix + ".R_centered.npz").tocsr()
        self.S = sp.load_npz(path_prefix + ".S_topk.npz").tocsr()
        self.user_means = np.load(path_prefix + ".user_means.npy")
        meta = np.load(path_prefix + ".meta.npy").astype(np.float64)
        self.n_users = int(meta[0])
        self.n_items = int(meta[1])
        self.k = int(meta[2])
        self.shrink = float(meta[3])
        self.global_mean = float(meta[4])
        clip_path = path_prefix + ".clip.npy"
        try:
            clip = np.load(clip_path)
            self.clip = (float(clip[0]), float(clip[1]))
        except FileNotFoundError:
            pass
        return self
