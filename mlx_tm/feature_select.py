"""RPCholesky feature selection — pick k representative columns from a wide
(e.g. ~20k molFTP) feature matrix, so they can be binarized into a Tsetlin Machine.

Randomly-Pivoted Cholesky (Chen, Epperly, Tropp 2022) builds a rank-k Nyström
approximation of a PSD matrix K by sampling pivot columns with probability
proportional to the residual diagonal. Applied to the *feature* Gram K = XᵀX
(F×F), the chosen pivots ARE the selected features. It never forms the full F×F
matrix — it only needs diag(K) (per-feature squared norm) and one column
K[:, s] = Xᵀ X[:, s] per pivot — so it scales to ~20k features.
"""
from __future__ import annotations

import numpy as np


def rpcholesky_select(X: np.ndarray, k: int, seed: int = 0,
                      standardize: bool = True, jitter: float = 1e-10):
    """Select k feature-column indices from X (n_samples, n_features) via RPCholesky.

    standardize: z-score columns first, so selection reflects correlation structure
                 rather than raw scale (high-variance columns won't dominate by units).
    Returns: pivots (k',) int — indices of selected features (k' <= k; stops early
             if residual variance is exhausted).
    """
    Xf = np.asarray(X, dtype=np.float64)
    n, F = Xf.shape
    if standardize:
        mu = Xf.mean(0)
        sd = Xf.std(0)
        sd[sd < 1e-12] = 1.0
        Xf = (Xf - mu) / sd
    k = min(k, F)
    d = np.einsum("ij,ij->j", Xf, Xf)            # diag(XᵀX): per-feature squared norm
    L = np.zeros((F, k), dtype=np.float64)        # Nyström factor (F × k)
    pivots: list[int] = []
    rng = np.random.RandomState(seed)
    chosen = np.zeros(F, dtype=bool)
    for i in range(k):
        dsum = d.sum()
        if dsum <= jitter:
            break
        p = np.clip(d, 0, None)
        p = p / p.sum()
        s = int(rng.choice(F, p=p))
        if chosen[s]:                             # avoid duplicate pivot
            d[s] = 0.0
            continue
        col = Xf.T @ Xf[:, s]                      # K[:, s]  (F,)
        if i > 0:
            col = col - L[:, :i] @ L[s, :i]        # subtract prior factors
        piv = col[s]
        if piv <= jitter:
            d[s] = 0.0
            continue
        L[:, i] = col / np.sqrt(piv)
        d = np.maximum(d - L[:, i] ** 2, 0.0)
        chosen[s] = True
        pivots.append(s)
    return np.array(pivots, dtype=np.int64)
