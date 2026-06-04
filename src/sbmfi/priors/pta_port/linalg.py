"""Rank-deficient linear algebra utilities.

Replaces component_contribution.linalg.LINALG which requires C++ binaries.
"""
from __future__ import annotations

import numpy as np


def qr_rank_deficient(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Thin factorisation of A retaining only rank-k components.

    Returns B of shape (k, n) where k = effective rank of A, satisfying
    ``B.T @ B ≈ A.T @ A`` (drops near-zero singular values).

    Convention matches component_contribution LINALG.qr_rank_deficient:
    used as ``result = qr_rank_deficient(X.T).T`` to get a (n, k) covariance
    square root from a (n, q) input square root X.

    Parameters
    ----------
    A   : (m, n) input matrix
    tol : singular values below ``tol * s_max`` are dropped

    Returns
    -------
    B : (k, n) with k = rank(A)
    """
    if A.size == 0:
        return A.copy()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    threshold = tol * s[0] if s[0] > 0 else tol
    rank = int(np.sum(s > threshold))
    if rank == 0:
        return np.zeros((0, A.shape[1]))
    return np.diag(s[:rank]) @ Vt[:rank, :]
