"""Numerical utilities — pure-Python port of pta.utils (no cobra/enkie deps)."""
from __future__ import annotations

import numpy as np


def covariance_square_root(
    covariance: np.ndarray,
    min_eigenvalue: float = 1e-3,
) -> np.ndarray:
    """Full-rank square root of a covariance matrix via truncated EVD.

    Returns V @ diag(sqrt(w)) of shape (n, k) where k = number of eigenvalues
    >= min_eigenvalue.  Satisfies ``result @ result.T ≈ covariance``.
    """
    w, v = np.linalg.eigh(covariance)
    mask = w >= min_eigenvalue
    if not np.any(mask):
        return np.zeros((covariance.shape[0], 0))
    w, v = w[mask], v[:, mask]
    return v * np.sqrt(w)


def apply_transform(
    value: np.ndarray,
    transform: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Apply affine transform (T, s): return T @ value + s."""
    return transform[0] @ value + transform[1]
