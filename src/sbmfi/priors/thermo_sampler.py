"""
Pure-Python hit-and-run MCMC sampler for the truncated multivariate-normal
distribution over reaction Gibbs free energies (dG).

Replaces pta's sample_drg() with no Gurobi / Docker dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from scipy.special import ndtr, ndtri

from .thermo_data import ThermoSamplingData


@dataclass
class DRGSamplingResult:
    """Mirrors the pta FreeEnergiesSamplingResult interface used in thermo_prior.py."""
    samples: pd.DataFrame         # (n_total_samples, n_reactions) — dG values
    basis_samples: pd.DataFrame   # (n_total_samples, n_reactions) — same but in MVN basis


# ------------------------------------------------------------------ #
#  Initial point finder (replaces pta's PmoProblemPool / _find_point)
# ------------------------------------------------------------------ #

def find_initial_points(
    data: ThermoSamplingData,
    n: int,
    reaction_ids: list[str] | None = None,
) -> np.ndarray:
    """Find n feasible starting points in dG space using scipy linprog.

    Parameters
    ----------
    data : ThermoSamplingData
    n    : number of initial points to find
    reaction_ids : subset of data.T_reaction_ids to return; defaults to all

    Returns
    -------
    points : (n_reactions, n) array — each column is a feasible dG vector
    """
    T_ids = list(data.T_reaction_ids)
    F_ids = list(data.F_reaction_ids)
    dim = len(T_ids)

    # Build sign constraints from flux bounds
    T_idx_in_F = [F_ids.index(rid) for rid in T_ids]
    only_fwd = [i for i in range(dim) if data.F_lb[T_idx_in_F[i]] >= 0]
    only_rev = [i for i in range(dim) if data.F_ub[T_idx_in_F[i]] <= 0]
    reversible = [i for i in range(dim) if i not in only_fwd and i not in only_rev]

    # Inequality constraints: dG[fwd] <= -eps, dG[rev] >= +eps
    eps = 1e-6
    rows, rhs = [], []
    for i in only_fwd:
        r = np.zeros(dim); r[i] = 1.0
        rows.append(r); rhs.append(-eps)
    for i in only_rev:
        r = np.zeros(dim); r[i] = -1.0
        rows.append(r); rhs.append(-eps)

    A_ub = np.array(rows) if rows else np.zeros((0, dim))
    b_ub = np.array(rhs) if rhs else np.zeros(0)

    # Bounds: use mean ± 3 std as box constraints to help the solver
    drg_mean, drg_cov = data.drg_mvn_params()
    drg_std = np.sqrt(np.diag(drg_cov))
    bounds = list(zip(drg_mean - 5 * drg_std, drg_mean + 5 * drg_std))

    points = []
    rng = np.random.default_rng()

    # Candidate objectives: maximise/minimise each dimension in turn
    objectives = []
    for i in reversible:
        objectives.append((i, -1.0))
        objectives.append((i, 1.0))
    for i in only_fwd:
        objectives.append((i, 1.0))
    for i in only_rev:
        objectives.append((i, -1.0))
    if not objectives:
        objectives = [(0, 1.0)]

    # Cycle through objectives until we have n points
    idx = 0
    while len(points) < n:
        i, sign = objectives[idx % len(objectives)]
        c = np.zeros(dim); c[i] = sign
        res = scipy.optimize.linprog(
            c, A_ub=A_ub if A_ub.size else None,
            b_ub=b_ub if b_ub.size else None,
            bounds=bounds, method='highs',
        )
        if res.success:
            points.append(res.x.copy())
        else:
            # Fall back to mean as a feasible point
            points.append(drg_mean.copy())
        idx += 1

    arr = np.column_stack(points)  # (dim, n)

    if reaction_ids is not None:
        sel = [T_ids.index(rid) for rid in reaction_ids]
        arr = arr[sel, :]
    return arr


# ------------------------------------------------------------------ #
#  Hit-and-run MCMC sampler
# ------------------------------------------------------------------ #

def _sign_constraints(data: ThermoSamplingData) -> tuple[list[int], list[int]]:
    """Return indices of forward-only and reverse-only reactions."""
    T_ids = list(data.T_reaction_ids)
    F_ids = list(data.F_reaction_ids)
    T_idx_in_F = [F_ids.index(rid) for rid in T_ids]
    fwd = [i for i in range(len(T_ids)) if data.F_lb[T_idx_in_F[i]] >= 0]
    rev = [i for i in range(len(T_ids)) if data.F_ub[T_idx_in_F[i]] <= 0]
    return fwd, rev


def _step_bounds(x: np.ndarray, direction: np.ndarray,
                 fwd_idx: list[int], rev_idx: list[int]) -> tuple[float, float]:
    """Compute [lo, hi] such that x + t*direction satisfies sign constraints."""
    lo, hi = -np.inf, np.inf
    for i in fwd_idx:
        # x[i] + t*d[i] <= 0  (dG must be negative for forward reactions)
        if direction[i] > 0:
            hi = min(hi, -x[i] / direction[i])
        elif direction[i] < 0:
            lo = max(lo, -x[i] / direction[i])
    for i in rev_idx:
        # x[i] + t*d[i] >= 0  (dG must be positive for reverse reactions)
        if direction[i] < 0:
            hi = min(hi, -x[i] / direction[i])
        elif direction[i] > 0:
            lo = max(lo, -x[i] / direction[i])
    return lo, hi


def sample_drg(
    data: ThermoSamplingData,
    initial_points: np.ndarray | None = None,
    num_samples: int = 20_000,
    num_chains: int = 4,
    num_initial_steps: int | None = None,
    max_psrf: float = 1.05,
    seed: int | None = None,
) -> DRGSamplingResult:
    """Hit-and-run MCMC sampler for the truncated MVN over dG.

    Parameters
    ----------
    data            : ThermoSamplingData
    initial_points  : (n_reactions, num_chains) starting points, or None
    num_samples     : number of post-burnin samples per chain
    num_chains      : number of independent chains
    num_initial_steps : burn-in steps (default: num_samples)
    max_psrf        : not enforced, stored for compatibility
    seed            : RNG seed for reproducibility

    Returns
    -------
    DRGSamplingResult with .samples and .basis_samples DataFrames,
    each of shape (num_samples * num_chains, n_reactions).
    """
    rng = np.random.default_rng(seed)
    if num_initial_steps is None:
        num_initial_steps = num_samples

    drg_mean, drg_cov = data.drg_mvn_params()
    L = np.linalg.cholesky(drg_cov)          # dG = mean + L @ z,  z ~ N(0,I)
    L_inv = np.linalg.inv(L)
    dim = len(data.T_reaction_ids)

    fwd_idx, rev_idx = _sign_constraints(data)

    if initial_points is None:
        initial_points = find_initial_points(data, num_chains)

    num_chains = initial_points.shape[1]

    # Run chains
    all_samples = []
    for c in range(num_chains):
        x = initial_points[:, c].copy()
        chain = []
        total_steps = num_initial_steps + num_samples
        for step in range(total_steps):
            # Random direction in original space, whitened for better mixing
            z = rng.standard_normal(dim)
            d = L @ (z / np.linalg.norm(z))  # direction in dG space

            lo, hi = _step_bounds(x, d, fwd_idx, rev_idx)

            # 1-D truncated normal along the ray x + t*d
            # The unnorm log-density along t: -0.5*(x+t*d - mean)' Cov^{-1} (x+t*d - mean)
            # In terms of z_t = L_inv @ (x - mean) + t * (L_inv @ d):
            # = -0.5 * ||z_t||^2  →  Gaussian in t with mean -<z0,Ld>/<Ld,Ld>, std 1/||Ld||
            Ld = L_inv @ d
            z0 = L_inv @ (x - drg_mean)
            t_mean = -np.dot(z0, Ld) / np.dot(Ld, Ld)
            t_std  = 1.0 / np.sqrt(np.dot(Ld, Ld))

            # Clamp bounds and sample from truncated normal
            lo_t = max(lo, t_mean - 10 * t_std)
            hi_t = min(hi, t_mean + 10 * t_std)

            if lo_t >= hi_t:
                chain.append(x.copy())
                continue

            # Inverse-CDF sampling of truncated normal on [lo_t, hi_t]
            a = (lo_t - t_mean) / t_std
            b = (hi_t - t_mean) / t_std
            u = rng.uniform(ndtr(a), ndtr(b))
            t = t_mean + t_std * ndtri(u)

            x = x + t * d

            if step >= num_initial_steps:
                chain.append(x.copy())

        all_samples.append(np.array(chain))  # (num_samples, dim)

    samples = np.concatenate(all_samples, axis=0)  # (num_chains*num_samples, dim)

    T_ids = list(data.T_reaction_ids)
    df_samples = pd.DataFrame(samples, columns=T_ids)
    # basis_samples: project into whitened (basis) space
    basis = ((samples - drg_mean[None, :]) @ L_inv.T)
    df_basis = pd.DataFrame(basis, columns=T_ids)

    return DRGSamplingResult(samples=df_samples, basis_samples=df_basis)
