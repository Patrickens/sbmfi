"""Thermodynamic space parameters and basis construction.

Pure-Python port of pta.ThermodynamicSpace + ThermodynamicSpaceBasis
(explicit_drg=True, explicit_drg0=False, explicit_log_conc=False).

No enkie, no cobra, no component_contribution required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .linalg import qr_rank_deficient
from .utils import covariance_square_root

R_GAS = 8.314472e-3  # kJ mol⁻¹ K⁻¹


@dataclass
class ThermodynamicSpaceParams:
    """All raw parameters needed to build a dG basis — no enkie objects.

    Conventions
    -----------
    dfg0_prime_cov_sqrt : (n_met, q_dfg) — already a square root, i.e.
        ``dfg0_prime_cov_sqrt @ dfg0_prime_cov_sqrt.T`` = covariance of
        standard formation energies.  Use ``covariance_square_root(cov)``
        to obtain this from a full covariance matrix.

    log_conc_cov : (n_met, n_met) — full covariance matrix (NOT a sqrt).

    S_constraints : (n_met, n_rxn) — stoichiometric sub-matrix for the
        thermodynamically constrained reactions only.
    """

    reaction_ids: list[str]
    metabolite_ids: list[str]
    T_kelvin: float

    dfg0_prime_mean: np.ndarray        # (n_met,)
    dfg0_prime_cov_sqrt: np.ndarray    # (n_met, q_dfg)
    log_conc_mean: np.ndarray          # (n_met,)
    log_conc_cov: np.ndarray           # (n_met, n_met)
    S_constraints: np.ndarray          # (n_met, n_rxn)


# ------------------------------------------------------------------ #
#  Basis construction
# ------------------------------------------------------------------ #

def build_drg_basis(
    params: ThermodynamicSpaceParams,
    min_eigenvalue: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the affine map  ΔrG' = T @ m + s,  m ~ N(0, I_q).

    This is a port of ThermodynamicSpaceBasis._compute_basis with
    explicit_drg=True, explicit_drg0=False, explicit_log_conc=False.

    The joint distribution of ΔrG' follows from Eq. (2)-(6) in
    Gollub et al. 2020:

        ΔrG' = Sᵀ (ΔfG'° + RT ln c)
        ΔfG'° ~ N(μ_dfg0, Σ_dfg0),   ln c ~ N(μ_lc, Σ_lc)

    Marginal mean:   s = Sᵀ (μ_dfg0 + RT μ_lc)
    Marginal cov:  Σ_r = Sᵀ (Σ_dfg0 + RT² Σ_lc) S

    T is a (n_rxn, q) matrix satisfying T @ T.T = Σ_r with minimal q.

    Parameters
    ----------
    params        : ThermodynamicSpaceParams
    min_eigenvalue: threshold for rank reduction

    Returns
    -------
    T : (n_rxn, q) covariance square root — columns are principal directions
    s : (n_rxn,)   mean vector of ΔrG'
    """
    RT = R_GAS * params.T_kelvin
    S = params.S_constraints  # (n_met, n_rxn)

    # Mean of ΔrG' = Sᵀ (μ_dfg0 + RT μ_lc)
    s = S.T @ (params.dfg0_prime_mean + RT * params.log_conc_mean)

    # Build joint covariance square root [RT Sᵀ Σ_lc^{1/2} | Sᵀ Σ_dfg0^{1/2}]
    # so that (cov_sqrt)(cov_sqrt)ᵀ = Σ_r
    lc_sqrt = covariance_square_root(params.log_conc_cov, min_eigenvalue)  # (n_met, k_lc)
    drg_cov_sqrt = np.hstack([
        RT * S.T @ lc_sqrt,            # (n_rxn, k_lc)
        S.T @ params.dfg0_prime_cov_sqrt,  # (n_rxn, q_dfg)
    ])  # (n_rxn, k_lc + q_dfg)

    # Rank-reduce to minimal orthonormal basis
    T = qr_rank_deficient(drg_cov_sqrt.T, tol=min_eigenvalue).T  # (n_rxn, q)

    return T, s


# ------------------------------------------------------------------ #
#  Polytope for irreversible sign constraints
# ------------------------------------------------------------------ #

def build_drg_polytope(
    T: np.ndarray,
    s: np.ndarray,
    T_reaction_ids: list[str],
    flux_lb: np.ndarray,
    flux_ub: np.ndarray,
    F_reaction_ids: list[str],
    drg_epsilon: float = 1e-1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the linear constraints G @ m <= h encoding irreversible reactions.

    For a forward-only reaction i (flux_lb[i] >= 0):
        ΔrG'_i = T[i,:] @ m + s[i]  must be <= -drg_epsilon
        → T[i,:] @ m  <= -s[i] - drg_epsilon

    For a backward-only reaction i (flux_ub[i] <= 0):
        ΔrG'_i must be >= +drg_epsilon
        → -T[i,:] @ m <= s[i] - drg_epsilon

    Parameters
    ----------
    T              : (n_rxn, q) from build_drg_basis
    s              : (n_rxn,) from build_drg_basis
    T_reaction_ids : reaction IDs corresponding to rows of T/s
    flux_lb        : lower bounds for ALL reactions (length n_F)
    flux_ub        : upper bounds for ALL reactions (length n_F)
    F_reaction_ids : all reaction IDs (length n_F)
    drg_epsilon    : minimum |ΔrG'| for irreversible reactions

    Returns
    -------
    G : (n_constraints, q)
    h : (n_constraints,)
    """
    F_idx = {rid: i for i, rid in enumerate(F_reaction_ids)}
    fwd_rows, bwd_rows = [], []
    fwd_h, bwd_h = [], []

    for i, rid in enumerate(T_reaction_ids):
        fi = F_idx[rid]
        if flux_lb[fi] >= 0:           # forward-only
            fwd_rows.append(T[i, :])
            fwd_h.append(-s[i] - drg_epsilon)
        elif flux_ub[fi] <= 0:         # backward-only
            bwd_rows.append(-T[i, :])
            bwd_h.append(s[i] - drg_epsilon)

    rows = fwd_rows + bwd_rows
    hs   = fwd_h   + bwd_h

    if not rows:
        return np.zeros((0, T.shape[1])), np.zeros(0)

    return np.array(rows), np.array(hs)


# ------------------------------------------------------------------ #
#  Reaction-level basis construction (for use with equilibrator_api)
# ------------------------------------------------------------------ #

def build_drg_basis_from_reactions(
    drg0_mean: np.ndarray,
    drg0_cov_sqrt: np.ndarray,
    log_conc_mean: np.ndarray,
    log_conc_cov: np.ndarray,
    S_constraints: np.ndarray,
    T_kelvin: float,
    min_eigenvalue: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the affine map ΔrG' = T @ m + s from reaction-level inputs.

    Use this when you already have reaction-level standard Gibbs energies
    (e.g. from :func:`~pta_port.gibbs.estimate_drg0_prime`).  Mathematically
    equivalent to :func:`build_drg_basis` but takes ΔrG'° directly instead
    of metabolite formation energies.

    Parameters
    ----------
    drg0_mean     : (n_rxn,) — mean ΔrG'° values (kJ/mol)
    drg0_cov_sqrt : (n_rxn, q_drg) — sqrt of Cov(ΔrG'°) from component contribution
    log_conc_mean : (n_met,) — mean of log metabolite concentrations
    log_conc_cov  : (n_met, n_met) — covariance of log concentrations
    S_constraints : (n_met, n_rxn) — stoichiometric sub-matrix
    T_kelvin      : temperature in K
    min_eigenvalue: threshold for rank reduction

    Returns
    -------
    T : (n_rxn, q) — columns are principal dG' directions
    s : (n_rxn,)   — mean ΔrG' = drg0_mean + RT * S.T @ log_conc_mean
    """
    RT = R_GAS * T_kelvin
    S  = S_constraints

    # Mean: ΔrG' = ΔrG'° + RT * S.T @ ln c
    s = drg0_mean + RT * S.T @ log_conc_mean

    # Joint covariance sqrt: [RT * S.T @ Σ_lc^{1/2}  |  Σ_drg0^{1/2}]
    lc_sqrt      = covariance_square_root(log_conc_cov, min_eigenvalue)
    drg_cov_sqrt = np.hstack([RT * S.T @ lc_sqrt, drg0_cov_sqrt])

    T = qr_rank_deficient(drg_cov_sqrt.T, tol=min_eigenvalue).T
    return T, s
