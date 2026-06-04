"""
Pure-Python TFS sampler — Algorithm 1 from Gollub et al. (2020).

Core idea: work in the reduced basis  ΔrG' = T m + s,  m ~ N(0, I_q).
The confidence region is a sphere ||m||² ≤ r² in m-space.

One hit-and-run step:
  1. Random direction d on S^{q-1}.
  2. Sphere intersection: ||m + τd||² = r²  →  [τ_lo, τ_hi].
  3. Sign-change breakpoints: τ_i where ΔrG'_i(τ) = 0 for each T-constrained reaction.
  4. For each sub-segment (between adjacent breakpoints, inside sphere):
       (a) Orthant key  = sign pattern of ΔrG' at the midpoint.
       (b) LP feasibility: ∃ v  s.t.  Sv = 0,  lb ≤ v ≤ ub,
                            sign(v_i) = −sign(ΔrG'_i)  ∀ i ∈ Γ.
  5. Weight valid segments by their N(−m·d, 1) probability mass.
  6. Sample segment ∝ weights, then τ from truncated N(−m·d, 1) on it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.special import ndtr, ndtri
from scipy.stats.distributions import chi2

from .thermo_data import ThermoSamplingData


# ──────────────────────────────────────────────────────────────────────────────
#  Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DRGSamplingResult:
    """Output of :func:`sample_drg`.

    samples       : (n_total, n_rxn) DataFrame — dG' values
    basis_samples : (n_total, q)     DataFrame — m-space coordinates
    orthants      : DataFrame with columns [rxn_id …, 'weight'] listing each
                    visited sign pattern and its empirical probability
    """
    samples:       pd.DataFrame
    basis_samples: pd.DataFrame
    orthants:      pd.DataFrame = field(default_factory=pd.DataFrame)


# ──────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sphere_bounds(m: np.ndarray, d: np.ndarray, r: float):
    """τ bounds for ||m + τd||² ≤ r².  Returns (τ_lo, τ_hi) or None."""
    md   = float(m @ d)
    disc = md * md - float(m @ m) + r * r
    if disc < 0.0:
        return None
    sq = float(np.sqrt(disc))
    return -md - sq, -md + sq


def _sign_breakpoints(m: np.ndarray, d: np.ndarray,
                      T: np.ndarray, s: np.ndarray) -> np.ndarray:
    """τ values where ΔrG'_i(τ) = T[i]·(m + τd) + s[i] = 0 for any reaction i."""
    Td   = T @ d
    dg_m = T @ m + s
    mask = np.abs(Td) > 1e-12
    if not np.any(mask):
        return np.empty(0)
    return -dg_m[mask] / Td[mask]


def _segment_mass(t_lo: float, t_hi: float, mu_t: float) -> float:
    """CDF mass of [t_lo, t_hi] under N(mu_t, 1)."""
    return float(max(0.0, ndtr(t_hi - mu_t) - ndtr(t_lo - mu_t)))


def _sample_trunc_normal(mu: float, lo: float, hi: float,
                         rng: np.random.Generator) -> float:
    """Sample from N(mu, 1) truncated to [lo, hi]."""
    Pa = ndtr(lo - mu)
    Pb = ndtr(hi - mu)
    if Pb - Pa < 1e-300:
        return 0.5 * (lo + hi)
    return float(mu + ndtri(float(rng.uniform(Pa, Pb))))


# ──────────────────────────────────────────────────────────────────────────────
#  LP feasibility checker
# ──────────────────────────────────────────────────────────────────────────────

def _make_feasibility_checker(data: ThermoSamplingData,
                               flux_eps: float,
                               cache_size: int):
    """Return check(sign_tuple) → bool.

    sign_tuple[i] = −1  →  ΔrG'_i < 0  →  v_i must be > 0  (forward)
    sign_tuple[i] = +1  →  ΔrG'_i > 0  →  v_i must be < 0  (backward)
    sign_tuple[i] =  0  →  ΔrG'_i ≈ 0  →  infeasible (equilibrium ⟹ v_i = 0)
    """
    S     = data.flux_S
    F_lb  = data.F_lb.astype(float, copy=True)
    F_ub  = data.F_ub.astype(float, copy=True)
    F_ids = list(data.F_reaction_ids)
    T_ids = list(data.T_reaction_ids)
    T_in_F = np.array([F_ids.index(rid) for rid in T_ids], dtype=int)
    n_F   = len(F_ids)
    c_obj = np.zeros(n_F)
    b_eq  = np.zeros(S.shape[0])
    cache: dict[tuple, bool] = {}

    def check(sign_tuple: tuple) -> bool:
        if sign_tuple in cache:
            return cache[sign_tuple]

        lb_mod = F_lb.copy()
        ub_mod = F_ub.copy()
        ok = True
        for local_i, sgn in enumerate(sign_tuple):
            gi = int(T_in_F[local_i])
            if sgn == 0:
                ok = False; break
            elif sgn < 0:                             # forward: v_i > 0
                lb_mod[gi] = max(F_lb[gi], flux_eps)
            else:                                     # backward: v_i < 0
                ub_mod[gi] = min(F_ub[gi], -flux_eps)
            if lb_mod[gi] > ub_mod[gi] + 1e-12:
                ok = False; break

        if ok:
            res = scipy.optimize.linprog(
                c_obj, A_eq=S, b_eq=b_eq,
                bounds=list(zip(lb_mod, ub_mod)),
                method="highs",
                options={"presolve": True, "disp": False},
            )
            ok = (res.status == 0)

        if len(cache) >= cache_size:           # simple LRU eviction
            for k in list(cache)[:cache_size // 2]:
                del cache[k]
        cache[sign_tuple] = ok
        return ok

    return check


# ──────────────────────────────────────────────────────────────────────────────
#  Confidence radius
# ──────────────────────────────────────────────────────────────────────────────

def _get_r(data: ThermoSamplingData) -> float:
    if data.confidence_radius is not None:
        return float(data.confidence_radius)
    return float(np.sqrt(chi2.ppf(0.95, data.basis_dimensionality)))


# ──────────────────────────────────────────────────────────────────────────────
#  Finding initial m-space points
# ──────────────────────────────────────────────────────────────────────────────

def find_initial_points(data: ThermoSamplingData, n: int) -> np.ndarray:
    """Find n feasible starting points in m-space.

    Satisfies:
    * sign constraints for irreversible reactions (T[i]·m + s[i] ≤ −ε)
    * inside the confidence sphere (approximated by box |m_k| ≤ r)

    Returns
    -------
    (q, n) array of m-space coordinates.
    """
    if data.basis_to_drg_T is None:
        raise ValueError("basis_to_drg_T is required; build ThermoSamplingData "
                         "via from_model_data() or extract from a pta TFSModel.")

    T   = data.basis_to_drg_T
    s   = data.basis_to_drg_shift[:, 0]
    q   = T.shape[1]
    r   = _get_r(data)
    eps = float(data.drg_epsilon)

    # Sign constraints in m-space  G @ m ≤ h
    _tG = data.thermo_G
    if isinstance(_tG, np.ndarray) and _tG.shape[0] > 0:
        G_irrev = _tG
        h_irrev = np.asarray(data.thermo_h).ravel() - eps
    else:
        F_ids = list(data.F_reaction_ids)
        T_ids = list(data.T_reaction_ids)
        F_idx = {rid: i for i, rid in enumerate(F_ids)}
        rows, hs = [], []
        for li, rid in enumerate(T_ids):
            fi = F_idx[rid]
            if data.F_lb[fi] >= 0:
                rows.append(T[li]);  hs.append(-s[li] - eps)
            elif data.F_ub[fi] <= 0:
                rows.append(-T[li]); hs.append(s[li] - eps)
        if rows:
            G_irrev = np.array(rows)
            h_irrev = np.array(hs)
        else:
            G_irrev = np.zeros((0, q))
            h_irrev = np.zeros(0)

    box_G = np.vstack([np.eye(q), -np.eye(q)])
    box_h = np.full(2 * q, r)
    A_ub  = np.vstack([G_irrev, box_G]) if G_irrev.shape[0] > 0 else box_G
    b_ub  = np.concatenate([h_irrev, box_h]) if G_irrev.shape[0] > 0 else box_h

    points = []
    for i in range(n):
        c = np.zeros(q)
        c[i % q] = 1.0 if i % 2 == 0 else -1.0
        res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub,
                                     bounds=[(-r, r)] * q, method="highs")
        if res.success:
            m = res.x.copy()
            nm = np.linalg.norm(m)
            if nm > r:
                m *= r * 0.95 / nm
        else:
            m = np.zeros(q)
        points.append(m)

    return np.column_stack(points)  # (q, n)


# ──────────────────────────────────────────────────────────────────────────────
#  PSRF  (split Gelman–Rubin)
# ──────────────────────────────────────────────────────────────────────────────

def _psrf(chains: np.ndarray) -> np.ndarray:
    """Split R-hat per variable.  chains: (n_chains, n_samples, n_vars)."""
    n_c, n, p = chains.shape
    if n_c < 2 or n < 4:
        return np.full(p, np.nan)
    h = n // 2
    sp = chains[:, :h, :]
    chain_means = sp.mean(axis=1)
    grand_mean  = chain_means.mean(axis=0)
    B = h / (n_c - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = sp.var(axis=1, ddof=1).mean(axis=0)
    var_plus = (h - 1) / h * W + B / h
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(var_plus / np.where(W > 0, W, np.nan))


# ──────────────────────────────────────────────────────────────────────────────
#  Main TFS sampler
# ──────────────────────────────────────────────────────────────────────────────

def sample_drg(
    data: ThermoSamplingData,
    initial_points: np.ndarray | None = None,
    num_samples: int = 1_000,
    num_chains: int = 4,
    num_initial_steps: int | None = None,
    max_psrf: float = 1.1,
    seed: int | None = None,
    feasibility_cache_size: int = 10_000,
    min_segment_length: float = 1e-6,
    flux_eps: float = 1e-4,
) -> DRGSamplingResult:
    """Hit-and-Run sampler for the steady-state thermodynamic space T.

    Implements Algorithm 1 from Gollub et al. 2020.  Requires
    ``data.basis_to_drg_T`` to be set (built by :func:`pta_port.build_drg_basis`
    or extracted from a pta TFSModel pickle).

    Parameters
    ----------
    data                   ThermoSamplingData with basis and flux information.
    initial_points         (q, n_chains) m-space starting points, or None.
    num_samples            Post-burn-in samples per chain.
    num_chains             Independent chains.
    num_initial_steps      Burn-in steps (default = num_samples).
    feasibility_cache_size LRU cache size for LP feasibility results.
    min_segment_length     Skip segments shorter than this × r.
    flux_eps               Minimum |flux| for feasibility LP.

    Returns
    -------
    DRGSamplingResult
    """
    if data.basis_to_drg_T is None:
        raise ValueError("basis_to_drg_T is required.")

    rng  = np.random.default_rng(seed)
    T    = data.basis_to_drg_T           # (n_rxn, q)
    s    = data.basis_to_drg_shift[:, 0] # (n_rxn,)
    q    = T.shape[1]
    r    = _get_r(data)
    eps  = float(data.drg_epsilon)
    T_ids = list(data.T_reaction_ids)

    if num_initial_steps is None:
        num_initial_steps = num_samples

    if initial_points is None:
        initial_points = find_initial_points(data, num_chains)

    num_chains = initial_points.shape[1]
    feasible   = _make_feasibility_checker(data, flux_eps, feasibility_cache_size)
    min_seg    = min_segment_length * r

    all_m:  list[np.ndarray] = []                     # (num_samples, q) per chain
    orthant_counts: dict[tuple, int] = defaultdict(int)

    for c in range(num_chains):
        m = initial_points[:, c].copy()
        chain: list[np.ndarray] = []
        total_steps = num_initial_steps + num_samples

        for step in range(total_steps):
            # ── 1. random direction ────────────────────────────────────────
            d    = rng.standard_normal(q)
            norm = np.linalg.norm(d)
            if norm < 1e-14:
                continue
            d /= norm

            # ── 2. sphere bounds ───────────────────────────────────────────
            sb = _sphere_bounds(m, d, r)
            if sb is None:
                if step >= num_initial_steps:
                    chain.append(m.copy())
                    _record(m, T, s, T_ids, orthant_counts)
                continue
            t_lo_s, t_hi_s = sb

            # ── 3. sign-change breakpoints ─────────────────────────────────
            bps = _sign_breakpoints(m, d, T, s)
            # keep only those strictly inside the sphere chord
            bps = bps[(bps > t_lo_s + 1e-14) & (bps < t_hi_s - 1e-14)]
            bps.sort()
            edges = np.concatenate([[t_lo_s], bps, [t_hi_s]])

            # ── 4. evaluate sub-segments ───────────────────────────────────
            mu_t = float(-(m @ d))
            seg_lo: list[float] = []
            seg_hi: list[float] = []
            seg_wt: list[float] = []

            for k in range(len(edges) - 1):
                a, b = float(edges[k]), float(edges[k + 1])
                if b - a < min_seg:
                    continue
                dg_mid = T @ (m + 0.5 * (a + b) * d) + s
                key    = tuple(int(np.sign(x)) for x in dg_mid)
                if not feasible(key):
                    continue
                mass = _segment_mass(a, b, mu_t)
                if mass < 1e-300:
                    continue
                seg_lo.append(a)
                seg_hi.append(b)
                seg_wt.append(mass)

            if not seg_lo:
                # No valid segment — stay put
                if step >= num_initial_steps:
                    chain.append(m.copy())
                    _record(m, T, s, T_ids, orthant_counts)
                continue

            # ── 5. sample segment ∝ weights ────────────────────────────────
            wts = np.array(seg_wt)
            wts /= wts.sum()
            idx  = int(rng.choice(len(seg_lo), p=wts))
            a_s, b_s = seg_lo[idx], seg_hi[idx]

            # ── 6. sample τ from truncated N(mu_t, 1) on [a_s, b_s] ───────
            t_new = _sample_trunc_normal(mu_t, a_s, b_s, rng)
            m     = m + t_new * d

            # Numerical safety: project back if drifted outside sphere
            nm = np.linalg.norm(m)
            if nm > r * (1.0 + 1e-9):
                m *= r / nm

            if step >= num_initial_steps:
                chain.append(m.copy())
                _record(m, T, s, T_ids, orthant_counts)

        if chain:
            all_m.append(np.array(chain))            # (≤ num_samples, q)

    if not all_m:
        raise RuntimeError("All chains produced zero samples — check initial points "
                           "or increase burn-in steps.")

    # ── assemble ───────────────────────────────────────────────────────────────
    m_all  = np.concatenate(all_m, axis=0)           # (total, q)
    dg_all = (T @ m_all.T).T + s[None, :]            # (total, n_rxn)

    df_samp  = pd.DataFrame(dg_all, columns=T_ids)
    df_basis = pd.DataFrame(m_all,  columns=[f"m{i}" for i in range(q)])

    # R-hat across chains (requires all same length)
    lengths = [len(c) for c in all_m]
    min_len = min(lengths)
    if min_len > 3 and len(all_m) > 1:
        chains_arr = np.stack([c[:min_len] for c in all_m])  # (n_chains, min_len, q)
        rhat = _psrf(chains_arr)
    else:
        rhat = np.full(q, np.nan)

    # Orthant table
    total_c = sum(orthant_counts.values())
    if total_c > 0:
        rows = [{"weight": cnt / total_c, **dict(zip(T_ids, signs))}
                for signs, cnt in orthant_counts.items()]
        df_orth = (pd.DataFrame(rows)
                   .sort_values("weight", ascending=False)
                   .reset_index(drop=True))
    else:
        df_orth = pd.DataFrame()

    result = DRGSamplingResult(samples=df_samp, basis_samples=df_basis, orthants=df_orth)
    result.psrf = pd.Series(rhat, index=[f"m{i}" for i in range(q)])
    return result


def _record(m, T, s, T_ids, counter):
    dg  = T @ m + s
    key = tuple(int(np.sign(x)) for x in dg)
    counter[key] += 1
