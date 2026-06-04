"""Tests for pta_port — run with:  pytest src/sbmfi/priors/pta_port/test_pta_port.py -v"""
from __future__ import annotations

import numpy as np
import pytest

from sbmfi.priors.pta_port.linalg import qr_rank_deficient
from sbmfi.priors.pta_port.utils import covariance_square_root, apply_transform
from sbmfi.priors.pta_port.thermo_space import (
    ThermodynamicSpaceParams,
    build_drg_basis,
    build_drg_polytope,
    R_GAS,
)
from sbmfi.priors.pta_port.flux_space import FluxSpaceParams

# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def two_rxn_params():
    """2-metabolite / 2-reaction toy network (M1→M2 forward-only, M2⇌∅ reversible)."""
    S = np.array([[-1.0, 0.0],
                  [ 1.0,-1.0]])
    dfg0_cov = np.array([[4.0, 1.5],
                         [1.5, 2.0]])
    dfg0_cov_sqrt = covariance_square_root(dfg0_cov, min_eigenvalue=1e-8)
    return ThermodynamicSpaceParams(
        reaction_ids=["R1", "R2"],
        metabolite_ids=["M1", "M2"],
        T_kelvin=310.15,
        dfg0_prime_mean=np.array([5.0, -2.0]),
        dfg0_prime_cov_sqrt=dfg0_cov_sqrt,
        log_conc_mean=np.array([-7.0, -7.0]),
        log_conc_cov=np.diag([0.25, 0.25]),
        S_constraints=S,
    )


@pytest.fixture
def three_rxn_params():
    """3-reaction network with identity stoichiometry (decoupled reactions)."""
    return ThermodynamicSpaceParams(
        reaction_ids=["R1", "R2", "R3"],
        metabolite_ids=["M1", "M2", "M3"],
        T_kelvin=310.15,
        dfg0_prime_mean=np.array([-2.0, 2.5, 0.0]),
        dfg0_prime_cov_sqrt=np.diag([0.3, 0.2, 0.5]),
        log_conc_mean=np.zeros(3),
        log_conc_cov=np.diag([1e-6, 1e-6, 1e-6]),
        S_constraints=np.eye(3),
    )


# ──────────────────────────────────────────────
#  linalg.qr_rank_deficient
# ──────────────────────────────────────────────

class TestQrRankDeficient:

    def test_full_rank(self, rng):
        A = rng.standard_normal((4, 6))
        B = qr_rank_deficient(A)
        assert B.shape == (4, 6)
        np.testing.assert_allclose(B.T @ B, A.T @ A, atol=1e-10)

    def test_rank_deficient_row(self, rng):
        A = rng.standard_normal((5, 8))
        A[2] = A[0] + A[1]          # one linearly dependent row → rank 4
        B = qr_rank_deficient(A)
        assert B.shape[0] == 4, f"expected rank 4, got {B.shape[0]}"
        np.testing.assert_allclose(B.T @ B, A.T @ A, atol=1e-10)

    def test_rank_deficient_column(self, rng):
        A = rng.standard_normal((6, 5))
        A[:, 3] = A[:, 0] + A[:, 2]  # rank 4
        B = qr_rank_deficient(A)
        assert B.shape[0] == 4
        np.testing.assert_allclose(B.T @ B, A.T @ A, atol=1e-10)

    def test_covariance_sqrt_reduction(self):
        """Primary usage: qr_rank_deficient(C.T).T gives a minimal covariance sqrt."""
        cov = np.array([[4.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0]])   # rank 2
        C = covariance_square_root(cov, min_eigenvalue=1e-8)  # (3, 2)
        T = qr_rank_deficient(C.T).T         # (3, q) with q = rank(C)
        assert T.shape[1] <= C.shape[1]
        np.testing.assert_allclose(T @ T.T, cov, atol=1e-10)

    def test_empty_matrix(self):
        A = np.zeros((0, 5))
        B = qr_rank_deficient(A)
        assert B.shape[1] == 5

    def test_all_zeros(self):
        A = np.zeros((3, 4))
        B = qr_rank_deficient(A)
        assert B.shape[0] == 0


# ──────────────────────────────────────────────
#  utils.covariance_square_root
# ──────────────────────────────────────────────

class TestCovarianceSquareRoot:

    def test_diagonal(self):
        cov = np.diag([1.0, 4.0, 9.0])
        sqrt = covariance_square_root(cov, min_eigenvalue=1e-8)
        np.testing.assert_allclose(sqrt @ sqrt.T, cov, atol=1e-14)
        assert sqrt.shape == (3, 3)

    def test_full_rank_symmetric(self):
        cov = np.array([[4.0, 1.5], [1.5, 2.0]])
        sqrt = covariance_square_root(cov, min_eigenvalue=1e-8)
        np.testing.assert_allclose(sqrt @ sqrt.T, cov, atol=1e-14)

    def test_rank_deficient_drops_small_eigenvalues(self):
        # Rank-1 covariance: outer product
        v = np.array([1.0, 2.0, 3.0])
        cov = np.outer(v, v)
        sqrt = covariance_square_root(cov, min_eigenvalue=1e-6)
        assert sqrt.shape == (3, 1), f"expected (3,1), got {sqrt.shape}"
        np.testing.assert_allclose(sqrt @ sqrt.T, cov, atol=1e-12)

    def test_returns_zeros_when_all_eigenvalues_below_threshold(self):
        cov = np.eye(3) * 1e-10
        sqrt = covariance_square_root(cov, min_eigenvalue=1e-6)
        assert sqrt.shape == (3, 0)


class TestApplyTransform:

    def test_basic(self):
        T = np.array([[2.0, 0.0], [0.0, 3.0]])
        s = np.array([1.0, -1.0])
        x = np.array([[1.0], [2.0]])
        result = apply_transform(x, (T, s[:, None]))
        np.testing.assert_allclose(result, T @ x + s[:, None])

    def test_batch(self):
        T = np.eye(3)
        s = np.ones(3)
        X = np.random.default_rng(0).standard_normal((3, 100))
        np.testing.assert_allclose(apply_transform(X, (T, s[:, None])), X + s[:, None])


# ──────────────────────────────────────────────
#  thermo_space.build_drg_basis
# ──────────────────────────────────────────────

class TestBuildDrgBasis:

    def test_covariance_roundtrip_two_rxn(self, two_rxn_params):
        p = two_rxn_params
        T, s = build_drg_basis(p, min_eigenvalue=1e-8)
        RT = R_GAS * p.T_kelvin
        S = p.S_constraints
        Sigma_r_ref = S.T @ (
            p.dfg0_prime_cov_sqrt @ p.dfg0_prime_cov_sqrt.T
            + RT**2 * p.log_conc_cov
        ) @ S
        np.testing.assert_allclose(T @ T.T, Sigma_r_ref, atol=1e-10)

    def test_mean_two_rxn(self, two_rxn_params):
        p = two_rxn_params
        T, s = build_drg_basis(p, min_eigenvalue=1e-8)
        RT = R_GAS * p.T_kelvin
        s_ref = p.S_constraints.T @ (p.dfg0_prime_mean + RT * p.log_conc_mean)
        np.testing.assert_allclose(s, s_ref, atol=1e-13)

    def test_dimensionality_at_most_n_rxn(self, two_rxn_params):
        T, _ = build_drg_basis(two_rxn_params, min_eigenvalue=1e-8)
        assert T.shape[0] == 2
        assert T.shape[1] <= 2

    def test_rank_reduction_three_rxn_tiny_lc(self, three_rxn_params):
        """With identity S and tiny log-conc noise, rank = rank(dfg0_cov_sqrt)."""
        T, s = build_drg_basis(three_rxn_params, min_eigenvalue=1e-4)
        # dfg0_prime_cov_sqrt is diagonal 3×3 → rank 3; lc is negligible
        assert T.shape == (3, 3)

    def test_covariance_roundtrip_three_rxn(self, three_rxn_params):
        p = three_rxn_params
        T, s = build_drg_basis(p, min_eigenvalue=1e-8)
        RT = R_GAS * p.T_kelvin
        S = p.S_constraints
        Sigma_r_ref = S.T @ (
            p.dfg0_prime_cov_sqrt @ p.dfg0_prime_cov_sqrt.T
            + RT**2 * p.log_conc_cov
        ) @ S
        np.testing.assert_allclose(T @ T.T, Sigma_r_ref, atol=1e-10)

    def test_correlated_dfg0_reduces_rank(self):
        """If Σ_dfg0 is rank-1 and Σ_lc is negligible, q should be 1."""
        v = np.array([1.0, 1.0])
        cov = np.outer(v, v) * 4.0  # rank 1
        cov_sqrt = covariance_square_root(cov, min_eigenvalue=1e-8)
        p = ThermodynamicSpaceParams(
            reaction_ids=["R1", "R2"],
            metabolite_ids=["M1", "M2"],
            T_kelvin=310.15,
            dfg0_prime_mean=np.array([0.0, 0.0]),
            dfg0_prime_cov_sqrt=cov_sqrt,
            log_conc_mean=np.zeros(2),
            log_conc_cov=np.eye(2) * 1e-12,
            S_constraints=np.eye(2),
        )
        T, _ = build_drg_basis(p, min_eigenvalue=1e-6)
        assert T.shape[1] == 1, f"expected q=1, got q={T.shape[1]}"

    def test_temperature_scaling(self, two_rxn_params):
        """Higher T → larger RT → log-conc contribution increases."""
        T_low,  _ = build_drg_basis(two_rxn_params, min_eigenvalue=1e-8)
        hot = ThermodynamicSpaceParams(**{
            **two_rxn_params.__dict__,
            "T_kelvin": 400.0,
        })
        T_hot, _ = build_drg_basis(hot, min_eigenvalue=1e-8)
        # Frobenius norm of Σ_r grows with T for non-zero log_conc_cov
        assert np.linalg.norm(T_hot @ T_hot.T) > np.linalg.norm(T_low @ T_low.T)


# ──────────────────────────────────────────────
#  thermo_space.build_drg_polytope
# ──────────────────────────────────────────────

class TestBuildDrgPolytope:

    def test_only_forward_reaction_generates_one_row(self, two_rxn_params):
        p = two_rxn_params
        T, s = build_drg_basis(p)
        # R1 forward-only (lb=0), R2 reversible (lb=-10)
        G, h = build_drg_polytope(T, s, ["R1","R2"],
                                   np.array([0.0,-10.0]), np.array([10.0,10.0]),
                                   ["R1","R2"])
        assert G.shape == (1, T.shape[1])
        assert h.shape == (1,)

    def test_backward_only_reaction_generates_one_row(self, two_rxn_params):
        p = two_rxn_params
        T, s = build_drg_basis(p)
        # R1 backward-only (ub=0), R2 reversible
        G, h = build_drg_polytope(T, s, ["R1","R2"],
                                   np.array([-10.0,-10.0]), np.array([0.0,10.0]),
                                   ["R1","R2"])
        assert G.shape == (1, T.shape[1])

    def test_all_reversible_empty_polytope(self, two_rxn_params):
        p = two_rxn_params
        T, s = build_drg_basis(p)
        G, h = build_drg_polytope(T, s, ["R1","R2"],
                                   np.array([-10.0,-10.0]), np.array([10.0,10.0]),
                                   ["R1","R2"])
        assert G.shape[0] == 0

    def test_constraint_rejects_infeasible_point(self, two_rxn_params):
        """G @ m <= h should be violated at m=0 when s[R1] > -eps for fwd reaction."""
        p = two_rxn_params
        T, s = build_drg_basis(p)
        G, h = build_drg_polytope(T, s, ["R1","R2"],
                                   np.array([0.0,-10.0]), np.array([10.0,10.0]),
                                   ["R1","R2"], drg_epsilon=0.1)
        # At m=0: ΔrG'_R1 = s[0]; if s[0] > -0.1 the point is infeasible
        m_zero = np.zeros(T.shape[1])
        violation = G @ m_zero - h
        drg_r1_at_zero = T[0] @ m_zero + s[0]
        if drg_r1_at_zero > -0.1:
            assert np.any(violation > 0), "origin should violate forward constraint"

    def test_constraint_satisfied_at_known_feasible_point(self, two_rxn_params):
        """A point deep inside the forward half-space must satisfy G m <= h."""
        p = two_rxn_params
        T, s = build_drg_basis(p)
        G, h = build_drg_polytope(T, s, ["R1","R2"],
                                   np.array([0.0,-10.0]), np.array([10.0,10.0]),
                                   ["R1","R2"], drg_epsilon=0.1)
        if G.shape[0] == 0:
            return   # no constraints to check
        # Find m such that ΔrG'_R1 = T[0]@m + s[0] = -10 (clearly feasible)
        # Simple approach: m along -T[0] direction scaled to get the target dG
        direction = -T[0] / (np.linalg.norm(T[0]) ** 2)
        target_drg_r1 = -10.0
        shift = target_drg_r1 - s[0]
        m_feas = direction * shift
        assert np.all(G @ m_feas <= h + 1e-9)


# ──────────────────────────────────────────────
#  flux_space.FluxSpaceParams
# ──────────────────────────────────────────────

class TestFluxSpaceParams:

    def test_construction(self):
        S = np.eye(3)
        f = FluxSpaceParams(S=S, lb=np.zeros(3), ub=np.ones(3)*10,
                            reaction_ids=["r1","r2","r3"],
                            metabolite_ids=["m1","m2","m3"])
        assert f.S.shape == (3, 3)
        assert f.reaction_ids == ["r1","r2","r3"]


# ──────────────────────────────────────────────
#  Integration: pta_port → ThermoSamplingData
# ──────────────────────────────────────────────

class TestIntegrationWithThermoSamplingData:

    def test_drg_mvn_params_matches_build_drg_basis(self, two_rxn_params):
        from sbmfi.priors.thermo_data import ThermoSamplingData

        p = two_rxn_params
        T, s = build_drg_basis(p, min_eigenvalue=1e-8)

        data = ThermoSamplingData(
            T_reaction_ids=p.reaction_ids,
            metabolite_ids=p.metabolite_ids,
            T_kelvin=p.T_kelvin,
            dfg0_prime_mean=p.dfg0_prime_mean,
            dfg0_prime_cov_sqrt=p.dfg0_prime_cov_sqrt,
            log_conc_mean=p.log_conc_mean,
            log_conc_cov=p.log_conc_cov,
            S_constraints=p.S_constraints,
            flux_S=p.S_constraints,
            F_reaction_ids=p.reaction_ids,
            F_lb=np.array([0.0, -10.0]),
            F_ub=np.array([10.0,  10.0]),
            basis_to_drg_T=T,
            basis_to_drg_shift=s.reshape(-1, 1),
        )
        mean_out, cov_out = data.drg_mvn_params()

        np.testing.assert_allclose(mean_out, s, atol=1e-12)
        np.testing.assert_allclose(cov_out, T @ T.T, atol=1e-10)
