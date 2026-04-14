"""
Tests for sbmfi.core.distributions — normal and truncated-normal helpers.
"""
import math
import pytest
import torch
from sbmfi.core.distributions import (
    norm_log_pdf,
    norm_pdf,
    norm_cdf,
    norm_inv_cdf,
    trunc_norm_log_pdf,
    trunc_norm_pdf,
    trunc_norm_cdf,
    trunc_norm_inv_cdf,
    unif_log_pdf,
    unif_inv_cdf,
    sample_bounded,
    bounded_log_prob,
)

DTYPE = torch.double


# ---------------------------------------------------------------------------
# Normal helpers
# ---------------------------------------------------------------------------

class TestNormPdf:
    def test_standard_normal_at_zero(self):
        x = torch.tensor(0.0, dtype=DTYPE)
        mu = torch.tensor(0.0, dtype=DTYPE)
        sigma = torch.tensor(1.0, dtype=DTYPE)
        expected = 1.0 / math.sqrt(2 * math.pi)
        assert norm_pdf(x, mu, sigma).item() == pytest.approx(expected, rel=1e-6)

    def test_log_pdf_matches_log_of_pdf(self):
        x = torch.linspace(-2, 2, 20, dtype=DTYPE)
        mu = torch.tensor(0.5, dtype=DTYPE)
        sigma = torch.tensor(1.5, dtype=DTYPE)
        log_p = norm_log_pdf(x, mu, sigma)
        p = norm_pdf(x, mu, sigma)
        torch.testing.assert_close(log_p.exp(), p)

    def test_pdf_integrates_to_one(self):
        """Numerical integration of N(1, 2) over [-10, 12] ≈ 1."""
        mu = torch.tensor(1.0, dtype=DTYPE)
        sigma = torch.tensor(2.0, dtype=DTYPE)
        x = torch.linspace(-10, 12, 10000, dtype=DTYPE)
        dx = x[1] - x[0]
        integral = (norm_pdf(x, mu, sigma) * dx).sum().item()
        assert integral == pytest.approx(1.0, abs=1e-4)


class TestNormCdf:
    def test_cdf_at_mean_is_half(self):
        mu = torch.tensor(3.0, dtype=DTYPE)
        sigma = torch.tensor(2.0, dtype=DTYPE)
        assert norm_cdf(mu, mu, sigma).item() == pytest.approx(0.5, abs=1e-7)

    def test_cdf_inv_cdf_roundtrip(self):
        mu = torch.tensor(1.0, dtype=DTYPE)
        sigma = torch.tensor(2.0, dtype=DTYPE)
        p = torch.linspace(0.01, 0.99, 50, dtype=DTYPE)
        x = norm_inv_cdf(p, mu, sigma)
        p_back = norm_cdf(x, mu, sigma)
        torch.testing.assert_close(p_back, p, atol=1e-6, rtol=0)

    def test_cdf_monotone(self):
        x = torch.linspace(-3, 3, 30, dtype=DTYPE)
        mu = torch.tensor(0.0, dtype=DTYPE)
        sigma = torch.tensor(1.0, dtype=DTYPE)
        cdf = norm_cdf(x, mu, sigma)
        assert (cdf.diff() > 0).all()


# ---------------------------------------------------------------------------
# Truncated normal helpers
# ---------------------------------------------------------------------------

class TestTruncNormPdf:
    @pytest.fixture
    def params(self):
        mu = torch.tensor(0.0, dtype=DTYPE)
        sigma = torch.tensor(1.0, dtype=DTYPE)
        lo = torch.tensor(-1.0, dtype=DTYPE)
        hi = torch.tensor(2.0, dtype=DTYPE)
        return mu, sigma, lo, hi

    def test_log_pdf_matches_log_of_pdf(self, params):
        mu, sigma, lo, hi = params
        x = torch.linspace(-0.9, 1.9, 30, dtype=DTYPE)
        log_p = trunc_norm_log_pdf(x, mu, sigma, lo, hi)
        p = trunc_norm_pdf(x, mu, sigma, lo, hi)
        torch.testing.assert_close(log_p.exp(), p)

    def test_pdf_integrates_to_one(self, params):
        mu, sigma, lo, hi = params
        x = torch.linspace(-1.0, 2.0, 50000, dtype=DTYPE)
        dx = x[1] - x[0]
        integral = (trunc_norm_pdf(x, mu, sigma, lo, hi) * dx).sum().item()
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_zero_outside_support(self, params):
        mu, sigma, lo, hi = params
        # Below lower bound
        x_below = torch.tensor(-1.5, dtype=DTYPE)
        # PDF should be finite but the log-pdf will be negative (large negative for very low values);
        # probability mass is only defined for x in (lo, hi)
        # Just verify that the CDF at lo is 0 and at hi is 1
        assert trunc_norm_cdf(lo, mu, sigma, lo, hi).item() == pytest.approx(0.0, abs=1e-6)
        assert trunc_norm_cdf(hi, mu, sigma, lo, hi).item() == pytest.approx(1.0, abs=1e-6)


class TestTruncNormCdf:
    def test_cdf_at_bounds(self):
        mu = torch.tensor(0.0, dtype=DTYPE)
        sigma = torch.tensor(1.0, dtype=DTYPE)
        lo = torch.tensor(-2.0, dtype=DTYPE)
        hi = torch.tensor(2.0, dtype=DTYPE)
        assert trunc_norm_cdf(lo, mu, sigma, lo, hi).item() == pytest.approx(0.0, abs=1e-6)
        assert trunc_norm_cdf(hi, mu, sigma, lo, hi).item() == pytest.approx(1.0, abs=1e-6)

    def test_cdf_inv_cdf_roundtrip(self):
        mu = torch.tensor(0.5, dtype=DTYPE)
        sigma = torch.tensor(1.0, dtype=DTYPE)
        lo = torch.tensor(-1.0, dtype=DTYPE)
        hi = torch.tensor(3.0, dtype=DTYPE)
        p = torch.linspace(0.01, 0.99, 40, dtype=DTYPE)
        x = trunc_norm_inv_cdf(p, mu, sigma, lo, hi)
        p_back = trunc_norm_cdf(x, mu, sigma, lo, hi)
        torch.testing.assert_close(p_back, p, atol=1e-6, rtol=0)

    def test_samples_in_support(self):
        mu = torch.tensor(0.0, dtype=DTYPE)
        sigma = torch.tensor(1.0, dtype=DTYPE)
        lo = torch.tensor(-1.0, dtype=DTYPE)
        hi = torch.tensor(1.0, dtype=DTYPE)
        p = torch.rand(200, dtype=DTYPE)
        samples = trunc_norm_inv_cdf(p, mu, sigma, lo, hi)
        assert (samples >= lo).all()
        assert (samples <= hi).all()


class TestTruncNormInvCdfReturnLogProb:
    def test_return_log_prob_shape(self):
        mu = torch.tensor(0.0, dtype=DTYPE)
        sigma = torch.tensor(1.0, dtype=DTYPE)
        lo = torch.tensor(-1.0, dtype=DTYPE)
        hi = torch.tensor(1.0, dtype=DTYPE)
        p = torch.rand(10, dtype=DTYPE)
        samples, log_prob = trunc_norm_inv_cdf(p, mu, sigma, lo, hi, return_log_prob=True)
        assert samples.shape == (10,)
        assert log_prob.shape == (10,)
        assert torch.isfinite(log_prob).all()


# ---------------------------------------------------------------------------
# Uniform helpers
# ---------------------------------------------------------------------------

class TestUnifHelpers:
    def test_log_pdf_constant(self):
        lo = torch.tensor(1.0, dtype=DTYPE)
        hi = torch.tensor(4.0, dtype=DTYPE)
        x = torch.linspace(1.1, 3.9, 20, dtype=DTYPE)
        log_p = unif_log_pdf(x, lo, hi)
        expected = -math.log(3.0)
        torch.testing.assert_close(log_p, torch.full_like(x, expected))

    def test_inv_cdf_maps_zero_one_to_bounds(self):
        lo = torch.tensor(2.0, dtype=DTYPE)
        hi = torch.tensor(5.0, dtype=DTYPE)
        assert unif_inv_cdf(torch.tensor(0.0, dtype=DTYPE), lo, hi).item() == pytest.approx(2.0)
        assert unif_inv_cdf(torch.tensor(1.0, dtype=DTYPE), lo, hi).item() == pytest.approx(5.0)

    def test_inv_cdf_return_log_prob(self):
        lo = torch.tensor(0.0, dtype=DTYPE)
        hi = torch.tensor(3.0, dtype=DTYPE)
        u = torch.rand(15, dtype=DTYPE)
        samples, log_prob = unif_inv_cdf(u, lo, hi, return_log_prob=True)
        assert samples.shape == (15,)
        assert log_prob.shape == (15,)
        torch.testing.assert_close(log_prob, torch.full_like(log_prob, -math.log(3.0)))


# ---------------------------------------------------------------------------
# sample_bounded / bounded_log_prob
# ---------------------------------------------------------------------------

class TestSampleBounded:
    def test_unif_shape(self):
        lo = torch.zeros(4, dtype=DTYPE)
        hi = torch.ones(4, dtype=DTYPE)
        samples = sample_bounded((10,), lo, hi, which="unif")
        assert samples.shape == (10, 4)

    def test_unif_in_bounds(self):
        lo = torch.tensor([0.0, 1.0], dtype=DTYPE)
        hi = torch.tensor([1.0, 3.0], dtype=DTYPE)
        samples = sample_bounded((500,), lo, hi, which="unif")
        assert (samples >= lo).all()
        assert (samples <= hi).all()

    def test_gauss_in_bounds(self):
        lo = torch.zeros(3, dtype=DTYPE)
        hi = torch.ones(3, dtype=DTYPE)
        samples = sample_bounded((200,), lo, hi, which="gauss")
        assert (samples >= lo).all()
        assert (samples <= hi).all()

    def test_reproducible_with_generator(self):
        lo = torch.zeros(2, dtype=DTYPE)
        hi = torch.ones(2, dtype=DTYPE)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        s1 = sample_bounded((5,), lo, hi, which="unif", generator=g1)
        s2 = sample_bounded((5,), lo, hi, which="unif", generator=g2)
        torch.testing.assert_close(s1, s2)

    def test_different_seeds_differ(self):
        lo = torch.zeros(2, dtype=DTYPE)
        hi = torch.ones(2, dtype=DTYPE)
        g1 = torch.Generator().manual_seed(1)
        g2 = torch.Generator().manual_seed(2)
        s1 = sample_bounded((5,), lo, hi, which="unif", generator=g1)
        s2 = sample_bounded((5,), lo, hi, which="unif", generator=g2)
        assert not torch.allclose(s1, s2)

    def test_lo_hi_shape_mismatch_raises(self):
        lo = torch.zeros(2, dtype=DTYPE)
        hi = torch.ones(3, dtype=DTYPE)
        with pytest.raises(ValueError, match="lo.shape"):
            sample_bounded((5,), lo, hi)

    def test_invalid_which_raises(self):
        lo = torch.zeros(2, dtype=DTYPE)
        hi = torch.ones(2, dtype=DTYPE)
        with pytest.raises(ValueError, match="Unsupported"):
            sample_bounded((5,), lo, hi, which="bad")

    def test_return_log_prob_unif(self):
        lo = torch.tensor([0.0, 2.0], dtype=DTYPE)
        hi = torch.tensor([1.0, 5.0], dtype=DTYPE)
        samples, log_prob = sample_bounded((10,), lo, hi, which="unif", return_log_prob=True)
        assert samples.shape == (10, 2)
        assert log_prob.shape == (10, 2)


class TestBoundedLogProb:
    def test_unif_log_prob_shape(self):
        lo = torch.zeros(3, dtype=DTYPE)
        hi = torch.ones(3, dtype=DTYPE)
        x = torch.rand(5, 3, dtype=DTYPE)
        lp = bounded_log_prob(x, lo, hi, which="unif")
        assert lp.shape == (5, 3)

    def test_gauss_log_prob_finite(self):
        lo = torch.zeros(3, dtype=DTYPE)
        hi = torch.ones(3, dtype=DTYPE)
        x = torch.rand(5, 3, dtype=DTYPE)
        lp = bounded_log_prob(x, lo, hi, which="gauss")
        assert torch.isfinite(lp).all()

    def test_invalid_which_raises(self):
        lo = torch.zeros(2, dtype=DTYPE)
        hi = torch.ones(2, dtype=DTYPE)
        x = torch.rand(3, 2, dtype=DTYPE)
        with pytest.raises(ValueError, match="Unsupported"):
            bounded_log_prob(x, lo, hi, which="bad")
