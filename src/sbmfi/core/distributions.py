"""
Probability distribution utilities built on torch.distributions.Normal.

Provides normal and truncated-normal PDF, log-PDF, CDF, and inverse-CDF as
standalone functions so the rest of the codebase can call them without any
backend abstraction layer.
"""
import torch
from torch.distributions import Normal

# Unit normal reused for all standardised calculations.
# Created lazily so import doesn't fail if torch is unavailable at test-
# collection time, but in practice torch is always present.
_U: Normal | None = None


def _unit_normal() -> Normal:
    global _U
    if _U is None:
        _U = Normal(
            torch.tensor(0.0, dtype=torch.double),
            torch.tensor(1.0, dtype=torch.double),
        )
    return _U


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

def norm_log_pdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Log-probability density of N(mu, sigma) at x."""
    return Normal(mu, sigma).log_prob(x)


def norm_pdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Probability density of N(mu, sigma) at x."""
    return norm_log_pdf(x, mu, sigma).exp()


def norm_cdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """CDF of N(mu, sigma) at x."""
    return Normal(mu, sigma).cdf(x)


def norm_inv_cdf(p: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Inverse CDF (quantile function) of N(mu, sigma) at probability p."""
    return Normal(mu, sigma).icdf(p)


# ---------------------------------------------------------------------------
# Truncated normal distribution helpers
#
# All functions parameterise as:
#   x / p  – the point / probability to evaluate
#   mu     – mean of the untruncated normal
#   sigma  – standard deviation of the untruncated normal
#   lo     – lower truncation bound
#   hi     – upper truncation bound
#
# Implementation uses the unit normal Φ (CDF) and Φ⁻¹ (icdf) for stability.
# ---------------------------------------------------------------------------

def trunc_norm_log_pdf(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """Log-PDF of TruncatedNormal(mu, sigma, lo, hi) at x."""
    U = _unit_normal()
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    log_Z = torch.log(U.cdf(b) - U.cdf(a))
    return Normal(mu, sigma).log_prob(x) - log_Z


def trunc_norm_pdf(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """PDF of TruncatedNormal(mu, sigma, lo, hi) at x."""
    return trunc_norm_log_pdf(x, mu, sigma, lo, hi).exp()


def trunc_norm_cdf(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """CDF of TruncatedNormal(mu, sigma, lo, hi) at x."""
    U = _unit_normal()
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    Z = U.cdf(b) - U.cdf(a)
    return (U.cdf((x - mu) / sigma) - U.cdf(a)) / Z


def trunc_norm_inv_cdf(
    p: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    return_log_prob: bool = False,
):
    """
    Inverse CDF (quantile) of TruncatedNormal(mu, sigma, lo, hi).

    Parameters
    ----------
    p:
        Uniform samples in (0, 1).
    return_log_prob:
        If True, also return the log-probability of each sample.

    Returns
    -------
    samples : torch.Tensor
    log_prob : torch.Tensor  (only when return_log_prob=True)
    """
    U = _unit_normal()
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    samples = mu + sigma * U.icdf(U.cdf(a) + p * (U.cdf(b) - U.cdf(a)))
    if return_log_prob:
        log_prob = trunc_norm_log_pdf(samples, mu, sigma, lo, hi)
        return samples, log_prob
    return samples


# ---------------------------------------------------------------------------
# Uniform helpers (kept here for symmetry with the old LinAlg API)
# ---------------------------------------------------------------------------

def unif_log_pdf(
    x: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """Log-PDF of Uniform(lo, hi) at x."""
    return -torch.log(hi - lo).expand_as(x)


def unif_inv_cdf(
    u: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    return_log_prob: bool = False,
):
    """
    Inverse CDF of Uniform(lo, hi).

    Parameters
    ----------
    u:
        Uniform samples in (0, 1).
    return_log_prob:
        If True, also return the log-probability of each sample.
    """
    samples = lo + u * (hi - lo)
    if return_log_prob:
        log_prob = unif_log_pdf(samples, lo, hi)
        return samples, log_prob
    return samples


def sample_bounded(
    shape: tuple,
    lo: torch.Tensor,
    hi: torch.Tensor,
    mu: torch.Tensor | None = None,
    sigma: float = 0.1,
    which: str = "unif",
    generator: torch.Generator | None = None,
    return_log_prob: bool = False,
):
    """
    Draw samples from a bounded distribution (uniform or truncated normal).

    Parameters
    ----------
    shape:
        Batch shape (not including the ``lo``/``hi`` dimension).
    lo, hi:
        Lower and upper bounds; shape determines the last dimension.
    mu:
        Mean for the truncated normal (unused for 'unif').
    sigma:
        Std for the truncated normal.
    which:
        ``'unif'`` or ``'gauss'``.
    generator:
        Optional torch.Generator for reproducibility.
    """
    if lo.shape != hi.shape:
        raise ValueError(f"lo.shape {lo.shape} != hi.shape {hi.shape}")
    u = torch.rand((*shape, *lo.shape), dtype=lo.dtype, generator=generator)
    if which == "unif":
        return unif_inv_cdf(u, lo, hi, return_log_prob)
    elif which == "gauss":
        if mu is None:
            mu = torch.zeros_like(lo)
        sigma_t = torch.full_like(lo, sigma)
        return trunc_norm_inv_cdf(u, mu, sigma_t, lo, hi, return_log_prob)
    else:
        raise ValueError(f"Unsupported distribution '{which}'. Use 'unif' or 'gauss'.")


def bounded_log_prob(
    x: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    mu: torch.Tensor | None = None,
    sigma: float = 0.1,
    which: str = "unif",
    old_is_new: bool = False,
    k: int = 1,
) -> torch.Tensor:
    """Log-probability of x under a bounded distribution (uniform or truncated normal).

    When ``old_is_new=True`` the function computes pairwise upper-triangular log
    probabilities and returns a symmetric matrix (used in SMC proposal evaluation).
    """
    if which == "unif":
        pdf = lambda a, m: unif_log_pdf(a, lo, hi)
    elif which == "gauss":
        if mu is None:
            mu = torch.zeros_like(lo)
        sigma_t = torch.full_like(lo, sigma) if not isinstance(sigma, torch.Tensor) else sigma
        pdf = lambda a, m: trunc_norm_log_pdf(a, m, sigma_t, lo, hi)
    else:
        raise ValueError(f"Unsupported distribution '{which}'.")

    if not old_is_new:
        return pdf(x, mu)

    # Pairwise upper-triangular computation used when old_is_new=True.
    rows, cols = torch.triu_indices(x.shape[1], x.shape[1], offset=k, device=x.device)
    diags = torch.arange(x.shape[1], device=x.device)
    uptri_x = x[rows, cols]
    uptri_mu = mu[cols, rows]
    uptri_probs = pdf(uptri_x, uptri_mu)
    out_shape = tuple(max(a, b) for a, b in zip(x.shape, mu.shape))
    log_probs = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    log_probs[rows, cols] = uptri_probs
    log_probs = log_probs + log_probs.transpose(0, 1)
    log_probs[diags, diags] /= 2
    return log_probs
