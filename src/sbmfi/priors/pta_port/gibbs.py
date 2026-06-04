"""
Gibbs free energy estimation via equilibrator_api — no enkie, no R/rpy2.

Public API
----------
make_component_contribution()   — offline-safe initialisation
estimate_drg0_prime()           — standard reaction energies + covariance sqrt
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_RMSE_INF_KJ = 3000.0


def make_component_contribution(rmse_inf_kJ: float = _RMSE_INF_KJ):
    """Return an offline-safe ``ComponentContribution`` instance.

    Uses locally cached files (``cc_params.npz``, ``compounds.sqlite``) if
    they already exist, bypassing the Zenodo network check.  Falls back to
    the default online init if the cache is absent.
    """
    import pooch

    cache_dir   = Path(pooch.os_cache("equilibrator"))
    params_file = cache_dir / "cc_params.npz"
    sqlite_file = cache_dir / "compounds.sqlite"

    if params_file.exists() and sqlite_file.exists():
        try:
            import sqlalchemy
            from component_contribution.parameters import CCModelParameters
            from equilibrator_api import ComponentContribution, Q_
            from equilibrator_api.component_contribution import GibbsEnergyPredictor
            from equilibrator_cache import CompoundCache

            params    = CCModelParameters.from_npz(params_file)
            engine    = sqlalchemy.create_engine(f"sqlite:///{sqlite_file}")
            ccache    = CompoundCache(engine)
            predictor = GibbsEnergyPredictor(
                parameters=params,
                rmse_inf=Q_(rmse_inf_kJ, "kJ/mol"),
            )
            cc = ComponentContribution(ccache=ccache, predictor=predictor)
            logger.debug("ComponentContribution initialised from local cache.")
            return cc
        except Exception as exc:
            logger.warning(f"Offline init failed ({exc!r}); trying default init.")

    from equilibrator_api import ComponentContribution, Q_
    return ComponentContribution(rmse_inf=Q_(rmse_inf_kJ, "kJ/mol"))


def estimate_drg0_prime(
    reaction_formulas: list[str],
    pH: float = 7.0,
    ionic_strength_M: float = 0.25,
    T_kelvin: float = 310.15,
    pMg: float = 3.0,
    cc=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard reaction Gibbs energies and their covariance sqrt via eQuilibrator.

    Parameters
    ----------
    reaction_formulas : list of reaction strings in equilibrator_api formula
        format, e.g. ``['kegg:C00092 = kegg:C00085', 'kegg:C00111 = kegg:C00118']``.
        Compound namespaces supported: ``kegg:``, ``bigg.metabolite:``,
        ``chebi:``, ``metanetx.chemical:``.
    pH, ionic_strength_M, T_kelvin, pMg : physiological conditions.
    cc : optional pre-built ``ComponentContribution`` instance.

    Returns
    -------
    mean     : (n_rxn,) array  — ΔrG'° means in kJ/mol
    cov_sqrt : (n_rxn, q) array — T s.t. T @ T.T = Cov(ΔrG'°)  (kJ/mol)

    Notes
    -----
    The returned values are pH-corrected standard energies.  To obtain the
    full ΔrG' distribution (including concentration uncertainty), pass these
    into :func:`~pta_port.thermo_space.build_drg_basis_from_reactions`.
    """
    from equilibrator_api import Q_

    if cc is None:
        cc = make_component_contribution()

    cc.p_h            = Q_(pH,              "")
    cc.ionic_strength = Q_(ionic_strength_M, "M")
    cc.temperature    = Q_(T_kelvin,         "K")
    cc.p_mg           = Q_(pMg,             "")

    rxns = [cc.parse_reaction_formula(f) for f in reaction_formulas]
    mus, cov_sqrt_qty = cc.standard_dg_prime_multi(rxns, "fullrank")

    mean     = np.asarray(mus.m_as("kJ/mol"),           dtype=float)
    cov_sqrt = np.asarray(cov_sqrt_qty.m_as("kJ/mol"),  dtype=float)

    return mean, cov_sqrt
