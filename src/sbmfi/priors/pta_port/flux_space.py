"""Minimal flux-space container — no cobra dependency."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FluxSpaceParams:
    """Flux space description: stoichiometry and bounds.

    S   : (n_met, n_rxn) stoichiometric matrix
    lb  : (n_rxn,) lower bounds
    ub  : (n_rxn,) upper bounds
    reaction_ids  : length-n_rxn list of reaction identifiers
    metabolite_ids: length-n_met list of metabolite identifiers
    """

    S: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    reaction_ids: list[str]
    metabolite_ids: list[str]

    # ------------------------------------------------------------------ #
    #  Convenience constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_cobrapy_model(cls, model) -> "FluxSpaceParams":
        """Build from a cobrapy Model (cobra is an optional dependency)."""
        import numpy as np
        from cobra.util.array import create_stoichiometric_matrix

        S = create_stoichiometric_matrix(model)
        lb = np.array([r.lower_bound for r in model.reactions])
        ub = np.array([r.upper_bound for r in model.reactions])
        return cls(
            S=S,
            lb=lb,
            ub=ub,
            reaction_ids=[r.id for r in model.reactions],
            metabolite_ids=[m.id for m in model.metabolites],
        )
