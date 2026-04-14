"""
ThermoSamplingData: portable container for thermodynamic parameters extracted
from a pta TFSModel pickle.  Stores everything as plain numpy arrays so that
pta (Gurobi, Docker) is not required at runtime.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

R_GAS = 8.314472e-3  # kJ mol⁻¹ K⁻¹


@dataclass
class ThermoSamplingData:
    """All thermodynamic parameters needed for dG sampling, serialisable to HDF5."""

    # Reaction IDs (T-space, i.e. the reactions with thermodynamic constraints)
    T_reaction_ids: list[str]
    # Metabolite IDs (rows of S_constraints)
    metabolite_ids: list[str]

    T_kelvin: float

    # Metabolite-level parameters  (n_metabolites,)
    dfg0_prime_mean: np.ndarray
    log_conc_mean: np.ndarray
    # (n_metabolites, n_metabolites)
    dfg0_prime_cov_sqrt: np.ndarray
    log_conc_cov: np.ndarray
    # Stoichiometric matrix  (n_metabolites, n_T_reactions)
    S_constraints: np.ndarray

    # Flux-space bounds for all model reactions (used to determine sign constraints)
    F_reaction_ids: list[str]
    F_lb: np.ndarray   # (n_F_reactions,)
    F_ub: np.ndarray   # (n_F_reactions,)

    # ------------------------------------------------------------------ #
    #  Serialisation
    # ------------------------------------------------------------------ #

    def save_hdf5(self, path: Union[str, Path]) -> None:
        import tables
        path = Path(path)
        with tables.open_file(str(path), mode='w') as h5:
            h5.root._v_attrs['T_reaction_ids'] = self.T_reaction_ids
            h5.root._v_attrs['metabolite_ids'] = self.metabolite_ids
            h5.root._v_attrs['F_reaction_ids'] = self.F_reaction_ids
            h5.root._v_attrs['T_kelvin'] = float(self.T_kelvin)
            for name in (
                'dfg0_prime_mean', 'log_conc_mean',
                'dfg0_prime_cov_sqrt', 'log_conc_cov',
                'S_constraints', 'F_lb', 'F_ub',
            ):
                h5.create_array(h5.root, name, getattr(self, name))

    @classmethod
    def load_hdf5(cls, path: Union[str, Path]) -> ThermoSamplingData:
        import tables
        path = Path(path)
        with tables.open_file(str(path), mode='r') as h5:
            return cls(
                T_reaction_ids=list(h5.root._v_attrs['T_reaction_ids']),
                metabolite_ids=list(h5.root._v_attrs['metabolite_ids']),
                F_reaction_ids=list(h5.root._v_attrs['F_reaction_ids']),
                T_kelvin=float(h5.root._v_attrs['T_kelvin']),
                dfg0_prime_mean=h5.root.dfg0_prime_mean[:],
                log_conc_mean=h5.root.log_conc_mean[:],
                dfg0_prime_cov_sqrt=h5.root.dfg0_prime_cov_sqrt[:],
                log_conc_cov=h5.root.log_conc_cov[:],
                S_constraints=h5.root.S_constraints[:],
                F_lb=h5.root.F_lb[:],
                F_ub=h5.root.F_ub[:],
            )

    # ------------------------------------------------------------------ #
    #  One-time migration from a pta TFSModel pickle
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pta_pickle(cls, path: Union[str, Path]) -> ThermoSamplingData:
        """Load a pta TFSModel pickle and extract fields as plain numpy arrays.

        Requires pta to be installed (Docker workflow).  Run once, then call
        save_hdf5() so pta is no longer needed at runtime.
        """
        path = Path(path)
        with open(path, 'rb') as f:
            tfs_model = pickle.load(f)

        try:
            from pta.constants import R as R_pta
            R_val = float(R_pta.model)
        except ImportError:
            R_val = R_GAS

        T_val = float(tfs_model.T.parameters.T().model)

        return cls(
            T_reaction_ids=list(tfs_model.T.reaction_ids),
            metabolite_ids=list(tfs_model.T.S_constraints.index
                                if hasattr(tfs_model.T.S_constraints, 'index')
                                else range(tfs_model.T.S_constraints.shape[0])),
            T_kelvin=T_val,
            dfg0_prime_mean=np.asarray(tfs_model.T.dfg0_prime_mean.model, dtype=float),
            log_conc_mean=np.asarray(tfs_model.T.log_conc_mean.model, dtype=float),
            dfg0_prime_cov_sqrt=np.asarray(tfs_model.T._dfg0_prime_cov_sqrt.model, dtype=float),
            log_conc_cov=np.asarray(tfs_model.T.log_conc_cov.model, dtype=float),
            S_constraints=np.asarray(tfs_model.T.S_constraints, dtype=float),
            F_reaction_ids=list(tfs_model.F.reaction_ids),
            F_lb=np.asarray(tfs_model.F.lb, dtype=float),
            F_ub=np.asarray(tfs_model.F.ub, dtype=float),
        )

    # ------------------------------------------------------------------ #
    #  Derived quantities
    # ------------------------------------------------------------------ #

    def drg_mvn_params(self, epsilon: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, cov) of the marginal dG distribution over T_reaction_ids.

        Applies a small diagonal correction to ensure the covariance is PSD.
        """
        T = self.T_kelvin
        RT = R_GAS * T

        dfg0_cov = self.dfg0_prime_cov_sqrt @ self.dfg0_prime_cov_sqrt.T
        dfg_cov = dfg0_cov + self.log_conc_cov * RT ** 2
        dfg_mean = self.dfg0_prime_mean + self.log_conc_mean * RT

        S = self.S_constraints
        drg_mean = (S.T @ dfg_mean)
        drg_cov  = S.T @ dfg_cov @ S

        # Nudge towards PSD
        eye = np.eye(drg_cov.shape[0])
        while True:
            try:
                np.linalg.cholesky(drg_cov)
                break
            except np.linalg.LinAlgError:
                drg_cov += epsilon * eye
                epsilon *= 10

        return drg_mean, drg_cov
