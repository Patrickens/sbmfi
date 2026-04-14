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

try:
    import tables as _tables
except ImportError:
    _tables = None

try:
    from pta.constants import R as _R_pta
    _PTA_R_GAS = float(_R_pta.model)
except ImportError:
    _R_pta = None
    _PTA_R_GAS = None

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
    flux_S: np.ndarray
    F_reaction_ids: list[str]
    F_lb: np.ndarray   # (n_F_reactions,)
    F_ub: np.ndarray   # (n_F_reactions,)

    # Basis-space representation used by PTA's steady-state thermodynamic sampler.
    basis_to_drg_T: np.ndarray | None = None
    basis_to_drg_shift: np.ndarray | None = None
    constrained_reaction_ids: np.ndarray | None = None
    thermo_G: np.ndarray | None = None
    thermo_h: np.ndarray | None = None
    confidence_radius: float | None = None
    drg_epsilon: float = 1e-1

    # ------------------------------------------------------------------ #
    #  Serialisation
    # ------------------------------------------------------------------ #

    def save_hdf5(self, path: Union[str, Path]) -> None:
        if _tables is None:
            raise ImportError("tables (PyTables) is required for HDF5 serialisation")
        path = Path(path)
        with _tables.open_file(str(path), mode='w') as h5:
            h5.root._v_attrs['T_reaction_ids'] = self.T_reaction_ids
            h5.root._v_attrs['metabolite_ids'] = self.metabolite_ids
            h5.root._v_attrs['F_reaction_ids'] = self.F_reaction_ids
            h5.root._v_attrs['T_kelvin'] = float(self.T_kelvin)
            for name in (
                'dfg0_prime_mean', 'log_conc_mean',
                'dfg0_prime_cov_sqrt', 'log_conc_cov',
                'S_constraints', 'flux_S', 'F_lb', 'F_ub',
                'basis_to_drg_T', 'basis_to_drg_shift',
                'constrained_reaction_ids', 'thermo_G', 'thermo_h',
            ):
                value = getattr(self, name)
                if value is not None:
                    h5.create_array(h5.root, name, value)
            h5.root._v_attrs['drg_epsilon'] = float(self.drg_epsilon)
            if self.confidence_radius is not None:
                h5.root._v_attrs['confidence_radius'] = float(self.confidence_radius)

    @classmethod
    def load_hdf5(cls, path: Union[str, Path]) -> ThermoSamplingData:
        if _tables is None:
            raise ImportError("tables (PyTables) is required for HDF5 serialisation")
        path = Path(path)
        with _tables.open_file(str(path), mode='r') as h5:
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
                flux_S=h5.root.flux_S[:],
                F_lb=h5.root.F_lb[:],
                F_ub=h5.root.F_ub[:],
                basis_to_drg_T=getattr(h5.root, 'basis_to_drg_T', None)[:] if hasattr(h5.root, 'basis_to_drg_T') else None,
                basis_to_drg_shift=getattr(h5.root, 'basis_to_drg_shift', None)[:] if hasattr(h5.root, 'basis_to_drg_shift') else None,
                constrained_reaction_ids=getattr(h5.root, 'constrained_reaction_ids', None)[:] if hasattr(h5.root, 'constrained_reaction_ids') else None,
                thermo_G=getattr(h5.root, 'thermo_G', None)[:] if hasattr(h5.root, 'thermo_G') else None,
                thermo_h=getattr(h5.root, 'thermo_h', None)[:] if hasattr(h5.root, 'thermo_h') else None,
                confidence_radius=getattr(h5.root._v_attrs, 'confidence_radius', None),
                drg_epsilon=float(getattr(h5.root._v_attrs, 'drg_epsilon', 1e-1)),
            )

    # ------------------------------------------------------------------ #
    #  One-time migration from a pta TFSModel pickle
    # ------------------------------------------------------------------ #

    @classmethod
    def from_tfs_model(cls, tfs_model) -> ThermoSamplingData:
        """Extract portable sampling data from an in-memory pta TFSModel-like object."""
        T_constraints = tfs_model.T.S_constraints
        metabolite_ids = (
            list(T_constraints.index)
            if hasattr(T_constraints, 'index')
            else list(range(T_constraints.shape[0]))
        )
        basis_to_drg_T, basis_to_drg_shift = tfs_model.B.to_drg_transform
        basis_to_drg_shift = np.asarray(basis_to_drg_shift, dtype=float).reshape(-1, 1)
        constrained_reaction_ids = np.asarray(
            [tfs_model.F.reaction_ids.index(rid) for rid in tfs_model.T.reaction_ids],
            dtype=int,
        )

        reaction_idxs_T = list(range(len(tfs_model.T.reaction_ids)))
        reaction_idxs_F = [tfs_model.F.reaction_ids.index(rid) for rid in tfs_model.T.reaction_ids]
        only_forward_ids_T = [i for i in reaction_idxs_T if tfs_model.F.lb[reaction_idxs_F[i]] >= 0]
        only_backward_ids_T = [i for i in reaction_idxs_T if tfs_model.F.ub[reaction_idxs_F[i]] <= 0]
        thermo_G = np.vstack(
            (
                basis_to_drg_T[only_forward_ids_T, :],
                -basis_to_drg_T[only_backward_ids_T, :],
            )
        )
        thermo_h = -np.vstack(
            (
                basis_to_drg_shift[only_forward_ids_T, :],
                -basis_to_drg_shift[only_backward_ids_T, :],
            )
        )
        return cls(
            T_reaction_ids=list(tfs_model.T.reaction_ids),
            metabolite_ids=metabolite_ids,
            T_kelvin=float(tfs_model.T.parameters.T().model),
            dfg0_prime_mean=np.asarray(tfs_model.T.dfg0_prime_mean.model, dtype=float),
            log_conc_mean=np.asarray(tfs_model.T.log_conc_mean.model, dtype=float),
            dfg0_prime_cov_sqrt=np.asarray(tfs_model.T._dfg0_prime_cov_sqrt.model, dtype=float),
            log_conc_cov=np.asarray(tfs_model.T.log_conc_cov.model, dtype=float),
            S_constraints=np.asarray(T_constraints, dtype=float),
            flux_S=np.asarray(tfs_model.F.S, dtype=float),
            F_reaction_ids=list(tfs_model.F.reaction_ids),
            F_lb=np.asarray(tfs_model.F.lb, dtype=float),
            F_ub=np.asarray(tfs_model.F.ub, dtype=float),
            basis_to_drg_T=np.asarray(basis_to_drg_T, dtype=float),
            basis_to_drg_shift=basis_to_drg_shift,
            constrained_reaction_ids=constrained_reaction_ids,
            thermo_G=np.asarray(thermo_G, dtype=float),
            thermo_h=np.asarray(thermo_h, dtype=float),
            confidence_radius=float(tfs_model.confidence_radius),
            drg_epsilon=float(tfs_model.drg_epsilon),
        )

    @classmethod
    def from_pta_pickle(cls, path: Union[str, Path]) -> ThermoSamplingData:
        """Load a pta TFSModel pickle and extract fields as plain numpy arrays.

        Requires pta to be installed (Docker workflow).  Run once, then call
        save_hdf5() so pta is no longer needed at runtime.
        """
        path = Path(path)
        with open(path, 'rb') as f:
            tfs_model = pickle.load(f)

        return cls.from_tfs_model(tfs_model)

    # ------------------------------------------------------------------ #
    #  Derived quantities
    # ------------------------------------------------------------------ #

    def drg_mvn_params(self, epsilon: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, cov) of the marginal dG distribution over T_reaction_ids.

        Applies a small diagonal correction to ensure the covariance is PSD.
        """
        if self.basis_to_drg_T is not None and self.basis_to_drg_shift is not None:
            drg_mean = self.basis_to_drg_shift[:, 0].copy()
            drg_cov = self.basis_to_drg_T @ self.basis_to_drg_T.T
        else:
            RT = R_GAS * self.T_kelvin

            dfg0_cov = self.dfg0_prime_cov_sqrt @ self.dfg0_prime_cov_sqrt.T
            dfg_cov = dfg0_cov + self.log_conc_cov * RT ** 2
            dfg_mean = self.dfg0_prime_mean + self.log_conc_mean * RT

            S = self.S_constraints
            drg_mean = S.T @ dfg_mean
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

    @property
    def basis_dimensionality(self) -> int:
        if self.basis_to_drg_T is not None:
            return self.basis_to_drg_T.shape[1]
        return len(self.T_reaction_ids)
