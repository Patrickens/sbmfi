import psutil
import multiprocessing as mp
import contextlib, io
import math
import cvxpy as cp
import numpy as np
import pandas as pd
import warnings
import torch
import threading
import random
import scipy
import inspect
from torch.distributions.constraints import Constraint, _Dependent, _Interval
from PolyRound.api import PolyRoundApi
from sbmfi.core.model import LabellingModel, RatioMixin
from sbmfi.core.reaction import LabellingReaction
from sbmfi.core.linalg import LinAlg, TorchBackend, NumpyBackend
from sbmfi.core.polytopia import LabellingPolytope, FluxCoordinateMapper, \
    PolytopeSamplingModel, coordinate_hit_and_run_cpp, extract_labelling_polytope, thermo_2_net_polytope, \
    project_polytope, fast_FVA, rref_and_project, transform_polytope_keep_transform, \
    compute_polytope_halfspaces, V_representation, sample_polytope

from collections import OrderedDict
from typing import Iterable, Union, Optional, Dict
from torch.distributions import constraints
from torch.distributions import Distribution

# from pta.sampling.tfs import (
#     FreeEnergiesSamplingResult, sample_drg,
#     TFSModel, _find_point,
#     sample_fluxes_from_drg, PmoProblemPool
# )
# from pta.constants import R

# TODO another idea is to find a vertex representation of a polytope. Every flux vector can be represented as a
#   convex combination of these vertices, keyword: barycentric coordinates. The coefficients of the verteces lie
#   on a simplex and can thus be transformed through compositional data analysis.
#   Log-ratio transformed coordinates are unconstrained and therefore no bleeding!
#   https://www.researchgate.net/post/How_can_I_sample_points_from_a_convex_polytope_according_to_a_specified_probability_distribution
#   https://math.stackexchange.com/questions/4484178/computing-barycentric-coordinates-for-convex-n-dimensional-polytope-that-is-not

# TODO these should be priors that can be sampled such that we can use them with SBI without amortization
#   maybe move these things to priors.py in core

class _CannonicalPolytopeSupport(_Dependent):  #
    _VTOL = 1e-13
    def __init__(
            self,
            polytope: LabellingPolytope,
            validation_tol = _VTOL,
    ):
        if polytope.S is not None:
            raise ValueError('only for cannonical polytopes, Av <= b')

        self._constraint_id = polytope.A.columns
        self._A = torch.from_numpy(polytope.A.values)
        self._b = torch.atleast_2d(torch.from_numpy(polytope.b.values + validation_tol)).T
        super().__init__(is_discrete=False, event_dim=self._A.shape[1])

    def to(self, *args, **kwargs):
        self._A = self._A.to(*args, **kwargs)
        self._b = self._b.to(*args, **kwargs)

    @property
    def constraint_id(self) -> pd.Index:
        return self._constraint_id.copy()

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if value.dtype != self._A.dtype:
            value = value.to(self._A.dtype)
        vape    = value.shape
        viewlue = value.permute((-1, *range(value.ndim -1))).view(vape[-1], vape[:-1].numel())
        viewlue = viewlue[:self._A.shape[1], :]
        valid   = (self._A @ viewlue <= self._b).T
        return valid.view(*vape[:-1], self._A.shape[0])

    def euclidian_distance(self, value):
        # https://github.com/cvxgrp/cvxpylayers
        # https://github.com/cvxgrp/diffcp
        # https://github.com/locuslab/qpth
        #
        in_polytope = self.check(value).all(-1)
        not_in_pol = value[~in_polytope, :]  # values that are not in the polytope for which to compute distance
        raise NotImplementedError


class _BasePrior(Distribution):
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            cache_size: int = 20000,
            num_processes: int = 0,
    ):
        # prior sampling variables
        self._ic = cache_size  # current index in the cache
        if isinstance(model, LabellingModel):
            kwargs = {}
            linalg = LinAlg('torch', seed=model._la._backwargs['seed'])
            if model._fcm is not None:
                kwargs = dict(
                    kernel_basis=model._fcm._sampler.kernel_basis,
                    basis_coordinates=model._fcm._sampler.basis_coordinates,
                    logit_xch_fluxes=model._fcm.logit_xch_fluxes,
                    free_reaction_id=model.labelling_reactions.list_attr('id'),
                )
            model = FluxCoordinateMapper(model, linalg=linalg, **kwargs)

        self._fcm = model

        if num_processes < 0:
            num_processes = psutil.cpu_count(logical=False)
        self._num_processes = num_processes
        if num_processes > 0:
            self._mp_pool = self._get_mp_pool(num_processes)

        self._cache_fill_kwargs = {'n': cache_size}
        self._sample_shape = None

        self._theta_cache = torch.zeros((cache_size, self.n_theta), dtype=torch.double) # cache to store dependent variables
        self._flux_cache  = torch.zeros((cache_size, self.n_fluxes), dtype=torch.double)  # cache to store corresponding fluxes
        # NOTE passing validate_args={} will trigger support checking
        Distribution.__init__(self, event_shape=torch.Size((self.n_theta,)), validate_args={})

    def _close_pool(self):
        self._mp_pool.close()
        self._mp_pool.join()

    def __getstate__(self):
        dct = self.__dict__
        if self._num_processes > 0:
            self._close_pool()
            dct.pop['_mp_pool']
        return dct

    @property
    def n_theta(self):
        # number of theta elements, depends on coordinate system for fluxes or number of ratios
        raise NotImplementedError

    @property
    def n_fluxes(self):
        return self._fcm._F.A.shape[1]

    @property
    def theta_id(self):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        # this is useful for when we would like to sample on GPU
        raise NotImplementedError

    def _fill_caches(self, n=20000, **kwargs):
        # this function fills the cache with dependent variables in self._cache and fluxes in self._flux_cache
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size([])):
        # NB this always returns free fluxes in the thermodynamic coordinate system
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        self._sample_shape = sample_shape

        n = sample_shape.numel()
        if n > self._theta_cache.shape[0]:
            self._cache_fill_kwargs['n'] = n

        jc = self._ic + n
        if jc > self._theta_cache.shape[0]:
            self._fill_caches(**self._cache_fill_kwargs)
            self._ic = 0
            jc = n
        sample = self._theta_cache[self._ic: jc].view(self._extended_shape(sample_shape))
        self._ic = jc
        return sample

    def sample_fluxes(self):
        if self._sample_shape is None:
            raise ValueError('first use rsample, this function returns the corresponding labelling fluxes')
        n = self._sample_shape.numel()
        return self._flux_cache[self._ic - n: self._ic].view(self._sample_shape + self._batch_shape + self.n_fluxes)

    def sample_dataframes(self, n=10000, **kwargs) -> (pd.DataFrame, pd.DataFrame):
        self._fill_caches(n=n, **kwargs)
        return pd.DataFrame(self._theta_cache.numpy(), columns=self.theta_id),\
               pd.DataFrame(self._flux_cache.numpy(), columns=self._fcm._F.A.columns)

    def _run_tasks(self, tasks, fn=sample_polytope, break_i=-1, close_pool=True, format=True, scramble=True):
        if self._num_processes > 0:
            if hasattr(self._mp_pool, '_state') and self._mp_pool._state == 'CLOSE':
                self._mp_pool = mp.Pool(self._num_processes)
            results = self._mp_pool.starmap(fn, tasks)
            if close_pool:
                self._mp_pool.close()
        else:
            results = []
            for i, task in enumerate(tasks):
                results.append(fn(**task))
                if (break_i > -1) and (i > break_i):
                    break

        if format:
            net_fluxes = self._fcm._la.cat([torch.as_tensor(r['fluxes']) for r in results])
            net_basis_samples = self._fcm._la.cat([torch.as_tensor(r['basis_samples']) for r in results])
            theta, fluxes = self._fcm.append_xch_flux_samples(net_fluxes, net_basis_samples)
            if 'new_basis_points' in results[0]:
                self._basis_points = results[0]['new_basis_points']
            if scramble:
                scramble_indices = self._fcm._la.randperm(fluxes.shape[0])
                fluxes = fluxes[scramble_indices]
                theta = theta[scramble_indices]
            self._flux_cache = fluxes
            self._theta_cache = theta
        else:
            return results


class _FluxPrior(_BasePrior):
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            cache_size: int = 20000,
            num_processes: int = 0,
    ):
        self._basis_points = None
        super(_FluxPrior, self).__init__(model, cache_size, num_processes)

    # NB event_dim=1 means that the right-most dimension defines an event!
    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        supp = _CannonicalPolytopeSupport(polytope=self._fcm.make_theta_polytope())
        supp.to(dtype=torch.float32)  # TODO maybe pass dtype as a kwarg or maybe always enforce float32
        return supp

    @property
    def theta_id(self) -> pd.Index:
        return self._fcm.theta_id

    @property
    def n_theta(self):
        return len(self._fcm.theta_id)


class UniFluxPrior(_FluxPrior):
    def __init__(
            self,
            model,
            cache_size: int = 20000,
            **kwargs,
    ):
        super(UniFluxPrior, self).__init__(model, cache_size, 0)

    def _fill_caches(self, n=20000, **kwargs):
        # this one is without pool always
        task = dict(
            model=self._fcm._sampler, initial_points=self._basis_points, n=n, n_burn=200,
            new_basis_points=True, return_basis_samples=True
        )
        self._run_tasks(tasks=[task], format=True, scramble=True)

    def log_prob(self, value):
        # multiply constant by distance?
        # log prob for uniform distribution is log(1 / vol(polytope))
        # for non-uniform xch flux distribution, return log(1 / vol(net_polytope)) + log_prob(xch_flux)
        if self._validate_args:
            self._validate_sample(value)
        # place-holder until we can compute polytope volumes
        return torch.zeros((*value.shape[:-1], 1))


# def get_initial_points(self: TFSModel, num_points: int) -> np.ndarray:
#
#     # NB this has to be called with processes = 1 due to memory errors
#     pool = PmoProblemPool(1, *self._pmo_args)
#
#     # Find candidate optimization direactions.
#     reaction_idxs_T = list(range(len(self.T.reaction_ids)))
#     reaction_idxs_F = [self.F.reaction_ids.index(id) for id in self.T.reaction_ids]
#     only_forward_ids_T = [
#         i for i in reaction_idxs_T if self.F.lb[reaction_idxs_F[i]] >= 0
#     ]
#     only_backward_ids_T = [
#         i for i in reaction_idxs_T if self.F.ub[reaction_idxs_F[i]] <= 0
#     ]
#     reversible_ids_T = [
#         i
#         for i in reaction_idxs_T
#         if self.F.lb[reaction_idxs_F[i]] < 0 and self.F.ub[reaction_idxs_F[i]] > 0
#     ]
#
#     reversible_dirs = [(i, -1) for i in reversible_ids_T] + [
#         (i, 1) for i in reversible_ids_T
#     ]
#     irreversible_dirs = [(i, -1) for i in only_backward_ids_T] + [
#         (i, 1) for i in only_forward_ids_T
#     ]
#
#     # Select optimization directions, giving precedence to the reversible reactions.
#     if num_points >= len(reversible_dirs):
#         directions = reversible_dirs
#         directions_pool = irreversible_dirs
#         to_sample = min(num_points - len(reversible_dirs), len(irreversible_dirs))
#     else:
#         directions = []
#         directions_pool = reversible_dirs
#         to_sample = min(num_points, len(reversible_dirs))
#     optimization_directions = directions + random.sample(directions_pool, to_sample)
#
#     # Run the optimizations in the pool.
#     initial_points = pool.map(_find_point, optimization_directions)
#     assert all(p is not None for p in initial_points), (
#         "One or more initial points could not be found. This could be due to "
#         "an overconstrained model or numerical inaccuracies."
#     )
#     points_array = np.hstack(initial_points)
#     pool.close()
#
#     return points_array
# TFSModel.get_initial_points = get_initial_points


def sampling_tasks(
        polytope: LabellingPolytope, # this is a basis polytope that will be modified using b_constraint_df and A_constraint_dct
        transform_type = 'svd',
        basis_coordinates = 'rounded',
        to_basis_fn = None,
        return_basis_samples=False,
        counts: Union[int, pd.Series] = 20,  # number of fluxes to sample from yielded polytope
        A_constraint_df: pd.DataFrame = None, # this should have a multiindex with level 1 being names, and 2 being constraint_names
        S_constraint_df: pd.DataFrame = None, # this should have a multiindex with level 1 being names, and 2 being constraint_names
        b_constraint_df: pd.DataFrame = None,
        return_kwargs: bool = False,
        n_burn: int = 100,
        thinning_factor: int = 5,
        n_chains: int = 4,
        sampling_function = sample_polytope,
        linalg: LinAlg = None,
):
    if (A_constraint_df is None) and (S_constraint_df is None) and (b_constraint_df is None):
        raise ValueError
    if (S_constraint_df is not None) and (polytope.S is None):
        raise ValueError

    for thing in [A_constraint_df, S_constraint_df, b_constraint_df]:
        if thing is not None:
            index = thing.index
            if isinstance(index, pd.MultiIndex):  # for S and A constraint dfs
                index = index.levels[0]

    if isinstance(counts, int):
        counts = pd.Series(counts, index=index)
    if A_constraint_df is None:
        A_constraint_df = pd.DataFrame(None, index=index)
    if S_constraint_df is None:
        S_constraint_df = pd.DataFrame(None, index=index)
    if b_constraint_df is None:
        b_constraint_df = pd.DataFrame(None, index=index)

    func_kwargs = inspect.getfullargspec(sampling_function).args
    for name, row in b_constraint_df.iterrows():
        pol = polytope.copy()
        if row.size > 0:
            pol.b.loc[b_constraint_df.columns] = row

        A = A_constraint_df.loc[name]
        if A.size > 0:
            pol.A.loc[A.index, A.columns] = A

        S = S_constraint_df.loc[name]
        if S.size > 0:
            pol.S.loc[S.index, S.columns] = S

        if counts is None:
            count = name
        else:
            count = counts.loc[name]

        kwargs = {  # these are the kwargs to coordinate_hit_and_run
            'model': pol,
            'n': count,
            'n_burn': n_burn,
            'initial_points': None,
            'thinning_factor': thinning_factor,
            'n_chains': n_chains,
            'new_basis_points': False,
            'return_basis_samples': return_basis_samples,
            'transform_type': transform_type,
            'basis_coordinates': basis_coordinates,
            'to_basis_fn': to_basis_fn,
            'linalg': linalg,
            'return_vsm': False,
        }

        kwargs = {key: kwargs.get(key) for key in func_kwargs}

        if return_kwargs:
            yield kwargs
        else:
            yield tuple(kwargs.values())  # because cannot pickle dict_values...


# class ThermoPrior(_FluxPrior):
#     def __init__(
#             self,
#             model: FluxCoordinateMapper,
#             tfs_model: TFSModel,
#             coordinates='thermo',
#             cache_size: int = 20000,
#             num_processes: int = 0,
#     ):
#         super(ThermoPrior, self).__init__(model, cache_size, num_processes)
#         self._coords = coordinates
#         self._tfs_model = tfs_model
#         self._fulabel_pol = None
#         self._drg_mvn = self.extract_drg_mvn(tfs_model)
#         self._thermo_basis_points = None
#         self._orthant_volumes = {}
#
#     @staticmethod
#     def reset_rhos(model):
#         reset_rhos = []
#         for r in model.reactions:
#             if isinstance(r, LabellingReaction) and (r.rho_max > 0.0) and (r.rho_max < r._RHO_MAX):
#                 # this is necessary to make sure the polytope of the support has the correct bounds
#                 r.rho_min = 0.0
#                 r.rho_max = r._RHO_MAX
#                 reset_rhos.append(r.id)
#         if len(reset_rhos) > 0:
#             print(f'The rho_bounds of reactions: {reset_rhos} has been set to ({0.0, LabellingReaction._RHO_MAX})')
#         return model
#
#     def extract_drg_mvn(self, tfs_model: TFSModel, epsilon=1e-12, as_tensor=False) -> Distribution:
#         # NB extract mv parameters, this is a shit show due to pint and pickling...
#         tfs_reaction_ids = pd.Index(tfs_model.T.reaction_ids)
#         indices = np.array([tfs_reaction_ids.get_loc(rid) for rid in self._fcm.fwd_id])
#
#         T = tfs_model.T.parameters.T().model
#
#         dfg0_prime_mean = tfs_model.T.dfg0_prime_mean.model
#         log_conc_mean = tfs_model.T.log_conc_mean.model
#         dfg_prime_mean = dfg0_prime_mean + log_conc_mean * R.model * T
#         S_constraints = tfs_model.T.S_constraints
#         drg_prime_mean = (S_constraints.T @ dfg_prime_mean)[indices]
#
#         dfg0_prime_cov_sqrt = tfs_model.T._dfg0_prime_cov_sqrt.model
#         dfg0_prime_cov = dfg0_prime_cov_sqrt @ dfg0_prime_cov_sqrt.T
#         dfg_prime_cov = dfg0_prime_cov + tfs_model.T.log_conc_cov.model * (R.model * T) ** 2
#         drg_prime_cov = (S_constraints.T @ dfg_prime_cov @ S_constraints)[:, indices][indices, :]
#
#         psd = False
#         eye = np.eye(drg_prime_cov.shape[0])
#         tot_eps = 0.0
#         while not psd:
#             try:
#                 np.linalg.cholesky(drg_prime_cov)
#                 psd = True
#                 if self._fcm._pr_settings.verbose:
#                     print(f'total correction epsilon to make matrix PSD: {tot_eps}')
#             except:
#                 # TODO this is a fancier way of fixing non-PSD, but I dont get it: https://stackoverflow.com/a/66902455
#                 # u, s, v = np.linalg.svd(drg_prime_cov)
#                 # s[s < 1e-12] += epsilon
#                 # drg_prime_cov = u @ (np.diag(s)) @ v.T
#                 tot_eps += epsilon
#                 drg_prime_cov += epsilon * eye
#
#         if as_tensor:
#             return torch.distributions.MultivariateNormal(
#                 # TODO this covariance matrix is degenerate, this is why we need to work in the other basis...
#                 loc=torch.as_tensor(drg_prime_mean, dtype=torch.double).squeeze(),
#                 covariance_matrix=torch.as_tensor(drg_prime_cov, dtype=torch.double)
#             )
#         else:
#             return scipy.stats.multivariate_normal(mean=drg_prime_mean.squeeze(), cov=drg_prime_cov)
#
#     def _compute_orthant_volume(self, thermo_fluxes):
#         pass
#         # orthants = thermo_fluxes.loc[:, fcm.fwd_id] > 0
#         # orthants.index = orthants.apply(lambda row: hash(tuple(row)), raw=True, axis=1)
#
#     def log_prob(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         if self._coords == 'labelling':
#             #
#             raise NotImplementedError(
#                 'this would mean that we need to compute the volume of a thermo_constrained_label_pol, '
#                 'which is a biatch due to large number of dimensions')
#         # this one will be very difficult to implement, since we need to convert exchange fluxes to dG and evaluate those
#         thermo_fluxes = self.map_theta_2_fluxes(value, return_thermo=True)
#         drg = self._fcm.compute_dgibbsr(thermo_fluxes)
#         orhtant_vols = self._compute_orthant_volume(thermo_fluxes)
#
#     def _generate_thermo_constrained_label_pol(self, abs_xch_flux_tol=0.0):
#         pol = self._fcm._F.copy()
#
#         if abs_xch_flux_tol == 0.0:
#             S_df = pd.DataFrame(
#                 0.0,
#                 index=(self._fcm.fwd_id + '_rho'),
#                 columns=pol.A.columns,
#             )
#             rho_rev_idx = np.zeros(S_df.shape, dtype=bool)
#             rho_rev_idx[
#                 [S_df.index.get_loc(key) for key in self._fcm.fwd_id + '_rho'],
#                 [S_df.columns.get_loc(key) for key in self._fcm.fwd_id + '_rev']
#             ] = True
#             S_df[rho_rev_idx] = -1.0
#             rho_constraints = pol.S.index.str.contains('_rho')
#             pol.S = pd.concat([pol.S.loc[~rho_constraints], S_df], axis=0).sort_index(axis=0)
#             pol.h = pd.concat([pol.h.loc[~rho_constraints], pd.Series(0.0, index=S_df.index)]).loc[pol.S.index]
#         else:
#             A_df = pd.DataFrame(
#                 0.0,
#                 index=(self._fcm.fwd_id + '_rho_min|lb').union((self._fcm.fwd_id + '_rho_max|ub')),
#                 columns=pol.A.columns,
#             )
#             rho_rev_idx = np.zeros(A_df.shape, dtype=bool)
#             rho_rev_idx[
#                 [A_df.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_min|lb'],
#                 [A_df.columns.get_loc(key) for key in self._fcm.fwd_id + '_rev']
#             ] = True
#             A_df[rho_rev_idx] = -1.0
#             rho_rev_idx[:] = False
#             rho_rev_idx[
#                 [A_df.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_max|ub'],
#                 [A_df.columns.get_loc(key) for key in self._fcm.fwd_id + '_rev']
#             ] = True
#             A_df[rho_rev_idx] = 1.0
#
#             rho_constraints = pol.A.index.str.contains('_rho')
#             pol.A = pd.concat([pol.A.loc[~rho_constraints], A_df], axis=0).sort_index(axis=0)
#             pol.b = pd.concat([pol.b.loc[~rho_constraints], pd.Series(0.0, index=A_df.index)]).loc[pol.A.index]
#         return pol
#
#     @staticmethod
#     def _sample_drg_suppress_output(
#             tfs_model: TFSModel,
#             n: int = 20000,
#             num_chains: int = 4,
#             thermo_basis_points: np.array = None
#     ):
#         if thermo_basis_points is None:
#             thermo_basis_points = tfs_model.get_initial_points(num_chains)
#
#         num_chains = thermo_basis_points.shape[1]
#         with contextlib.redirect_stdout(io.StringIO()):
#             result = sample_drg(
#                 tfs_model,
#                 initial_points=thermo_basis_points,
#                 num_direction_samples=n,
#                 num_samples=n,
#                 max_psrf=1.0 + 1e-12,
#                 num_chains=num_chains,
#                 max_steps = 32 * n,
#                 num_initial_steps = n * 2,
#                 # max_threads=num_chains,
#             )
#         new_points_idx = np.random.choice(n, num_chains)
#         new_thermo_basis_points = result.basis_samples.values[new_points_idx, :].T
#         return result, new_thermo_basis_points
#
#     def _make_b_constraint_df(self, orthants):
#         forward = orthants.astype(int)
#         reverse = (~orthants).astype(int)
#         forward.columns += '|ub'
#         reverse.columns += '|lb'
#
#         b = self._fcm._Fn.b.copy()
#
#         ub = forward * b.loc[forward.columns]
#         lb = reverse * b.loc[reverse.columns]
#
#         # NB this dataframe contains all the bounds for the b vector
#         return pd.concat([lb, ub], axis=1).sort_index(axis=1)
#
#     def _make_Fn_tasks(self, drg_result: FreeEnergiesSamplingResult, n_flux=1):
#         drg_samples = drg_result.samples.loc[:, self._fcm.fwd_id]
#         drg_xch_fluxes = self._fcm.compute_xch_fluxes(dgibbsr=drg_samples)
#         drg_orthants = drg_samples < 0.0
#
#         orthants = (drg_orthants).value_counts().reset_index().rename({0: 'counts'}, axis=1).set_index('counts')
#         orthants = orthants.loc[orthants.index > 1]
#         orthants.index *= n_flux  # this means we take subsamples of the net space
#
#         b_constraint_df = self._make_b_constraint_df(orthants)
#         task_generator = sampling_tasks(  # transform_type=self._fcm.transform_type, basis_coordinates=self._fcm.basis_coordinates
#             self._fcm._Fn, counts=None, to_basis_fn=self._fcm.to_basis_fn,
#             b_constraint_df=b_constraint_df, return_basis_samples=True,
#             return_kwargs=self._num_processes == 0,
#         )
#         return dict(
#             orthants=orthants, drg_orthants=drg_orthants, b_constraint_df=b_constraint_df,
#             drg_xch_fluxes=drg_xch_fluxes, sampling_task_generator=task_generator
#         )
#
#     def _make_S_constraint_df(self, drg_result: FreeEnergiesSamplingResult, S: pd.DataFrame):
#         drg_samples = drg_result.samples.loc[:, self._fcm.fwd_id]
#         drg_xch_fluxes = self._fcm.compute_xch_fluxes(dgibbsr=drg_samples)
#         S_template = pd.DataFrame(0.0, index=S.index[S.index.str.contains('rho')], columns=self._fcm.fwd_id)
#         drg_xch_fluxes.columns += '_rho'
#         drg_xch_fluxes = drg_xch_fluxes.clip(0.0, LabellingReaction._RHO_MAX)
#         reverse_dir = (drg_samples > 0.0).values
#         drg_xch_fluxes.values[reverse_dir] = (1.0 / drg_xch_fluxes).values[reverse_dir]
#         rho_idx = np.zeros(S_template.shape, dtype=bool)
#         rho_idx[
#             [S_template.index.get_loc(key) for key in self._fcm.fwd_id + '_rho'],
#             [S_template.columns.get_loc(key) for key in self._fcm.fwd_id]
#         ] = True
#
#         S_constraints = OrderedDict()
#         for i, name in enumerate(drg_samples.index):
#             S = np.zeros(S_template.shape).T
#             S[rho_idx.T] = drg_xch_fluxes.iloc[i].values
#             S_constraints[name] = pd.DataFrame(S.T, index=S_template.index, columns=S_template.columns)
#
#         return pd.concat(S_constraints.values(), keys=drg_samples.index)
#
#     def _make_A_constraint_df(self, drg_result: FreeEnergiesSamplingResult, A: pd.DataFrame, abs_xch_flux_tol=0.05):
#         drg_samples = drg_result.samples.loc[:, self._fcm.fwd_id]
#         drg_xch_fluxes = self._fcm.compute_xch_fluxes(dgibbsr=drg_samples)
#
#         A_template = pd.DataFrame(0.0, index=A.index[A.index.str.contains('rho')], columns=self._fcm.fwd_id)
#
#         xch_fluxes_max = drg_xch_fluxes + abs_xch_flux_tol / 2
#         drg_xch_fluxes -= abs_xch_flux_tol / 2
#
#         drg_xch_fluxes.columns += '_rho_min|lb'
#         xch_fluxes_max.columns += '_rho_max|ub'
#
#         drg_xch_fluxes = drg_xch_fluxes.clip(0.0, LabellingReaction._RHO_MAX)
#         xch_fluxes_max = xch_fluxes_max.clip(0.0, LabellingReaction._RHO_MAX)
#
#         reverse_dir = (drg_samples > 0.0).values
#         drg_xch_fluxes.values[reverse_dir] = (1.0 / drg_xch_fluxes).values[reverse_dir]
#         xch_fluxes_max.values[reverse_dir] = (1.0 / xch_fluxes_max).values[reverse_dir]
#
#         rho_fwd_min_idx = np.zeros(A_template.shape, dtype=bool)
#         rho_fwd_min_idx[
#             [A_template.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_min|lb'],
#             [A_template.columns.get_loc(key) for key in self._fcm.fwd_id]
#         ] = True
#         rho_fwd_max_idx = np.zeros(A_template.shape, dtype=bool)
#         rho_fwd_max_idx[
#             [A_template.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_max|ub'],
#             [A_template.columns.get_loc(key) for key in self._fcm.fwd_id]
#         ] = True
#
#         A_constraints = OrderedDict()
#         for i, name in enumerate(drg_samples.index):
#             A = np.zeros(A_template.shape).T
#             A[rho_fwd_min_idx.T] = drg_xch_fluxes.iloc[i].values
#             A[rho_fwd_max_idx.T] = -xch_fluxes_max.iloc[i].values
#             A = pd.DataFrame(A.T, index=A_template.index, columns=A_template.columns)
#             # NOTE this is essential, otherwise the bounds conflict for rho > 1.0
#             A[A < -1.0] = -A[A < -1.0].values
#             A[A > 1.0] = -A[A > 1.0].values
#             A_constraints[name] = A
#
#         return pd.concat(A_constraints.values(), keys=drg_samples.index)
#
#     def _make_F_tasks(self, drg_result:FreeEnergiesSamplingResult, n_flux=1, abs_xch_flux_tol=0.05):
#         pol = self._generate_thermo_constrained_label_pol(abs_xch_flux_tol)
#         kwargs = {}
#         if abs_xch_flux_tol == 0.0:
#             kwargs['S_constraint_df'] = self._make_S_constraint_df(drg_result, pol.S)
#         else:
#             # TODO not sure this is correct!
#             kwargs['A_constraint_df'] = self._make_A_constraint_df(drg_result, pol.A, abs_xch_flux_tol)
#
#         return sampling_tasks(
#             pol, counts=n_flux, to_basis_fn=None, return_basis_samples=False,
#             return_kwargs=self._num_processes == 0, **kwargs
#         )
#
#     def _fill_caches(
#             self, n=20000, n_flux=1, drg_result: FreeEnergiesSamplingResult = None,
#             abs_xch_flux_tol=0.00, break_i=-1, close_pool=True,
#     ):
#         if drg_result is None:
#             drg_result, self._thermo_basis_points = self._sample_drg_suppress_output(
#                 self._tfs_model, n, thermo_basis_points=self._thermo_basis_points
#             )
#         else:
#             new_points_idx = np.random.choice(n, self._num_processes)
#             self._thermo_basis_points = drg_result.basis_samples.values[new_points_idx, :].T
#
#         if self._coords == 'thermo':
#             tasks_dct = self._make_Fn_tasks(drg_result, n_flux)
#             orthants = tasks_dct['orthants']
#             drg_orthants = tasks_dct['drg_orthants']
#             drg_xch_fluxes = tasks_dct['drg_xch_fluxes']
#             sampling_task_generator = tasks_dct['sampling_task_generator']
#
#             results = self._run_tasks(sampling_task_generator, break_i=break_i, close_pool=close_pool)
#             net_fluxes = pd.concat([r['fluxes'] for r in results], ignore_index=True).loc[:, self._fcm._Fn.A.columns]
#             xch_cols = self._fcm.fwd_id + '_xch'
#             xch_fluxes = pd.DataFrame(np.nan, index=net_fluxes.index, columns=xch_cols)
#
#             flux_orthants = net_fluxes.loc[:, orthants.columns] > 0.0
#             drg_orthants = drg_orthants.loc[:, orthants.columns]
#
#             for (count, orthant), res in zip(orthants.iterrows(), results):
#                 self._orthant_volumes[hash(tuple(orthant))] = res['log_det_E']
#                 which_drg = (drg_orthants == orthant).all(1)
#                 which_flux = (flux_orthants == orthant).all(1)
#                 if which_flux.sum() < which_drg.sum() * n_flux:
#                     # NOTE these orthants have 0 hypervolume I guess, thus flux sampling fails
#                     raise ValueError('some orthant was not sampled correctly, make sure that the Fluxspace '
#                                      'of self._tfs_model and self._model are the same!')
#
#                 # print(which_flux.sum(), which_drg.sum() * n_flux, count, which_flux.sum() < which_drg.sum() * n_flux)
#                 xch_fluxes.loc[which_flux, xch_cols] = pd.concat(
#                     [drg_xch_fluxes.loc[which_drg, :], ] * n_flux, axis=0
#                 ).values
#
#             net_basis_samples = pd.concat([r['basis_samples'] for r in results], ignore_index=True).loc[:, self._fcm.net_basis_id]
#             theta, fluxes = self._fcm.append_xch_flux_samples(net_fluxes, net_basis_samples, xch_fluxes)
#             # self.ding = self._fcm.map_theta_2_fluxes(self.theta)
#
#         elif self._coords == 'labelling':
#             if self._fcm.include_cofactors:
#                 raise ValueError('mapping from fluxes to thermo wont work if cofactors are included')
#             sampling_task_generator = self._make_F_tasks(drg_result, n_flux, abs_xch_flux_tol)
#             results = self._run_tasks(sampling_task_generator, break_i=break_i, close_pool=close_pool)
#             fluxes = pd.concat([r['fluxes'] for r in results], ignore_index=True)
#             theta = self._fcm.map_fluxes_2_theta(fluxes)
#
#         self._flux_cache  = torch.as_tensor(fluxes.values, dtype=torch.double)
#         self._theta_cache = torch.as_tensor(theta.values, dtype=torch.double)


class _RatioSupport(Constraint):
    def __init__(
            self,
            fcm: FluxCoordinateMapper,
            ratio_repo: dict,  # TODO need to figure out condensation and such
            ratio_tol: float = 0.0,
            min_denom_sum: float = 0.0001,
            project=False,
    ):
        # TODO project polytope on ratio-reactions!
        self._ratol = ratio_tol
        self._mds = min_denom_sum

        polytope = fcm._Fn
        normalize = fcm._sampler.kernel_basis != 'rref'
        simpol = PolyRoundApi.simplify_polytope(polytope, settings=fcm._sampler._pr_settings, normalize=normalize)
        polytope = LabellingPolytope.from_Polytope(simpol, polytope)
        if project:
            ratio_reactions = []
            for ratio_id, num_den in ratio_repo.items():
                ratio_reactions.extend(list(num_den['denominator'].keys()))
            ratio_reactions = pd.Index(set(ratio_reactions))
            P = pd.DataFrame(0.0, index=ratio_reactions, columns=polytope.A.columns)
            P.loc[ratio_reactions, ratio_reactions] = np.eye(len(ratio_reactions))
            polytope = project_polytope(polytope, P, number_type='float')
            polytope._objective = {polytope.A.columns[0]: 1.0}

        if len(polytope.objective) == 0:
            # no objective is set; automatically set one or raise error
            raise ValueError('set an objective')

        n_vars = polytope.A.shape[1]  # number of fluxes
        v = cp.Variable(n_vars, name='fluxes')

        locations = [polytope.A.columns.get_loc(reaction_id) for reaction_id in polytope.objective.keys()]
        c = np.zeros(n_vars)
        c[locations] = list(polytope.objective.values())  # what to optimize for
        objective = cp.Maximize(c @ v)

        self._nrat = len(ratio_repo)

        self._numarr = np.zeros((self._nrat, polytope.A.shape[1]), dtype=np.double)
        self._denarr = np.zeros((self._nrat, polytope.A.shape[1]), dtype=np.double)
        for i, (ratio_id, vals) in enumerate(ratio_repo.items()):
            num_idxs = np.array([polytope.A.columns.get_loc(key) for key in vals['numerator'].keys()])
            self._numarr[i, num_idxs] = np.array(list(vals['numerator'].values()), dtype=np.double)

            conden = OrderedDict((key, val) for key, val in vals['denominator'].items() if key not in vals['numerator'])
            den_idxs = np.array([polytope.A.columns.get_loc(key) for key in conden.keys()])
            self._denarr[i, den_idxs] = np.array(list(conden.values()), dtype=np.double)

        self._nlhs = self._nrat
        if ratio_tol > 0.0:
            self._nlhs = self._nrat * 2

        self._lhs = cp.Parameter(shape=(self._nlhs, polytope.A.shape[1]), name='ratio_constraints')
        rhs = cp.Constant(value=np.zeros(self._nlhs))

        if ratio_tol > 0.0:
            ratio_constraints = self._lhs @ v <= rhs
        else:
            ratio_constraints = self._lhs @ v == rhs

        # this is necessary to avoid numerical issues where we get a lot of fluxes in the denominator < 1e-12
        denominator_lhs = cp.Parameter(
            shape=self._numarr.shape, value=self._numarr + self._denarr, name='denominator_constraint'
        )
        denominator_rhs = cp.Constant(value=np.ones(self._nrat) * min_denom_sum)

        constraints = [
            polytope.A.values @ v <= polytope.b.values,
            denominator_lhs   @ v >= denominator_rhs,
            ratio_constraints,
        ]
        if polytope.S is not None:
            constraints.append(
                polytope.S.values @ v == polytope.h.values
            )

        self._problem = cp.Problem(objective=objective, constraints=constraints)

        # now we construct a polytope with ratio constraints
        index = pd.Index(ratio_repo.keys())
        den_lhs_df = pd.DataFrame(-denominator_lhs.value, index=index + '_den', columns=polytope.A.columns)
        den_rhs_df = pd.Series(min_denom_sum, index=index + '_den')
        polytope.A = pd.concat([polytope.A, den_lhs_df], axis=0)
        polytope.b = pd.concat([polytope.b, den_rhs_df])

        if self._ratol > 0.0:
            index  = (index + '_min').append(index + '_max')

        lhs_df = pd.DataFrame(self._lhs.value, index=index, columns=polytope.A.columns)
        rhs_sr = pd.Series(0.0, index=index)
        if ratio_tol > 0.0:
            polytope.A = pd.concat([polytope.A, lhs_df], axis=0)
            polytope.b = pd.concat([polytope.b, rhs_sr])
        else:
            polytope.S = pd.concat([polytope.S, lhs_df], axis=0)
            polytope.h = pd.concat([polytope.h, rhs_sr])

        self._ratio_pol = polytope
        self._reaction_ids = polytope.A.columns
        self._constraint_ids = index

    def construct_polytope_constraints(self, ratio_sample: np.array) -> np.array:
        ratio_sample = np.atleast_2d(ratio_sample)  # means that it is now 2D

        vape = ratio_sample.shape
        viewlue = ratio_sample.reshape((math.prod(vape[:-1]), vape[-1]))

        lhs_proposal = np.zeros(shape=(viewlue.shape[0], *self._lhs.shape))
        lhs_proposal[:, :self._nrat, :] = ((ratio_sample[..., None] - self._ratol) - 1.0) * self._numarr[None, ...]
        lhs_proposal[:, :self._nrat, :] += (ratio_sample[..., None] - self._ratol) * self._denarr[None, ...]

        if self._ratol > 0.0:
            # upper bounds
            lhs_proposal[:, self._nrat:, :] = (1.0 - (ratio_sample[..., None]  + self._ratol)) * self._numarr[None, ...]
            lhs_proposal[:, self._nrat:, :] += -(ratio_sample[..., None]  + self._ratol) * self._denarr[None, ...]
        return lhs_proposal

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if len(value.shape) == 1:
            value = value[None, :]  # means that it is now at least 2D

        vape = value.shape
        viewlue = value.view(vape[:-1].numel(), vape[-1]).to(dtype=torch.double, device='cpu').numpy()

        nv = viewlue.shape[0]
        self._accepted = torch.zeros((nv, ), dtype=torch.bool)
        self._optima = np.zeros((nv, ), dtype=np.double)

        for i in range(viewlue.shape[0]):
            ratio_sample = viewlue[i, :]
            self._lhs.value = self.construct_polytope_constraints(ratio_sample=ratio_sample)[0, ...]
            try:
                optimum = self._problem.solve(solver=cp.GUROBI, verbose=False, max_iter=1000)
                # NOTE sometimes the polytope is not empty according to cvxpy but sampling still fails
                if optimum is not None:
                    self._accepted[i] = True
                    self._optima[i] = optimum
            except:
                pass
        return self._accepted.view(vape[:-1])


def _init_worker(input_q: mp.Queue, output_q: mp.Queue, ratsupp: _RatioSupport):
    global _IQ, _OQ, _RS
    _IQ = input_q
    _OQ = output_q
    _RS = ratsupp


def _ratio_worker():
    global _IQ, _OQ, _RS
    warnings.simplefilter("ignore")
    nacceptot, ntotal = 0, 0
    # print(f'begin {mp.current_process().name}')
    while True:
        task = _IQ.get()
        if isinstance(task, int) and (task == 0):
            _OQ.put(task)
            return
        try:
            ratio_samples = task[1]
            accepted = _RS.check(value=ratio_samples)
            naccepted = accepted.sum()
            nacceptot += naccepted
            ntotal += ratio_samples.shape[0]
            _OQ.put(ratio_samples[accepted])
            if (ntotal > 500) and (nacceptot / ntotal < 0.01):
                raise ValueError(f'Acceptance fraction is below 1%: {nacceptot / ntotal}')
        except Exception as e:
            _OQ.put(e)


def _ratio_listener(output_q: mp.Queue, n, result):
    i = 0
    while i < n:
        oput = output_q.get()
        result.append(oput)
        if isinstance(oput, Exception):
            break
        i += oput.shape[0]


class RatioPrior(_BasePrior):
    def __init__(
            self,
            model: RatioMixin,
            # TODO should always be uniform, otherwise log_probs does not hold!
            cache_size: int = 20000,
            fluxes_subsamples: int = 10,
            num_processes: int = 0,
            algorithm: Union[str, torch.distributions.Distribution] = 'hypercube',
            ratio_tol: float = 0.05,
            min_denom_sum: float = 0.0001,
            coef=0,
    ):
        self._ratio_repo = model.ratio_repo
        self._model = model
        super().__init__(model, cache_size, num_processes)
        if len(model.ratio_repo) == 0:
            raise ValueError('set ratio_repo')
        self._theta_id = model.ratios_id

        if ratio_tol < 0.0:
            raise ValueError('bruegh')
        ratio_tol /= 2.0
        if ratio_tol < RatioMixin._RATIO_ATOL:
            ratio_tol = 0.0
        self._ratol = ratio_tol

        self._mds = min_denom_sum
        self._n_flux = fluxes_subsamples
        self._support = self.support

        self._cache_fill_kwargs['n_flux'] = self._n_flux

        self._naccepted = 9
        self._ntotal = 10
        if coef > 0:
            raise ValueError
        self._coef = coef

        self._lhsides = np.zeros((cache_size, *self._support._lhs.shape), dtype=np.double)

        self._fill_caches = self._fill_caches_rejection
        if algorithm in ('ratio', 'numden'):
            if algorithm == 'ratio':
                # NOTE this one is too restrictive, meaning that there are ratios that are valid
                #   according to rejection sampling that fall outside of this polytope
                self._pol = self.construct_ratio_polytope(self._fcm, model)
            else:
                # NOTE this "works" for sampling ratios, but the distribution is not uniform at all and looks more
                #   like the distribution we get from uniform sampling
                self._pol = self.construct_numden_polytope(self._fcm, model, coef=coef)
            self._vsm = PolytopeSamplingModel(self._pol)
            self._bsp = None  # these are the basis points
            self._fill_caches = self._fill_caches_usm
        elif algorithm == 'hypercube':
            ratio_bounds_df = self.ratio_variability_analysis(self._fcm, model, min_denom_sum=min_denom_sum)
            self._ratio_dist = self.construct_uniform_ratio_sampler(ratio_bounds_df=ratio_bounds_df)
        elif isinstance(algorithm, torch.distributions.Distribution):
            self._ratio_dist = algorithm
        else:
            raise ValueError
        self._algo = algorithm
        self._cache_fill_kwargs['ratio_dist'] = algorithm

    def _get_mp_pool(self, num_processes):
        self._input_q = mp.Queue(maxsize=20)
        self._output_q = mp.Queue()
        return mp.Pool(
            processes=num_processes, initializer=_init_worker,
            initargs=[self._input_q, self._output_q, self._support]
        )

    @staticmethod
    def construct_uniform_ratio_sampler(ratio_bounds_df: pd.DataFrame):
        lo = torch.as_tensor(ratio_bounds_df['min'].values, dtype=torch.double)
        hi = torch.as_tensor(ratio_bounds_df['max'].values, dtype=torch.double)
        return torch.distributions.Uniform(low=lo, high=hi, validate_args=None)

    @staticmethod
    def ratio_variability_analysis(fcm: FluxCoordinateMapper, model: RatioMixin, min_denom_sum = 0.0, positive_numerator=True) -> pd.DataFrame:
        """
        TODO rewrite this such that the objective is a parameter and its value is reset 1 or -1 for the different directions!
            this would be equal to sbmfi.util.generate_cvxpy_LP
        figure out the min and max of every ratio when not constraining others; this helps excluding stuff
        https://en.wikipedia.org/wiki/Fractional_programming
        https://en.wikipedia.org/wiki/Linear-fractional_programming
        :return:
        """
        # net_pol = thermo_2_net_polytope(thermo_pol).H_representation(simplify=True)
        net_pol = fcm._Fn

        n = net_pol.A.shape[1]  # number of fluxes
        objective = cp.Parameter(shape=n, value=np.zeros(n))
        v = cp.Variable(n, name='fluxes')
        t = cp.Variable(1, name='t')  # auxiliary variable for linear fractional programming
        A = net_pol.A.values
        b = net_pol.b.values
        ratio_bounds = {}
        ratio_repo = model.ratio_repo  # this is necessary to get the uncondensed representation

        for ratio_id in model.ratios_id:
            objective.value[:] = 0.0
            d = np.zeros(n)
            c = np.zeros(n)

            num = ratio_repo[ratio_id]['numerator']
            den = ratio_repo[ratio_id]['denominator']
            for flux_id, coeff in den.items():
                index = net_pol.A.columns.get_loc(flux_id)
                d[index] = coeff
                if flux_id in num:
                    objective.value[index] = coeff
                    c[index] = coeff

            constraints = [
                A @ v <= b * t,
                d @ v == 1.0,
                t >= 0.0,
                d @ v >= min_denom_sum, # make sure the denominator is positive
            ]
            if positive_numerator:
                constraints.append(c @ v >= 0.0)  # means that the numerator is also definitely positive
            lfp = cp.Problem(  # linear fractional programme
                objective=cp.Minimize(objective @ v),
                constraints=constraints
            )
            lfp.solve(solver=cp.GUROBI)
            val_min = lfp.value
            if lfp.status != 'optimal':
                raise ValueError(f'ish not a valid ratio: {ratio_id}')
            if val_min < -1e-3:
                print(f'minimum ratio {ratio_id} is: {round(val_min, 3)}')
            ratio_min = max(round(val_min, 3), 0.0)
            objective.value *= -1
            lfp.solve(solver=cp.GUROBI)
            val_max = lfp.value
            if lfp.status != 'optimal':
                raise ValueError(f'ish not a valid ratio: {ratio_id}')
            ratio_max = round(val_max, 3)
            ratio_bounds[ratio_id] = (ratio_min, -ratio_max)
        return pd.DataFrame(ratio_bounds, index=['min', 'max']).T

    @staticmethod
    def construct_ratio_polytope(fcm: FluxCoordinateMapper, model:RatioMixin, tolerance=1e-10):
        raise NotImplementedError
        F_simp = PolyRoundApi.simplify_polytope(fcm._sampler, settings=self._pr_settings, normalize=normalize)
        F_simp = LabellingPolytope.from_Polytope(F_simp, polytope)

        rref_pol, T_1, x = transform_polytope_keep_transform(F_simp, fcm._sampler._pr_settings, 'rref')
        net_flux_vertices = V_representation(rref_pol, number_type='float')
        linalg = NumpyBackend()
        num = model._sum_getter('numerator', model._ratio_repo, linalg, rref_pol.A.columns)
        den = model._sum_getter('denominator', model._ratio_repo, linalg, rref_pol.A.columns)
        ratio_num = num @ net_flux_vertices.T
        ratio_den = den @ net_flux_vertices.T
        ratio_den[ratio_den <= 0.0] = tolerance
        ratios = pd.DataFrame((ratio_num.values / ratio_den.values).T, columns=model.ratios_id).drop_duplicates()
        ratios[ratios < 0.0] = 0.0
        ratios[ratios > 1.0] = 1.0
        A, b = compute_polytope_halfspaces(vertices=ratios.values)
        return LabellingPolytope(
            A=pd.DataFrame(A, columns=model.ratios_id),
            b=pd.Series(b),
        )

    @staticmethod
    def construct_numden_polytope(fcm:FluxCoordinateMapper, model: RatioMixin, coef=0):
        F_simp = PolyRoundApi.simplify_polytope(fcm._sampler, settings=self._pr_settings, normalize=normalize)
        F_simp = LabellingPolytope.from_Polytope(F_simp, polytope)
        rref_pol = transform_polytope_keep_transform(F_simp, fcm._sampler._pr_settings, 'rref')
        linalg = NumpyBackend()
        num = model._sum_getter('numerator', model._ratio_repo, linalg, rref_pol.A.columns)
        den = model._sum_getter('denominator', model._ratio_repo, linalg, rref_pol.A.columns)
        num_sum = pd.DataFrame(num, columns=rref_pol.A.columns, index=model.ratios_id + '_num')
        den_sum = pd.DataFrame(den, columns=rref_pol.A.columns, index=model.ratios_id + '_den')
        P = pd.concat([
            num_sum,
            den_sum + coef * num_sum.values,
        ])
        return project_polytope(rref_pol, P)

    @property
    def theta_id(self) -> pd.Index:
        return self._theta_id.rename('theta_id')

    @property
    def n_theta(self):
        return len(self._ratio_repo)

    @constraints.dependent_property(is_discrete=False, event_dim=1)  # NB event_dim=1 means that the right-most dimension defines an event!
    def support(self):
        return _RatioSupport(
            fcm=self._fcm,
            ratio_repo=self._ratio_repo,
            ratio_tol=self._ratol,
            min_denom_sum=self._mds
        )

    def _fill_caches_rejection(
            self,
            ratio_dist: torch.distributions.Distribution=None,
            n=1000,
            batch_size=100,  # batches of samples from ratio_dist to check at once; influences m
            n_flux=20,
            close_pool=True,
            break_i=-1
    ):
        # by making ratio_dist an argument, we can later pass intermediate posteriors!

        if ratio_dist is None:
            ratio_dist = self._ratio_dist


        result = []
        nacceptot, ntotal = 0, 0
        m = math.ceil(n * 1.1)  # this is because some will still fail when rounding the polytope!
        while nacceptot < m:
            ratio_samples = ratio_dist.sample((batch_size, ))
            accepted = self._support.check(value=ratio_samples)
            naccepted = accepted.sum()
            nacceptot += naccepted
            ntotal += ratio_samples.shape[0]
            if (ntotal > 500) and (nacceptot / ntotal < 0.01):
                raise ValueError(f'Acceptance fraction is below 1%: {nacceptot / ntotal}')
            result.append(ratio_samples[accepted])

        result = torch.cat(result)
        if n_flux == 0:
            # this is to only fill the ratio_cache, useful for plotting
            self._theta_cache = result[:n, ...]
        else:
            constraint_df = pd.concat([
                pd.DataFrame(ar, index=self._support._constraint_ids, columns=self._support._reaction_ids
                ) for ar in self._support.construct_polytope_constraints(result)[:m]
            ], keys=np.arange(m))
            if self._ratol == 0.0:
                kwargs = {'S_constraint_df': constraint_df}
            else:
                kwargs = {'A_constraint_df': constraint_df}
            sampling_task_generator = sampling_tasks(
                polytope=self._support._ratio_pol, transform_type=self._fcm._sampler.kernel_basis,
                basis_coordinates=self._fcm._sampler.basis_coordinates, linalg=self._fcm._la,
                counts=n_flux, return_kwargs=self._num_processes == 0, return_basis_samples=True,
                **kwargs
            )
            self._run_tasks(sampling_task_generator, break_i=break_i, close_pool=close_pool, format=True)
            # net_fluxes = pd.concat([r['fluxes'] for r in results], ignore_index=True).loc[:, self._fcm._Fn.A.columns]
            # pool = mp.Pool(processes=3)
            # result = pool.starmap(sample_polytope, iterable=sampling_task_generator)
            # result = pd.concat(result, ignore_index=True)
            # self._flux_cache = torch.as_tensor(result.values[:n * n_flux], dtype=torch.double)
            # self._flux_cache = self._fcm._map_thermo_2_fluxes(thermo_fluxes=self._flux_cache)
            print(self._flux_cache.shape)
            print(self._theta_cache.shape)
            self._theta_cache = self._model.compute_ratios(self._flux_cache)
            # # otherwise subsequent samples will have similar ratio values
            # scramble_indices = self._fcm._la.randperm(n * n_flux)
            # self._theta_cache = self._theta_cache[scramble_indices]
            # self._flux_cache = self._flux_cache[scramble_indices]

    def _fill_caches_usm(self, n=1000, close_pool=True, n_flux=20):
        # TODO doesnt fucking work...
        samples, self._bsp = coordinate_hit_and_run_cpp(
            self._vsm, n=n, initial_points=self._bsp, return_basis_samples=True
        )
        samples = torch.as_tensor(samples.values, dtype=torch.double)
        if self._algo == 'numden':
            n_ratios = len(self._model.ratios_id)
            num = samples[:, :n_ratios]
            den = samples[:, n_ratios:] + abs(self._coef) * num
            self._theta_cache = num / den
        else:
            self._theta_cache = samples

        if n_flux > 0:
            raise NotImplementedError

    def log_prob(self, value):
        if self._algo != 'hypercube':
            raise NotImplementedError
        if self._validate_args:
            self._validate_sample(value)

        return self._ratio_dist.log_prob(value=value).sum(-1)


class ProjectionPrior(_FluxPrior):
    # TODO I noticed that for the biomass flux, it is rarely sampled over 0.3, thus here we
    #  sample boundary fluxes in a projected polytope and then constrain and sample just like with ratios

    # PROBLEM if we sample uniformely in xch space, the resulting polytopes have different volumes
    #   which implies a different log_prob, we could compute the vol
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            projected_fluxes: Iterable,
            projection_pol: LabellingPolytope = None,
            cache_size: int = 20000,
            num_processes: int = 0,
            number_type='float',
    ):
        super(ProjectionPrior, self).__init__(model, cache_size, num_processes)

        if projection_pol is None:
            pol = self._fcm._Fn
            settings = self._fcm._sampler._pr_settings
            spol = PolyRoundApi.simplify_polytope(pol, settings=settings, normalize=False)
            pol = LabellingPolytope.from_Polytope(spol, pol)
            P = pd.DataFrame(0.0, index=projected_fluxes, columns=pol.A.columns)
            P.loc[projected_fluxes, projected_fluxes] = np.eye(len(projected_fluxes))
            self._projection_pol = rref_and_project(
                pol, P=P, number_type=number_type, settings=self._fcm._sampler._pr_settings
            )
        else:
            if not projection_pol.A.columns.isin(projected_fluxes).all():
                raise ValueError(f'wrong projection pol: {projection_pol.A.columns}, '
                                 f'wrt projected fluxes {projected_fluxes}')
            self._projection_pol = projection_pol

        self._boundary_psm = PolytopeSamplingModel(self._projection_pol)
        self._projection_basis_points = None
        self._projection_fva = fast_FVA(self._projection_pol)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        thermo_fluxes = self._fcm.map_theta_2_fluxes(value, return_thermo=True, pandalize=True)
        projected_fluxes = thermo_fluxes.loc[:, ]
        b_constraint_df = pd.concat([-projected_fluxes, projected_fluxes], axis=1)
        sampling_generator = sampling_tasks(
            self._fcm._Fn, b_constraint_df=b_constraint_df, counts=None,
            return_basis_samples=True, return_kwargs=self._num_processes == 0, to_basis_fn=self._fcm.to_basis_fn
        )
        # volume_task_generator = volume_tasks(sampling_generator)
        # results = self.run_tasks(volume_task_generator, fn=compute_volume)
        # voldf = pd.DataFrame(results)
        raise NotImplementedError

    def _fill_caches(self, n=100, n_flux=10, rel_tol=0.01, break_i=-1, close_pool=True):
        # boundary_result = coordinate_hit_and_run_cpp(
        #     self._boundary_psm, n=n, initial_points=self._projection_basis_points,
        #     new_basis_points=True
        # )
        boundary_result = sample_polytope(
            self._boundary_psm, n=n, initial_points=self._projection_basis_points, new_basis_points=True, pandalize=True
        )
        self._projection_basis_points = boundary_result['new_basis_points']
        lb = boundary_result['fluxes']
        ub = lb.copy()
        lb.columns += '|lb'
        ub.columns += '|ub'

        if rel_tol > 0.0:
            bounds_tol = (self._projection_fva['max'] - self._projection_fva['min']) * rel_tol * 0.5
            lb -= bounds_tol.values
            ub += bounds_tol.values
            lb = lb.clip(lower=self._projection_fva['min'].values, upper=self._projection_fva['max'].values, axis=1)
            ub = ub.clip(lower=self._projection_fva['min'].values, upper=self._projection_fva['max'].values, axis=1)

        b_constraint_df = pd.concat([-lb, ub], axis=1)  # NB this dataframe contains all the bounds for the b vector
        sampling_task_generator = sampling_tasks(
            self._fcm._Fn, b_constraint_df=b_constraint_df, counts=n_flux,
            return_basis_samples=True, return_kwargs=self._num_processes == 0,
        )
        self._run_tasks(sampling_task_generator, break_i=break_i, close_pool=close_pool, format=True)



if __name__ == "__main__":
    import pickle, os
    from sbmfi.settings import MODEL_DIR, BASE_DIR
    from sbmfi.models.build_models import build_e_coli_anton_glc, build_e_coli_tomek
    from sbmfi.models.small_models import spiro
    from equilibrator_api import *

    # model, kwargs = spiro()
    # fcm = FluxCoordinateMapper(model)
    # pickle.dump(fcm, open('spiro_fcm.p', 'wb'))

    model, kwargs = spiro(backend='torch', add_biomass=True, ratios=False)

    up = UniFluxPrior(model)
    # pp = ProjectionPrior(model, projected_fluxes=kwargs['measured_boundary_fluxes'])
    t,f = up.sample_dataframes(n=2)

    # model, kwargs = build_e_coli_anton_glc(build_simulator=False)
    # model.reactions.get_by_id('EX_glc__D_e').bounds = (-15.0, 0.0)
    # model.reactions.get_by_id('biomass_rxn').bounds = (0.05, 1000.0)
    # model.reactions.get_by_id('EX_ac_e').bounds = (0.0, 1000.0)
    # model.build_simulator(free_reaction_id=['biomass_rxn', 'EX_glc__D_e', 'EX_ac_e'])
    #
    # fcm = FluxCoordinateMapper(model=model, basis_coordinates='transformed', transform_type='rref')
    # pickle.dump(fcm, open('fcm.p', 'wb'))
    # fcm = pickle.load(open('fcm.p', 'rb'))
    # pp = ProjectionPrior(fcm, projected_fluxes=['biomass_rxn', 'EX_glc__D_e', 'EX_ac_e'])
    # t,f = pp.sample_dataframes(n=20, n_flux=20)
    # prior_u = UniFluxPrior(fcm=fcm, num_processes=0)
    # tdf, fdf = prior_u.sample_dataframes(50000)

    # _make_voldf()

    # model, kwargs = build_e_coli_tomek()
    # free_rid = ['EX_ac_e', model.biomass_id,'EX_glc__D_e']
    # model.reactions.get_by_id('EX_glc__D_e').bounds = (-12.0, 0.0)
    # model.reactions.get_by_id(model.biomass_id).bounds = (0.05, 1000.0)
    # model.reactions.get_by_id('EX_ac_e').bounds = (0.0, 1000.0)
    # fcm = FluxCoordinateMapper(model, transform_type='rref', free_reaction_id=['EX_ac_e'])
    #
    # pp = ProjectionPrior(fcm, projected_fluxes=free_rid)
    # pickle.dump((fcm, pp._projection_pol), open('proj.p', 'wb'))
    # fcm, projection_pol = pickle.load(open('proj.p', 'rb'))
    # pp = ProjectionPrior(fcm, projected_fluxes=projection_pol.A.columns, projection_pol=projection_pol)
    # t,f = pp.sample_dataframes(n=100)

    # model, kwargs = build_e_coli_anton_glc(build_simulator=False)
    # model.reactions.get_by_id('EX_glc__D_e').bounds = (-12.0, 0.0)
    # model.reactions.get_by_id('biomass_rxn').bounds = (0.05, 1000.0)
    # model.reactions.get_by_id('EX_ac_e').bounds = (0.0, 1000.0)
    # model.build_simulator(free_reaction_id=['EX_ac_e'])
    # ubmp = ProjectionPrior(fcm=fcm, projected_fluxes=['biomass_rxn', 'EX_glc__D_e', 'EX_ac_e'], num_processes=0)

    # model, kwargs = build_e_coli_anton_glc(build_simulator=False)
    # model.reactions.get_by_id('EX_glc__D_e').bounds = (-15.0, 0.0)
    # model.reactions.get_by_id('biomass_rxn').bounds = (0.05, 1000.0)
    # model.reactions.get_by_id('EX_ac_e').bounds = (0.0, 1000.0)
    # model.build_simulator(free_reaction_id=['biomass_rxn', 'EX_glc__D_e', 'EX_ac_e'])
    #
    # fcm = FluxCoordinateMapper(model=model)
    # pickle.dump(fcm, open('fcm.p', 'wb'))
    # fcm = pickle.load(open('fcm.p', 'rb'))
    #
    # ubmp = ProjectionPrior(fcm=fcm, projected_fluxes=['biomass_rxn', 'EX_glc__D_e', 'EX_ac_e'], num_processes=2)
    # t, f = ubmp.sample_dataframes(n=10, n_flux=25, rel_tol=0.05)


    # model, kwargs = spiro()
    # fcm = FluxCoordinateMapper(model=model, free_reaction_id=['d_out', 'h_out'])
    # ubmp = ProjectionPrior(fcm=fcm, projected_fluxes=['d_out', 'h_out'], num_processes=0)
    # t, f = ubmp.sample_dataframes(n=20, n_flux=25, rel_tol=0.05)  # REL_TOL == 0.0 is essential, otherwise sampled support is outside polytope

    # prior_u = UniFluxPrior(fcm=fcm)
    # tdf, fdf = prior_u.sample_dataframes(50000)


    # tfs = pickle.load(open("C:\python_projects\pysumo\chapriors\e_coli_tomek_gluc_aero_tfs.p", 'rb'))
    # tp = ThermoPrior(fcm, tfs, coordinates='labelling')
    # t,f = tp.sample_dataframes(n=100, break_i=1, abs_xch_flux_tol=0.00)


    #
    # val = torch.rand((4,1,5,6, len(fcm.theta_id)))
    # tp.log_prob(val)

    # tp = ThermoPrior(fcm, tfs, num_processes=0)
    # t, f = tp.sample_dataframes(n=20, coordinates='labelling')
    # _make_voldf()