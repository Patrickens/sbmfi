import psutil
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import inspect
from torch.distributions.constraints import Constraint, _Dependent, _Interval
from PolyRound.api import PolyRoundApi
from sbmfi.core.model import LabellingModel
from sbmfi.core.linalg import LinAlg
from sbmfi.core.polytopia import LabellingPolytope, FluxCoordinateMapper, \
    PolytopeSamplingModel, fast_FVA, rref_and_project, sample_polytope, project_polytope, compute_volume
from typing import Iterable, Union, List
from torch.distributions import constraints
from torch.distributions import Distribution
import math
#   https://math.stackexchange.com/questions/4484178/computing-barycentric-coordinates-for-convex-n-dimensional-polytope-that-is-not


def sampling_tasks(
        polytope: LabellingPolytope, # this is a basis polytope that will be modified using b_constraint_df and A_constraint_dct
        kernel_basis = 'svd',
        basis_coordinates = 'rounded',
        counts: Union[int, pd.Series] = 20,  # number of fluxes to sample from yielded polytope
        A_constraint_df: pd.DataFrame = None, # this should have a multiindex with level 1 being names, and 2 being constraint_names
        S_constraint_df: pd.DataFrame = None, # this should have a multiindex with level 1 being names, and 2 being constraint_names
        b_constraint_df: pd.DataFrame = None,
        # return_kwargs: bool = False,
        n_burn: int = 100,
        thinning_factor: int = 5,
        n_chains: int = 4,
        sampling_function = sample_polytope,
        linalg: LinAlg = None,
        return_psm = False,
        return_what = 'basis',
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
            'new_initial_points': False,
            'return_psm': return_psm,
            'phi': None,
            'linalg': linalg,
            'kernel_basis': kernel_basis,
            'basis_coordinates': basis_coordinates,
            'density': None,
            'n_cdf': 1,
            'return_arviz': False,
            'return_what': return_what,
        }

        kwargs = {key: kwargs.get(key) for key in func_kwargs}

        yield kwargs
        # if return_kwargs:
        #     yield kwargs
        # else:
        #     yield tuple(kwargs.values())  # because cannot pickle dict_values... god I hate dict_keys and dict_values


def volume_tasks(
        models: Union[PolytopeSamplingModel, LabellingPolytope],
        n: int = -1,
        n0_multiplier: int = 5,
        thinning_factor: int = 1,
        epsilon: float = 1.0,
        enumerate_vertices: bool = False,
        return_all_ratios: bool = False,
        quadratic_program: bool = False,
):
    for i, model in enumerate(models):
        yield {
            'model': model,
            'n': n,
            'n0_multiplier': n0_multiplier,
            'thinning_factor': thinning_factor,
            'epsilon': epsilon,
            'enumerate_vertices': enumerate_vertices,
            'return_all_ratios': return_all_ratios,
            'quadratic_program': quadratic_program,
        }

class _CannonicalPolytopeSupport(_Dependent):  #
    _VTOL = 1e-6
    def __init__(
            self,
            fcm: FluxCoordinateMapper,
            validation_tol = _VTOL,
    ):
        polytope = fcm.make_theta_polytope()
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
        vape = value.shape
        if len(vape) > 2:
            value = value.view((math.prod(vape[:-1]), vape[-1]))
        value = value[:, :self._A.shape[1]]
        valid   = (self._A @ value.T <= self._b).T
        return valid.view(*vape[:-1], self._A.shape[0])


class _SphereSupport(_Dependent):
    def __init__(
            self,
            fcm: FluxCoordinateMapper,
            validation_tol=_CannonicalPolytopeSupport._VTOL,
    ):
        self._vtol = validation_tol
        self._nx = fcm._nx
        self._logxch = fcm._logxch
        self._constraint_id = fcm.theta_id
        if not self._logxch and (self._nx > 0):
            self._rho_bounds = torch.Tensor(fcm._rho_bounds)

        polytope = fcm.make_theta_polytope()
        super().__init__(is_discrete=False, event_dim=polytope.A.shape[1])

    def to(self, *args, **kwargs):
        if not self._logxch and (self._nx > 0):
            self._rho_bounds = self._rho_bounds.to(*args, **kwargs)

    def check(self, value: torch.Tensor) -> torch.Tensor:
        ball = value
        if self._nx > 0:
            ball = value[..., :-self._nx]
        sphere = ball[..., :-1]
        distance = ball[..., [-1]]
        dist_check = (distance > 0.0) & (distance < 1.0)
        norm = torch.norm(sphere, p=2, dim=-1, keepdim=True)
        norm_check = (norm > 1.0 - self._vtol) & (norm < 1.0 + self._vtol)
        if (self._nx > 0) and not self._logxch:
            xch_vars = value[..., -self._nx:]
            xch_check = (xch_vars > self._rho_bounds[:, 0]) & (xch_vars < self._rho_bounds[:, 1])
            return torch.cat([norm_check, dist_check, xch_check], dim=-1)
        return torch.cat([norm_check, dist_check], dim=-1)


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
            if not model._is_built:
                raise ValueError('First build the model; choose kernel basis and basis coordinate system!')
            if model._la.backend != 'torch':
                linalg = LinAlg('torch', seed=model._la._backwargs['seed'])
                model = FluxCoordinateMapper(
                    model,
                    linalg=linalg,
                    free_reaction_id=model.labelling_reactions.list_attr('id'),
                    **model._fcm.fcm_kwargs
                )
            else:
                model = model._fcm

        self._fcm = model
        self._la = model._la

        if num_processes < 0:
            num_processes = psutil.cpu_count(logical=False)
        self._num_processes = num_processes
        if num_processes > 0:
            self._mp_pool = self._get_mp_pool(num_processes)

        self._cache_fill_kwargs = {'n': cache_size}

        self._theta_cache = torch.zeros((cache_size, self.n_theta), dtype=torch.double) # cache to store dependent variables

        # passing validate_args={} will trigger support checking
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

    def sample_pandalize(self, n):
        result = self.sample((n, ))
        return pd.DataFrame(self._la.tonp(result), columns=self.theta_id)

    def _run_tasks(
            self, tasks, fn=sample_polytope, break_i=-1, close_pool=True, scramble=True,
            what='basis', return_results=False
    ):
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

        if fn == sample_polytope:
            whatensor = self._la.cat([torch.as_tensor(r[what]) for r in results])
            if 'new_initial_points' in results[0]:
                self._initial_points = results[0]['new_initial_points']
            if scramble:
                scramble_indices = self._la.randperm(whatensor.shape[0])
                whatensor = whatensor[scramble_indices]

            if return_results:
                log_det_E = [r['log_det_E'] for r in results]
                if 'psm' in results[0]:
                    psms = [r['psm'] for r in results]
                return whatensor, {'log_det_E': log_det_E, 'psms': psms}
            return whatensor
        elif fn == compute_volume:
            return pd.DataFrame(results)
        else:
            return results


class _NetFluxPrior(_BasePrior):
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            cache_size: int = 20000,
            num_processes: int = 0,
    ):
        self._basis_points = None
        super(_NetFluxPrior, self).__init__(model, cache_size, num_processes)

    # NB event_dim=1 means that the right-most dimension defines an event!
    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        if self._fcm._sampler.basis_coordinates in ['spherical', 'semi_spherical']:
            supp = _SphereSupport(self._fcm)
        else:
            supp = _CannonicalPolytopeSupport(fcm=self._fcm)
        supp.to(dtype=torch.float32)  # TODO maybe pass dtype as a kwarg or maybe always enforce float32
        return supp

    @property
    def theta_id(self) -> pd.Index:
        return self._fcm.theta_id

    @property
    def n_theta(self):
        return len(self._fcm.theta_id)


class _XchFluxPrior(Distribution):
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
    ):
        # we do not have a support for these priors, since they are checked by the support of
        #   UniFluxPrior anyways
        if isinstance(model, LabellingModel):
            if not model._is_built:
                raise ValueError('First build the model; choose whether to logit exchange fluxes!')
            if model._la.backend != 'torch':
                linalg = LinAlg('torch', seed=model._la._backwargs['seed'])
                model = FluxCoordinateMapper(
                    model,
                    linalg=linalg,
                    free_reaction_id=model.labelling_reactions.list_attr('id'),
                    **model._fcm.fcm_kwargs
                )
            else:
                model = model._fcm

        if model._nx == 0:
            raise ValueError('no boundary fluxes')

        self._la = model._la
        self._xch_id = model.xch_basis_id
        self._logxch = model._logxch
        self._rho_bounds = model._rho_bounds
    
    def theta_id(self) -> pd.Index:
        return self._xch_id
    
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UniXchFluxPrior(_XchFluxPrior):
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
    ):
        super().__init__(model)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        xch_fluxes = self._la.sample_bounded_distribution(
            shape=sample_shape, lo=self._rho_bounds[:, 0], hi=self._rho_bounds[:, 1]
        )
        if self._logxch:
            xch_fluxes = self._la._logit_xch(xch_fluxes)
        return xch_fluxes

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # we do not check any support here, since that has been done in UniFluxPrior
        return torch.zeros((*value.shape[:-1], 1))


class TruncNormXchFluxPrior(_XchFluxPrior):
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            mu_sigma: pd.DataFrame,
    ):
        super().__init__(model)
        fva = fast_FVA(model._Fn)

    @staticmethod
    def _rando_mu_sigma(fcm):
        fva = fast_FVA(model._Fn)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        pass

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pass


class UniformNetPrior(_NetFluxPrior):
    def __init__(
            self,
            model,
            xch_prior: _XchFluxPrior = None,
            cache_size: int = 20000,
            **kwargs,
    ):
        super(UniformNetPrior, self).__init__(model, cache_size, **kwargs)
        if (self._fcm._nx > 0) and (xch_prior is None):
            xch_prior = UniXchFluxPrior(self._fcm)
        self._xch_prior = xch_prior

    def _fill_caches(self, n=20000, **kwargs):
        # this one is without pool always
        task = dict(
            model=self._fcm._sampler, initial_points=self._basis_points, n=n, n_burn=200, new_initial_points=True,
        )
        theta = self._run_tasks(tasks=[task], scramble=True)
        if self._fcm._nx > 0:
            xch_basis_samples = self._xch_prior.sample((n, ))
            theta = self._la.cat([theta, xch_basis_samples], dim=-1)
        self._theta_cache = theta

    def log_prob(self, value):
        # log prob for uniform distribution is log(1 / vol(polytope))
        # for non-uniform xch flux distribution, return log(1 / vol(net_polytope)) + log_prob(xch_flux)
        if self._validate_args:
            self._validate_sample(value)
        # place-holder until we can compute polytope volumes

        if (self._fcm._nx > 0) and not isinstance(self._xch_prior, UniXchFluxPrior):
            xch_fluxes = value[..., -self._fcm._nx:]
            return self._xch_prior.log_prob(xch_fluxes)
        return torch.zeros((*value.shape[:-1], 1))


class ProjectionPrior(UniformNetPrior):
    # TODO I noticed that for the biomass flux, it is rarely sampled over 0.3, thus here we
    #  sample boundary fluxes in a projected polytope and then constrain and sample just like with ratios

    # PROBLEM if we sample uniformely in xch space, the resulting polytopes have different volumes
    #   which implies a different log_prob, we could compute the vol
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            projected_fluxes: Iterable,
            xch_prior: _XchFluxPrior = None,
            projection_pol: LabellingPolytope = None,
            cache_size: int = 20000,
            num_processes: int = 0,
            number_type='float',
    ):
        super(ProjectionPrior, self).__init__(model, xch_prior, cache_size, num_processes=num_processes)

        if projection_pol is None:
            pol = self._fcm._Fn
            settings = self._fcm._sampler._pr_settings
            spol = PolyRoundApi.simplify_polytope(pol, settings=settings, normalize=False)
            pol = LabellingPolytope.from_Polytope(spol, pol)
            P = pd.DataFrame(0.0, index=projected_fluxes, columns=pol.A.columns)
            P.loc[projected_fluxes, projected_fluxes] = np.eye(len(projected_fluxes))
            try:
                self._projection_pol = rref_and_project(pol, P=P, number_type=number_type, settings=settings)
            except:
                self._projection_pol = project_polytope(pol, P=P, number_type=number_type)
            else:
                raise ValueError('somethings off')
        else:
            if not projection_pol.A.columns.isin(projected_fluxes).all():
                raise ValueError(f'wrong projection pol: {projection_pol.A.columns}, '
                                 f'wrt projected fluxes {projected_fluxes}')
            self._projection_pol = projection_pol

        self._volumes = None
        self._boundary_psm = PolytopeSamplingModel(self._projection_pol)
        self._projection_initial_points = None
        self._projection_fva = fast_FVA(self._projection_pol)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        raise NotImplementedError

    def _fill_caches(
            self, n=100, n_flux=10, rel_tol=0.01, break_i=-1, close_pool=True, scramble=False,
            what_volume='polytope', enumerate_vertices=True,
    ):
        boundary_result = sample_polytope(
            self._boundary_psm, n=n, initial_points=self._projection_initial_points, new_initial_points=True,
            return_what='basis',
        )
        self._projection_initial_points = boundary_result['new_initial_points']
        boundary_samples = self._boundary_psm.to_net_fluxes(boundary_result['basis'], pandalize=True)
        lb = boundary_samples.copy()
        ub = boundary_samples.copy()
        lb.columns += '|lb'
        ub.columns += '|ub'

        if rel_tol > 0.0:
            bounds_tol = (self._projection_fva['max'] - self._projection_fva['min']) * rel_tol * 0.5
            lb -= bounds_tol.values
            ub += bounds_tol.values
            lb = lb.clip(lower=self._projection_fva['min'].values, upper=self._projection_fva['max'].values, axis=1)
            ub = ub.clip(lower=self._projection_fva['min'].values, upper=self._projection_fva['max'].values, axis=1)

        b_constraint_df = pd.concat([-lb, ub], axis=1)  # NB this dataframe contains all the bounds for the b vector

        kernel_basis = 'rref' if enumerate_vertices else 'svd'
        basis_coordinates = 'transformed' if enumerate_vertices else 'rounded'
        sampling_task_generator = sampling_tasks(
            self._fcm._Fn, b_constraint_df=b_constraint_df, counts=n_flux, return_what='net_fluxes',
            return_psm=True, kernel_basis=kernel_basis, basis_coordinates=basis_coordinates,
        )

        net_fluxes = self._run_tasks(
            sampling_task_generator, break_i=break_i, close_pool=close_pool and what_volume is None, scramble=True,
            what='net_fluxes', return_results=True,
        )

        net_fluxes, results = net_fluxes
        if what_volume is None:
            pass
        else:
            if what_volume == 'polytope':
                volume_task_generator = volume_tasks(results['psms'], enumerate_vertices=enumerate_vertices)
                volume_df = self._run_tasks(volume_task_generator, fn=compute_volume, break_i=break_i, close_pool=close_pool)
                volume_df = pd.concat([boundary_samples, volume_df], axis=1)
            elif what_volume == 'log_det_E':
                boundary_samples['log_det_E'] = results['log_det_E']
                volume_df = boundary_samples

            if self._volumes is None:
                self._volumes = volume_df
            else:
                self._volumes = pd.concat([self._volumes, volume_df], axis=0, ignore_index=True)

        theta = self._fcm._sampler.to_net_basis(net_fluxes)
        if self._fcm._nx > 0:
            xch_basis_samples = self._xch_prior.sample((n * n_flux, ))
            theta = self._la.cat([theta, xch_basis_samples], dim=-1)

        self._theta_cache = theta



if __name__ == "__main__":
    import pickle, os
    from sbmfi.settings import MODEL_DIR, BASE_DIR
    from sbmfi.models.build_models import build_e_coli_anton_glc, build_e_coli_tomek
    from sbmfi.models.small_models import spiro
    from equilibrator_api import *

    # model, kwargs = spiro()
    # fcm = FluxCoordinateMapper(model)
    # pickle.dump(fcm, open('spiro_fcm.p', 'wb'))

    model, kwargs = spiro(
        backend='numpy', add_biomass=True, ratios=False, build_simulator=True, v2_reversible=True, v5_reversible=False,
        basis_coordinates='transformed', kernel_basis='rref',
    )
    up = ProjectionPrior(model, projected_fluxes=kwargs['measured_boundary_fluxes'][1:], cache_size=100, number_type='fraction', num_processes=1)
    # TruncNormXchFluxPrior._rando_mu_sigma(model._fcm)

    # # pp = ProjectionPrior(model, projected_fluxes=kwargs['measured_boundary_fluxes'])
    aa = up.sample(sample_shape=(5, ))
    aa = up.sample(sample_shape=(5, ))
    print(up._volumes)

    print(up.sample_pandalize(2))
    # print(aa)
    # supp = up.support
    #
    # inn = supp.check(aa.view(2,3,6))