from sbmfi.priors.uniform import BaseRoundedPrior
import numpy as np
import pandas as pd
import torch
import scipy
from sbmfi.core.reaction import LabellingReaction
from sbmfi.core.polytopia import sample_polytope
from sbmfi.core.coordinater import (
    FluxCoordinateMapper,
    map_gibbs_2_xch_fluxes,
    map_thermo_2_gibbs,
    make_net_theta_id,
)
from sbmfi.priors.uniform import sampling_tasks
from sbmfi.core.util import get_tensor
from sbmfi.core.distributions import sample_bounded
from sbmfi.priors.thermo_data import ThermoSamplingData, R_GAS
from sbmfi.priors.thermo_sampler import (
    DRGSamplingResult, sample_drg as _sample_drg_pure,
    find_initial_points,
)
from collections import OrderedDict
from torch.distributions import Distribution
try:
    from pta.sampling.tfs import TFSModel
except ImportError:
    TFSModel = None
FreeEnergiesSamplingResult = DRGSamplingResult
R = R_GAS


def compute_xch_fluxes(self: FluxCoordinateMapper, dgibbsr: pd.DataFrame) -> pd.DataFrame:
    return map_gibbs_2_xch_fluxes(self, dgibbsr)


def compute_dgibbsr(self: FluxCoordinateMapper, thermo_fluxes) -> pd.DataFrame:
    dgibbsr = map_thermo_2_gibbs(self, thermo_fluxes, pandalize=True)
    dgibbsr.columns = self._fwd_id
    return dgibbsr


def append_xch_flux_samples(  # TODO make this work without things being dataframes
        self: FluxCoordinateMapper, net_fluxes: pd.DataFrame = None, net_basis_samples: pd.DataFrame = None,
        xch_fluxes: pd.DataFrame = None, pandalize=False, return_type='both'
):
    index = None

    if isinstance(net_fluxes, pd.DataFrame):
        index = net_fluxes.index
        net_fluxes = get_tensor(values=net_fluxes.loc[:, self._Fn.A.columns].values)
    if isinstance(net_basis_samples, pd.DataFrame):
        index = net_basis_samples.index
        net_basis_samples = get_tensor(values=net_basis_samples.loc[:, make_net_theta_id(self._sampler)].values)

    if net_fluxes is None:
        n = net_basis_samples.shape[0]
        if return_type in ['fluxes', 'both']:
            net_fluxes = self._sampler.map_rounded_2_fluxes(net_basis_samples, is_rounded=self._sampler._coor_id == 'rounded')
    elif net_basis_samples is None:
        n = net_fluxes.shape[0]
        if return_type in ['theta', 'both']:
            net_basis_samples = self._sampler.map_fluxes_2_rounded(net_fluxes)
    else:
        n = net_fluxes.shape[0]

    if self._nx > 0:
        if xch_fluxes is None:
            xch_fluxes = sample_bounded(
                shape=(n,), lo=self._rho_bounds[:, 0], hi=self._rho_bounds[:, 1]
            )
        elif isinstance(xch_fluxes, pd.DataFrame):
            xch_fluxes = get_tensor(values=xch_fluxes.loc[:, self._fwd_id + '_xch'].values)
        if return_type in ['fluxes', 'both']:
            thermo_fluxes = torch.cat([net_fluxes, xch_fluxes], dim=-1)
    else:
        thermo_fluxes = net_fluxes

    if return_type in ['fluxes', 'both']:
        fluxes = self.map_thermo_2_fluxes(thermo_fluxes)

    if pandalize and (return_type in ['fluxes', 'both']):
        fluxes = pd.DataFrame(fluxes.detach().cpu().numpy(), index=index, columns=self._F.A.columns)
    if return_type == 'fluxes':
        return fluxes

    if self._nx > 0:
        theta = torch.cat([net_basis_samples, xch_fluxes], dim=-1)
    else:
        theta = net_basis_samples

    if pandalize:
        theta = pd.DataFrame(theta.detach().cpu().numpy(), index=index, columns=self.theta_id())
    if return_type == 'theta':
        return theta
    elif return_type == 'both':
        return theta, fluxes


FluxCoordinateMapper.append_xch_flux_samples = append_xch_flux_samples
FluxCoordinateMapper.compute_xch_fluxes = compute_xch_fluxes
FluxCoordinateMapper.compute_dgibbsr = compute_dgibbsr

class ThermoPrior(BaseRoundedPrior):
    def __init__(
            self,
            model: FluxCoordinateMapper,
            tfs_model,   # TFSModel | ThermoSamplingData
            coordinates='thermo',
            cache_size: int = 20000,
            num_processes: int = 0,
    ):
        super(ThermoPrior, self).__init__(model, num_processes=num_processes)
        self._cache_size = cache_size
        self._coords = coordinates
        self._tfs_model = tfs_model
        self._fulabel_pol = None
        self._drg_mvn = self.extract_drg_mvn(tfs_model)
        self._thermo_basis_points = None
        self._orthant_volumes = {}
        self._orthant_log_probs = {}

    @staticmethod
    def _coerce_sampling_data(tfs_model) -> ThermoSamplingData:
        if isinstance(tfs_model, ThermoSamplingData):
            return tfs_model
        return ThermoSamplingData.from_tfs_model(tfs_model)

    @staticmethod
    def reset_rhos(model):
        reset_rhos = []
        for r in model.reactions:
            if isinstance(r, LabellingReaction) and (r.rho_max > 0.0) and (r.rho_max < r._RHO_MAX):
                # this is necessary to make sure the polytope of the support has the correct bounds
                r.rho_min = 0.0
                r.rho_max = r._RHO_MAX
                reset_rhos.append(r.id)
        if len(reset_rhos) > 0:
            print(f'The rho_bounds of reactions: {reset_rhos} has been set to ({0.0, LabellingReaction._RHO_MAX})')
        return model

    def extract_drg_mvn(self, tfs_model, epsilon=1e-12, as_tensor=False) -> Distribution:
        """Extract the marginal dG MVN for the reactions used by the FCM."""
        data = self._coerce_sampling_data(tfs_model)
        drg_mean, drg_cov = data.drg_mvn_params(epsilon=epsilon)
        tfs_reaction_ids = pd.Index(data.T_reaction_ids)
        indices = np.array([tfs_reaction_ids.get_loc(rid) for rid in self._fcm.fwd_id])
        drg_prime_mean = drg_mean[indices]
        drg_prime_cov = drg_cov[np.ix_(indices, indices)]

        if as_tensor:
            return torch.distributions.MultivariateNormal(
                loc=torch.as_tensor(drg_prime_mean, dtype=torch.double).squeeze(),
                covariance_matrix=torch.as_tensor(drg_prime_cov, dtype=torch.double)
            )
        return scipy.stats.multivariate_normal(mean=drg_prime_mean.squeeze(), cov=drg_prime_cov)

    def _compute_orthant_volume(self, thermo_fluxes):
        if isinstance(thermo_fluxes, pd.DataFrame):
            net_fluxes = thermo_fluxes.loc[:, self._fcm.fwd_id]
        else:
            net_fluxes = pd.DataFrame(
                thermo_fluxes[..., :len(self._fcm.fwd_id)].detach().cpu().numpy(),
                columns=self._fcm.fwd_id,
            )
        orthant_keys = net_fluxes.gt(0.0).apply(lambda row: hash(tuple(row)), raw=True, axis=1)
        values = np.array([self._orthant_volumes.get(key, 0.0) for key in orthant_keys], dtype=float)
        return torch.as_tensor(values[:, None], dtype=torch.double)

    def _estimate_orthant_log_prob(self, orthant, n_samples=50_000):
        key = hash(tuple(orthant))
        if key not in self._orthant_log_probs:
            draws = self._drg_mvn.rvs(size=n_samples)
            draws = np.atleast_2d(draws)
            draws_orthants = draws < 0.0
            match = np.all(draws_orthants == np.asarray(orthant, dtype=bool), axis=1).mean()
            self._orthant_log_probs[key] = np.log(max(match, 1.0 / n_samples))
        return self._orthant_log_probs[key]

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        if self._coords == 'labelling':
            #
            raise NotImplementedError(
                'this would mean that we need to compute the volume of a thermo_constrained_label_pol, '
                'which is a biatch due to large number of dimensions')
        thermo_fluxes = self._fcm.map_theta_2_fluxes(value, return_thermo=True, pandalize=True)
        drg = self._fcm.compute_dgibbsr(thermo_fluxes)
        orthant_vols = self._compute_orthant_volume(thermo_fluxes)
        orthants = drg.lt(0.0)
        orthant_log_probs = orthants.apply(
            lambda row: self._estimate_orthant_log_prob(tuple(row.values)),
            axis=1,
        ).to_numpy(dtype=float)
        log_prob = self._drg_mvn.logpdf(drg.loc[:, self._fcm.fwd_id].values)
        log_prob = np.atleast_1d(log_prob) - orthant_log_probs - orthant_vols[:, 0].detach().cpu().numpy()
        return torch.as_tensor(log_prob[:, None], dtype=torch.double)

    def _generate_thermo_constrained_label_pol(self, abs_xch_flux_tol=0.0):
        pol = self._fcm._F.copy()

        if abs_xch_flux_tol == 0.0:
            S_df = pd.DataFrame(
                0.0,
                index=(self._fcm.fwd_id + '_rho'),
                columns=pol.A.columns,
            )
            rho_rev_idx = np.zeros(S_df.shape, dtype=bool)
            rho_rev_idx[
                [S_df.index.get_loc(key) for key in self._fcm.fwd_id + '_rho'],
                [S_df.columns.get_loc(key) for key in self._fcm.fwd_id + '_rev']
            ] = True
            S_df[rho_rev_idx] = -1.0
            rho_constraints = pol.S.index.str.contains('_rho')
            pol.S = pd.concat([pol.S.loc[~rho_constraints], S_df], axis=0).sort_index(axis=0)
            pol.h = pd.concat([pol.h.loc[~rho_constraints], pd.Series(0.0, index=S_df.index)]).loc[pol.S.index]
        else:
            A_df = pd.DataFrame(
                0.0,
                index=(self._fcm.fwd_id + '_rho_min|lb').union((self._fcm.fwd_id + '_rho_max|ub')),
                columns=pol.A.columns,
            )
            rho_rev_idx = np.zeros(A_df.shape, dtype=bool)
            rho_rev_idx[
                [A_df.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_min|lb'],
                [A_df.columns.get_loc(key) for key in self._fcm.fwd_id + '_rev']
            ] = True
            A_df[rho_rev_idx] = -1.0
            rho_rev_idx[:] = False
            rho_rev_idx[
                [A_df.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_max|ub'],
                [A_df.columns.get_loc(key) for key in self._fcm.fwd_id + '_rev']
            ] = True
            A_df[rho_rev_idx] = 1.0

            rho_constraints = pol.A.index.str.contains('_rho')
            pol.A = pd.concat([pol.A.loc[~rho_constraints], A_df], axis=0).sort_index(axis=0)
            pol.b = pd.concat([pol.b.loc[~rho_constraints], pd.Series(0.0, index=A_df.index)]).loc[pol.A.index]
        return pol

    @staticmethod
    def _sample_drg_suppress_output(
            tfs_model,   # TFSModel | ThermoSamplingData
            n: int = 20000,
            num_chains: int = 4,
            thermo_basis_points: np.ndarray | None = None,
    ):
        sampling_data = ThermoPrior._coerce_sampling_data(tfs_model)
        if thermo_basis_points is None:
            thermo_basis_points = find_initial_points(sampling_data, num_chains)
        num_chains = thermo_basis_points.shape[1]
        result = _sample_drg_pure(
            sampling_data,
            initial_points=thermo_basis_points,
            num_samples=n,
            num_chains=num_chains,
            num_initial_steps=n * 2,
        )
        new_points_idx = np.random.choice(n, num_chains)
        new_thermo_basis_points = result.basis_samples.values[new_points_idx, :].T
        return result, new_thermo_basis_points

    def _run_tasks(self, tasks, fn=sample_polytope, break_i=-1, close_pool=True):
        del close_pool
        if self._num_processes > 0:
            return self._mp_pool.starmap(fn, tasks)

        results = []
        for i, task in enumerate(tasks):
            if (break_i >= 0) and (i > break_i):
                break
            results.append(fn(*task))
        return results

    def _make_b_constraint_df(self, orthants):
        forward = orthants.astype(int)
        reverse = (~orthants).astype(int)
        forward.columns += '|ub'
        reverse.columns += '|lb'

        b = self._fcm._Fn.b.copy()

        ub = forward * b.loc[forward.columns]
        lb = reverse * b.loc[reverse.columns]

        # NB this dataframe contains all the bounds for the b vector
        return pd.concat([lb, ub], axis=1).sort_index(axis=1)

    def _make_Fn_tasks(self, drg_result: FreeEnergiesSamplingResult, n_flux=1):
        drg_samples = drg_result.samples.loc[:, self._fcm.fwd_id]
        drg_xch_fluxes = self._fcm.compute_xch_fluxes(dgibbsr=drg_samples)
        drg_orthants = drg_samples < 0.0

        orthants = drg_orthants.value_counts().rename_axis(index=list(drg_orthants.columns)).reset_index(name='counts')
        orthants = orthants.set_index('counts')
        orthants = orthants.loc[orthants.index > 1]
        orthants.index *= n_flux  # this means we take subsamples of the net space

        b_constraint_df = self._make_b_constraint_df(orthants)
        task_generator = sampling_tasks(
            self._fcm._Fn,
            counts=None,
            b_constraint_df=b_constraint_df,
            return_what='all',
        )
        return dict(
            orthants=orthants, drg_orthants=drg_orthants, b_constraint_df=b_constraint_df,
            drg_xch_fluxes=drg_xch_fluxes, sampling_task_generator=task_generator
        )

    def _make_S_constraint_df(self, drg_result: FreeEnergiesSamplingResult, S: pd.DataFrame):
        drg_samples = drg_result.samples.loc[:, self._fcm.fwd_id]
        drg_xch_fluxes = self._fcm.compute_xch_fluxes(dgibbsr=drg_samples)
        S_template = pd.DataFrame(0.0, index=S.index[S.index.str.contains('rho')], columns=self._fcm.fwd_id)
        drg_xch_fluxes.columns += '_rho'
        drg_xch_fluxes = drg_xch_fluxes.clip(0.0, LabellingReaction._RHO_MAX)
        reverse_dir = (drg_samples > 0.0).values
        drg_xch_fluxes.values[reverse_dir] = (1.0 / drg_xch_fluxes).values[reverse_dir]
        rho_idx = np.zeros(S_template.shape, dtype=bool)
        rho_idx[
            [S_template.index.get_loc(key) for key in self._fcm.fwd_id + '_rho'],
            [S_template.columns.get_loc(key) for key in self._fcm.fwd_id]
        ] = True

        S_constraints = OrderedDict()
        for i, name in enumerate(drg_samples.index):
            S = np.zeros(S_template.shape).T
            S[rho_idx.T] = drg_xch_fluxes.iloc[i].values
            S_constraints[name] = pd.DataFrame(S.T, index=S_template.index, columns=S_template.columns)

        return pd.concat(S_constraints.values(), keys=drg_samples.index)

    def _make_A_constraint_df(self, drg_result: FreeEnergiesSamplingResult, A: pd.DataFrame, abs_xch_flux_tol=0.05):
        drg_samples = drg_result.samples.loc[:, self._fcm.fwd_id]
        drg_xch_fluxes = self._fcm.compute_xch_fluxes(dgibbsr=drg_samples)

        A_template = pd.DataFrame(0.0, index=A.index[A.index.str.contains('rho')], columns=self._fcm.fwd_id)

        xch_fluxes_max = drg_xch_fluxes + abs_xch_flux_tol / 2
        drg_xch_fluxes -= abs_xch_flux_tol / 2

        drg_xch_fluxes.columns += '_rho_min|lb'
        xch_fluxes_max.columns += '_rho_max|ub'

        drg_xch_fluxes = drg_xch_fluxes.clip(0.0, LabellingReaction._RHO_MAX)
        xch_fluxes_max = xch_fluxes_max.clip(0.0, LabellingReaction._RHO_MAX)

        reverse_dir = (drg_samples > 0.0).values
        drg_xch_fluxes.values[reverse_dir] = (1.0 / drg_xch_fluxes).values[reverse_dir]
        xch_fluxes_max.values[reverse_dir] = (1.0 / xch_fluxes_max).values[reverse_dir]

        rho_fwd_min_idx = np.zeros(A_template.shape, dtype=bool)
        rho_fwd_min_idx[
            [A_template.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_min|lb'],
            [A_template.columns.get_loc(key) for key in self._fcm.fwd_id]
        ] = True
        rho_fwd_max_idx = np.zeros(A_template.shape, dtype=bool)
        rho_fwd_max_idx[
            [A_template.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_max|ub'],
            [A_template.columns.get_loc(key) for key in self._fcm.fwd_id]
        ] = True

        A_constraints = OrderedDict()
        for i, name in enumerate(drg_samples.index):
            A = np.zeros(A_template.shape).T
            A[rho_fwd_min_idx.T] = drg_xch_fluxes.iloc[i].values
            A[rho_fwd_max_idx.T] = -xch_fluxes_max.iloc[i].values
            A = pd.DataFrame(A.T, index=A_template.index, columns=A_template.columns)
            # NOTE this is essential, otherwise the bounds conflict for rho > 1.0
            A[A < -1.0] = -A[A < -1.0].values
            A[A > 1.0] = -A[A > 1.0].values
            A_constraints[name] = A

        return pd.concat(A_constraints.values(), keys=drg_samples.index)

    def _make_F_tasks(self, drg_result:FreeEnergiesSamplingResult, n_flux=1, abs_xch_flux_tol=0.05):
        pol = self._generate_thermo_constrained_label_pol(abs_xch_flux_tol)
        kwargs = {}
        if abs_xch_flux_tol == 0.0:
            kwargs['S_constraint_df'] = self._make_S_constraint_df(drg_result, pol.S)
        else:
            # TODO not sure this is correct!
            kwargs['A_constraint_df'] = self._make_A_constraint_df(drg_result, pol.A, abs_xch_flux_tol)

        return sampling_tasks(
            pol,
            counts=n_flux,
            return_what='fluxes',
            **kwargs,
        )

    def _fill_caches(
            self, n=20000, n_flux=1, drg_result: FreeEnergiesSamplingResult = None,
            abs_xch_flux_tol=0.00, break_i=-1, close_pool=True,
    ):
        if drg_result is None:
            drg_result, self._thermo_basis_points = self._sample_drg_suppress_output(
                self._tfs_model, n, thermo_basis_points=self._thermo_basis_points
            )
        else:
            new_points_idx = np.random.choice(n, self._num_processes)
            self._thermo_basis_points = drg_result.basis_samples.values[new_points_idx, :].T

        if self._coords == 'thermo':
            tasks_dct = self._make_Fn_tasks(drg_result, n_flux)
            orthants = tasks_dct['orthants']
            drg_orthants = tasks_dct['drg_orthants']
            drg_xch_fluxes = tasks_dct['drg_xch_fluxes']
            sampling_task_generator = tasks_dct['sampling_task_generator']

            results = self._run_tasks(sampling_task_generator, break_i=break_i, close_pool=close_pool)
            net_fluxes = pd.DataFrame(
                torch.cat([r['fluxes'] for r in results], dim=0).detach().cpu().numpy(),
                columns=self._fcm._Fn.A.columns,
            )
            xch_cols = self._fcm.fwd_id + '_xch'
            xch_fluxes = pd.DataFrame(np.nan, index=net_fluxes.index, columns=xch_cols)

            flux_orthants = net_fluxes.loc[:, orthants.columns] > 0.0
            drg_orthants = drg_orthants.loc[:, orthants.columns]

            for (count, orthant), res in zip(orthants.iterrows(), results):
                self._orthant_volumes[hash(tuple(orthant))] = res['log_det_E']
                which_drg = (drg_orthants == orthant).all(1)
                which_flux = (flux_orthants == orthant).all(1)
                if which_flux.sum() < which_drg.sum() * n_flux:
                    # NOTE these orthants have 0 hypervolume I guess, thus flux sampling fails
                    raise ValueError('some orthant was not sampled correctly, make sure that the Fluxspace '
                                     'of self._tfs_model and self._model are the same!')

                # print(which_flux.sum(), which_drg.sum() * n_flux, count, which_flux.sum() < which_drg.sum() * n_flux)
                xch_fluxes.loc[which_flux, xch_cols] = pd.concat(
                    [drg_xch_fluxes.loc[which_drg, :], ] * n_flux, axis=0
                ).values

            net_basis_samples = pd.DataFrame(
                torch.cat([r['rounded'] for r in results], dim=0).detach().cpu().numpy(),
                columns=make_net_theta_id(self._fcm._sampler),
            )
            theta, fluxes = self._fcm.append_xch_flux_samples(
                net_fluxes,
                net_basis_samples,
                xch_fluxes,
                pandalize=True,
            )
            # self.ding = self._fcm.map_theta_2_fluxes(self.theta)

        elif self._coords == 'labelling':
            if self._fcm.include_cofactors:
                raise ValueError('mapping from fluxes to thermo wont work if cofactors are included')
            sampling_task_generator = self._make_F_tasks(drg_result, n_flux, abs_xch_flux_tol)
            results = self._run_tasks(sampling_task_generator, break_i=break_i, close_pool=close_pool)
            fluxes = pd.DataFrame(
                torch.cat([r['fluxes'] for r in results], dim=0).detach().cpu().numpy(),
                columns=self._fcm._F.A.columns,
            )
            theta = self._fcm.map_fluxes_2_theta(fluxes, pandalize=True)

        if isinstance(fluxes, pd.DataFrame):
            fluxes = fluxes.values
        if isinstance(theta, pd.DataFrame):
            theta = theta.values

        self._flux_cache  = torch.as_tensor(fluxes, dtype=torch.double)
        self._theta_cache = torch.as_tensor(theta, dtype=torch.double)

if __name__ == "__main__":
    pass
