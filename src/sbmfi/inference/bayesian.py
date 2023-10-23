from sbmfi.core.simulator import _BaseSimulator, DataSetSim
from sbmfi.inference.priors import _BasePrior, _NetFluxPrior
from sbmfi.core.observation import MDV_ObservationModel, BoundaryObservationModel
from sbmfi.core.model import LabellingModel
import math
import arviz as az
import numpy as np
import pandas as pd
import tqdm
import multiprocessing as mp
import psutil
from typing import Dict
from functools import partial

from sbi.inference.posteriors.base_posterior import NeuralPosterior


# from line_profiler import line_profiler
# prof2 = line_profiler.LineProfiler()
# from sbmfi.core.util import profile
class _BaseBayes(_BaseSimulator):

    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _NetFluxPrior,
            boundary_observation_model: BoundaryObservationModel = None,
    ):
        super(_BaseBayes, self).__init__(model, substrate_df, mdv_observation_models, boundary_observation_model)

        self._prior = prior
        if prior is not None:
            if not prior._fcm.labelling_fluxes_id.equals(model.labelling_fluxes_id):
                raise ValueError('prior has different labelling fluxes than model')
            if not model._fcm.theta_id.equals(prior.theta_id):
                raise ValueError('theta of model and prior are different')

        self._sampler = self._fcm._sampler
        self._K = self._sampler.dimensionality
        self._nx = self._fcm._nx

        self._x_meas = None
        self._x_meas_id = None
        self._true_theta = None
        self._true_theta_id = None

        self._potentype = None
        self._potential_fn = None
        self._sphere_samples = None
        self._A_dist = None

    @property
    def potentype(self):
        if self._potentype is not None:
            return self._potentype[:]

    @property
    def measurements(self):
        return pd.DataFrame(self._la.tonp(self._x_meas), index=self._x_meas_id, columns=self.data_id)

    @property
    def true_theta(self):
        if self._true_theta is None:
            return
        return pd.Series(self._la.tonp(self._true_theta[0]), index=self.theta_id, name=self._true_theta_id)

    def set_measurement(self, x_meas: pd.Series, atol=1e-3):
        if isinstance(x_meas, pd.Series):
            name = 'measurement' if not x_meas.name else x_meas.name
            x_meas = x_meas.to_frame(name=name).T
        x_meas_index = None
        if isinstance(x_meas, pd.DataFrame):
            x_meas_index = x_meas.index
            x_meas = x_meas.values
        if x_meas_index is None:
            x_meas_index = pd.RangeIndex(x_meas.shape[0])
        elif isinstance(x_meas_index, pd.MultiIndex):
            raise ValueError
        self._x_meas = self._la.atleast_2d(self._la.get_tensor(values=x_meas))
        self._x_meas_id = x_meas_index
        if (self._bomsize > 0) and self._bom._check:
            if not self._la.transax((self._bom._A @ self._x_meas[:, -self._bomsize].T <= self._bom._b)).all():
                raise ValueError('boundary measurements are outside polytope')
        x_meas_df = pd.DataFrame(self._la.tonp(self._x_meas), index=x_meas_index, columns=self.data_id)
        for labelling_id, obmod in self._obmods.items():
            obmod.check_x_meas(x_meas_df.loc[:, labelling_id], atol=atol)

    def set_true_theta(self, theta: pd.Series):
        if theta is None:
            return
        if isinstance(theta, pd.DataFrame):
            if theta.shape[0] > 1:
                raise ValueError
            theta = theta.iloc[0]
        self._true_theta = self._la.atleast_2d(self._la.get_tensor(values=theta.loc[self.theta_id].values))
        self._true_theta_id = theta.name

    def simulate_true_data(self, n_obs=0, pandalize=True):
        if self._true_theta is None:
            raise ValueError('set true_theta')
        tt = self._la.tile(self._true_theta.T, (self._la._batch_size, )).T
        true_data = self.simulate(tt, n_obs, pandalize=pandalize)
        if not pandalize:
            return true_data[[0]]
        true_data = true_data.iloc[[0]]
        true_data.index = pd.RangeIndex(true_data.shape[0])
        return true_data

    def log_lik(
            self,
            theta,
            return_data=False,
            sum=True,  # for debugging its useful to have log_lik split out per observation model
    ):

        if not self._is_exact:
            raise ValueError(
                'some observation models do not have a .log_prob, meaning that exact inference is impossible'
            )
        if self._x_meas is None:
            raise ValueError('set measurement')

        mu_o = self.simulate(theta, n_obs=0)

        n_f = self._model._fluxes.shape[0]
        n_meas = self._x_meas.shape[0]
        n_bom = 1 if self._bomsize > 0 else 0

        log_lik = self._la.get_tensor(shape=(n_f, n_meas, len(self._obmods) + n_bom))

        if self._bomsize > 0:
            bo_meas = self._x_meas[:, -self._bomsize:]
            mu_bo = mu_o[:, 0, -self._bomsize:]
            log_lik[..., -1] = self._bom.log_lik(bo_meas=bo_meas, mu_bo=mu_bo)

        for i, (labelling_id, obmod) in enumerate(self._obmods.items()):
            j, k = self._obsize[labelling_id]
            x_meas_o = self._x_meas[..., j:k]
            mu_o_i = mu_o[:, 0, j:k]
            ll = obmod.log_lik(x_meas_o, mu_o_i)
            log_lik[..., i] = ll

        if sum:
            log_lik = self._la.sum(log_lik, axis=(1, 2), keepdims=False)

        if return_data:
            return log_lik, mu_o
        return log_lik

    def log_prob(
            self,
            theta,
            return_data=False,
            evaluate_prior=False,
    ):
        if self._x_meas is None:
            raise ValueError('set an observation first')
        # NB we do not evaluate the log_prob of the measured boundary fluxes, since it is a constant for _x_meas

        vape = theta.shape
        if len(vape) > 2:
            theta = self._la.view(theta, shape=(math.prod(vape[:-1]), vape[-1]))

        n_f = theta.shape[0]
        k = len(self._obmods) + (1 if self._bom is None else 2)  # the 2 is for a column of prior and boundary probabilities
        n_meas = self._x_meas.shape[0]
        log_prob = self._la.get_tensor(shape=(n_f, n_meas, k))

        if evaluate_prior:
            # NB not necessary for uniform prior
            # NB this also checks support! the hr is guaranteed to sample within the support
            # NB since priors are currently torch objects, this will not work with numpy backend
            #   which has proven the faster option for the hr-sampler
            log_prob[..., -1] = self._prior.log_prob(theta)

        log_lik = self.log_lik(theta, return_data, False)
        if return_data:
            log_lik, mu_o = log_lik

        log_prob[..., :-1] = log_lik
        log_prob = self._la.view(self._la.sum(log_prob, axis=(1, 2), keepdims=False), shape=vape[:-1])
        if return_data:
            return log_prob, self._la.view(mu_o, shape=(*vape[:-1], len(self._did)))
        return log_prob

    def compute_distance(
            self,
            theta,
            n_obs=5,
            metric='rmse',
            epsilon=None,
            return_data=False,
            evaluate_prior=False,
            **kwargs
    ):
        if self._x_meas is None:
            raise ValueError('set an observation first')
        # NB we do not evaluate the log_prob of the measured boundary fluxes, since it is a constant for _x_meas

        vape = theta.shape
        if len(vape) > 2:
            theta = self._la.view(theta, shape=(math.prod(vape[:-1]), vape[-1]))
        data = self.__call__(theta, n_obs=n_obs, **kwargs)
        data = self._la.unsqueeze(data, 0)  # artificially add a chains dimension!
        if metric == 'rmse':
            distances = self._fobmod.rmse(data, self._x_meas).squeeze(0)
        else:
            # TODO think of other distance metrics
            raise ValueError

        n_obshape = max(1, n_obs)
        distances = self._la.view(distances, shape=vape[:-1])
        if epsilon is not None:
            distances <= epsilon
            raise NotImplementedError
            # TODO reject all particles that have an
        if evaluate_prior:
            pass
        if return_data:
            return distances, self._la.view(data, shape=(*vape[:-1], n_obshape, len(self._did)))
        return distances

    def evaluate_neural_density(
            self,
            theta,
            potential_fn,
            evaluate_prior=False,
            return_data=False,
    ):
        vape = theta.shape
        theta = self._la.view(theta, shape=(math.prod(vape[:-1]), vape[-1]))
        if return_data:
            raise NotImplementedError(
                'its more efficient to sample all paramters and then use a DataSim to simulate all data'
            )

        raise NotImplementedError

    def _set_potential(
            self,
            potentype,
            potential_fn=None,
            **kwargs
    ):
        if potentype == 'exact':
            if not self._is_exact:
                self.log_lik(None)  # this raises the error
            fun = self.log_prob
            kwargs = dict(
                return_data=kwargs.get('return_data', True),
                evaluate_prior=kwargs.get('evaluate_prior', True),
            )
        elif potentype == 'approx':
            fun = self.compute_distance
            kwargs = dict(
                n_obs=kwargs.get('n_obs', 5),
                metric=kwargs.get('metric', 'rmse'),
                return_data=kwargs.get('return_data', True),
                evaluate_prior=kwargs.get('evaluate_prior', True),
            )
        elif potentype == 'density':
            self.evaluate_neural_density
            kwargs = dict(
                track_gradients=False,
                # when we use sequential neural likelihood, we need to evaluate the prior, in SNPE not
                # evaluate_prior=kwargs.get('evaluate_prior', False),
            )
            fun = potential_fn.__call__
        else:
            raise ValueError
        self._potentype = potentype
        self.potential = partial(fun, **kwargs)

    def set_density(self, density: NeuralPosterior):
        raise NotImplementedError

    def mvn_kernel_variance(
            self,
            theta,
            weights=None,
            samples_per_dim: int = 100,
            kernel_std_scale: float = 1.0,
            prev_cov=True,
            exclude_xch=True,
    ):
        vape = theta.shape
        # shape into matrix with variables along rows and samples in columns
        if len(vape) > 2:
            theta = self._la.view(theta, shape=(math.prod(vape[:-1]), vape[-1])).T  # TODO why .T

        if exclude_xch:
            theta = theta[:self._K]
        else:
            raise NotImplementedError('we dont yet have the option to change the kernel for exchange fluxes')
        if prev_cov:
            # Calculate weighted covariance of particles.
            # For variant C, Beaumont et al. 2009, the kernel variance comes from the
            # previous population.
            population_cov = self._la.cov(theta, aweights=weights)  # rowvar=False,
            # Make sure variance is nonsingular.
            # I'd rather have this crash out if the singular, means that the parameters are not independent
            #    or constrained to a single value
            self._la.cholesky(kernel_std_scale * population_cov)
            return kernel_std_scale * population_cov
        else:
            # Toni et al. and Sisson et al. it comes from the parameter ranges.
            indices = self._la.multinomial(samples_per_dim * theta.shape[1], p=weights)
            samples = theta[indices]
            particle_ranges = self._la.max(samples, 0) - self._la.min(samples, 0)
            return kernel_std_scale * self._la.diag(particle_ranges)

    def _format_line_kernel_kwargs(self, alpha, line_variance, directions):
        alpha_min = self._la.vecopy(alpha)
        alpha_max = self._la.vecopy(alpha)
        alpha_max[alpha_max < 0.0] = alpha_max.max()
        alpha_max = self._la.min(alpha_max, -1)

        alpha_min[alpha_min > 0.0] = alpha_min.min()
        alpha_min = self._la.max(alpha_min, -1)

        # construct proposals along the line-segment and compute the empirical CDF from which we select the next step
        if not isinstance(line_variance, float):
            line_variance = self._la.sum(((directions @ line_variance) * directions), -1)
        return alpha_min, alpha_max, line_variance

    def choose_direction(self):
        pass

    def perturb_particles(
            self,
            theta,
            i,
            batch_shape,
            n_cdf=5,
            chord_proposal='unif',
            chord_std=2.0,
            xch_proposal='gauss',
            xch_std=0.4,
            return_chords=True,
            logit_xch=False,  # for most tasks where this is needed, we enter this function via the hyper-rectangle, so no logit is necessary
    ):
        # TODO implement random coordinate instead of random direction
        # given x, the next point in the chain is x+alpha*r
        #   it also satisfies A(x+alpha*r)<=b which implies A*alpha*r<=b-Ax
        #   so alpha<=(b-Ax)/ar for ar>0, and alpha>=(b-Ax)/ar for ar<0.
        #   b - A @ x is always >= 0, clamping for numerical tolerances

        batch_n = batch_shape[0]
        ii = i % batch_n
        if ii == 0:
            # TODO: https://link.springer.com/article/10.1007/BF02591694
            #  implement coordinate hit-and-run (might be faster??)
            # uniform samples from unit ball in batch_shape dims
            self._sphere_samples = self._la.sample_hypersphere(shape=(*batch_shape, self._K))
            # batch compute distances to all planes
            self._A_dist = self._la.tensormul_T(self._sampler._G, self._sphere_samples)

        sphere_sample = self._sphere_samples[[ii]]
        A_dist = self._A_dist[ii]

        pol_dist = self._sampler._h.T - self._la.tensormul_T(self._sampler._G, theta[..., :self._K])
        pol_dist[pol_dist < 0.0] = 0.0
        allpha = pol_dist / A_dist
        alpha_min, alpha_max = self._la.min_pos_max_neg(allpha, return_what=0)

        # if not isinstance(line_variance, float):
        #     line_variance = self._la.sum(((directions @ line_variance) * directions), -1)

        chord_alphas = self._la.sample_bounded_distribution(
            shape=(n_cdf,), lo=alpha_min, hi=alpha_max, which=chord_proposal, std=chord_std,
        )

        perturbed_particles = theta[..., :self._K] + chord_alphas[..., None] * sphere_sample

        if self._nx > 0:
            # in case there are exchange fluxes, construct them here
            current_xch = theta[..., -self._nx:]
            if self._fcm.logit_xch_fluxes and logit_xch:
                current_xch = self._fcm._expit_xch(current_xch)
            xch_fluxes = self._la.sample_bounded_distribution(
                shape=perturbed_particles.shape[:-1], lo=self._fcm._rho_bounds[:, 0], hi=self._fcm._rho_bounds[:, 1],
                mu=current_xch, which=xch_proposal, std=xch_std,
            )
            if self._fcm.logit_xch_fluxes and logit_xch:
                xch_fluxes = self._fcm._logit_xch(xch_fluxes)
            perturbed_particles = self._la.cat([perturbed_particles, xch_fluxes], dim=-1)

        if return_chords:
            return perturbed_particles, dict(
                chord_alphas=chord_alphas, alpha_min=alpha_min, alpha_max=alpha_max, directions=sphere_sample
            )
        return perturbed_particles

    def quantile_indices(self, distances, quantiles=0.8):
        dist_cumsum = self._la.cumsum(distances, 0)
        dist_cdf = dist_cumsum / dist_cumsum[-1]
        bigger_than_quantile = self._la.get_tensor(values=(dist_cdf >= quantiles), dtype=np.uint8)
        # select the first 1 in the bigger_than_quantile matrix above along the 0 axis
        return self._la.argmax(bigger_than_quantile, 0)

    def _format_dims_coords(self, n_obs=0):
        data_dims = ['data_id']
        coords = {
            'theta_id': self.theta_id.tolist(),
            'measurement_id': self._x_meas_id.tolist(),
            'data_id': [f'{i[0]}: {i[1]}' for i in self.data_id.tolist()],
        }
        if n_obs > 0:
            data_dims = ['obs_idx', 'data_id']
            coords['obs_idx'] = np.arange(n_obs)
        dims = {
            'theta': ['theta_id'],
            'observed_data': ['measurement_id', 'data_id'],
            'data': data_dims,
        }
        return dims, coords

    def simulate_data(
            self,
            inference_data: az.InferenceData = None,
            n=20000,
            theta=None,
            include_predictive=True,
            num_processes=2,
            n_obs=0,
            show_progress=True,
    ):
        model = self._model

        from_prior = theta is None
        if from_prior:
            theta = self._prior.sample(sample_shape=(n,))
            if model._la.backend != 'torch':
                # TODO inconsistency between model and prior LinAlg, where prior has torch backend and model has numpy backend
                theta = self._prior._fcm._la.tonp(theta)

        if (inference_data is None) or not from_prior:
            result = dict(theta=theta[None, :, :])
        else:
            prior_dataset = az.convert_to_dataset(
                {'theta': theta[None, :, :]},
                dims={'theta': ['theta_id']},
                coords={'theta_id': model._fcm.theta_id.tolist()},
            )
            inference_data.add_groups(
                group_dict={'prior': prior_dataset},
            )

        if include_predictive:
            dsim = DataSetSim(
                model=model,
                substrate_df=self._substrate_df,
                mdv_observation_models=self._obmods,
                boundary_observation_model=self._bom,
                num_processes=num_processes,
            )
            fluxes = model._fcm.map_theta_2_fluxes(theta)
            data = dsim.simulate_set(fluxes, n_obs=n_obs, show_progress=show_progress)['data']
            dims = {'data': ['data_id']}
            coords = {'data_id': [f'{i[0]}: {i[1]}' for i in self.data_id.tolist()]}
            if n_obs == 0:
                data = model._la.transax(data, 0, 1)
            else:
                dims['data'] = ['obs_idx', 'data_id']
                data = data[None, :, :, :]

            if (inference_data is None) or not from_prior:
                result['data'] = data
            else:
                prior_dataset = az.convert_to_dataset(
                    {'data': data},
                    dims=dims,
                    coords=coords,
                )
                inference_data.add_groups(
                    group_dict={'prior_predictive': prior_dataset},
                )

        if (inference_data is None) or not from_prior:
            if inference_data is not None:
                print('returning result instead of adding to inference data, since it is not clear where theta originates')
            return result

    def compute_proposal_prob(
            self,
            old_particles,
            new_particles,
            chord_alphas=None,
            chord=None,
            chord_proposal='unif',
            chord_std=1.0,
            xch_proposal='unif',
            xch_std=0.4,
            new_along_chord=True,
    ):
        old_pol = old_particles[..., :self._K]
        new_pol = new_particles[..., :self._K]

        diff = old_pol - new_pol

        dirr = diff
        if new_along_chord:
            dirr = diff[[0]]

        directions = dirr / self._la.norm(dirr, 2, -1, True)

        A_dist = self._la.tensormul_T(self._sampler._G, directions)

        particle_pol_dist = self._la.transax(
            self._sampler._h - self._sampler._G @ self._la.transax(old_pol)
        )

        allpha = particle_pol_dist / A_dist
        alpha_min, alpha_max = self._la.min_pos_max_neg(allpha, return_what=0)
        alpha = diff[..., 0] / directions[..., 0]
        log_probs = self._la.bounded_distribution_log_prob(x=alpha, lo=alpha_min, hi=alpha_max, mu=0.0, std=chord_std, which=chord_proposal)

        if self._nx > 0:
            x_probs = self._la.bounded_distribution_log_prob(x=old_particles[..., :self._K])

        print(directions.shape, diff.shape, alpha.shape, alpha_min.shape, log_probs.shape)
        print(chord['directions'].shape)
        #
        # alpha = diff[..., 0] / directions[..., 0]  # alpha is scalar and is thus the same along all polytope dimensions
        #
        # print(old_particles[..., :self._K])
        # print(self._la.norm(diff, 2, -1, True).shape, old_pol.shape, diff.shape, old_particles.shape)
        # directions = diff / self._la.norm(diff, 2, -1, True)
        # print(directions)


class MCMC(_BaseBayes):

    def _accept_reject(self, potentials, samples, line_alphas, line_kernel, line_variance, xch_kernel, xch_variance):
        # based on: https://www.math.ntnu.no/preprint/statistics/2004/S4-2004.pdf
        # other notation: https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/258340/348197_FULLTEXT01.pdf?sequence=2&isAllowed=y
        if (line_kernel == 'unif') & (xch_kernel == 'unif'):
            # TODO only potentials, since all the all the transition probabilities are equal
            return # TODO sample categorical according to potentials
        std_diff = (line_alphas[:, None, ...] - line_alphas[None, ...]) / line_variance # evaluate transition probability on by diff on standard normal!
        std_log_prob = self._la.eval_std_normal(std_diff)
        P = std_log_prob + potentials
        print(std_diff)
        print(std_log_prob.shape, potentials.shape)
        print(potentials)
        print(std_log_prob)
        print(P)
        if (xch_kernel == 'normal') and (self._nx > 0):
            xch_theta = samples[..., self._K:]
            P_hat += self._la.bounded_distribution_log_prob(
                x=xch_theta,
                lo=self._fcm._rho_bounds[:, 0],
                hi=self._fcm._rho_bounds[:, 1],
                mu=xch_theta, std=xch_variance, which=xch_kernel, log=True,
            )

    def run(
            self,
            initial_points=None,
            n: int = 50,
            n_burn=50,
            thinning_factor=3,
            n_chains: int = 4,
            potentype='exact',
            n_cdf=6,
            chord_proposal='gauss',
            chord_std=1.0,
            xch_proposal='gauss',
            xch_std=0.4,
            return_data=True,
            evaluate_prior=False,
            potential_kwargs={},
            return_az=True,
    ) -> az.InferenceData:
        # TODO: this publication talks about this algo, but has a different acceptance procedure:
        #  doi:10.1080/01621459.2000.10473908
        #  doi:10.1007/BF02591694  Rinooy Kan article

        chord_std = self._la.get_tensor(values=np.array([chord_std]))
        xch_std = self._la.get_tensor(values=np.array([xch_std]))

        if self._fcm._sampler.basis_coordinates == 'transformed':
            raise NotImplementedError('transform the chains to transformed')

        # log_rnds = False
        # accept_idx = self._la.get_tensor(shape=(n_chains,), dtype=np.int64)
        # if algorithm == 'mh':
        #     n_cdf = 1  # means we only evaluate the current and proposed parameters
        #
        #     log_rnds = True
        # elif algorithm not in ['cdf', 'peskun_nd']:
        #     raise ValueError

        batch_size = n_chains * n_cdf
        if (self._la._batch_size != batch_size) or not self._model._is_built:
            # this way the batch processing is corrected
            self._la._batch_size = batch_size
            self._model.build_simulator(**self._fcm.fcm_kwargs)

        chains = self._la.get_tensor(shape=(n, n_chains, len(self._fcm.theta_id)))
        post_probs = self._la.get_tensor(shape=(n, n_chains))
        accept_rate = self._la.get_tensor(shape=(n_chains,), dtype=np.int64)

        if return_data:
            sim_data = self._la.get_tensor(shape=(n, n_chains, len(self.data_id)))

        if initial_points is None:
            net_basis_points = self._sampler.get_initial_points(num_points=n_chains)
            xch_basis_points = self._prior._xch_prior.sample((n_chains, ))
            y = self._la.cat([net_basis_points, xch_basis_points], dim=-1)
        else:
            y = initial_points

        y = self._la.tile(y, (n_cdf, 1))  # remember that the new batch size is n_chains x n_cdf

        self._set_potential(potentype, **dict(return_data=return_data, evaluate_prior=evaluate_prior, **potential_kwargs))
        if (self._potentype == 'approx'):
            # https://www.biorxiv.org/content/10.1101/106450v1.full.pdf
            # TODO for approx, the MH acceptance is just the ratio between prior probabilities and proposal (which is symmetric, so falls out)
            if n_chains > 1:
                raise ValueError(
                    'currently not possible to simulate multiple chains at once '
                    'due to skipping distance under epsilon simulations'
                )
            raise NotImplementedError('this is complicated, since we need to weight samples by the prior somehow')

        chord_ys = self._la.get_tensor(shape=(1 + n_cdf, n_chains, len(self._fcm.theta_id)))
        chord_alphas = self._la.get_tensor(shape=(1 + n_cdf, n_chains))
        chord_post_prob = self._la.get_tensor(shape=(1 + n_cdf, n_chains))
        chord_prop_prob = self._la.get_tensor(shape=(1 + n_cdf, n_chains))
        post_prob = self.potential(y)

        if return_data:
            chord_data = self._la.get_tensor(shape=(1 + n_cdf, n_chains, len(self.data_id)))
            post_prob, data = post_prob
            chord_data[0] = data[:n_chains]

        chord_post_prob[0] = post_prob[:n_chains]  # ordering of the samples from the PDF does not matter for inverse sampling
        theta_selector = self._la.arange(n_chains)

        y = y[: n_chains, :]
        chord_ys[0] = y

        n_tot = n_burn + n * thinning_factor
        biatch = min(5000, n_tot)
        perturb_kwargs = dict(
            batch_shape=(biatch, n_chains),
            n_cdf=n_cdf,
            chord_proposal=chord_proposal,
            chord_std=chord_std,
            xch_proposal=xch_proposal,
            xch_std=xch_std,
            return_chords=True,
        )
        # for i in tqdm.trange(n_tot, ncols=100):
        pbar = tqdm.tqdm(total=n_tot, ncols=100)
        i = 0
        while i < n_tot:
            chord_ys[1:], chord = self.perturb_particles(y, i, **perturb_kwargs)
            chord_alphas[1:] = chord['chord_alphas']
            post_prob = self.potential(chord_ys[1:])

            if self.potentype == 'approx':
                raise NotImplementedError('reject ABC proposals if all are too distant!')

            if return_data:
                post_prob, data = post_prob
                chord_data[1:] = data

            chord_post_prob[1:] = post_prob

            self.compute_proposal_prob(
                old_particles=y,
                new_particles=chord_ys[1:],
                chord=chord,
                chord_proposal=chord_proposal,
                chord_std=chord_std,
                xch_proposal=xch_proposal,
                xch_std=xch_std,
                new_along_chord=True,
            )
            return


            accept_idx = self._accept_reject(chord_post_prob, chord_ys, chord_alphas, chord_proposal, chord_std, xch_proposal, xch_std)
            # if n_cdf == 0:
            #     accept_idx[:] = 0
            #     log_mh_ratio = line_pot[1] - line_pot[0]
            #     accept_idx[rnd <= log_mh_ratio] = 1  # rnd is already in log due to log_rnd=True
            # else:
            #     pass
            # if algorithm == 'cdf':
            #     max_line_pot = line_pot.max(0)
            #     normalized = line_pot - max_line_pot[None, :]
            #     probs = self._la.exp(normalized)  # TODO make sure this does not underflow!
            #     accept_idx = self.quantile_indices(distances=probs, quantiles=rnd)
            # elif algorithm == 'peskun_nd':
            #     # https://www.math.ntnu.no/preprint/statistics/2004/S4-2004.pdf
            #     raise NotImplementedError
            # elif algorithm == 'mh':
            #     accept_idx[:] = 0
            #     log_mh_ratio = line_pot[1] - line_pot[0]
            #     accept_idx[rnd <= log_mh_ratio] = 1  # rnd is already in log due to log_rnd=True
            # else:
            #     raise ValueError

            accepted_probs = chord_post_prob[accept_idx, theta_selector]
            chord_post_prob[0] = accepted_probs  # set the log-probs of the current sample
            y = chord_ys[accept_idx, theta_selector]
            chord_ys[0] = y  # set the log-probs of the current sample
            # alpha = chord_alphas[accept_idx, theta_selector]  # NOTE: the accepted alpha is always 0
            # chord_alphas[0] = alpha
            if return_data:
                data = chord_data[accept_idx, theta_selector]
                chord_data[0] = data

            j = i - n_burn

            if j > 0:
                if n_cdf > 1:
                    accept_idx[accept_idx > 0] = 1
                accept_rate += accept_idx
                if (j > 0) and (j % 1000 == 0):
                    avg_rate = self._la.mean(accept_rate / j)
                    if avg_rate < 0.05:
                        raise ValueError('average acceptance rate under 5%')

            if (j % thinning_factor == 0) and (j > -1):
                k = j // thinning_factor
                post_probs[k] = accepted_probs
                chains[k] = y
                if return_data:
                    sim_data[k] = data

            i += 1
            pbar.update(1)
        pbar.close()

        if not return_az:
            return chains

        posterior_predictive = None
        if return_data:
            posterior_predictive = {
                'data': self._la.transax(sim_data, dim0=1, dim1=0)
            }

        attrs = {
            'potentype': potentype,
            'evaluate_prior': str(evaluate_prior),
            'potential_kwargs': [(k, v if not isinstance(v, bool) else str(v)) for k, v in potential_kwargs.items()],
            'n_burn': n_burn,
            'acceptance_rate': self._la.tonp(accept_rate) / j,
            'thinning_factor': thinning_factor,
            'n_cdf': n_cdf,
            'line_kernel': chord_proposal,
            'line_variance': chord_std,
            'xch_kernel': xch_proposal,
            'xch_variance': xch_std,
        }
        if self.true_theta is not None:
            attrs['true_theta'] = self._la.tonp(self._true_theta)
            attrs['true_theta_id'] = self._true_theta_id

        n_obs = potential_kwargs.get('n_obs', 0)
        dims, coords = self._format_dims_coords(n_obs=n_obs if self.potentype == 'approx' else 0)
        return az.from_dict(
            posterior={
                'theta': self._la.transax(chains, dim0=1, dim1=0)  # chains x draws x param
            },
            dims=dims,
            coords=coords,
            observed_data={
                'observed_data': self.measurements.values
            },
            sample_stats={
                'lp': post_probs.T
            },
            posterior_predictive=posterior_predictive,
            attrs=attrs
        )

    # @staticmethod
    def run_parallel(
            self,
            num_processes=4,
            initial_points=None,
            n: int = 2000,
            n_burn=0,
            thinning_factor=3,
            n_chains: int = 7,
            kernel=None,
            n_cdf=5,
            line_how='unif',
            line_proposal_std=2.0,
            xch_how='gauss',
            xch_proposal_std=0.4,
            return_data=False,
            evaluate_prior=False,
            kernel_kwargs=None
    ) -> az.InferenceData:
        mcmc_kwargs = dict(
            initial_points=initial_points,
            n=n,
            n_burn=n_burn,
            thinning_factor=thinning_factor,
            n_chains=n_chains,
            kernel=kernel,
            n_cdf=n_cdf,
            line_how=line_how,
            line_proposal_std=line_proposal_std,
            xch_how=xch_how,
            xch_proposal_std=xch_proposal_std,
            return_data=return_data,
            evaluate_prior=evaluate_prior,
            kernel_kwargs=kernel_kwargs
        )
        if num_processes < 0:
            num_processes = psutil.cpu_count(logical=False)
        elif num_processes == 0:
            return self.run(**mcmc_kwargs)

        if num_processes > 0:
            pool = mp.Pool(num_processes)
            res = pool.starmap(self.run, [tuple(mcmc_kwargs.values()) for i in range(num_processes)])
            pool.close()
            pool.join()
            return az.concat(res, dim='chain')


class SMC(_BaseBayes):
    # https://www.annualreviews.org/doi/pdf/10.1146/annurev-ecolsys-102209-144621
    #  https://jblevins.org/notes/smc-intro
    #  https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.pdf
    _CHECK_TRANSFORM = False

    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _BasePrior = None,
            boundary_observation_model: BoundaryObservationModel = None,
            num_processes=0,
    ):
        if self._CHECK_TRANSFORM:  # can be switched off if we want to compare to ABC on raw simplex data
            for labelling_id, obsmod in mdv_observation_models.items():
                if obsmod._transformation is None:
                    raise ValueError(f'Observationmodel {obsmod} does not have a transformation and therefore '
                                     f'euclidian distance is not defined (data lies on simplices)')
        super(SMC, self).__init__(model, substrate_df, mdv_observation_models, prior, boundary_observation_model)
        self._num_processes = num_processes
        self._fobmod = next(iter(self._obmods.values()))
        self._dss = DataSetSim(
            model=model,
            substrate_df=self._substrate_df,
            mdv_observation_models=self._obmods,
            boundary_observation_model=self._bom,
            num_processes=num_processes,
        )
        # this is so that we can make compute_distances use the right simulation function
        self.__call__ = partial(self._dss.__call__, close_pool=False)

    def _calculate_new_log_weights(
            self,
            new_particles,
            old_particles,
            old_log_weights,
            line_kernel,
            chord_variance,
            xch_kernel,
            xch_std,
            evaluate_prior=True,
            pbatch=1000,
    ):
        # TODO memeory problems here, so we need to make sure that population batch is available here
        pbatch = min(old_particles.shape[0], pbatch)
        log_probs = self._la.get_tensor(shape=(*new_particles.shape[:-1], *old_particles.shape[:-1]))

        for i in range(0, old_particles.shape[0], pbatch):
            old_batch = old_particles[i: i + pbatch]
            new_pol = new_particles[..., :self._K]
            old_pol = old_batch[..., :self._K]

            diff = old_pol - self._la.unsqueeze(new_pol, -2)  # centered at old_pol!
            directions = diff / self._la.norm(diff, 2, -1, True)
            # A_dist = self._la.transax(self._sampler._G @ self._la.transax(directions))
            A_dist = self._la.tensormul_T(self._sampler._G, directions)

            allpha = self._particle_pol_dist[i: i + pbatch] / A_dist
            alpha_min, alpha_max = self._la.min_pos_max_neg(allpha, return_what=0)
            # alpha_min, alpha_max, line_var = self._format_line_kernel_kwargs(allpha, chord_variance, directions)

            alpha = diff[..., 0] / directions[..., 0]  # alpha is scalar and is thus the same along all polytope dimensions
            log_probs[..., i: i+pbatch] = self._la.bounded_distribution_log_prob(
                alpha, alpha_min, alpha_max, std=line_var, which=line_kernel, log=True,
            )

            if self._nx > 0:
                new_xch = new_particles[..., self._K:]
                old_xch = old_batch[..., self._K:]
                # we sample xch fluxes independently, so they can be evaluated independently!
                #   with correlations, this becomes messy and we would need to apply the sample polytope logic as above
                #   computing and evaluating alphas and all that
                # TODO for sampling and evaluating exchange fluxes, we use a constant kernel
                #   and do not adapt it to the previous population, check whether this leads to funny results in a model
                #   where we can identify xch fluxes
                log_probs[..., i: i+pbatch] += self._la.bounded_distribution_log_prob(
                    x=new_xch,
                    lo=self._fcm._rho_bounds[:, 0],
                    hi=self._fcm._rho_bounds[:, 1],
                    mu=old_xch, std=xch_std, which=xch_kernel, log=True,
                )

        log_weighted_sum = self._la.logsumexp(log_probs + old_log_weights, -1)  # computes importance weights

        if evaluate_prior:
            prior_log_probs = self._prior.log_prob(new_particles)
        else:
            prior_log_probs = -0.1  # for a uniform prior, dont need to evaluate

        return prior_log_probs - log_weighted_sum

    def _sample_next_population(
            self,
            particles,
            log_weights,
            epsilon: float,
            kernel_variance_scale=1.0,
            line_kernel='gauss',
            xch_kernel='gauss',
            xch_variance=0.4,
            population_batch=1000,
            n_cdf=1,
            return_data=True,
            evaluate_prior=False,
    ):
        """Return particles, weights and distances of new population."""

        new_particles = []
        new_log_weights = []
        new_distances = []
        new_data = []

        m = 0
        n = particles.shape[0]
        population_batch = min(n, population_batch)

        line_variance = None
        if line_kernel == 'gauss':
            line_variance = self.mvn_kernel_variance(
                particles,
                weights=self._la.exp(log_weights),
                samples_per_dim=500,
                kernel_std_scale=kernel_variance_scale,
                prev_cov=True,
            )

        self._particle_pol_dist = self._la.transax(
            self._sampler._h - self._sampler._G @ self._la.transax(particles[..., :self._K])
        )
        self._particle_pol_dist[self._particle_pol_dist < 0.0] = 0.0
        pbar = tqdm.tqdm(total=n, ncols=100)
        while m < n:
            # Sample from previous population and perturb.
            sample_indices = self._la.multinomial(population_batch, self._la.exp(log_weights))
            sampled_particles = particles[sample_indices]

            perturbed_particles = self.perturb_particles(
                theta=sampled_particles,
                i=0,
                batch_shape=(1, population_batch),
                n_cdf=n_cdf,
                chord_proposal=line_kernel,
                chord_std=line_variance,
                xch_proposal=xch_kernel,
                xch_std=xch_variance,
            ).squeeze(0)

            dist = self.potential(perturbed_particles)
            if return_data:
                dist, data = dist

            is_accepted = dist <= epsilon
            num_accepted_batch = is_accepted.sum()

            if num_accepted_batch > 0:
                pbar.update(self._la.tonp(num_accepted_batch))
                accepted_particles = perturbed_particles[is_accepted]
                new_particles.append(accepted_particles)
                new_distances.append(dist[is_accepted])
                new_log_weights.append(
                    self._calculate_new_log_weights(
                        new_particles=accepted_particles,
                        old_particles=particles,
                        old_log_weights=log_weights,
                        evaluate_prior=evaluate_prior,
                        line_kernel=line_kernel,
                        chord_variance=line_variance,
                        xch_kernel=xch_kernel,
                        xch_std=xch_variance,
                        pbatch=population_batch,
                    )
                )
                if return_data:
                    new_data.append(data[is_accepted])
                m += num_accepted_batch
        pbar.close()

        # collect lists of tensors into tensors
        new_distances   = self._la.cat(new_distances)
        sort_idx        = self._la.argsort(new_distances)

        new_distances   = new_distances[sort_idx][:n]
        new_particles   = self._la.cat(new_particles)[sort_idx][:n]
        new_log_weights = self._la.cat(new_log_weights)[sort_idx][:n]
        if return_data:
            new_data    = self._la.cat(new_data)[sort_idx][:n]

        # normalize the new weights
        new_log_weights -= self._la.logsumexp(new_log_weights, dim=0)

        return (
            new_particles,
            new_log_weights,
            new_distances,
            new_data,
        )

    def run(
            self,
            n_smc_steps=3,
            n=100,
            n_obs=5,
            n0_multiplier=2,
            population_batch=1000,
            distance_based_decay=True,
            epsilon_decay=0.8,
            kernel_variance_scale=1.0,
            evaluate_prior=False,
            potentype='approx',
            return_data=True,
            potential_kwargs={},
            metric='rmse',
            line_kernel='gauss',
            xch_kernel='gauss',
            xch_variance=0.4,
            return_all_populations=False,
            return_az=True,
    ):

        self._set_potential(potentype, **dict(n_obs=n_obs, metric=metric, return_data=return_data, **potential_kwargs))
        if self._potentype != 'approx':
            raise NotImplementedError(
                'think about what it means for non-approximate potential '
                'where we do not need to reject stuff below epsilon'
            )

        data = None
        if n_smc_steps < 2:
            raise ValueError
        try:
            for i in range(n_smc_steps):
                if i == 0:
                    prior_theta = self._prior.sample(sample_shape=(n * n0_multiplier, ))
                    if self._la.backend != 'torch':
                        prior_theta = self._prior._fcm._la.tonp(prior_theta)
                    dist = self.potential(prior_theta, fluxes_per_task=500, show_progress=True)
                    if return_data:
                        dist, data = dist
                        prior_data = data

                    sortidx = self._la.argsort(dist)
                    particles = prior_theta[sortidx][:n]
                    dist = dist[sortidx][:n]
                    epsilon = dist[-1]
                    log_weights = self._la.log(1 / n * self._la.ones(n))

                    if return_all_populations:
                        all_particles = [particles]
                        all_log_weights = [log_weights]
                        all_distances = [dist]
                        all_epsilons = [epsilon]
                        if return_data:
                            all_data = [data[sortidx][:n]]
                else:
                    if distance_based_decay:
                        # Quantile of last population
                        epsidx = self.quantile_indices(dist, quantiles=epsilon_decay)
                        epsilon = dist[epsidx]
                    else:
                        # Constant decay.
                        epsilon *= epsilon_decay

                    particles, log_weights, dist, data = self._sample_next_population(
                        particles=particles,
                        log_weights=log_weights,
                        epsilon=epsilon,
                        population_batch=population_batch,
                        kernel_variance_scale=kernel_variance_scale,
                        line_kernel=line_kernel,
                        xch_kernel=xch_kernel,
                        xch_variance=xch_variance,
                        return_data=return_data,
                        evaluate_prior=evaluate_prior,
                    )
                    if return_all_populations:
                        all_particles.append(particles)
                        all_log_weights.append(log_weights)
                        all_distances.append(dist)
                        all_epsilons.append(epsilon)
                        if return_data:
                            all_data.append(data)
        except:
            if return_all_populations:
                kwargs = dict(
                    all_particles=all_particles,
                    all_log_weights=all_log_weights,
                    all_distances=all_distances,
                    all_epsilons=all_epsilons
                )
                if return_data:
                    kwargs['all_data'] = all_data
            else:
                kwargs = dict(particles=particles, log_weights=log_weights, dist=dist, epsilon=epsilon)
                if return_data:
                    kwargs['data'] = data
            # could be used to reset the SMC algo!
            pickle.dump(kwargs, open('FAILUREPOPULATIONS.p', 'wb'))
        if return_all_populations:
            particles = self._la.stack(all_particles, 0)
            log_weights = self._la.stack(all_log_weights, 0)
            dist = self._la.stack(all_distances, 0)
            epsilon = self._la.stack(all_epsilons, 0)
            if return_data:
                data = self._la.stack(all_data, 0)
        else:
            # add the 'chains' dimension
            particles = particles[None, ...]
            log_weights = log_weights[None, ...]
            dist = dist[None, ...]
            if return_data:
                data = data[None, ...]

        if not return_az:
            return particles

        posterior_predictive, prior_predictive = None, None
        if return_data:
            posterior_predictive = {
                'data': data
            }
            prior_predictive = {
                'data': prior_data[None, ...],  # add the 'chains' dimension
            }

        attrs = {
            'potentype': potentype,
            'population_batch': population_batch,
            'epsilons': self._la.tonp(epsilon),
            'evaluate_prior': str(evaluate_prior),
            'n_smc_steps': n_smc_steps,
            'potential_kwargs': [(k, v if not isinstance(v, bool) else str(v)) for k, v in potential_kwargs.items()],
            'n_obs': n_obs,
            'kernel_variance_scale': kernel_variance_scale,
            'n0_multiplier': n0_multiplier,
            'distance_based_decay': str(distance_based_decay),
            'metric': metric,
            'epsilon_decay': epsilon_decay,
            'line_kernel': line_kernel,
            'xch_kernel': xch_kernel,
            'xch_variance': xch_variance,
        }
        if self.true_theta is not None:
            attrs['true_theta'] = self._la.tonp(self._true_theta)
            attrs['true_theta_id'] = self._true_theta_id

        dims, coords = self._format_dims_coords(n_obs=n_obs if self.potentype == 'approx' else 0)

        return az.from_dict(
            posterior={
                'theta': particles  # chains x draws x param
            },
            prior={
                'theta': prior_theta[None, ...],  # add the 'chains' dimension
            },
            dims=dims,
            coords=coords,
            observed_data={
                'observed_data': self.measurements.values,
            },
            sample_stats={
                'log_weights': log_weights,
                'distances': dist,
            },
            posterior_predictive=posterior_predictive,
            prior_predictive=prior_predictive,
            attrs=attrs
        )


def check_stuff():
    model, kwargs = spiro(
        backend='numpy',
        batch_size=1, which_measurements='lcms', build_simulator=True, which_labellings=list('CD'),
        v2_reversible=True, logit_xch_fluxes=False, include_bom=True, seed=3,
    )
    sdf = kwargs['substrate_df']
    dss = kwargs['datasetsim']
    simm = dss._obmods
    bom = dss._bom
    up = UniformNetPrior(model, cache_size=2000)
    # from botorch.utils.sampling import sample_polytope
    mcmc = SMC(
        model=model,
        substrate_df=sdf,
        mdv_observation_models=simm,
        boundary_observation_model=bom,
        prior=up,
    )
    algo = 'rej'
    mcmc.set_measurement(x_meas=kwargs['measurements'])
    mcmc.set_true_theta(theta=kwargs['theta'])
    run_kwargs = dict(
        epsilon=5.0,
        n=50, n_burn=0, n_chains=4, thinning_factor=2, n_cdf=4, return_data=True, line_proposal_std=5.0,
        evaluate_prior=False, algorithm=algo
    )

    n_obs = 2
    result = dss.simulate_prior_predictive(mcmc, num_processes=0, n_obs=n_obs, n=500, include_prior_predictive=False)
    fluxes = model._fcm.map_theta_2_fluxes(result['theta'])[0]
    sims = dss.simulate_set(fluxes, n_obs=n_obs, what='mdv')
    mdvs = sims['mdv']
    results = []
    for i, (lid, obmod) in enumerate(dss._obmods.items()):
        part_mdvss = obmod.compute_observations(mdvs[:, i, :], pandalize=True)
        part_mdvss_df = part_mdvss.loc[np.sort(np.repeat(np.arange(part_mdvss.shape[0]), n_obs))].reset_index(drop=True)
        logobs = obmod._la.log10(obmod._la.get_tensor(values=part_mdvss.values) + 1e-12)
        logI = logobs + obmod._log_scaling  # in log space, multiplication is addition
        noisy_observations = obmod._la.tile(logI[:, None, :], (1, n_obs, 1))
        sigma_x = obmod.construct_sigma(logI=logI)
        obmod.set_sigma(sigma=sigma_x, verify=False)
        noise_samples = obmod.sample_sigma(shape=(n_obs,))
        noisy_observations += noise_samples
        noisy_observations = 10 ** noisy_observations
        noisy_observations1 = obmod._la.clip(noisy_observations, 750.0, None)
        noisy_observations2 = obmod.compute_observations(noisy_observations1, select=False)

        transformed = obmod._transformation(noisy_observations2)

        back = obmod._transformation.inv(transformed)

        shaper = noisy_observations.shape
        newshape = (math.prod(shaper[:-1]), shaper[-1])

        shaper2 = transformed.shape
        newshape2 = (math.prod(shaper2[:-1]), shaper2[-1])

        logobs = pd.DataFrame(logobs, columns=part_mdvss.columns)
        logI = pd.DataFrame(logI, columns=part_mdvss.columns)
        noise_samples = pd.DataFrame(noise_samples.reshape(newshape), columns=part_mdvss.columns)
        sigma_x = pd.concat([pd.DataFrame(sigma_x[i], index=part_mdvss.columns, columns=part_mdvss.columns) for i in
                             range(fluxes.shape[0])], axis=0)
        # sigma_x.to_excel('sx.xlsx')
        noisy_observations = pd.DataFrame(noisy_observations.reshape(newshape), columns=part_mdvss.columns)
        noisy_observations1 = pd.DataFrame(noisy_observations1.reshape(newshape), columns=part_mdvss.columns)
        noisy_observations2 = pd.DataFrame(noisy_observations2.reshape(newshape), columns=part_mdvss.columns)
        transformed = pd.DataFrame(transformed.reshape(newshape2), columns=obmod._transformation.transformation_id)
        back = pd.DataFrame(back.reshape(newshape), columns=part_mdvss.columns)

        diff = abs(back - part_mdvss_df)

    printo = True
    if printo:
        data = result['data']  # data.flatten(start_dim=0, end_dim=-2)
        shaper = data.shape
        new_shape = (math.prod(shaper[:-1]), shaper[-1])
        transformed = data.reshape(new_shape)

        part_mdvs = mcmc.to_partial_mdvs(transformed)

        fluxes = model._fcm.map_theta_2_fluxes(result['theta'])[0]
        sims = dss.simulate_set(fluxes, n_obs=2, what='all')

        dang = mcmc.to_partial_mdvs(sims['mdv'], is_mdv=True)
        dang = dang.loc[np.sort(np.repeat(np.arange(dang.shape[0]), n_obs))].reset_index(drop=True)

        mdvs = pd.concat([pd.DataFrame(mdvs[:, i, :], columns=model.state_id) for i in range(mdvs.shape[1])], axis=1)
        diff = abs(part_mdvs - dang)


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    np.set_printoptions(linewidth=500)

    from sbmfi.models.small_models import spiro
    from sbmfi.models.build_models import build_e_coli_anton_glc
    from sbmfi.inference.priors import UniformNetPrior, ProjectionPrior
    from sbmfi.core.polytopia import FluxCoordinateMapper, PolytopeSamplingModel, sample_polytope, fast_FVA
    import pickle

    model, kwargs = spiro(backend='torch', v2_reversible=True, ratios=False, build_simulator=True, which_measurements='com', which_labellings=['A'])
    sdf = kwargs['substrate_df']
    bb = kwargs['basebayes']
    up = UniformNetPrior(model._fcm)
    mcmc = MCMC(model, sdf, bb._obmods, prior=up, boundary_observation_model=bb._bom)
    mcmc.set_measurement(x_meas=kwargs['measurements'])
    mcmc.set_true_theta(theta=kwargs['theta'])
    res = mcmc.run(
        n=2,
        n_burn=0,
        thinning_factor=1,
        n_chains=3,
        n_cdf=2,
    )


    # model, kwargs = build_e_coli_anton_glc(backend='torch', which_measurements=None)
    # projected_fluxes = kwargs['measured_boundary_fluxes']
    # model.reactions.get_by_id('EX_glc__D_e').bounds = (-12.0, 0.0)

    # fcm = FluxCoordinateMapper(model, kernel_basis='svd', basis_coordinates='rounded', free_reaction_id=projected_fluxes)

    # up = ProjectionPrior(fcm, projected_fluxes=projected_fluxes, cache_size=2000, number_type='fraction', num_processes=0)
    # projection_pol = up._projection_pol
    # pickle.dump(projection_pol, open('anton_glc_projection_pol.p', 'wb'))

    # projection_pol = pickle.load(open('anton_glc_projection_pol.p', 'rb'))
    # up = ProjectionPrior(model, projection_pol=projection_pol, projected_fluxes=projected_fluxes, cache_size=2000, number_type='fraction', num_processes=3)
    # up._fill_caches(n=5000, enumerate_vertices=True)
    # pickle.dump(up, open('anton_glc_projection_prior_w_volumes.p', 'wb'))
    # up = pickle.load(open('anton_glc_projection_prior_w_volumes.p', 'rb'))

    # model, kwargs = build_e_coli_anton_glc(
    #     backend='torch',
    #     auto_diff=False,
    #     build_simulator=True,
    #     ratios=False,
    #     batch_size=25,
    #     which_measurements='tomek',
    #     which_labellings=['20% [U]Glc', '[1]Glc'],
    #     measured_boundary_fluxes=[_bmid_ANTON, 'EX_glc__D_e', 'EX_ac_e'],
    #     seed=1,
    # )
    #
    # sdf = kwargs['substrate_df']
    # dss = kwargs['basebayes']
    # simm = dss._obmods
    # bom = dss._bom
    # up = UniFluxPrior(model, cache_size=2000)
    #
    # smc = SMC(
    #     model=model,
    #     substrate_df=sdf,
    #     mdv_observation_models=simm,
    #     boundary_observation_model=bom,
    #     prior=up,
    #     num_processes=0,
    # )
    # smc.set_measurement(x_meas=kwargs['measurements'])
    # smc.set_true_theta(theta=kwargs['theta'])

    # a = False
    # if a:
    #     which = 'lcms'
    #     # which = 'tomek'
    #     ding = SMC
    #     run_kwargs = dict(
    #         n_smc_steps=3,
    #         n=50,
    #         n_obs=5,
    #         population_batch=7,
    #         n0_multiplier=2,
    #         distance_based_decay=True,
    #         epsilon_decay=0.8,
    #         kernel_variance_scale=1.0,
    #         evaluate_prior=False,
    #         potentype='approx',
    #         return_data=True,
    #         potential_kwargs={},
    #         metric='rmse',
    #         line_kernel='gauss',
    #         xch_kernel='gauss',
    #         xch_std=0.4,
    #         return_all_populations=False,
    #     )
    # else:
    #     which = 'com'
    #     # which = 'anton'
    #     ding = MCMC
    #     run_kwargs = dict(
    #         n= 20,
    #         n_burn = 10,
    #         thinning_factor = 2,
    #         n_chains = 1,
    #         potentype = 'exact',  # TODO: this should either accpet a normalizing flow or even a distance for MCMC-ABC
    #         n_cdf = 2,
    #         # algorithm = 'mh',  # TODO: https://www.math.ntnu.no/preprint/statistics/2004/S4-2004.pdf different acceptance step from eCDF!!
    #         chord_proposal = 'gauss',
    #         chord_std = 1.0,
    #         xch_proposal = 'gauss',
    #         xch_std = 0.4,
    #         return_data = True,
    #         evaluate_prior = False,
    #         potential_kwargs = {},
    #         return_az = True,
    #     )
    #
    #
    # model, kwargs = spiro(
    #     backend='numpy',
    #     batch_size=3, which_measurements=which, build_simulator=True, which_labellings=list('CD'),
    #     v2_reversible=True, logit_xch_fluxes=False, include_bom=True, seed=2, L_12_omega=1.0,
    #     v5_reversible=True
    # )
    # # model, kwargs = build_e_coli_anton_glc(
    # #     backend='torch',
    # #     auto_diff=False,
    # #     build_simulator=True,
    # #     ratios=False,
    # #     batch_size=10,
    # #     which_measurements=which,
    # #     which_labellings=['20% [U]Glc', '[1]Glc'],
    # #     measured_boundary_fluxes=[_bmid_ANTON, 'EX_glc__D_e', 'EX_ac_e'],
    # #     seed=1,
    # # )
    # sdf = kwargs['substrate_df']
    # dss = kwargs['basebayes']
    # simm = dss._obmods
    # bom = dss._bom
    # up = UniFluxPrior(model, cache_size=2000)
    # # from botorch.utils.sampling import sample_polytope
    #
    # mcmc = ding(
    #     model=model,
    #     substrate_df=sdf,
    #     mdv_observation_models=simm,
    #     boundary_observation_model=bom,
    #     prior=up,
    # )
    # mcmc.set_measurement(x_meas=kwargs['measurements'])
    # mcmc.set_true_theta(theta=kwargs['theta'])
    #
    # post = mcmc.run(**run_kwargs)
    # # az.to_netcdf(post, 'MCMC_e_coli_glc_anton_obsmod.nc')
    # # print(prof2.print_stats())
    #
    # # mcmc_data = az.from_netcdf(r"C:\python_projects\sbmfi\MCMC_e_coli_glc_anton_obsmod.nc")
    # # mcmc_data = az.from_netcdf(r"C:\python_projects\sbmfi\src\sbmfi\inference\MCMC_e_coli_glc_anton_obsmod.nc")
    #
    # # if 'prior' in mcmc_data:
    # #     del mcmc_data.prior
    # # if 'prior_predictive' in mcmc_data:
    # #     del mcmc_data.prior_predictive
    #
    # # make_model = True
    # # if make_model:
    # #     model, kwargs = build_e_coli_anton_glc(
    # #         backend='torch',
    # #         auto_diff=False,
    # #         build_simulator=True,
    # #         ratios=False,
    # #         batch_size=25,
    # #         which_measurements='anton',
    # #         which_labellings=['20% [U]Glc', '[1]Glc'],
    # #         measured_boundary_fluxes=[_bmid_ANTON, 'EX_glc__D_e', 'EX_ac_e'],
    # #         seed=1,
    # #     )
    # #     sdf = kwargs['substrate_df']
    # #     dss = kwargs['basebayes']
    # #     simm = dss._obmods
    # #     bom = dss._bom
    # #     up = UniFluxPrior(model, cache_size=2000)
    # #
    # #     mcmc = MCMC(
    # #         model=model,
    # #         substrate_df=sdf,
    # #         mdv_observation_models=simm,
    # #         boundary_observation_model=bom,
    # #         prior=up,
    # #     )
    # #     mcmc.set_measurement(x_meas=kwargs['measurements'])
    # #     mcmc.set_true_theta(theta=kwargs['theta'])
    # #     mcmc.simulate_data(mcmc_data, n=60000)


