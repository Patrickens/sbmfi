from sbi.inference.abc.smcabc import SMCABC
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbmfi.inference.simulator import _BaseSimulator, simulate_prior_predictive, DataSetSim
from sbmfi.inference.priors import _BasePrior, UniFluxPrior
from sbmfi.core.model import LabellingModel
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
import torch


class _BaseBayes(_BaseSimulator):
    def log_lik(
            self,
            fluxes,
            return_data=False,
            sum=True,  # for debugging its useful to have log_lik split out per observation model
    ):
        if not self._is_exact:
            raise ValueError(
                'some observation models do not have a .log_prob, meaning that exact inference is impossible'
            )
        if self._x_meas is None:
            raise ValueError('set measurement')

        mu_o = self.simulate(fluxes, n_obs=0)

        n_f = fluxes.shape[0]
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
            value,
            return_data=False,
            evaluate_prior=False,
    ):
        if self._x_meas is None:
            raise ValueError('set an observation first')
        # NB we do not evaluate the log_prob of the measured boundary fluxes, since it is a constant for _x_meas

        vape = value.shape
        theta = self._la.view(value, shape=(math.prod(vape[:-1]), vape[-1]))

        n_f = theta.shape[0]
        k = len(self._obmods) + (
            1 if self._bom is None else 2)  # the 2 is for a column of prior and boundary probabilities
        n_meas = self._x_meas.shape[0]
        log_prob = self._la.get_tensor(shape=(n_f, n_meas, k))
        fluxes = self._fcm.map_theta_2_fluxes(theta)

        if evaluate_prior:
            # NB not necessary for uniform prior
            # NB this also checks support! the hr is guaranteed to sample within the support
            # NB since priors are currently torch objects, this will not work with numpy backend
            #   which has proven the faster option for the hr-sampler
            log_prob[..., -1] = self._prior.log_prob(theta)

        log_lik = self.log_lik(fluxes, return_data, False)
        if return_data:
            log_lik, mu_o = log_lik

        log_prob[..., :-1] = log_lik
        log_prob = self._la.view(self._la.sum(log_prob, axis=(1, 2), keepdims=False), shape=vape[:-1])
        if return_data:
            return log_prob, self._la.view(mu_o, shape=(*vape[:-1], len(self._did)))
        return log_prob

    def compute_distance(
            self,
            value,
            n_obs=5,
            metric='rmse',
            return_data=False,
    ):
        if self._x_meas is None:
            raise ValueError('set an observation first')
        # NB we do not evaluate the log_prob of the measured boundary fluxes, since it is a constant for _x_meas

        vape = value.shape
        theta = self._la.view(value, shape=(math.prod(vape[:-1]), vape[-1]))
        fluxes = self._fcm.map_theta_2_fluxes(theta)
        result = self._dss.simulate_set(fluxes, n_obs=n_obs, close_pool=False)
        data = self._la.unsqueeze(result['data'], 0)  # artificially add a chains dimension!
        if metric == 'rmse':
            distances = self._fobmod.rmse(data, self._x_meas).squeeze(0)
        else:
            # TODO think of other distance metrics
            raise ValueError

        n_obshape = max(1, n_obs)
        distances = self._la.view(distances, shape=vape[:-1])
        if return_data:
            return distances, self._la.view(data, shape=(*vape[:-1], n_obshape, len(self._did)))
        return distances

    def evaluate_neural_density(self):
        raise NotImplementedError

    def _set_potential(self, potentype, **kwargs):
        if potentype == 'exact':
            if not self._is_exact:
                self.log_lik(None)  # this raises the error
            potentype = self.log_prob
            pot_kwargs = dict(
                return_data=kwargs.get('return_data', True),
                evaluate_prior=kwargs.get('evaluate_prior', True),
            )
        elif potentype == 'approx':
            potentype = self.compute_distance
            pot_kwargs = dict(
                n_obs=kwargs.get('n_obs', 5),
                metric=kwargs.get('metric', 'rmse'),
                return_data=kwargs.get('return_data', True)
            )
        elif potentype == 'density':
            raise NotImplementedError
        else:
            raise ValueError

        self.potential = partial(potentype, **pot_kwargs)

    def set_density(self, density: NeuralPosterior):
        pass

    def _get_directions(self, i, shape: tuple, randos=True, log_rnd=False):
        # TODO: https://link.springer.com/article/10.1007/BF02591694
        #  implement coordinate hit-and-run (might be faster??)
        # uniform samples from unit ball in d dims
        batch = shape[0]
        if i % batch == 0:
            self._sphere_samples = self._la.sample_hypersphere(shape=(*shape, self._sampler.dimensionality))
            # batch compute distances to all planes
            self._A_dist = self._la.transax(self._sampler._G @ self._la.transax(self._sphere_samples))
            if randos:
                self._rnds = self._la.randu(shape, dtype=self._sampler._G.dtype)
                if log_rnd:
                    self._rnds = self._la.log(self._rnds)
        if randos:
            return self._sphere_samples[i % batch], self._A_dist[i % batch], self._rnds[i % batch]
        return self._sphere_samples[i % batch], self._A_dist[i % batch]

    def mvn_kernel_variance(
            self,
            value,
            weights=None,
            samples_per_dim: int = 100,
            kernel_variance_scale: float = 1.0,
            prev_cov=True,
            exclude_xch=True,
    ):
        vape = value.shape
        # shape into matrix with variables along rows and samples in columns
        theta = self._la.view(value, shape=(math.prod(vape[:-1]), vape[-1])).T
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
            self._la.cholesky(kernel_variance_scale * population_cov)
            return kernel_variance_scale * population_cov
        else:
            # Toni et al. and Sisson et al. it comes from the parameter ranges.
            indices = self._la.multinomial(samples_per_dim * theta.shape[1], p=weights)
            samples = theta[indices]
            particle_ranges = self._la.max(samples, 0) - self._la.min(samples, 0)
            return kernel_variance_scale * self._la.diag(particle_ranges)

    def _format_line_kernel_kwargs(self, alpha, line_variance, directions):
        alpha_min = alpha
        alpha_max = self._la.vecopy(alpha)
        alpha_max[alpha_max < 0.0] = alpha_max.max()
        alpha_max = self._la.min(alpha_max, -1)

        alpha_min[alpha_min > 0.0] = alpha_min.min()
        alpha_min = self._la.max(alpha_min, -1)

        # construct proposals along the line-segment and compute the empirical CDF from which we select the next step
        if not isinstance(line_variance, float):
            line_variance = self._la.sum(((directions @ line_variance) * directions), -1)
        return alpha_min, alpha_max, line_variance

    def perturb_particles(
            self,
            theta,
            sphere_sample,
            A_dist,
            n_cdf=5,
            line_kernel='uniform',
            line_variance=2.0,
            xch_kernel='gauss',
            xch_variance=0.4,
    ):
        # TODO implement random coordinate instead of random direction
        # given x, the next point in the chain is x+alpha*r
        #   it also satisfies A(x+alpha*r)<=b which implies A*alpha*r<=b-Ax
        #   so alpha<=(b-Ax)/ar for ar>0, and alpha>=(b-Ax)/ar for ar<0.
        #   b - A @ x is always >= 0, clamping for numerical tolerances

        pol_dist = self._la.transax(self._sampler._h - self._sampler._G @ theta[..., :self._K].T)
        pol_dist[pol_dist < 0.0] = 0.0
        alpha = pol_dist / A_dist
        print(alpha.shape)
        alpha_min, alpha_max, line_variance = self._format_line_kernel_kwargs(alpha, line_variance, sphere_sample)
        print(alpha_min.shape, alpha_max.shape, line_variance.shape)

        line_alphas = self._la.sample_bounded_distribution(
            shape=(n_cdf,), lo=alpha_min, hi=alpha_max, which=line_kernel, std=line_variance
        )
        print(123, alpha.shape, line_variance.shape, line_alphas.shape, pol_dist.shape, A_dist.shape)
        net_basis_points = theta[..., :self._K] + line_alphas[..., None] * sphere_sample

        xch_fluxes = None
        if self._n_rev > 0:
            # in case there are exchange fluxes, construct them here
            current_xch = theta[..., -self._n_rev:]
            if self._fcm.logit_xch_fluxes:
                current_xch = self._fcm._sigmoid_xch(current_xch)
            xch_fluxes = self._la.sample_bounded_distribution(
                shape=net_basis_points.shape[:-1], lo=self._fcm._rho_bounds[:, 0], hi=self._fcm._rho_bounds[:, 1],
                mu=current_xch, which=xch_kernel, std=xch_variance
            )

        return self._fcm.append_xch_flux_samples(
            net_basis_samples=net_basis_points, xch_fluxes=xch_fluxes, return_type='theta'
        )

    def quantile_indices(self, distances, quantiles=0.8):
        dist_cumsum = self._la.cumsum(distances, 0)
        dist_cdf = dist_cumsum / dist_cumsum[-1]
        bigger_than_quantile = self._la.get_tensor(values=(dist_cdf >= quantiles), dtype=np.uint8)
        # select the first 1 in the bigger_than_quantile matrix above along the 0 axis
        return self._la.argmax(bigger_than_quantile, 0)


class MCMC(_BaseBayes):
    def run(
            self,
            initial_points=None,
            n: int = 50,
            n_burn=50,
            thinning_factor=3,
            n_chains: int = 9,
            potentype='exact',  # TODO: this should either accpet a normalizing flow or even a distance for MCMC-ABC
            n_cdf=6,
            algorithm='mh',
            line_kernel='gauss',
            line_variance=2.0,
            xch_kernel='gauss',
            xch_variance=0.4,
            return_data=True,
            evaluate_prior=False,
            potential_kwargs={},
    ) -> az.InferenceData:
        # TODO: this publication talks about this algo, but has a different acceptance procedure:
        #  doi:10.1080/01621459.2000.10473908
        #  doi:10.1007/BF02591694  Rinooy Kan article

        if self._fcm._sampler.basis_coordinates == 'transformed':
            raise NotImplementedError('transform the chains to transformed')

        log_rnds = False
        if algorithm == 'mh':
            n_cdf = 1  # means we only evaluate the current and proposed parameters
            accept_idx = self._la.get_tensor(shape=(n_chains, ), dtype=np.int64)
            log_rnds = True
        elif algorithm != 'cdf':
            raise ValueError

        batch_size = n_chains * n_cdf
        if (self._la._batch_size != batch_size) or not self._model._is_built:
            # this way the batch processing is corrected
            self._la._batch_size = batch_size
            self._model.build_simulator(**self._fcm.fcm_kwargs)

        chains = self._la.get_tensor(shape=(n, n_chains, len(self._fcm.theta_id)))
        kernel_dist = self._la.get_tensor(shape=(n, n_chains))
        accept_rate = self._la.get_tensor(shape=(n_chains,), dtype=np.int64)

        sim_data = None
        if return_data:
            sim_data = self._la.get_tensor(shape=(n, n_chains, len(self.data_id)))

        if initial_points is None:
            net_basis_points = self._sampler.get_initial_points(num_points=n_chains)
            theta = self._fcm.append_xch_flux_samples(net_basis_samples=net_basis_points, return_type='theta')
        else:
            theta = initial_points

        theta = self._la.tile(theta, (n_cdf, 1))  # remember that the new batch size is n_chains x n_cdf

        self._set_potential(potentype, **dict(return_data=return_data, evaluate_prior=evaluate_prior, **potential_kwargs))

        line_thetas = self._la.get_tensor(shape=(1 + n_cdf, n_chains, len(self._fcm.theta_id)))
        line_dist = self._la.get_tensor(
            shape=(1 + n_cdf, n_chains))
        dist = self.potential(theta)
        if return_data:
            line_data = self._la.get_tensor(shape=(1 + n_cdf, n_chains, len(self.data_id)))
            dist, data = dist
        line_dist[0] = dist[:n_chains]  # ordering of the samples from the PDF does not matter for inverse sampling
        theta_selector = self._la.arange(n_chains)

        theta = theta[: n_chains, :]

        n_tot = n_burn + n * thinning_factor
        biatch = min(2500, n_tot)
        perturb_kwargs = dict(
            n_cdf=n_cdf,
            line_kernel=line_kernel,
            line_variance=line_variance,
            xch_kernel=xch_kernel,
            xch_variance=xch_variance
        )
        for i in tqdm.trange(n_tot, ncols=100):
            sphere_sample, A_dist, rnd = self._get_directions(i, (biatch, n_chains), True, log_rnds)
            line_thetas[1:] = self.perturb_particles(theta, sphere_sample, A_dist, **perturb_kwargs)

            dist = self.potential(theta)

            if return_data:
                dist, data = dist
                line_data[1:] = data

            line_dist[1:] = dist

            if algorithm == 'cdf':
                max_line_dist = line_dist.max(0)
                if isinstance(max_line_dist, tuple):
                    max_line_dist = max_line_dist[0]
                normalized = line_dist - max_line_dist[None, :]
                probs = self._la.exp(normalized)  # TODO make sure this does not underflow!
                accept_idx = self.quantile_indices(distances=probs, quantiles=rnd)
            else:
                accept_idx[:] = 0
                log_mh_ratio = line_dist[1] - line_dist[0]
                accept_idx[rnd <= log_mh_ratio] = 1  # rnd is already in log due to log_rnd=True

            accepted_probs = line_dist[accept_idx, theta_selector]
            line_dist[0] = accepted_probs  # set the log-probs of the current sample
            theta = line_thetas[accept_idx, theta_selector]
            line_thetas[0, ...] = theta  # set the log-probs of the current sample
            if return_data:
                data = line_data[accept_idx, theta_selector]
                line_data[0, ...] = data
            j = i - n_burn

            if j > 0:
                if algorithm == 'cdf':
                    accept_idx[accept_idx > 0] = 1
                accept_rate += accept_idx

            if (j % thinning_factor == 0) and (j > -1):
                k = j // thinning_factor
                kernel_dist[k] = accepted_probs
                chains[k] = theta
                if return_data:
                    sim_data[k] = data

        if return_data:
            sim_data = {
                'simulated_data': self._la.transax(sim_data, dim0=1, dim1=0)
            }

        attrs = {
            'n_burn': n_burn,
            'acceptance_rate': self._la.tonp(accept_rate) / j,
            'thinning_factor': thinning_factor,
            'n_cdf': n_cdf,
            'line_kernel': line_kernel,
            'line_variance': line_variance,
            'xch_kernel': xch_kernel,
            'xch_variance': xch_variance,
        }
        if self.true_theta is not None:
            attrs['true_theta'] = self._la.tonp(self._true_theta)
            attrs['true_theta_id'] = self._true_theta_id

        return az.from_dict(
            posterior={
                'theta': self._la.transax(chains, dim0=1, dim1=0)  # chains x draws x param
            },
            dims={
                'theta': ['theta_id'],
                'observed_data': ['measurement_id', 'data_id'],
                'simulated_data': ['data_id'],
            },
            coords={
                'theta_id': self.theta_id.tolist(),
                'measurement_id': self._x_meas_id.tolist(),
                'data_id': [f'{i[0]}: {i[1]}' for i in self.data_id.tolist()],
            },
            observed_data={
                'observed_data': self.measurements.values
            },
            sample_stats={
                'lp': kernel_dist.T
            },
            posterior_predictive=sim_data,
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
            line_how='uniform',
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
    #
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

    def _calculate_new_log_weights(
            self,
            new_particles,
            old_particles,
            sample_indices,
            old_log_weights,
            line_variance,
            evaluate_prior=True,
    ):
        new_pol = new_particles[..., :self._K]
        old_pol = old_particles[..., :self._K]

        diff = old_pol - self._la.unsqueeze(new_pol, -2)  # centered at old_pol!
        directions = diff / self._la.norm(diff, 2, -1, True)
        A_dist = self._la.transax(self._sampler._G @ self._la.transax(directions))

        alpha = self._particle_pol_dist / A_dist
        alpha_min, alpha_max, line_variance = self._format_line_kernel_kwargs(alpha, line_variance, directions)
        # ding = self._la.sample_bounded_distribution(shape=(12,), lo=alpha_min, hi=alpha_max, std=line_variance, which='gauss') #TODO WORKS!!!
        def log_prob_bounded_distribution(x, lo=alpha_min, hi=alpha_max, std=line_variance, which='gauss'):
            # TODO
            pass

        if self._n_rev > 0:
            new_xch = new_particles[..., self._K:]
            old_xch = old_particles[..., self._K:]

        raise ValueError
        # TODO holy FUUUUUCK this is complicated, need to
        """Return new log weights following formulas in publications A,B anc C."""

        # Prior can be batched across new particles.
        if evaluate_prior:
            prior_log_probs = self.prior.log_prob(new_particles)
        else:
            prior_log_probs = -0.1  # for a uniform prior, dont need to evaluate

        # Contstruct function to get kernel log prob for given old particle.
        # The kernel is centered on each old particle as in all three variants (A,B,C).
        def kernel_log_prob(new_particle):
            return self.get_new_kernel(old_particles).log_prob(new_particle)

        # We still have to loop over particles here because
        # the kernel log probs are already batched across old particles.
        log_weighted_sum = tensor(
            [
                torch.logsumexp(old_log_weights + kernel_log_prob(new_particle), dim=0)
                for new_particle in new_particles
            ],
            dtype=torch.float32,
        )
        # new weights are prior probs over weighted sum:
        return prior_log_probs - log_weighted_sum

    def _sample_next_population(
            self,
            particles,
            log_weights,
            distances,
            epsilon: float,
            data,
            kernel_variance_scale=1.0,
            line_kernel='gauss',
            xch_kernel='gauss',
            xch_variance=0.4,
            population_batch=6,
            n_cdf=1,
            return_data=True,
            algorithm='cdf',
            evaluate_prior=False,
    ):
        """Return particles, weights and distances of new population."""

        new_particles = []
        new_log_weights = []
        new_distances = []
        new_data = []

        num_accepted_particles = 0
        num_particles = particles.shape[0]
        population_batch = min(num_particles, population_batch)

        line_variance = self.mvn_kernel_variance(
            particles,
            weights=self._la.exp(log_weights),
            samples_per_dim=500,
            kernel_variance_scale=kernel_variance_scale,
            prev_cov=True,
        )

        self._particle_pol_dist = self._la.transax(self._sampler._h - self._sampler._G @ particles[..., :self._K].T)
        self._particle_pol_dist[self._particle_pol_dist < 0.0] = 0.0

        while num_accepted_particles < num_particles:
            # Sample from previous population and perturb.
            sample_indices = self._la.multinomial(population_batch, self._la.exp(log_weights))
            sampled_particles = particles[sample_indices]

            sphere_sample, A_dist, rnd = self._get_directions(0, shape=(1, population_batch), randos=True)
            perturbed_particles = self.perturb_particles(
                theta=sampled_particles,
                sphere_sample=sphere_sample,
                A_dist=A_dist,
                n_cdf=1 if algorithm == 'smc' else n_cdf,
                line_kernel=line_kernel,
                line_variance=line_variance,
                xch_kernel=xch_kernel,
                xch_variance=xch_variance,
            ).squeeze(0)
            dist = self.potential(perturbed_particles)
            if return_data:
                dist, data = dist

            if algorithm == 'cdf':
                self.quantile_indices(dist)
                raise NotImplementedError('need to select the distances<=epsilon and then sample the cdf or perhaps just select a bunch of particles')
            elif algorithm != 'smc':
                raise ValueError

            is_accepted = dist <= epsilon
            num_accepted_batch = is_accepted.sum()

            if num_accepted_batch > 0:
                new_particles.append(perturbed_particles[is_accepted])
                new_distances.append(dist[is_accepted])
                new_log_weights.append(
                    self._calculate_new_log_weights(
                        new_particles=perturbed_particles[is_accepted],
                        old_particles=particles,
                        sample_indices=sample_indices,
                        old_log_weights=log_weights,
                        evaluate_prior=evaluate_prior,
                        line_variance=line_variance,
                    )
                )
                if return_data:
                    new_data.append(data[is_accepted])
                num_accepted_particles += num_accepted_batch

        # collect lists of tensors into tensors
        new_distances = self._la.cat(new_distances)[:num_particles]
        sort_idx = torch.argsort(new_distances)
        new_particles = self._la.cat(new_particles)[:num_particles][sort_idx]
        new_log_weights = self._la.cat(new_log_weights)[:num_particles][sort_idx]
        if return_data:
            new_data = self._la.cat(new_data)[:num_particles][sort_idx]

        # normalize the new weights
        new_log_weights -= torch.logsumexp(new_log_weights, dim=0)

        return (
            new_particles,
            new_log_weights,
            new_distances,
            new_data,
        )

    def run(
            self,
            n_smc_steps=10,
            n=50,
            n_obs=5,
            n0_multiplier=2,
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
            algorithm='smc',
    ):

        self._set_potential(potentype, **dict(n_obs=n_obs, metric=metric, return_data=return_data, **potential_kwargs))

        data = None
        for i in range(n_smc_steps):
            if i == 0:
                prior_theta = self._prior.sample(sample_shape=(n * n0_multiplier,))
                if self._la.backend != 'torch':
                    prior_theta = self._prior._fcm._la.tonp(prior_theta)
                dist = self.potential(prior_theta)
                if return_data:
                    dist, data = dist

                sortidx = self._la.argsort(dist)
                particles = prior_theta[sortidx][:n]
                dist = dist[sortidx][:n]
                epsilon = dist[-1]
                if return_data:
                    data = data[sortidx][:n]
                    all_data = [data]

                log_weights = self._la.log(1 / n * self._la.ones(n))
                all_particles = [particles]
                all_log_weights = [log_weights]
                all_distances = [dist]
                all_epsilons = [epsilon]
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
                    distances=dist,
                    epsilon=epsilon,
                    data=data,
                    kernel_variance_scale=kernel_variance_scale,
                    line_kernel=line_kernel,
                    xch_kernel=xch_kernel,
                    xch_variance=xch_variance,
                    return_data=return_data,
                    algorithm=algorithm,
                )
                all_particles.append(particles)
                all_log_weights.append(log_weights)
                all_distances.append(dist)
                all_epsilons.append(epsilon)
                if return_data:
                    all_data.append(data)


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
    up = UniFluxPrior(model, cache_size=2000)
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
    result = simulate_prior_predictive(mcmc, num_processes=0, n_obs=n_obs, n=500, include_prior_predictive=False)
    fluxes = model._fcm.map_theta_2_fluxes(result['theta'])[0]
    sims = dss.simulate_set(fluxes, n_obs=n_obs, what='mdv')
    mdvs = sims['mdv']
    results = []
    for i, (lid, obmod) in enumerate(dss._obmods.items()):
        part_mdvss = obmod.compute_observations(mdvs[:, i, :], pandalize=True)
        part_mdvss_df = part_mdvss.loc[np.sort(np.repeat(np.arange(part_mdvss.shape[0]), n_obs))].reset_index(drop=True)
        logobs = obmod._la.log10(obmod._la.get_tensor(values=part_mdvss.values) + 1e-12)
        logI = logobs + obmod._log_scaling[None, :]  # in log space, multiplication is addition
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
        data = result['simulated_data']  # data.flatten(start_dim=0, end_dim=-2)
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
    # from pta.sampling.uniform import sample_flux_space_uniform, UniformSamplingModel
    # from pta.sampling.tfs import TFSModel
    import pickle, os
    from sbmfi.models.small_models import spiro, multi_modal
    from sbmfi.inference.priors import UniFluxPrior
    from sbmfi.models.build_models import build_e_coli_anton_glc, build_e_coli_tomek
    from sbmfi.core.util import _excel_polytope
    from sbmfi.core.observation import MVN_BoundaryObservationModel
    from sbmfi.settings import SIM_DIR
    from sbmfi.core.polytopia import FluxCoordinateMapper, compute_volume
    # from sbmfi.inference.simulator import MCMC
    from cdd import Fraction
    from bokeh.plotting import show, output_file
    from arviz import plot_density
    from holoviews.operation.stats import univariate_kde
    # hv.extension('bokeh')

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    np.set_printoptions(linewidth=500)

    a = True
    if a:
        which = 'lcms'
        ding = SMC
    else:
        which = 'com'
        ding = MCMC


    model, kwargs = spiro(
        backend='numpy',
        batch_size=5, which_measurements=which, build_simulator=True, which_labellings=list('CD'),
        v2_reversible=True, logit_xch_fluxes=False, include_bom=True, seed=3, L_12_omega=1.0,
    )
    sdf = kwargs['substrate_df']
    dss = kwargs['datasetsim']
    simm = dss._obmods
    bom = dss._bom
    up = UniFluxPrior(model, cache_size=2000)
    from botorch.utils.sampling import sample_polytope

    mcmc = ding(
        model=model,
        substrate_df=sdf,
        mdv_observation_models=simm,
        boundary_observation_model=bom,
        prior=up,
        # num_processes=0,
    )
    algo = 'rej'
    mcmc.set_measurement(x_meas=kwargs['measurements'])
    mcmc.set_true_theta(theta=kwargs['theta'])
    # run_kwargs = dict(
    #     epsilon=5.0,
    #     n=50, n_burn=0, n_chains=4, thinning_factor=2, n_cdf=4, return_data=True, line_proposal_std=5.0,
    #     evaluate_prior=False, algorithm=algo
    # )

    post = mcmc.run()