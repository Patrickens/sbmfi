import math
import pickle

import tqdm
from sbi.inference.snpe.snpe_c import SNPE_C
from sbmfi.inference.bayesian import _BaseBayes, MCMC, SMC
import torch
from sbmfi.core.model import LabellingModel
from sbmfi.core.observation import BoundaryObservationModel, MDV_ObservationModel
from sbmfi.inference.priors import _NetFluxPrior
from sbmfi.core.simulator import _BaseSimulator
import pandas as pd
from typing import Dict, Optional, Callable, Any, Union
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions import Distribution
from functools import partial
from torch import Tensor, nn, optim
from sbi.neural_nets.flow import build_maf, build_nsf
from sbi.utils import (
    del_entries,
)
from sbi.utils import (
    test_posterior_net_for_multi_d_x,
    x_shape_from_simulation,
)
from copy import deepcopy
from ray.air import session
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi import utils
from sbi.inference.potentials import posterior_estimator_based_potential


def _fix_nn(
        batch_x: Tensor,
        batch_y: Tensor,
        nn_fn,
        **kwargs
):
    neural_net = nn_fn(batch_x, batch_y, **kwargs)
    neural_net.to(batch_y.dtype)
    return neural_net


class SBI_HDF_Helper(_BaseSimulator):
    def prepare_for_sbi(
            self,
            theta,
            data,
            randomize=True,
            **kwargs # this allows to pass **result
    ):
        # cast to correct datatype and I guess do some other pre-processing?
        # TODO think of device!
        data = torch.as_tensor(data, dtype=torch.float32)
        theta = torch.as_tensor(theta, dtype=torch.float32)
        n_t = theta.shape[-1]
        n_samples, n_obs, n_x = data.shape
        theta = theta.unsqueeze(1).repeat((1, n_obs, 1)).view(n_obs * n_samples, n_t)
        data = data.view(n_obs * n_samples, n_x)
        if randomize:
            ridx = torch.as_tensor(self._la.choice(n_samples, n_samples))
            data = data[ridx]
            theta = theta[ridx]
        return theta, data

    def sbi_data_from_hdf(
            self,
            hdf: str,
            dataset_id: str,
            n: int = None,
            device=None,
    ):
        data, validx = self.read_hdf(hdf, dataset_id, what='data')
        data = data[validx.all(-1)]
        theta, validx = self.read_hdf(hdf, dataset_id, what='theta')
        theta = theta[validx.all(-1)]

        if n is not None:
            if n > data.shape[0]:
                raise ValueError(f'cannot sample {n} from dataset of lenghth {data.shape[0]}')
            ridx = self._la.tonp(self._la.choice(n, data.shape[0]))
            data = data[ridx]
            theta = theta[ridx]
        return self.prepare_for_sbi(theta, data, device)

# prof = line_profiler.LineProfiler()

from sbmfi.core.linalg import LinAlg
class NeuralPolytopePosterior(NeuralPosterior):
    def __init__(
            self,
            potential_fn,
            sample_with,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _NetFluxPrior,
            boundary_observation_model: BoundaryObservationModel = None,
            x_shape=None,
            device='cpu',
    ):
        super(NeuralPolytopePosterior, self).__init__(
            potential_fn=potential_fn, theta_transform=None, device=device, x_shape=x_shape
        )
        if sample_with == 'mcmc':
            self._default_kwargs = dict(
                initial_points = None,
                n_burn = 500,
                thinning_factor = 3,
                n_chains = 4,
                n_cdf = 6,
                algorithm = 'mh',
                line_kernel = 'gauss',
                line_variance = 2.0,
                xch_kernel = 'gauss',
                xch_variance = 0.4,
                return_az = True
            )
            self._constant_kwargs = dict(
                potentype='density',
                potential_kwargs=dict(potential_fn=self.potential_fn, track_gradients=False),
                n=0,  # this is set in the .sample(...)
                return_data=False,
                evaluate_prior=False,
            )
            self._sampler = MCMC(
                model, substrate_df, mdv_observation_models, prior, boundary_observation_model
            )
        elif sample_with == 'smc':
            self._default_kwargs = dict(
                n_smc_steps=10,
                n0_multiplier=2,
                epsilon_decay=0.8,
                kernel_variance_scale=1.0,
                line_kernel='gauss',
                xch_kernel='gauss',
                xch_variance=0.4,
                return_all_populations=False,
            )
            self._constant_kwargs = dict(
                potentype='density',
                potential_kwargs=dict(potential_fn=self.potential_fn, track_gradients=False),
                n=0,  # this is set in the .sample(...)
                return_data=False,
                distance_based_decay=False,
                n_obs=0, # ignored anyways
                evaluate_prior=False,
                metric='rmse',
                algorithm='smc',
            )
            self._sampler = SMC(
                model, substrate_df, mdv_observation_models, prior, boundary_observation_model
            )
        else:
            raise ValueError

        # sbi works with float32, this is ugly but works for now
        float32_linalg = LinAlg(
            backend='torch', device=device, seed=self._sampler._la._backwargs['seed'], dtype=np.float32
        )
        self._sampler._la = float32_linalg
        self._sampler._fcm._la = float32_linalg
        self._sampler._fcm._rho_bounds = self._sampler._fcm._rho_bounds.to(torch.float32)
        self._sampler._fcm._sampler = self._sampler._fcm._sampler.to_linalg(float32_linalg, dtype=np.float32)
        self._sampler._sampler = self._sampler._fcm._sampler

        # tt = self._sampler._la.get_tensor(shape=(4,4))
        # print(123123, tt.dtype)

    def sample(
        self,
        sample_shape = torch.Size(),
        x: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        self.potential_fn.set_x(self._x_else_default_x(x))
        kwargs = {**self._default_kwargs, **kwargs}
        kwargs = {**kwargs, **self._constant_kwargs}
        kwargs['n'] = math.prod(sample_shape)
        return_az = kwargs.get('return_az')
        result = self._sampler.run(**kwargs)
        if return_az:
            return result
        raise NotImplementedError  # reshape into sample_shape, chains and all that
        return self._la.view(result, (sample_shape, ))


    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Tensor:
        pass

    def log_prob(
            self, theta: Tensor, x: Optional[Tensor] = None,
    ):
        pass

    def set_default_x(self, x):
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            self._sampler.set_measurement(x_meas=x)
            if self._sampler._x_meas.shape[0] > 1:
                raise ValueError('can only handle single observation')
        NeuralPosterior.set_default_x(self, x=self._sampler._x_meas)


class SNPE_P(_BaseBayes, SNPE_C):
    def __init__(
            self,
            nn_fn: Callable,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _NetFluxPrior,
            boundary_observation_model: BoundaryObservationModel = None,
    ):
        # https://arxiv.org/abs/2210.04815
        # https://arxiv.org/abs/1905.07488
        # https://github.com/mackelab/tsnpe_neurips/tree/main
        if model._la.backend != 'torch':
            raise ValueError('can only work with torch for now!')
        if isinstance(prior, RatioPrior):
            raise NotImplementedError
        super(SNPE_P, self).__init__(
            model, substrate_df, mdv_observation_models, prior, boundary_observation_model
        )
        SNPE_C.__init__(self, prior, density_estimator=nn_fn, device=str(self._la._BACKEND._device))

    @staticmethod
    def constuct_nn_fn(
            density_estimator = 'nsf',
            hidden_features: int = 10,
            num_transforms: int = 3,
            num_bins: int = 5,
            tail_bound: float = 3.0,
            hidden_layers_spline_context: int = 1,
            num_blocks: int = 3,
            dropout_probability: float = 0.0,
            use_batch_norm: bool = False,
    ):
        kwargs = locals()
        if density_estimator == 'nsf':
            nn_fn = build_nsf
        elif density_estimator == 'maf':
            nn_fn = build_maf
        else:
            ValueError('we chose to only use normalizing flows in this package')
        kwargs['nn_fn'] = nn_fn
        return partial(_fix_nn, **kwargs)


    def build_posterior(
        self,
        density_estimator: Optional[nn.Module] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "rejection",
        sampling_parameters: Dict[str, Any] = {},
    ) -> Union[NeuralPolytopePosterior, RejectionPosterior, DirectPosterior]:
        r"""Build posterior from the neural density estimator.

        For SNPE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:
        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        - alternatively, if leakage is very high (which can happen for multi-round
            SNPE), sample from the posterior with MCMC.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection` | `vi`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior` or `DirectPosterior`. By default,
                `DirectPosterior` is used. Only if `rejection_sampling_parameters`
                contains `proposal`, a `RejectionPosterior` is instantiated.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """
        if prior is None:
            assert self._prior is not None, (
                "You did not pass a prior. You have to pass the prior either at "
                "initialization `inference = SNPE(prior)` or to "
                "`.build_posterior(prior=prior)`."
            )
            prior = self._prior
        else:
            utils.check_prior(prior)

        if density_estimator is None:
            posterior_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            posterior_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device.type

        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior_estimator=posterior_estimator,
            prior=prior,
            x_o=None,
            enable_transform=False,
        )

        if sample_with == 'direct':
            self._posterior = DirectPosterior(
                posterior_estimator=posterior_estimator,
                prior=prior,
                x_shape=self._x_shape,
                device=device,
            )
        elif sample_with == "vi":
            raise ValueError('This is not implemented for polytopes and ratios')
        elif sample_with in ["mcmc", "smc"]:
            if isinstance(self._prior, RatioPrior):
                raise NotImplementedError(
                    f'because of the difficult support of the the ratio-prior, '
                    f'its not currently possible to use {sample_with}'
                )
            elif isinstance(self._prior, _NetFluxPrior):
                npp_kwargs = dict(
                    model=self._model,
                    substrate_df=self._substrate_df,
                    prior=prior,
                    mdv_observation_models=self._obmods,
                    boundary_observation_model=self._bom,
                )
                if sample_with in ["mcmc", "smc"]:
                    self._posterior = NeuralPolytopePosterior(
                        potential_fn=potential_fn,
                        sample_with=sample_with,
                        x_shape=self._x_shape,
                        device=device,
                        **npp_kwargs
                    )
        elif sample_with == "rejection":
            if isinstance(self._prior, RatioPrior):
                # TODO construct a box-uniform proposal distribution!
                raise NotImplementedError
            elif isinstance(self._prior, _NetFluxPrior):
                proposal = UniformNetPrior(self._fcm)
            self._posterior = RejectionPosterior(
                potential_fn=potential_fn,
                proposal=proposal,
                device=device,
                x_shape=self._x_shape,
            )
        else:
            raise ValueError('not a valid posterior argument')

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))

        return deepcopy(self._posterior)


    def _train(
            self,
            training_batch_size: int = 50,
            learning_rate: float = 5e-4,
            validation_fraction: float = 0.1,
            max_num_epochs: int = 3,
            clip_max_norm: Optional[float] = 5.0,
            calibration_kernel: Optional[Callable] = None,
            dataloader_kwargs: Optional[dict] = None,
            discard_prior_samples: bool = False,
            retrain_from_scratch: bool = False,
            force_first_round_loss: bool = False,
    ):
        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: torch.ones([len(x)], device=self._device)

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            False,
            dataloader_kwargs=dataloader_kwargs,
        )
        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.

        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring transforms)
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )
            self._x_shape = x_shape_from_simulation(x.to("cpu"))
            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta.to("cpu"),
                x.to("cpu"),
            )

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        self.optimizer = optim.Adam(  # TODO make this optional, can be a hyperparameter
            list(self._neural_net.parameters()), lr=learning_rate
        )

        for epoch in tqdm.trange(max_num_epochs):
            # Train for a single epoch.
            self._neural_net.train()
            train_log_probs_sum = 0
            for batch in tqdm.tqdm(train_loader):
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch, masks_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )

                train_losses = self._loss(  # SLOWEST
                    theta_batch,
                    x_batch,
                    masks_batch,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                )
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            train_log_prob_average = train_log_probs_sum / (
                    len(train_loader) * train_loader.batch_size  # type: ignore
            )

            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch, masks_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                    )
                    val_log_prob_sum -= val_losses.sum().item()

            # Take mean over all validation samples.
            val_log_prob = val_log_prob_sum / (
                    len(val_loader) * val_loader.batch_size  # type: ignore
            )
            session.report(
                {'epoch': epoch, 'val_log_prob': val_log_prob, 'train_log_prob_average': train_log_prob_average}
            )

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def train(
        self,
        num_atoms: int = 10,
        training_batch_size: int = 100,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        max_num_epochs: int = 3,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        use_combined_loss: bool = False,
        retrain_from_scratch: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> nn.Module:

        self._num_atoms = num_atoms
        self._use_combined_loss = use_combined_loss
        kwargs = del_entries(
            locals(), entries=("self", "__class__", "num_atoms", "use_combined_loss")
        )
        self._round = max(self._data_round_index)
        return self._train(**kwargs)


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro

    from sbmfi.core.simulator import DataSetSim
    from sbmfi.inference.priors import UniformNetPrior
    import os
    import numpy as np

    model, kwargs = spiro(
        backend='torch',
        batch_size=5, which_measurements='lcms', build_simulator=True, which_labellings=list('CD'),
        v2_reversible=True, logit_xch_fluxes=False, include_bom=True, seed=3, L_12_omega=1.0,
        v5_reversible=True
    )
    bbs = kwargs['basebayes']
    sdf = kwargs['substrate_df']
    dss = DataSetSim(model, sdf, bbs._obmods, bbs._bom, num_processes=0)
    n = 20000
    prior = UniformNetPrior(model, cache_size=n)
    h5_file = 'spiro.h5'
    dataset_id = 'test'
    create_data = False
    if create_data:
        if os.path.exists(h5_file):
            os.remove(h5_file)
        theta = prior.sample((n,))
        fluxes = prior._flux_cache
        result = dss.simulate_set(fluxes, n_obs=3, what='all')
        result['theta'] = theta
        data = result['data']
        dss.to_hdf(h5_file, dataset_id=dataset_id, result=result)
    else:
        theta, data = dss.sbi_data_from_hdf(h5_file, dataset_id, n=1000)

    train_nn = True
    nn_fn = SNPE_P.constuct_nn_fn()
    nfi = SNPE_P(nn_fn, model, sdf, bbs._obmods, prior, bbs._bom)
    nfi.append_simulations(theta, data)
    if train_nn:
        neural_net = nfi.train()
        # pickle.dump(neural_net, open('nn2.p', 'wb'))
    else:
        neural_net = pickle.load(open('nn2.p', 'rb'))
    nfi.set_measurement(kwargs['measurements'])
    # post = nfi.build_posterior(neural_net, sample_with='rejection')
    # post.set_default_x(x=nfi._x_meas)
    # azz = post.sample((1000, ))





