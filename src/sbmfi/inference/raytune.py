import ray
from ray import tune
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper, Stopper
from ray.tune import Callback
from ray.air._internal.session import _get_session
from ray.tune.utils import validate_save_restore

import logging
import warnings
import inspect
import numpy as np
import os, pickle
from collections import OrderedDict
import time
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union, Tuple, Literal, List

from sbi.inference import SNPE_C, SNPE_B, SNPE_A, DirectPosterior, MCMCPosterior, VIPosterior, RejectionPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.neural_nets.flow import build_nsf  # neural spline flow
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi.analysis import check_sbc, run_sbc, get_nltp, sbc_rank_plot
from sbi.types import TensorboardSummaryWriter
from sbi.inference.potentials import posterior_estimator_based_potential
from sbi.utils import (
    process_prior,
    test_posterior_net_for_multi_d_x,
    x_shape_from_simulation,
)

from sbmfi.settings import SIM_DIR
from sbmfi.models.build_models import ecoli_builder, spiro, multi_modal
from sbmfi.core.observation import (
    construct_model_annot_df,
    exclude_low_massiso,
)
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import shutil
import pandas as pd
from sbmfi.core.model import RatioMixin
from sbmfi.estimate.priors import (
    _BasePrior,
    UniFluxPrior,
    ThermoPrior,
    RatioPrior,
    _CannonicalPolytopeSupport,
)
from sbmfi.estimate.simulator import (
    BoundaryObservationModel,
    FluxSimulator
)

import torch
from torch import Tensor, as_tensor
from torch.utils import data
from torch.distributions.transforms import identity_transform
from torch.distributions import Distribution, MultivariateNormal, Uniform
from torch import Tensor, nn, ones, optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm.auto import tqdm
from sbi.utils import gradient_ascent, within_support
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.utils import (
    check_dist_class,
    del_entries,
)
from sbi.inference.posteriors import (
    DirectPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
)

from scipy.stats import kstest, uniform

def check_uniformity_frequentist(ranks, num_posterior_samples) -> Tensor:
    # TODO kstest sometimes has an underflow error, monkeypatch this sheiss
    kstest_pvals = torch.tensor(
        [
            kstest(rks, uniform(loc=0, scale=num_posterior_samples).cdf)[1]
            for rks in ranks.T
        ],
        dtype=torch.float32,
    )

    return kstest_pvals


def _load_data(hdf, dataset_id, x_coordinate_sys='transformation', start=None, step=None, stop=None):
    result = OrderedDict()
    result[x_coordinate_sys] = FluxSimulator.read_hdf(hdf, dataset_id, x_coordinate_sys, None, start, step, stop, aspandas=False)
    result['theta'] = FluxSimulator.read_hdf(hdf, dataset_id, 'theta', None, start, step, stop, aspandas=False)
    return FluxSimulator.prepare_for_sbi(result, x_coordinate_sys, device='cpu')  # TODO transfer to GPU here already?


def fraction_samples_within_support(
        posterior: NeuralPosterior,
        x: torch.Tensor,
        num_samples: int = 1000,
        average: bool = True
):
    x = torch.atleast_2d(x)
    fractions_in_support = []
    prior = posterior.potential_fn.prior
    for x_o in x:
        potential_samples = posterior.potential_fn.posterior_estimator.sample(
            num_samples=num_samples, context=x_o[None, :]
        )
        fractions_in_support.append(prior.support.check(value=potential_samples).all(-1).sum() / num_samples)
    if average:
        return np.mean(fractions_in_support)
    return fractions_in_support


class MyCallBack(Callback):
    def __init__(
            self,
            hdf,
            validation_id,
            min_support=0.7,
            num_posterior_samples=1000,
            num_c2st_repetitions=1,
            sbc=False,
    ):
        self._hdf = hdf
        self._vid = validation_id
        self._minsupp = min_support
        self._npost = num_posterior_samples
        self._nc2st = num_c2st_repetitions
        self._sbc = sbc  # whether to run sbc
        self._data = {}

    def _load_validation_data(self, x_coordinate_sys):
        if x_coordinate_sys in self._data:
            return self._data[x_coordinate_sys]
        data_o = _load_data(hdf=self._hdf, dataset_id=self._vid, x_coordinate_sys=x_coordinate_sys)
        self._data[x_coordinate_sys] = data_o[x_coordinate_sys], data_o['theta']
        return self._data[x_coordinate_sys]

    def on_trial_complete(
        self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        # trial.last_result['done'] can be used along with trial.last_result['epoch'] to see whether the trial has converged
        checkpoint = trial.checkpoint.to_air_checkpoint().to_dict()
        posterior = torch.load(checkpoint['checkpoint_path'])
        checkpoint_dir = checkpoint['checkpoint_dir']
        xcs = checkpoint['x_coordinate_sys']
        xs, thetas = self._load_validation_data(x_coordinate_sys=xcs)
        frac_in_support = fraction_samples_within_support(posterior, x=xs, num_samples=self._npost)

        stats = {}
        if (frac_in_support > self._minsupp) and self._sbc:
            ranks, dap_samples = run_sbc(
                thetas=thetas,
                xs=xs,
                posterior=posterior,
                num_posterior_samples=self._npost,
                num_workers=1,
                sbc_batch_size=1,  # TODO what is this??
                show_progress_bar=False,
            )
            stats = {**stats, **check_sbc(
                ranks=ranks,
                prior_samples=thetas,
                dap_samples=dap_samples,
                num_posterior_samples=self._npost,
                num_c2st_repetitions=self._nc2st,
            ), 'ranks': ranks, 'dap_samples': dap_samples, 'frac_in_support': frac_in_support}

        stats['frac_in_support'] = frac_in_support
        pickle.dump(stats, open(os.path.join(checkpoint_dir, "stats.p"), 'wb'))


class FixedSNPE_C(SNPE_C):
    # NB the goal if this class is to fix a bunch of functions of SNPE_C
    #   mainly those related to transforms of prior support
    def build_posterior(
        self,
        density_estimator: Optional[nn.Module] = None,
        prior: Optional[Distribution] = None,
        theta_transform = identity_transform,  # NB ESSENTIAL, OTHERWISE WE GET A PRIOR SUPPORT ERROR!
        sample_with: str = "rejection",
        mcmc_method: str = "slice_np",
        vi_method: str = "rKL",
        mcmc_parameters: Dict[str, Any] = {},
        vi_parameters: Dict[str, Any] = {},
        rejection_sampling_parameters: Dict[str, Any] = {},
    ) -> Union[MCMCPosterior, RejectionPosterior, VIPosterior, DirectPosterior]:
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

        potential_fn, _ = posterior_estimator_based_potential(  # NB HERE WE ARE PASSING A THETA_TRANSFORM!
            posterior_estimator=posterior_estimator, prior=prior, x_o=None, theta_transform=theta_transform,
        )

        if sample_with == "rejection":
            if "proposal" in rejection_sampling_parameters.keys():
                self._posterior = RejectionPosterior(
                    potential_fn=potential_fn,
                    device=device,
                    x_shape=self._x_shape,
                    **rejection_sampling_parameters,
                )
            else:
                self._posterior = DirectPosterior(
                    posterior_estimator=posterior_estimator,
                    theta_transform=theta_transform,
                    prior=prior,
                    x_shape=self._x_shape,
                    device=device,
                )
        elif sample_with == "mcmc":
            self._posterior = MCMCPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                proposal=prior,
                method=mcmc_method,
                device=device,
                x_shape=self._x_shape,
                **mcmc_parameters,
            )
        elif sample_with == "vi":
            self._posterior = VIPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                prior=prior,  # type: ignore
                vi_method=vi_method,
                device=device,
                x_shape=self._x_shape,
                **vi_parameters,
            )
        else:
            raise NotImplementedError

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))

        return deepcopy(self._posterior)

    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 50,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        theta, x, prior_masks = self.get_simulations(starting_round)

        dataset = data.TensorDataset(theta, x, prior_masks)
        num_examples = theta.size(0)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        if not resume_training:
            # Seperate indicies for training and validation
            permuted_indices = torch.randperm(num_examples)
            self.train_indices, self.val_indices = (
                permuted_indices[:num_training_examples],
                permuted_indices[num_training_examples:],
            )

        train_loader_kwargs = {
            "batch_size": int(min(training_batch_size, num_training_examples)),
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.train_indices.tolist()),
        }
        val_loader_kwargs = {
            "batch_size": int(min(training_batch_size, num_validation_examples)),
            "shuffle": False,
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.val_indices.tolist()),
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        train_loader = data.DataLoader(dataset, **train_loader_kwargs)

        val_loader = None
        if validation_fraction > 0.0:
            val_loader = data.DataLoader(dataset, **val_loader_kwargs)

        return train_loader, val_loader


class SNPE_Trainable(tune.Trainable):
    def _converged_since_last_improvement(self):
        pass

    def _converged_coeff_var(self):
        pass

    def _sbi_step(self):
        # NB this funciton uses the methods provided by sbi
        #   we reimplement a bunch of them to make it easier to integrate the stuff that ray tune provides
        #   this step method is largely for comparison purposes

        results = {}
        self._snpec.train(**self._parsed_config['train'])
        results['epochs'] = self._snpec._summary["epochs_trained"][-1]
        results['val_log_prob'] = self._snpec._summary["best_validation_log_prob"][-1]

        self.posterior = self._snpec.build_posterior(**self._parsed_config['build_posterior'])
        results['c2st_dap'] = 0.0
        if self._vid is not None:
            raise NotImplementedError
            frac_in_support = self._snpec.fraction_samples_within_support(
                num_samples=1000,
                x=self._parsed_config['run_sbc']['xs']
            )
            results['avg_frac_in_support'] = frac_in_support
            if frac_in_support > 0.8:  # NB require that at least 80% of samples are within prior support
                ranks, dap_samples = run_sbc(
                    posterior=self.posterior,
                    **self._parsed_config['run_sbc'],
                )
                check_stats = check_sbc(
                    ranks,
                    prior_samples=self._parsed_config['run_sbc']['thetas'],
                    dap_samples=dap_samples,
                    num_posterior_samples=self._parsed_config['run_sbc']['num_posterior_samples'],
                    **self._parsed_config['check_sbc']
                )
                results['c2st_dap'] = check_stats['c2st_dap']
        return results

    def _get_dataloaders(self):
        if self._data_noise == 'static':
            # this is where we load MDVs and noise them up dynamically
            if self._snpec.epoch == 0:
                data_loader_kwargs = {
                    'starting_round': 0,
                    **{
                        k: self._parsed_config['train'][k] for k in
                        ['training_batch_size', 'validation_fraction', 'resume_training', 'dataloader_kwargs']
                    }
                }
                self._dataloaders = self._snpec.get_dataloaders(**data_loader_kwargs)
            return self._dataloaders
        elif self._data_noise == 'observation':
            raise NotImplementedError
            indep_vars = self._simulator.sample_observation_noise(mdv=self._mdv, n_obs=3, x_coordinate_sys=self._data)
            data = self._simulator.prepare_for_sbi(result={self._data: indep_vars, 'theta': self._theta}, x_coordinate_sys=self._data, device='cpu')
            self._snpec.append_simulations(  # NB this is only necessary when using the SBI functions
                theta=data['theta'],
                x=data[self._data],
            )

            # TODO problem is that this may end up eating a lot of memory; we need to clear the 4 lists below to free up memory
            self._theta_roundwise.append(theta)
            self._x_roundwise.append(x)
            self._prior_masks.append(prior_masks)
            self._proposal_roundwise.append(proposal)
            return self._snepc.get_dataloaders(**data_loader_kwargs)
        elif self._data_noise == 'flux':
            raise NotImplementedError

    def _epoch_train_step(self):
        self._snpec._neural_net.train()
        train_log_probs_sum = 0
        train_loader, val_loader = self._get_dataloaders()
        proposal = self._snpec._proposal_roundwise[-1]
        calibration_kernel = self._parsed_config['train']['calibration_kernel']
        force_first_round_loss = self._parsed_config['train']['force_first_round_loss']
        epoch_start_time = time.time()

        for batch in train_loader:
            self._snpec.optimizer.zero_grad()
            # Get batches on current device.
            theta_batch, x_batch, masks_batch = (
                batch[0].to(self._snpec._device),
                batch[1].to(self._snpec._device),
                batch[2].to(self._snpec._device),
            )

            train_losses = self._snpec._loss(
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
            clip_max_norm = self._parsed_config['train']['clip_max_norm']
            if clip_max_norm is not None:
                clip_grad_norm_(
                    self._snpec._neural_net.parameters(), max_norm=clip_max_norm
                )
            self._snpec.optimizer.step()

        self._snpec.epoch += 1

        train_log_prob_average = train_log_probs_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
        )
        return {
            'epoch': self._snpec.epoch,
            'epoch_start_time': epoch_start_time,
            'epoch_train_time': time.time(),
            'train_log_probs': train_log_prob_average
        }

    def _epoch_val_step(self):
        train_result = self._epoch_train_step()
        train_loader, val_loader = self._get_dataloaders()
        self._snpec._neural_net.eval()

        proposal = self._snpec._proposal_roundwise[-1]
        calibration_kernel = self._parsed_config['train']['calibration_kernel']
        force_first_round_loss = self._parsed_config['train']['force_first_round_loss']

        val_log_prob_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                theta_batch, x_batch, masks_batch = (
                    batch[0].to(self._snpec._device),
                    batch[1].to(self._snpec._device),
                    batch[2].to(self._snpec._device),
                )
                # Take negative loss here to get validation log_prob.
                val_losses = self._snpec._loss(
                    theta_batch,
                    x_batch,
                    masks_batch,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                )
                val_log_prob_sum -= val_losses.sum().item()
        val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
        )
        return {
            **train_result,
            'epoch_valid_time': time.time(),
            'val_log_probs': val_log_prob,
        }

    def _fill_config_defaults(self, config):
        default_kwargs = {
            'FixedSNPE_C': {
                'show_progress_bars': False,
                'device': 'cpu',
            },

            'posterior_nn': {
                'model': 'nsf',
                'z_score_theta': 'independent',
                'z_score_x': 'independent',
                'hidden_features': 50,  # number of neurons
                'num_transforms': 5,
                'num_bins': 10,

                # specific to build_nsf or build_maf
                'tail_bound': 3.0,  # padding on either side of the spline?
                'num_blocks': 2,  # depth of neural net
                'dropout_probability': 0.0,
                'use_batch_norm': False,

                # specific to build_mdn
                'num_mixture_components': 10,
            },

            'train': {
                'training_batch_size': 128,
                'learning_rate': 5e-4,
                'validation_fraction': 0.05,  # NB 5% of data is kept as a validation set!
                'stop_after_epochs': 20,
                'max_num_epochs': 500,
                'clip_max_norm': 5.0,
                'dataloader_kwargs': None,

                'calibration_kernel': None,  # TODO
                'resume_training': False,
                'discard_prior_samples': False,
                'force_first_round_loss': False,
                'use_combined_loss': True,
                'retrain_from_scratch': False,
                'show_train_summary': False,
            },

            'build_posterior': {
                'sample_with': 'rejection',
                'rejection_sampling_parameters': {},
                'mcmc_method': 'slice_np',
                'mcmc_parameters': {},
                'vi_method': 'rKL',
                'vi_parameters': {},
            },

            'run_sbc': {
                'num_posterior_samples': 1000,
                'num_workers': 1,
                'sbc_batch_size': 1,
                'show_progress_bar': False,
            },

            'check_sbc':{
                'num_c2st_repetitions': 1,
            }
        }
        updated_default_kwargs = deepcopy(default_kwargs)
        for fn, fn_kwargs in default_kwargs.items():
            for k, v in config.items():
                if k in fn_kwargs:
                    updated_default_kwargs[fn][k] = v
        return updated_default_kwargs

    def setup(self, config: Dict):
        self._simulator: FluxSimulator = pickle.load(open(config['sim_pickle'], 'rb'))
        self._hdf = config['hdf']

        self._xcs = config.get('x_coordinate_sys', 'transformation')  # what data to load from the HDF
        self._data_noise = config.get('data_noise', 'static')
        self._tid = config.get('training_id', 'training')  # dataset_id inside hdf
        self._vid = config.get('validation_id', None)  # dataset_id inside hdf

        self._parsed_config = self._fill_config_defaults(config)

        validation_fraction = self._parsed_config['train']['validation_fraction']
        if validation_fraction > 0.0:
            # NB this means that we care
            self.step = self._epoch_val_step
        else:
            self.step = self._epoch_train_step

        build_fn = posterior_nn(**self._parsed_config['posterior_nn'])

        self._snpec = FixedSNPE_C(
            prior=self._simulator._prior,
            density_estimator=build_fn,
            summary_writer=SummaryWriter(os.path.join(SIM_DIR, 'sbi_logs', f'{self._simulator._model.name}_log')),
            **self._parsed_config['FixedSNPE_C'],
        )
        calibration_kernel = self._parsed_config['train']['calibration_kernel']
        if calibration_kernel is None:
            self._parsed_config['train']['calibration_kernel'] = lambda x: ones([len(x)], device=self._snpec._device)

        if self._data_noise == 'static':
            data = _load_data(hdf=self._hdf, dataset_id=self._tid, x_coordinate_sys=self._xcs)
            self._snpec.append_simulations(  # NB this is only necessary when using the SBI functions
                theta=data['theta'],
                x=data[self._xcs],
            )
        elif self._data_noise == 'observation':
            # NB here we load the MDVs and reample observation errors at every epoch of training
            raise NotImplementedError
            data = _load_data(hdf=self._hdf, dataset_id=self._tid, x_coordinate_sys='mdv')
            self._mdv = data['mdv']
            self._theta = data['theta']
            training_batch_size = self._parsed_config['train']['training_batch_size']
            xs = self._simulator.sample_observation_noise(mdv=self._mdv[:training_batch_size], n_obs=1)
            self._snpec.append_simulations(  # NB this is only necessary when using the SBI functions
                theta=self._theta[:training_batch_size],
                x=xs,
            )
        elif self._data_noise == 'flux':
            # NB this means we resample and simulate data every epoch as a form of regularization;
            raise NotImplementedError
        else:
            raise ValueError()

        # NB prepare a bunch of things that are typicaly done inside of self._snpec.train(...)
        learning_rate = self._parsed_config['train']['learning_rate']
        theta, x, _ = self._snpec.get_simulations()
        self._snpec._neural_net = self._snpec._build_neural_net(theta.to("cpu"), x.to("cpu"),)
        test_posterior_net_for_multi_d_x(self._snpec._neural_net,theta.to("cpu"),x.to("cpu"),)
        self._snpec.epoch, self._snpec._val_log_prob = 0, float("-Inf")
        self._snpec.optimizer = optim.Adam(
            list(self._snpec._neural_net.parameters()), lr=learning_rate
        )

        # NB dont need to load validation data if we use ray functions instead of sbi
        if (self._vid is not None) and (self.step == self._sbi_step):
            data_o = _load_data(hdf=self._hdf, dataset_id=self._vid, x_coordinate_sys=self._xcs)
            self._parsed_config['run_sbc'] = {
                'xs': data_o[self._xcs],
                'thetas': data_o['theta'],
                **self._parsed_config['run_sbc'],
            }

    def reset_config(self, new_config: Dict):
        pass

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        checkpoint_path = os.path.join(checkpoint_dir, "posterior.p")
        self._posterior = self._snpec.build_posterior()
        torch.save(self._posterior, checkpoint_path)
        return {
            'checkpoint_dir': checkpoint_dir,
            'checkpoint_path': checkpoint_path,
            'x_coordinate_sys': self._xcs,
            'epoch': self._snpec.epoch,
        }

    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        checkpoint_path = checkpoint['checkpoint_path']
        self._xcs = checkpoint['x_coordinate_sys']
        # TODO restore posterior to self._snpec
        self._posterior = torch.load(open(checkpoint_path, 'rb'))

    def cleanup(self):
        pass


def main(
        hdf,
        sim_pickle,
        # tune_type: Literal['epoch', 'sbi'] = 'epoch',
        training_id: str = 'training',
        validation_id: str = None,
        tune_id: str = 'test',
        num_samples=100,
        max_t=75,
        gpus_per_trial=0,
        cpus_per_trial=1,
        validation_fraction=0.0,  # signals whether to use train_step or val_step
        num_posterior_samples=1000,
        num_c2st_repetitions=1,
        **kwargs
) -> ray.air.Result:

    param_space = {
        'hdf': hdf,
        # 'tune_type': tune_type,
        'sim_pickle': sim_pickle,
        'training_id': training_id,
        'validation_id': validation_id,
        'x_coordinate_sys': 'transformation', #tune.choice(('transformation', 'observation')),
        'z_score_theta': 'independent',
        #'z_score_theta': tune.choice((None, 'independent')),
        'z_score_x': 'independent',
        #'z_score_x': tune.choice((None, 'independent')),
        'use_batch_norm': True,
        #'use_batch_norm': tune.choice((True, False)),
        'hidden_features': tune.qrandint(10, 100, 5),
        'num_transforms': tune.qrandint(2, 12, 2),
        'num_bins': tune.qrandint(4, 20, 2),    # paper tested K ∈ [8, 10]
        # 'tail_bound': tune.uniform(np.arange(0.05, 5.0)),  # paper says results not too sensitive to tail bound, tested B ∈ [1, 5]
        'num_blocks': tune.randint(2, 5),  #
        'dropout_probability': tune.uniform(0.0, 0.5),
        'training_batch_size': tune.choice(2**np.arange(3, 9)),
        'learning_rate': tune.loguniform(1e-5, 3e-1),
        'validation_fraction': validation_fraction,
        **kwargs,  # NB this way we can set some t
    }

    callback = None
    if validation_id is not None:
        callback = [
            MyCallBack(
                hdf=hdf,
                validation_id=validation_id,
                num_posterior_samples=num_posterior_samples,
                num_c2st_repetitions=num_c2st_repetitions
            )
        ]

    if validation_fraction == 0.0:
        metric = 'train_log_probs'
    else:
        metric = 'val_log_probs'

    local_dir = os.path.join(SIM_DIR, 'ray_logs')
    dirpath = Path(local_dir) / tune_id
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    tuner = tune.Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(SNPE_Trainable),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial, "memory": 1e9},
        ),
        tune_config=tune.TuneConfig(
            search_alg=None,  # NB defaults to random search, think of doing BOHB or BayesOptSearch
            metric=metric,
            mode='max',
            scheduler=ASHAScheduler(
                time_attr='epoch',
                max_t=max_t,
                grace_period=1,
                reduction_factor=2
            ),
            num_samples=num_samples,
        ),
        run_config=ray.air.RunConfig(
            local_dir=local_dir,
            name=tune_id,
            callbacks=callback,
            stop=TrialPlateauStopper(  # NB checks for convergence based on CV
                metric=metric,
                std=0.05, # TODO THIS SHOULD ACTUALLY THE CV!
                num_results=5,
                grace_period=4,
                metric_threshold=None,
                mode='min',
            ),
            log_to_file=True,
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_at_end=True
            )
        ),
        param_space=param_space,
    )
    result = tuner.fit()
    return result


# NB functions below are helpers to quickly create a dataset
def _from_scratch(model, substrate_df, prior, hdf):
    obsmods = FluxSimulator.build_models(model, annotation_dfs={k: None for k in substrate_df.index})
    sim = FluxSimulator(
        prior=prior,
        substrate_df=substrate_df,
        mdv_observation_models=obsmods,
        boundary_observation_model=None,
        num_processes=0,
    )
    res = sim.sample_and_simulate(
        n=1000,
        n_obs=3,
        x_coordinate_sys='mdv',
    )
    if os.path.exists(hdf):
        os.remove(hdf)
    sim.to_hdf(hdf=hdf, result=res, dataset_id='selection_mdvs', append=False, expectedrows_multiplier=1)


def _create_multi_modal_dataset(
        hdf,
        sim_pickle,
        training_id='training',
        validation_id='validation',
        x_coordinate_sys='all',
        n_train=10000,
        n_validate=300,
        from_scratch=True,
        n_obs=3,
        **kwargs
):
    model, kwargs = multi_modal(backend='torch', batch_size=50, ratios=False)

    # NB boundary model has to be first, since it might change the order of labelling_reaction!
    prior = UniFluxPrior(
        model=model,
        cache_size=n_train + n_validate,
        num_processes=1,
        logit_xch_fluxes=True,
        rounded_net_coordinates=False,
    )
    substrate_df = kwargs['substrate_df']
    if from_scratch:
        _from_scratch(model, substrate_df, prior, hdf)
    mdvdf = FluxSimulator.read_hdf(hdf=hdf, dataset_id='selection_mdvs', what='mdv')
    annot_dfs = {lid: construct_model_annot_df(model, mdvdf.loc[:, lid]) for lid in substrate_df.index}
    obsmods = FluxSimulator.build_models(model, annot_dfs, total_intensities=kwargs['total_intensities'])
    sim = FluxSimulator(
        prior=prior,
        substrate_df=substrate_df,
        mdv_observation_models=obsmods,
        num_processes=0,
    )
    data = sim.sample_and_simulate(
        n=n_train,
        n_obs=n_obs,
        x_coordinate_sys=x_coordinate_sys,
    )
    sim.to_hdf(hdf=hdf, result=data, dataset_id=training_id, append=False, expectedrows_multiplier=5)

    if n_validate > 0:
        x_o = sim.sample_and_simulate(
            n=n_validate,
            n_obs=1,  # NB 1, we build 1 posterior per flux-sample!
            x_coordinate_sys=x_coordinate_sys,
        )
        sim.to_hdf(hdf=hdf, result=x_o, dataset_id=validation_id, append=False, expectedrows_multiplier=5)
    pickle.dump(sim, open(sim_pickle, 'wb'))


def _create_spiro_dataset(
        hdf,
        sim_pickle,
        training_id='training',
        validation_id='validation',
        x_coordinate_sys='all',
        n_train=50000,
        n_validate=50,
        from_scratch=True,
        n_obs=3,
        **kwargs
):
    labellings = ['A', 'B']
    # NB ratios==False is essential, because otherwise we run into trouble with the ordering of fluxes!!!!
    # TODO due to strange things with the RatioMixin, ordering of fluxes is scrambled and bound_idx
    #   of BoundaryObservationModel gets confused. This is messy and should be addressed
    model, kwargs = spiro(backend='torch', batch_size=50, ratios=False)

    # NB boundary model has to be first, since it might change the order of labelling_reaction!
    bom = BoundaryObservationModel(model, measured_boundary_fluxes=['a_in', 'd_out', 'bm'], check_noise=False)
    prior = UniFluxPrior(
        model=model,
        cache_size=n_train + n_validate,
        num_processes=1,
        logit_xch_fluxes=True,
        rounded_net_coordinates=False,
    )
    substrate_df = kwargs['substrate_df'].loc[labellings]

    if from_scratch:
        _from_scratch(model, substrate_df, prior, hdf)
    mdvdf = FluxSimulator.read_hdf(hdf=hdf, dataset_id='selection_mdvs', what='mdv')
    ser = exclude_low_massiso(mdvdf)
    mdvdf = mdvdf.loc[:, ser.index]
    annot_dfs = {lid: construct_model_annot_df(model, mdvdf.loc[:, lid]) for lid in substrate_df.index}
    obsmods = FluxSimulator.build_models(model, annot_dfs, total_intensities=kwargs['total_intensities'])
    sim = FluxSimulator(
        prior=prior,
        substrate_df=substrate_df,
        mdv_observation_models=obsmods,
        boundary_observation_model=bom,
        num_processes=0,
    )
    data = sim.sample_and_simulate(
        n=n_train,
        n_obs=n_obs,
        x_coordinate_sys=x_coordinate_sys,
    )
    sim.to_hdf(hdf=hdf, result=data, dataset_id=training_id, append=False, expectedrows_multiplier=5)

    if n_validate > 0:
        x_o = sim.sample_and_simulate(
            n=n_validate,
            n_obs=1,  # NB 1, we build 1 posterior per flux-sample!
            x_coordinate_sys=x_coordinate_sys,
        )
        sim.to_hdf(hdf=hdf, result=x_o, dataset_id=validation_id, append=False, expectedrows_multiplier=5)
    pickle.dump(sim, open(sim_pickle, 'wb'))


def _create_coli_glc_dataset(
        hdf,
        sim_pickle,
        training_id='training',
        validation_id='validation',
        x_coordinate_sys='transformation',
        n_train=50000,
        n_validate=50,  # number of validation samples from prior
        n_obs=3,
        from_scratch=True,
        **kwargs
):
    labellings = ['G1', 'GnGU1:1']
    model, kwargs = ecoli_builder(backend='torch', batch_size=50, min_biomass=0.1, ratios=False)

    model.reactions.get_by_id(model.biomass_id).bounds = (0.45, 0.8)
    model.reactions.get_by_id('EX_glc__D_e').bounds = (-10.0, -8.0)

    bom = BoundaryObservationModel(model, measured_boundary_fluxes=[model.biomass_id, 'EX_glc__D_e'], check_noise=False)
    prior = UniFluxPrior(model=model, cache_size=n_train, num_processes=1, logit_xch_fluxes=True)

    substrate_df = kwargs['substrate_df'].loc[labellings]

    if from_scratch:
        _from_scratch(model, substrate_df, prior, hdf)

    mdvdf = FluxSimulator.read_hdf(hdf=hdf, dataset_id='selection_mdvs', what='mdv')
    ser = exclude_low_massiso(mdvdf)
    mdvdf = mdvdf.loc[:, ser.index]
    annot_dfs = {lid: construct_model_annot_df(model, mdvdf.loc[:, lid]) for lid in substrate_df.index}
    obsmods = FluxSimulator.build_models(model, annot_dfs, total_intensities=kwargs['total_intensities'])

    sim = FluxSimulator(
        prior=prior,
        substrate_df=substrate_df,
        mdv_observation_models=obsmods,
        boundary_observation_model=bom,
        num_processes=4,
    )

    res = sim.sample_and_simulate(
        n=n_train,
        n_obs=n_obs,
        x_coordinate_sys=x_coordinate_sys,
    )
    sim.to_hdf(hdf=hdf, result=res, dataset_id=training_id, append=False, expectedrows_multiplier=10)

    if n_validate > 0:
        x_o = sim.sample_and_simulate(
            n=n_validate,
            n_obs=1,  # NB 1, we build 1 posterior per flux-sample!
            x_coordinate_sys=x_coordinate_sys,
        )
        sim.to_hdf(hdf=hdf, result=x_o, dataset_id=validation_id, append=False, expectedrows_multiplier=5)

    pickle.dump(sim, open(sim_pickle, 'wb'))


def _create_designs(from_scratch=True):
    # substrate = 'glc'
    # labellings = ['Gn', 'G1', 'GnGU1:1', 'G5', 'G456']
    # bm_bounds = (0.45, 0.75)
    # ex_id = 'EX_glc__D_e'

    substrate = 'pyr'
    labellings = None
    bm_bounds = (0.05, 0.3)
    ex_id = 'EX_pyr_e'

    # substrate = 'xyl'
    # labellings = None
    # bm_bounds = (0.40, 0.75)
    # ex_id = 'EX_xyl__D_e'

    hdf = os.path.join(SIM_DIR, f'design_{substrate}.h5')
    dataset_id = 'design'
    n = 5000

    pickle_file = 'modkwd.p'
    if os.path.exists(pickle_file):
        model, kwargs = pickle.load(open('modkwd.p', 'rb'))
    else:
        model, kwargs = ecoli_builder(substrate=substrate, backend='torch', batch_size=50, min_biomass=0.1)
        # pickle.dump((model, kwargs), open('modkwd.p', 'wb'))

    model.reactions.get_by_id(model.biomass_id).bounds = bm_bounds
    model.reactions.get_by_id(ex_id).bounds = (-10.0, -8.0)

    prior = UniFluxPrior(model=model, cache_size=n, num_processes=1, logit_xch_fluxes=True)
    substrate_df = kwargs['substrate_df']
    if labellings is not None:
        substrate_df = substrate_df.loc[labellings]

    if from_scratch:
        _from_scratch(model, substrate_df, prior, n, hdf)

    mdvdf = FluxSimulator.read_hdf(hdf=hdf, dataset_id='selection_mdvs', what='mdv')
    ser = exclude_low_massiso(mdvdf)
    mdvdf = mdvdf.loc[:, ser.index]
    annot_dfs = {lid: (construct_model_annot_df(model, mdvdf.loc[:, lid]), None) for lid in substrate_df.index}

    if 'Gn' in annot_dfs:
        annot_dfs['Gn'] = annot_dfs['G1']  # so that we have same signals for natural an d1 labelled glucose

    obsmods = FluxSimulator.build_models(model, annot_dfs)
    bom = BoundaryObservationModel(model, measured_boundary_fluxes=[model.biomass_id, ex_id], check_noise=False)

    sim = FluxSimulator(
        prior=prior,
        substrate_df=substrate_df,
        mdv_observation_models=obsmods,
        boundary_observation_model=bom,
        num_processes=4,
    )

    prior.sample(sample_shape=(n,))
    fluxes = prior.sample_fluxes()

    # res = {}
    res = sim.classical_design(
        fluxes=fluxes,
        x_coordinate_sys='sigma_r'
    )
    res['fluxes'] = fluxes

    if isinstance(model, RatioMixin):
        numerator = model._ratio_num_sum @ fluxes.T
        denominator = model._ratio_den_sum @ fluxes.T
        ratios = numerator / denominator
        res['ratios'] = ratios.T

    sim.to_hdf(hdf=hdf, result=res, dataset_id=dataset_id, append=False, expectedrows_multiplier=10)

if __name__ == "__main__":


    kwargs = {
        'hdf': os.path.join(SIM_DIR, f'multi_modal2.h5'),
        'sim_pickle': os.path.join(SIM_DIR, f'multi_modal_sim2.p'),
        'validation_id': 'validation',
        'x_coordinate_sys': 'all',
        'validation_fraction': 0.00,
        'show_progress_bars': True,
        'max_t': 15,
    }
    _create_multi_modal_dataset(**kwargs)

    # result = main(**kwargs)
    # ding = SNPE_Trainable(config=kwargs)

    # result = main(**kwargs)
    # post = ding._epoch_train_step()

    # post = torch.load(open(r"C:\python_projects\pysumo\simulations\ray_logs\test\SNPE_Trainable_10868_00018_18_dropout_probability=0.1387,hidden_features=90,learning_rate=0.0761,num_bins=16,num_blocks=4,num_tran_2022-11-07_12-36-28\checkpoint_000001\posterior.p", 'rb'))
    # data_o = _load_data(hdf=kwargs['hdf'], dataset_id=kwargs['validation_id'], x_coordinate_sys='transformation')
    # xs, theta = data_o['transformation'], data_o['theta']
    # # prior = pickle.load(open(kwargs['sim_pickle'], 'rb'))._prior
    # f = fraction_samples_within_support(post, x=xs)
    # snpec = train_fn(config=kwargs)

    # _create_spiro_dataset(**kwargs)
    # snpec = train_fn(kwargs)
    # train_fn(kwargs)

    # kwargs = {
    #     'hdf': os.path.join(SIM_DIR, f'ecoli.h5'),
    #     #     'sim_pickle': os.path.join(SIM_DIR, f'ecoli_sim.p'),
    #     'max_num_epochs': 1,
    # }
    # _create_coli_glc_dataset(**kwargs, n_train=2000, n_validate=50, n_obs=3)