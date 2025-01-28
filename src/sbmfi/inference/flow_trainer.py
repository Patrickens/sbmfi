import torch
# from sbi.neural_nets.flow import build_maf, build_nsf
from pyknos.nflows.transforms import PointwiseAffineTransform
from torch import nn
from typing import Optional
from normflows.distributions.base import Uniform, UniformGaussian
from normflows.flows import Permute, LULinearPermute
from sbmfi.inference.normflows_patch import (
    CircularAutoregressiveRationalQuadraticSpline,
    CircularCoupledRationalQuadraticSpline,
    EmbeddingConditionalNormalizingFlow,
    DiagGaussianScale
)
from normflows.core import ConditionalNormalizingFlow
from torch.utils.data import DataLoader
from normflows.flows.neural_spline.wrapper import (
    CoupledRationalQuadraticSpline,
    AutoregressiveRationalQuadraticSpline
)
from sbmfi.core.polytopia import FluxCoordinateMapper
from sbmfi.core.simulator import _BaseSimulator
from sbmfi.priors.uniform import _BasePrior
import numpy as np
import tqdm
from typing import Dict
import os
from sbmfi.settings import SIM_DIR
import shutil
from pathlib import Path
import ray
from ray import tune
from ray.tune.stopper import TrialPlateauStopper
from torch.utils.data import Dataset, DataLoader, random_split
from normflows.flows.neural_spline.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from normflows.nets.made import MADE

class Flow_Dataset(Dataset):
    def __init__(self, data: torch.Tensor, theta: torch.Tensor, standardize=True):
        if theta.shape[0] != data.shape[0]:
            raise ValueError

        if data.ndim == 3:
            n, n_obs, n_d = data.shape
            if theta.ndim == 2:
                theta = theta.tile(n_obs, 1, 1).transpose(0, 1)

        if (data.ndim == 3) and (theta.ndim == 3):
            data = data.view(n * n_obs, n_d)
            n_t = theta.shape[-1]
            theta = theta.contiguous().view(n * n_obs, n_t)

        self.data_mean = data.mean(0, keepdims=True)
        self.data_std = data.std(0, keepdims=True)

        if standardize:
            data = (data - self.data_mean) / self.data_std
            # data = (data * self.data_std) + self.data_mean # reverse

        self.data = data
        self.theta = theta

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.theta[idx]


def flow_constructor(
        fcm: FluxCoordinateMapper,
        coordinate_id,
        log_xch,
        rescale_val,
        embedding_net=None,
        num_context_channels=None,
        autoregressive=True,
        num_blocks=4,
        num_hidden_channels=5,
        num_bins=10,
        dropout_probability=0.0,
        num_transforms = 3,
        init_identity=True,
        permute='lu',
        p=None,
        device='cpu',
):
    # prior_flow just makes a normalizing flow that matches samples from a prior
    #   thus not needing to fuck around with context = conditioning on data

    n_theta = len(fcm.theta_id(coordinate_id, log_xch))

    if not torch.cuda.is_available():
        device = 'cpu'

    if coordinate_id not in ['cylinder', 'rounded']:
        raise ValueError(f'flow training only coordinate_id sho')
    elif coordinate_id == 'rounded':
        if log_xch:
            raise NotImplementedError
        ind = list(range(n_theta - fcm._nx, n_theta))
        scale = torch.ones(n_theta)
        scale[-fcm._nx:] *= rescale_val * 2  # need to pass the width!
        base = UniformGaussian(ndim=n_theta, ind=ind, scale=scale)
        base.scale.to(device)
    elif coordinate_id == 'cylinder':
        if rescale_val is None:
            raise ValueError(f'flow only works when all values are rescaleval')
        if (fcm._nx > 0) and log_xch:
            ind = list(range(n_theta - fcm._nx))
            scale = torch.ones(n_theta)
            scale[:-fcm._nx] *= rescale_val * 2  # need to pass the width!
            base = UniformGaussian(ndim=n_theta, ind=ind, scale=scale)
        else:
            base = Uniform(shape=n_theta, low=-rescale_val, high=rescale_val)
            base.low = base.low.to(device)
            base.high = base.high.to(device)

    transforms = []
    for i in range(num_transforms):
        common_kwargs = dict(
            num_input_channels=n_theta,
            num_blocks=num_blocks,
            num_hidden_channels=num_hidden_channels,
            num_context_channels=num_context_channels,
            num_bins=num_bins,
            tail_bound=rescale_val,
            activation=nn.ReLU,
            dropout_probability=dropout_probability,
            init_identity=init_identity,
        )
        if coordinate_id == 'cylinder':
            common_kwargs['ind_circ'] = [0]
        if (coordinate_id == 'cylinder') and autoregressive:
            transform = CircularAutoregressiveRationalQuadraticSpline(
                **common_kwargs,
                permute_mask=True,
            )
        elif (coordinate_id == 'cylinder') and not autoregressive:
            transform = CircularCoupledRationalQuadraticSpline(
                **common_kwargs,
                reverse_mask=False,
                mask=None,
            )
        elif (coordinate_id == 'rounded') and autoregressive:
            transform = AutoregressiveRationalQuadraticSpline(**common_kwargs)
        elif (coordinate_id == 'rounded') and not autoregressive:
            transform = CoupledRationalQuadraticSpline(**common_kwargs)

        if permute == 'lu':
            if isinstance(transform, CircularAutoregressiveRationalQuadraticSpline):
                raise ValueError('this CircularAutoregressiveRationalQuadraticSpline and LU transforms do not play together!')
            perm = LULinearPermute(num_channels=n_theta, identity_init=init_identity)
        elif permute == 'shuffle':
            perm = Permute(num_channels=n_theta, mode='shuffle')

        transform_sequence = [transform]
        if permute is not None:
            transform_sequence = [transform, perm]

        transforms.extend(transform_sequence)

    if permute is not None:
        transforms = transforms[:-1]


    flow = EmbeddingConditionalNormalizingFlow(q0=base, flows=transforms, embedding_net=embedding_net, p=p)
    flow.to(device=device)
    return flow


class MFA_Flow(tune.Trainable):
    def _prior_flow_step(self):
        theta = self._prior.sample((self._batch_size,))
        loss = self._flow.forward_kld(theta)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            self._optimizer.step()
        return {'forward_kld': loss.to('cpu').data.numpy().item()}

    def _posterior_flow_step(self):
        raise NotImplementedError

    def _get_data_loaders(self):
        pass

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        pass

    def setup(self, config: Dict, prior=None, simulator=None):
        self._prior = prior
        self._simulator = simulator

        # TODO load data and use a batch to parametrize z-scoring
        embedding = config.get('data_embedding')
        embedding_net = None
        if embedding == 'z_score_trainable':
            embedding_net = PointwiseAffineTransform  # TODO shift and scale are registered as buffers????
            # register as parameters
            raise NotImplementedError
        # TODO come up with other embedding nets?

        prior_flow = config.get('prior_flow', True)

        self._flow = flow_constructor(
            fcm=prior._fcm,
            simulator=simulator,
            embedding_net=embedding_net,
            prior_flow=prior_flow,
            autoregressive=config.get('autoregressive', True),
            num_blocks=config.get('num_blocks', 2),
            num_hidden_channels=config.get('num_hidden_channels', 10),
            num_bins=config.get('num_bins', 10),
            dropout_probability=config.get('dropout_probability', 0.1),
            use_batch_norm=config.get('use_batch_norm', False),
            num_transforms=config.get('num_transforms', 2),
            init_identity=config.get('init_identity', True),
        )

        self._optimizer = torch.optim.Adam(
            self._flow.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self._batch_size = config.get('batch_size', 512)

        if prior_flow:
            self.step = self._prior_flow_step
        else:
            self.step = self._posterior_flow_step


class Kanker(MFA_Flow):
    def __init__(
            self,
            prior: _BasePrior,
            simulator: _BaseSimulator = None,
    ):
        self._prior = prior
        self._flow = flow_constructor(prior._fcm)
        self._simulator = simulator
        self.step = self._prior_flow_step
        self._optimizer = torch.optim.Adam(
            self._flow.parameters(),
            lr=1e-3,
            weight_decay=1e-5,
        )
        self._batch_size = 512


class CVStopper(TrialPlateauStopper):

    def __init__(
            self,
            metric: str,
            std: float = 0.01,
            num_results: int = 4,
            grace_period: int = 4,
            metric_threshold: Optional[float] = None,
            mode: Optional[str] = None,
            max_kld = 5.0,
    ):
        super().__init__(metric, std, num_results, grace_period, metric_threshold, mode)
        self._cv = self._std
        self._max_kld = max_kld

    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)

        if metric_result  > self._max_kld:
            return True

        self._trial_results[trial_id].append(metric_result)
        self._iter[trial_id] += 1

        # If still in grace period, do not stop yet
        if self._iter[trial_id] < self._grace_period:
            return False

        # If not enough results yet, do not stop yet
        if len(self._trial_results[trial_id]) < self._num_results:
            return False

        # If metric threshold value not reached, do not stop yet
        if self._metric_threshold is not None:
            if self._mode == "min" and metric_result > self._metric_threshold:
                return False
            elif self._mode == "max" and metric_result < self._metric_threshold:
                return False

        # Calculate stdev of last `num_results` results
        try:
            current_std = np.std(self._trial_results[trial_id])
            current_mu = np.mean(self._trial_results[trial_id])
            current_cv = current_std / abs(current_mu)
        except Exception:
            current_cv = float("inf")

        # If stdev is lower than threshold, stop early.
        return current_cv < self._cv


def main(
        prior: _BasePrior,
        simulator: _BaseSimulator = None,
        tune_id: str = 'test',
        prior_flow=True,
        num_samples=200,
        max_t=300,

):
    if not prior_flow:
        raise NotImplementedError('have yet to think about doing posteriors')

    param_space = {
        'prior_flow': prior_flow,
        'autoregressive': tune.choice((True, False)),
        'num_blocks': tune.randint(2, 5),
        'num_hidden_channels': tune.qrandint(10, 100, 5),
        'num_bins': tune.qrandint(4, 20, 4),
        'dropout_probability': tune.uniform(0.0, 0.5),
        'use_batch_norm': tune.choice((True, False)),
        'num_transforms': tune.randint(2, 5),
        'init_identity': tune.choice((True, False)),
        'learning_rate': tune.loguniform(1e-5, 3e-1),
        'weight_decay': tune.loguniform(1e-7, 1e-5),
        'batch_size': tune.choice(2 ** np.arange(4, 10)),
    }
    whatune = (
        'autoregressive', 'num_blocks', 'num_hidden_channels', 'num_bins', 'dropout_probability', 'use_batch_norm',
        'num_transforms', 'learning_rate', 'batch_size'
    )
    param_space = {k: v for k, v in param_space.items() if k in whatune}
    local_dir = os.path.join(SIM_DIR, 'ray_logs')
    dirpath = Path(local_dir) / tune_id
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    callback = None  # TODO for posterior trainine, useful to do some calibration automatically!

    metric = 'forward_kld'
    grace_period = 5
    tuner = tune.Tuner(
        trainable=tune.with_parameters(MFA_Flow, prior=prior, simulator=simulator),
        # trainable=tune.with_resources(
        #     tune.with_parameters(MFA_Flow, prior=prior, simulator=simulator),
        #     resources={"cpu": 1, "gpu": 0, "memory": 1e9},
        # ),
        tune_config=tune.TuneConfig(
            search_alg=None,  # NB defaults to random search, think of doing BOHB or BayesOptSearch
            metric=metric,
            mode='min',
            scheduler=tune.schedulers.ASHAScheduler(
                max_t=max_t,
                grace_period=grace_period,
                reduction_factor=2
            ),
            num_samples=num_samples,
        ),
        run_config=ray.air.RunConfig(
            local_dir=local_dir,
            name=tune_id,
            callbacks=callback,
            stop=CVStopper(  # NB checks for convergence based on CV
                metric=metric,
                std=0.005,  # TODO THIS SHOULD ACTUALLY THE CV!
                num_results=20,
                grace_period=grace_period,
                metric_threshold=None,
                mode='min',
                max_kld=5.0,
            ),
            log_to_file=True,
            # checkpoint_config=ray.air.CheckpointConfig(
            #     checkpoint_at_end=True
            # )
        ),
        param_space=param_space,
    )
    result = tuner.fit()
    return result


def flow_trainer(
    flow,
    dataset,
    optimizer=torch.optim.Adam,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=64,
    max_iter=100,
    show_progress=True,
):
    optim = optimizer(flow.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if max_iter is None:
        max_iter = len(train_loader)

    if show_progress:
        pbar = tqdm.tqdm(total=max_iter, ncols=100, desc='loss')
    losses = []
    try:
        for i, (x, y) in enumerate(train_loader):
            loss = flow.forward_kld(y, context=x)
            optim.zero_grad()
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optim.step()
            losses.append(loss.to('cpu').data.numpy())
            if show_progress:
                pbar.update(1)
                pbar.set_postfix(forward_kld=np.round(losses[-1], 5))
            if i == max_iter:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if show_progress:
            pbar.close()
    return np.array(losses)


if __name__ == "__main__":

    import pickle
    import math

    batch_size = 1024

    dataset, fcm = pickle.load(open(r"C:\python_projects\sbmfi\dat_fcm.p",'rb'))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  # WORKS WITH batch_size=2048, try 1024

    # n_data = data.shape[-1]
    # n_hidden = math.ceil(n_data / 1.5)
    # n_latent = math.ceil(n_data / 3)
    # n_hidlay = 2
    # embedding_net = [torch.nn.Linear(n_data, n_hidden), torch.nn.LeakyReLU(0.01)]
    # for i in range(n_hidlay):
    #     embedding_net.extend([torch.nn.Linear(n_hidden, n_hidden), torch.nn.LeakyReLU(0.01)])
    # embedding_net.append(torch.nn.Linear(n_hidden, n_latent))
    # embedding_net = torch.nn.Sequential(*embedding_net)
    #

    common_kwargs = dict(
        fcm=fcm,
        coordinate_id='cylinder',
        rescale_val=1.0,
        log_xch=True,
        autoregressive=True,
        num_blocks=4,
        num_hidden_channels=128,
        num_bins=8,
        dropout_probability=0.1,
        num_transforms=10,
        init_identity=True,
        permute=None,
        p=None,
        device='cuda:0'
    )


    cond_flow = flow_constructor(
        num_context_channels=dataset[0][0].shape[-1],
        **common_kwargs
    )
    from torch.profiler import profile, record_function, ProfilerActivity
    x, theta = next(iter(dataloader))
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            cond_flow.forward_kld(theta, x)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


    def autoregressive_net_forward(self: MADE, inputs, context=None):
        outputs = self.preprocessing(inputs)
        outputs = self.initial_layer(outputs)
        if context is not None:
            outputs += self.context_layer(context)
        for block in self.blocks:
            outputs = block(outputs, context)
        outputs = self.final_layer(outputs)
        return outputs

    def mprqat_forward(self: MaskedPiecewiseRationalQuadraticAutoregressive, inputs, context=None):
        # autoregressive_params = self.autoregressive_net(inputs, context)
        autoregressive_params = autoregressive_net_forward(self.autoregressive_net, inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self: CircularAutoregressiveRationalQuadraticSpline, z, context=None):
        # z, log_det = self.mprqat(z, context=context)
        z, log_det = mprqat_forward(self.mprqat, z, context=context)
        return z, log_det.view(-1)

    def forward_kld(self: EmbeddingConditionalNormalizingFlow, x, context=None):
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            # z, log_det = self.flows[i].inverse(z, context=context)
            z, log_det = inverse(self.flows[i], z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return -torch.mean(log_q)

    def sample(self: ConditionalNormalizingFlow, num_samples, context=None):
        z, log_q = self.q0(num_samples, context=context)
        print(z)
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        return z, log_q

    def train_main(dataloader, flow: ConditionalNormalizingFlow, optimizer=None, losses=None, n_epoch=25, scheduler=None, learning_rate=1e-4, weight_decay=1e-4, LR_gamma=1.0):
        n_steps = n_epoch * len(dataloader)
        pbar = tqdm.tqdm(total=n_steps, ncols=120, position=0)

        if optimizer is None:
            optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if (LR_gamma < 1.0) and (scheduler is None):
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_gamma, last_epoch=-1)

        try:
            get_val = lambda x: x.to('cpu').data.numpy()
            if losses is None:
                losses = []
            for epoch in range(n_epoch):
                for i, (x_chunk, theta_chunk) in enumerate(dataloader):
                    context = None
                    if hasattr(flow.flows[0].mprqat.autoregressive_net, 'context_layer'):
                        context = x_chunk
                    # loss = forward_kld(flow, theta_chunk, context)
                    loss = flow.forward_kld(theta_chunk)
                    return
                    optimizer.zero_grad()
                    if ~(torch.isnan(loss) | torch.isinf(loss)):
                        loss.backward()
                        nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
                        optimizer.step()
                    else:
                        raise ValueError(f'loss: {loss}')
                    np_loss = get_val(loss)
                    losses.append(float(np_loss))
                    pbar.update()
                    pbar.set_postfix(loss=np_loss.round(4))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
            raise e
        finally:
            pbar.close()
        return flow, losses




    # n_epoch = 5
    # LR_gamma = 1.0
    #
    # learning_rate = 1e-4
    # weight_decay = 1e-4
    #
    # cond_optimizer = torch.optim.Adam(cond_flow.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # cond_losses = []
    # # train_main(dataloader, cond_flow, cond_optimizer, cond_losses)
    # x_chunk, theta_chunk = next(iter(dataloader))
    # a = cond_flow.forward_kld(x=theta_chunk.to('cuda:0'), context=x_chunk.to('cuda:0'))
    # x,lq = cond_flow.sample(4)
    # s = cond_flow.q0.log_prob(x)
    # print(s)
    # prior_losses = []
    # prior_optimizer = torch.optim.Adam(prior_flow.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # train_main(dataloader, prior_flow, prior_optimizer, prior_losses)




