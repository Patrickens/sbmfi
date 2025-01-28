import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sbmfi.core.simulator import _BaseSimulator
from sbmfi.settings import BASE_DIR
import math
import tqdm
import os
import normflows
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.tune import Trial

import numpy as np
import pandas as pd


class MDFVAE(nn.Module):
    # denoising: https://davidstutz.de/denoising-variational-auto-encoders/
    # https://arxiv.org/abs/1511.06406
    # denoising auto-encoder; pass in noisy features, decode to noise-free features; use latent variables for inference
    def __init__(
            self,
            n_data: int,
            n_latent = 0.3,
            n_hidden = 0.6,
            n_hidden_layers = 2,
            activation = nn.LeakyReLU(0.01),
            prior = None,
            flow = None,
            denoising = False,
            decoder_unit_var = True,
    ):
        super().__init__()

        bias = True
        device = 'cpu'  # still need to figure out GPU stuff

        if isinstance(n_latent, float):
            n_latent = math.ceil(n_latent * n_data)
        if isinstance(n_hidden, float):
            n_hidden = math.ceil(n_hidden * n_data)

        if (n_latent > n_hidden) or (n_latent > n_data):
            raise ValueError(f'n_data: {n_data}, n_hidden: {n_hidden}, n_latent: {n_latent}')

        endocer_layers = [nn.Linear(n_data, n_hidden, bias=bias), activation]
        decoder_layers = [nn.Linear(n_latent, n_hidden, bias=bias), activation]
        for i in range(n_hidden_layers):
            endocer_layers.extend([nn.Linear(n_hidden, n_hidden, bias=bias), activation])
            decoder_layers.extend([nn.Linear(n_hidden, n_hidden, bias=bias), activation])

        # encoder diagonal Gaussian
        self.encoder = nn.Sequential(*endocer_layers)
        self.encoder_mean = nn.Linear(n_hidden, n_latent)
        self.encoder_log_var = nn.Linear(n_hidden, n_latent)

        # decoder diagonal Gaussian
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder_mean = nn.Linear(n_hidden, n_data)
        self.decoder_unit_var = decoder_unit_var
        if decoder_unit_var:
            self.decoder_std = torch.Tensor(1)
        else:
            self.decoder_std = nn.Linear(n_hidden, n_data)

        if prior is None:
            prior = torch.distributions.MultivariateNormal(
                torch.zeros(n_latent, device=device), torch.eye(n_latent, device=device)
            )
        self.prior = prior
        self.flow = flow  # Normalizing flow
        self.denoising = denoising

    def encode(self, x, num_samples=1):
        hidden = self.encoder(x)
        mean = self.encoder_mean(hidden)
        log_var = self.encoder_log_var(hidden)

        # reparametrization trick
        std = torch.exp(0.5 * log_var)
        epsilon = torch.rand((num_samples,) +  std.size())
        z = mean + std * epsilon

        log_q = -0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(  # this is the KL-div term
            2 * np.pi
        ) - torch.sum(torch.log(std) + 0.5 * torch.pow(epsilon, 2), list(range(2, z.dim())))

        return z, log_q

    def decode(self, z, x, y=None):
        if self.denoising and (y is None):
            raise ValueError('pass y, this is a denoising VAE')

        if len(z) > len(x):  # this accounts for n_samples, which results in larger first dimension of z
            x = x.unsqueeze(1)
            x = x.repeat(1, z.size()[0] // x.size()[0], *((x.dim() - 2) * [1])).view(
                -1, *x.size()[2:]
            )

        if (y is not None) and (len(z) > len(y)):  # this accounts for n_samples, which results in larger first dimension of z
            y = y.unsqueeze(1)
            y = y.repeat(1, z.size()[0] // y.size()[0], *((y.dim() - 2) * [1])).view(
                -1, *y.size()[2:]
            )

        hidden = self.decoder(z)
        mean = self.decoder_mean(hidden)
        if self.decoder_unit_var:  # this is kinda ugly
            std = self.decoder_std
        else:
            std = self.decoder_std(hidden)
        var = torch.exp(std)

        if self.denoising:
            sample = y
        else:
            sample = x
        log_p = -0.5 * torch.prod(torch.tensor(z.size()[1:])) * np.log(  # this is the MSE, but weighted by gaussian
            2 * np.pi
        ) - 0.5 * torch.sum(
            std + (sample - mean) ** 2 / var, list(range(1, z.dim()))
        )
        return mean, log_p

    def forward(self, x, y=None, num_samples=1):
        if self.denoising and (y is None):
            raise ValueError('need to pass non-corrupted data y for denoising auto-encoder')
        z, log_q = self.encode(x, num_samples)
        z = z.view(-1, *z.size()[2:])
        log_q = log_q.view(-1, *log_q.size()[2:])

        if self.flow is not None:
            z, log_det = self.flow.forward_and_log_det(z)
            log_q -= log_det

        x_mean, log_p = self.decode(z, x, y)
        log_p += self.prior.log_prob(z)

        # Separate batch and sample dimension again
        z = z.view(num_samples, -1, *z.size()[1:])
        log_q = log_q.view(num_samples, -1, *log_q.size()[1:])
        log_p = log_p.view(num_samples, -1, *log_p.size()[1:])
        return z, x_mean, log_q, log_p


def linear_annealing(step, n_steps, min_beta=0.0, max_beta=1.0, start_step=0.2, stop_step=0.8):
    if isinstance(start_step, float):
        start_step = math.ceil(start_step * n_steps)

    if isinstance(stop_step, float):
        stop_step = math.floor(stop_step * n_steps)

    if (stop_step < start_step) or (stop_step > n_steps):
        raise ValueError(f'start {start_step}, stop {stop_step}, n {n_steps}')

    if step < start_step:
        return min_beta
    elif step > stop_step:
        return max_beta
    return ((step - start_step) / (stop_step - start_step) * (max_beta - min_beta)) + min_beta


def cyclical_annealing(step, n_steps, min_beta=0.0, max_beta=1.0, n_cycles=4, cycle_frac=0.75, start_step=0.2):
    if isinstance(start_step, float):
        start_step = math.ceil(start_step * n_steps)

    if step < start_step:
        return min_beta

    n_steps_wup = n_steps - start_step

    period = (n_steps_wup / n_cycles)  # N_iters/N_cycles
    internal_period = (step - start_step) % (period)  # Itteration_number/(Global Period)
    tau = internal_period / period

    if tau > cycle_frac:
        return max_beta
    else:
        return min(max_beta, max(tau / cycle_frac, min_beta))  # Linear function


class MDVAE_Dataset(Dataset):
    def __init__(self, data: torch.Tensor, mu: torch.Tensor = None, standardize=True):
        if data.ndim == 3:
            n, n_obs, n_d = data.shape
            shape = (n * n_obs, n_d)
            data = data.view(shape)
            if mu is not None:
                if mu.ndim != 3:
                    raise ValueError
                if mu.shape[1] != n_obs:
                    mu = torch.tile(mu, (1, n_obs, 1))
                mu = mu.view(shape)
        elif (data.ndim == 2) and (mu.ndim == 2) and (data.shape != mu.shape):
            raise ValueError
        else:
            raise ValueError

        self.data_mean = data.mean(0, keepdims=True)
        self.data_std = data.std(0, keepdims=True)
        if mu is not None:
            self.mu_mean = mu.mean(0, keepdims=True)
            self.mu_std = mu.std(0, keepdims=True)
            if (self.mu_std < 1e-6).any():
                raise ValueError('there are non-informative data-dimensions!')

        if standardize:
            data = (data - self.data_mean) / self.data_std
            # data = (data * self.data_std) + self.data_mean # reverse
            if mu is not None:
                mu = (mu - self.mu_mean) / self.mu_std
                # mu = (mu * self.mu_std) + self.mu_mean # reverse

        self.data = data
        self.mu = mu

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.mu is None:
            return self.data[idx], self.data[idx]
        return self.data[idx], self.mu[idx]


def ray_train_MDVAE(config, cwd, ray=False, show_progress=False):
    # n_epoch = int(config.get('n_epoch', 3))  # previously, ray config would return odd stuff, so I had to specify a type.
    # This is annoying for n_hidden/n_latent, since these can be either int or float
    n_epoch = int(config.get('n_epoch', 3))
    n_hidden = config.get('n_hidden', 0.6)
    n_latent = config.get('n_latent', 0.3)  # not specifying type
    n_hidden_layers = int(config.get('n_hidden_layers', 2))
    batch_size = int(config.get('batch_size', 32))
    learning_rate = float(config.get('learning_rate', 1e-4))
    weight_decay = float(config.get('weight_decay', 1e-4))
    LR_gamma = float(config.get('lr_gamma', 1.0))

    # this was all the stuff for beta annealing, which ended up doing jack-all ...
    beta_annealing = config.get('beta_annealing', 'constant')
    beta = config.get('beta', 1.0)  # if constant beta

    min_beta = config.get('min_beta', 0.0)
    max_beta = config.get('max_beta', 1.0)
    warmup_frac = config.get('warmup_frac', 0.15)
    cooldown_frac = config.get('cooldown_frac', 0.9)

    n_cycles = config.get('n_cycles', 4)
    cycle_frac = config.get('cycle_frac', 0.75)

    if beta_annealing == 'constant':
        f_beta = lambda step: beta
    elif beta_annealing == 'linear':
        f_beta = lambda step: linear_annealing(step, n_steps, min_beta, max_beta, warmup_frac, cooldown_frac)
    elif beta_annealing == 'cyclical':
        f_beta = lambda step: cyclical_annealing(step, n_steps, min_beta, max_beta, n_cycles, cycle_frac, warmup_frac)
    else:
        raise ValueError

    train_ds = torch.load(os.path.join(cwd, 'train_ds.pt'))
    val_ds = torch.load(os.path.join(cwd, 'val_ds.pt'))

    x_val, y_val = val_ds[:]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    flow = config.get('flow', None)
    num_samples = config.get('num_samples', 1)
    decoder_unit_var = config.get('decoder_unit_var', True)
    mdfvae = MDFVAE(
        n_data=x_val.shape[-1],
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_hidden_layers=n_hidden_layers,
        flow=flow,
        denoising=~(x_val == y_val).all().to('cpu').data.numpy(),
        decoder_unit_var=decoder_unit_var,
    )

    optimizer = torch.optim.Adam(mdfvae.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if LR_gamma < 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_gamma, last_epoch=-1)

    get_val = lambda x: x.to('cpu').data.numpy()
    n_steps = n_epoch * len(train_loader)
    if show_progress:
        pbar = tqdm.tqdm(total=n_steps, ncols=120, position=0)

    losses = []

    try:
        for epoch in range(n_epoch):
            for i, (x, y) in enumerate(train_loader):

                z, x_hat, log_q, log_p = mdfvae.forward(x.float(), y.float(), num_samples=num_samples)

                step = i + epoch * len(train_loader)
                beta = f_beta(step)
                mean_log_q = torch.mean(log_q)
                mean_log_p = torch.mean(log_p)
                loss = -mean_log_p + beta * mean_log_q

                optimizer.zero_grad()
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(mdvae.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    raise ValueError(f'loss: {loss}')
                pbar.update()
                losses.append((0, get_val(loss), get_val(mean_log_q), get_val(-mean_log_p), beta))

                if ray:
                    raise NotImplemented(' if necessary, still need to implement ray.tune.report')

                if (i % 100 == 0) and show_progress:
                    pbar.set_postfix(loss=losses[-1][1].round(4), log_q=losses[-1][2].round(4), log_p=losses[-1][3].round(4), beta=round(beta, 4))
            if LR_gamma < 1.0:
                scheduler.step()

            with torch.no_grad():
                z, x_hat, log_q, log_p = mdfvae.forward(x_val.float(), y_val.float())
                loss = log_p + beta * log_q
                losses.append((1, get_val(loss), get_val(mean_log_q), get_val(-mean_log_p), beta))

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        raise e
    finally:
        losses = pd.DataFrame(losses, columns=['train0_val1', 'loss', 'log_q', 'log_p', 'beta'])
        if show_progress:
            pbar.close()
        return mdfvae, losses


def ray_main(hdf, simulator: _BaseSimulator, dataset_id: str, denoising: bool=False, val_frac: float=0.1, max_n_hidden=0.8,):
    if not simulator._la.backend == 'torch':
        raise ValueError

    mdvs = simulator.read_hdf(hdf=hdf, dataset_id=dataset_id, what='mdv')
    data = simulator.read_hdf(hdf=hdf, dataset_id=dataset_id, what='data')
    theta = simulator.read_hdf(hdf=hdf, dataset_id=dataset_id, what='theta')
    mu = simulator.simulate(theta=theta, mdvs=mdvs, n_obs=0) if denoising else None
    dataset = MDVAE_Dataset(data, mu)

    n_validate = math.ceil(val_frac * len(dataset))

    # prepare datasets
    train_ds, val_ds = random_split(
        dataset,
        lengths=(len(dataset) - n_validate, n_validate),
        # generator=simulator._la._BACKEND._rng
    )
    train_file = os.path.join(BASE_DIR, 'train_ds.pt')
    val_file = os.path.join(BASE_DIR, 'val_ds.pt')
    torch.save(train_ds, train_file)
    torch.save(val_ds, val_file)

    # figure out dimensions of the VAE n_data -> n_hidden -> n_latent == n_free_vars
    n_data, n_latent = data.shape[-1], len(simulator.theta_id)
    min_n_hidden = n_latent / n_data
    if min_n_hidden > max_n_hidden:
        raise ValueError

    config = {
        'batch_size': tune.choice(list(2 ** np.arange(3, 9))),
        'n_hidden': tune.uniform(min_n_hidden, max_n_hidden),
        'n_data': n_data,
        'n_latent': n_latent,
        'n_hidden_layers': tune.randint(1, 4),
        'learning_rate': tune.loguniform(1e-5, 3e-1),
    }
    if not ray.is_initialized():
        ray.init(
            local_mode=True,
            log_to_driver=False,
            num_cpus=2,
            include_dashboard=True
        )
    metric = 'val_loss'
    mode = 'min'
    hyperopt_search = HyperOptSearch(metric=metric, mode=mode)

    def short_dirname(trial: Trial):
        return "trial_" + str(trial.trial_id)

    tuner = tune.Tuner(
        ray_train_MDVAE,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=40,
            search_alg=hyperopt_search,
            max_concurrent_trials=6,
            scheduler=ASHAScheduler(time_attr='epoch', metric=metric, mode=mode),
            trial_dirname_creator=short_dirname
        ),
        run_config=train.RunConfig(storage_path=os.path.join(BASE_DIR, 'ray_mdvae'), name="test_experiment")
    )
    results = tuner.fit()

    os.remove(train_file)
    os.remove(val_file)
    return results


def create_flow(n_latent, flow_type='mix', n_flows=15, ):

    if flow_type == 'planar':
        flows = [normflows.flows.Planar((n_latent,)) for k in range(n_flows)]
    elif flow_type == 'radial':
        flows = [normflows.flows.Radial((n_latent,)) for k in range(n_flows)]
    elif flow_type == 'real_nvp':
        b = torch.tensor(n_latent // 2 * [0, 1] + n_latent % 2 * [0])
        flows = []
        for i in range(n_flows):
            s = normflows.nets.MLP([n_latent, n_latent])
            t = normflows.nets.MLP([n_latent, n_latent])
            if i % 2 == 0:
                flows += [normflows.flows.MaskedAffineFlow(b, t, s)]
            else:
                flows += [normflows.flows.MaskedAffineFlow(1 - b, t, s)]
    elif flow_type == 'mix':
        raise NotImplementedError
    else:
        raise NotImplementedError

    base = normflows.distributions.base.DiagGaussian(n_latent)
    return normflows.NormalizingFlow(q0=base, flows=flows)


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    from normflows import NormalizingFlowVAE
    from normflows.distributions import NNDiagGaussian, NNDiagGaussianDecoder


    # n_data = 13
    # n_latent = 4

    # flow = create_flow(n_latent=n_latent, flow_type='real_nvp')
    # mdvae, losses = ray_train_MDVAE({
    #     'n_epoch': 25,
    #     'n_hidden': 0.7,
    #     'n_latent': n_latent,
    #     'n_hidden_layers': 2,
    #     'learning_rate': 1e-3,
    #     'batch_size': 64,
    #     'LR_gamma': 0.9,
    #     'decoder_unit_var': False,
    #     'flow': flow,
    #     'num_samples': 1,
    #     'beta': 0.0,
    #     'beta_annealing': 'constant',
    #     'warmup_frac': 0.2,
    #     'min_beta': 0.0,
    # }, cwd=BASE_DIR, show_progress=True)  # TTHIS ONE WORKED ONCE


    # flow = create_flow(n_latent=n_latent, flow_type='real_nvp')
    # mdvae, losses = ray_train_MDVAE({
    #     'n_epoch': 25,
    #     'n_hidden': 0.7,
    #     'n_latent': n_latent,
    #     'n_hidden_layers': 2,
    #     'learning_rate': 1e-4,
    #     'batch_size': 64,
    #     'LR_gamma': 0.9,
    #     'decoder_unit_var': False,  # decoder_unit_var=False makes for a working auto-encoder with beta=0.0 and flow=None and lr=1e-4!
    #     'num_samples': 1,
    #     'beta': 0.0,
    #     'beta_annealing': 'constant',
    #     'warmup_frac': 0.2,
    #     'min_beta': 0.0,
    # }, cwd=BASE_DIR, show_progress=True)


    n_data = 13
    n_latent = 7
    flow = create_flow(n_latent=n_latent, flow_type='real_nvp')
    mdvae, losses = ray_train_MDVAE({
        'n_epoch': 25,
        'n_hidden': 0.7,
        'n_latent': n_latent,
        'n_hidden_layers': 2,
        'learning_rate': 1e-4,
        'batch_size': 64,
        'LR_gamma': 0.9,
        'decoder_unit_var': False,  # decoder_unit_var=False makes for a working auto-encoder with beta=0.0 and flow=None and lr=1e-4!
        'flow': flow,
        'num_samples': 1,
        'beta': 1.0,
        'beta_annealing': 'constant',
        'warmup_frac': 0.2,
        'min_beta': 0.0,
    }, cwd=BASE_DIR, show_progress=True)

    torch.set_printoptions(linewidth=200)

    val_ds = torch.load(os.path.join(BASE_DIR, 'val_ds.pt'))
    x_in, y_in = val_ds[:]
    with torch.no_grad():
        z, x_hat, log_q, log_p = mdvae.forward(x_in.float(), y_in.float())
    print(y_in[[66, 12, 50]])
    print(x_hat[[66, 12, 50]])
    print((y_in - x_hat)[[66, 12, 50]].round(decimals=4))
    print((((y_in - x_hat)**2).mean()).round(decimals=4))

    from normflows.nets import MLP

    # model, kwargs = spiro(
    #     batch_size=50, include_bom=True,
    #     backend='torch', v2_reversible=True, build_simulator=True, which_measurements='lcms',
    #     which_labellings=list('AB')
    # )
    # sim = kwargs['basebayes']
    # hdf = r"C:\python_projects\sbmfi\spiro_mdvae_test.h5"
    # did = 'test1'
    #
    # results = ray_main(hdf, sim, did)

