import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sbmfi.core.simulator import _BaseSimulator
from sbmfi.core.util import hdf_opener_and_closer
import math
import tqdm
import os

import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.tune import Trial

import numpy as np

from sbmfi.settings import BASE_DIR


def ray_train_MDVAE(config, cwd, show_progress=False):
    n_epoch = int(config.get('n_epoch', 3))
    n_hidden = float(config.get('n_hidden', 0.6))
    n_latent = float(config.get('n_latent', 0.3))
    n_hidden_layers = int(config.get('n_hidden_layers', 2))
    batch_size = int(config.get('batch_size', 64))
    learning_rate = float(config.get('learning_rate', 1e-3))
    weight_decay = float(config.get('weight_decay', 1e-4))
    LR_gamma = float(config.get('LR_gamma', 1.0))
    beta = float(config.get('beta', 1.0))
    bias = bool(config.get('bias', True))

    train_ds = torch.load(os.path.join(cwd, 'train_ds.pt'))
    val_ds = torch.load(os.path.join(cwd, 'val_ds.pt'))
    x_val, y_val = val_ds[:]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    mdvae = MDVAE(
        n_data=x_val.shape[-1],
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_hidden_layers=n_hidden_layers,
        bias=bias,
    )

    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(mdvae.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if LR_gamma < 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_gamma, last_epoch=-1)

    get_val = lambda x: x.to('cpu').data.numpy().round(4)
    if show_progress:
        pbar = tqdm.tqdm(total=n_epoch * len(train_loader), ncols=100, position=0)

    losses = []
    try:
        for epoch in range(n_epoch):
            for i, (x, y) in enumerate(train_loader):
                x_hat, mean, log_var = mdvae.forward(x)
                reconstruct = loss_f(x_hat, y)
                KL_div = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = reconstruct + beta * KL_div

                optimizer.zero_grad()
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                pbar.update()
                losses.append((0, get_val(loss), get_val(KL_div), get_val(reconstruct)))

                if (i % 50 == 0) and show_progress:
                    pbar.set_postfix(loss=losses[-1][1].round(4), KL_div=losses[-1][2].round(4), mse=losses[-1][3].round(4))
            if LR_gamma < 1.0:
                scheduler.step()

            with torch.no_grad():
                x_val_hat, mean, log_var = mdvae.forward(x_val)
                reconstruct = loss_f(x_val_hat, y_val)
                KL_div = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = reconstruct + beta * KL_div
                losses.append((1, get_val(loss), get_val(KL_div), get_val(reconstruct)))
    # try:
    #     pass
    except KeyboardInterrupt:
        pass
    finally:
        losses = pd.DataFrame(losses, columns=['train0_val1', 'loss', 'KL_div', 'mse'])
        if show_progress:
            pbar.close()
        return mdvae, losses


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

        if standardize:
            ding = data if mu is None else mu

            self.mean = ding.mean(-1, keepdims=True)
            self.std = ding.std(-1, keepdims=True)
            data = (data - self.mean) / self.std
            if mu is not None:
                mu = (mu - self.mean) / self.std

        self.data = data
        self.mu = mu

    def standardize(self, data, reverse=False):
        mu = None
        if isinstance(data, MDVAE_Dataset):
            data, mu = data[:]

        if reverse:
            data = (data * self.std) + self.mean
            if (mu is not None) and (self.mu is not None):
                mu = (mu * self.std) + self.mean
        else:
            data = (data - self.mean) / self.std
            if (mu is not None) and (self.mu is not None):
                mu = (mu - self.mean) / self.std
        if mu is None:
            return data
        if self.mu is None:
            return MDVAE_Dataset(data)
        return MDVAE_Dataset(data, mu)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.mu is None:
            return self.data[idx], self.data[idx]
        return self.data[idx], self.mu[idx]


_prr = lambda x: x.to('cpu').data.numpy()
def short_dirname(trial: Trial):
    return "trial_" + str(trial.trial_id)


def ray_train_MDVAE(config, cwd, ray=False, show_progress=False):
    n_epoch = float(config.get('n_epoch', 3))
    n_hidden = float(config.get('n_hidden', 0.6))
    n_latent = int(config.get('n_latent', 0.3))
    n_hidden_layers = int(config.get('n_hidden_layers', 2))
    batch_size = int(config.get('batch_size', 64))
    learning_rate = float(config.get('learning_rate', 1e-3))
    weight_decay = float(config.get('weight_decay', 1e-4))
    LR_gamma = float(config.get('LR_gamma', 1.0))
    beta = float(config.get('beta', 1.0))
    bias = bool(config.get('bias', True))

    train_ds = torch.load(os.path.join(cwd, 'train_ds.pt'))
    val_ds = torch.load(os.path.join(cwd, 'val_ds.pt'))
    x_val, y_val = val_ds[:]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    mdvae = MDVAE(
        n_data=x_val.shape[-1],
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_hidden_layers=n_hidden_layers,
        bias=bias,
    )
    if ray:
        raise NotImplemented

    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(mdvae.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if LR_gamma < 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_gamma, last_epoch=-1)

    if show_progress:
        pbar = tqdm.tqdm(total=n_epoch * len(train_loader), ncols=100)
        get_val = lambda x: x.to('cpu').data.numpy().round(4)

    losses = []
    try:
        for epoch in range(n_epoch):
            for i, (x, y) in enumerate(train_loader):
                x_hat, mean, log_var = mdvae.forward(x)
                reconstruct = loss_f(x_hat, y)
                KL_div = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = reconstruct + beta * KL_div

                optimizer.zero_grad()
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()

                losses.append((0, get_val(loss), get_val(KL_div), get_val(reconstruct)))

                if (i % 50 == 0) and show_progress:
                    pbar.set_postfix(loss=losses[-1][0].round(4), KL_div=losses[-1][1].round(4), mse=losses[-1][2].round(4))
            if LR_gamma < 1.0:
                scheduler.step()

            with torch.no_grad():
                x_val_hat, mean, log_var = mdvae.forward(x_val)
                reconstruct_val = loss_f(x_val_hat, y_val)
                KL_div_val = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                val_loss = reconstruct_val + beta * KL_div_val
                losses.append((1, get_val(val_loss), get_val(KL_div_val), get_val(reconstruct_val)))

    except KeyboardInterrupt:
        pass
    finally:
        losses = pd.DataFrame(losses, index=['train0_val1', 'loss', 'KL_div', 'mse']).T
        if show_progress:
            pbar.close()
        return mdvae, losses


def ray_main(
        hdf,
        simulator: _BaseSimulator,
        dataset_id: str,
        denoising: bool=False,
        val_frac: float=0.1,
        max_n_hidden=0.8,
):
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
    tuner = tune.Tuner(
        ray_train,
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




if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    model, kwargs = spiro(
        batch_size=50, include_bom=True,
        backend='torch', v2_reversible=True, build_simulator=True, which_measurements='lcms',
        which_labellings=list('AB')
    )
    sim = kwargs['basebayes']
    hdf = r"C:\python_projects\sbmfi\spiro_mdvae_test.h5"
    did = 'test1'

    results = ray_main(hdf, sim, did)

