import math
import os
import numpy as np
from scipy.stats import random_correlation, loguniform

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn

import time

class MDVAE(nn.Module):
    # denoising auto-encoder; pass in noisy features, decode to noise-free features; use latent variables for inference
    def __init__(
            self,
            n_data: int,
            n_latent = 0.3,
            n_hidden = 0.6,
            n_hidden_layers = 1,
            bias = True,
            activation=nn.LeakyReLU(0.01),
    ):
        super().__init__()
        if isinstance(n_latent, float):
            n_latent = math.ceil(n_latent * n_data)
        if isinstance(n_hidden, float):
            n_hidden = math.ceil(n_hidden * n_data)

        if (n_latent > n_hidden) or (n_latent > n_data):
            raise ValueError

        endocer_layers = [nn.Linear(n_data, n_hidden, bias=bias), activation]
        for i in range(n_hidden_layers):
            endocer_layers.extend([nn.Linear(n_hidden, n_hidden, bias=bias), activation])

        self.encoder = nn.Sequential(*endocer_layers)
        self.mean_lay = nn.Linear(n_hidden, n_latent)
        self.log_var_lay  = nn.Linear(n_hidden, n_latent)

        decoder_layers = []
        for layer in endocer_layers:
            if layer == activation:
                continue
            decoder_layers.extend([nn.Linear(layer.out_features, layer.in_features), activation])
        decoder_layers.extend([nn.Linear(n_latent, n_hidden)])
        self.decoder = nn.Sequential(*decoder_layers[::-1])

    def encode_and_sample(self, x):
        hidden = self.encoder(x)
        mean = self.mean_lay(hidden)
        log_var = self.log_var_lay(hidden)

        # reparametrization trick
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z, mean, log_var

    def forward(self, x):
        z, mean, log_var = self.encode_and_sample(x)
        return self.decoder(z), mean, log_var

class VAE_Dataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]


_PRR = lambda x: float(x.to('cpu').data.numpy())
def ray_train(config, cwd, n_data, n_epoch=5, show_progress=False):
    n_hidden = float(config.get('n_hidden', 0.6))
    n_latent = int(config.get('n_latent', 0.3))
    n_hidden_layers = int(config.get('n_hidden_layers', 2))
    batch_size = int(config.get('batch_size', 64))
    learning_rate = float(config.get('learning_rate', 1e-3))
    weight_decay = float(config.get('weight_decay', 1e-4))
    beta = float(config.get('beta', 1.0))

    train_ds = torch.load(os.path.join(cwd, 'train_ds.pt'))
    val_ds = torch.load(os.path.join(cwd, 'val_ds.pt'))
    x_val, y_val = val_ds[:]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    mdvae = MDVAE(
        n_data=n_data,
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_hidden_layers=n_hidden_layers,
    )

    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(mdvae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if show_progress:
        raise NotImplementedError

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

        with torch.no_grad():
            x_val_hat, mean, log_var = mdvae.forward(x_val)
            reconstruct_val = loss_f(x_val_hat, y_val)
            KL_div_val = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            val_loss = reconstruct_val + beta * KL_div_val

            print(_PRR(val_loss))
    return mdvae


def prepare_datasets(
        cwd,
        n_data,
        n_samples=20000,
        val_frac=0.1
):
    # reproduciple random numbers
    rng = torch.Generator()
    rng.manual_seed(3)
    np_rng = np.random.RandomState(3)

    # create a data-set with correlated variables
    eigs = np_rng.rand(n_data)
    eigs = eigs * (n_data / sum(eigs))
    corr = torch.from_numpy(
        random_correlation.rvs(eigs, random_state=np_rng)
    )
    chol = torch.linalg.cholesky(corr)

    data = torch.randn((n_samples, n_data), generator=rng, dtype=torch.float64)
    data = data @ chol  # correlated data
    data = (data - torch.mean(data, 0)) / torch.std(data, 0)  # normalized data
    dataset = VAE_Dataset(data.float())

    n_validate = math.ceil(val_frac * n_samples)

    # prepare datasets
    train_ds, val_ds = random_split(
        dataset,
        lengths=(len(dataset) - n_validate, n_validate),
    )
    torch.save(train_ds, os.path.join(cwd, 'train_ds.pt'))
    torch.save(val_ds, os.path.join(cwd, 'val_ds.pt'))


if __name__ == "__main__":
    param_space = {
        'batch_size': 32,
        'n_hidden': .8,
        'n_latent': 10,
        'n_hidden_layers': 4,
        'learning_rate': 0.0001,
        'beta': 3,
    }

    n_samples = 20000
    n_data = 15
    n_latent = 5  # I make an assumption on the right latent space size, so this is kept fixed

    cwd = os.getcwd()
    if not (os.path.isfile(os.path.join(cwd, 'train_ds.pt')) and os.path.isfile(os.path.join(cwd, 'val_ds.pt'))):
        prepare_datasets(cwd, n_data, n_samples)

    mdvae = ray_train(param_space, n_epoch=15, cwd=os.getcwd(), n_data=15)

    cwd = os.getcwd()
    train_ds = torch.load(os.path.join(cwd, 'train_ds.pt'))
    val_ds = torch.load(os.path.join(cwd, 'val_ds.pt'))
    x_val, y_val = val_ds[:]
    with torch.no_grad():
        y_hat, mean, log_var = mdvae.forward(x_val)

    print(y_val[:3])
    print(y_hat[:3])
    print(torch.std(y_hat, 0))

