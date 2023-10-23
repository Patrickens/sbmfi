import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, random_split
from sbmfi.core.simulator import _BaseSimulator
from sbmfi.core.util import hdf_opener_and_closer
import math
import tqdm

class MDVAE(nn.Module):
    # denoising auto-encoder; pass in noisy features, decode to noise-free features; use latent variables for inference
    def __init__(
            self,
            n_data: int,
            n_latent = 0.3,
            n_hidden=0.6,
            n_hidden_layers = 1,
            bias = True,
            activation=nn.LeakyReLU(0.01),
    ):
        super().__init__()
        if isinstance(n_latent, float):
            n_latent = math.ceil(n_latent * n_data)
        if isinstance(n_hidden, float):
            n_hidden = math.ceil(n_hidden * n_data)

        if n_latent > n_hidden:
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

    def forward(self, x):
        hidden = self.encoder(x)
        mean = self.mean_lay(hidden)
        log_var = self.log_var_lay(hidden)

        # reparametrization trick
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return self.decoder(z), mean, log_var


class _MDV_Dataset(Dataset):
    def __init__(self, data: torch.Tensor, mu: torch.Tensor):
        if (len(data.shape) != 3) or (len(mu.shape) != 3):
            raise ValueError
        mu = torch.tile(mu, (1, data.shape[1], 1)).view((math.prod(data.shape[:-1]), data.shape[-1]))
        data = data.view(mu.shape)
        self.data = data
        self.mu = mu

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.mu[idx]

class Affine(nn.Module):
    def __init__(self, mu, std, inverse=False):
        self.mu = mu
        self.std = torch.clip(std, min=1e-5)
        self.inverse = inverse

    def forward(self):
        pass


@hdf_opener_and_closer()
def train_mdv_encoder(
        hdf,
        simulator: _BaseSimulator,
        dataset_id,
        standardize=False,
        n_epochs = 3,
        lr = 1e-4,
        weight_decay=1e-5,
        batch_size=32,
        **kwargs,
):
    if not simulator._la.backend == 'torch':
        raise ValueError
    mdvs = simulator.read_hdf(hdf=hdf, dataset_id=dataset_id, what='mdv')
    data = simulator.read_hdf(hdf=hdf, dataset_id=dataset_id, what='data')
    theta = simulator.read_hdf(hdf=hdf, dataset_id=dataset_id, what='theta')
    mu = simulator.simulate(theta=theta, mdvs=mdvs, n_obs=0)
    dataset = _MDV_Dataset(data, mu)

    n_validate = math.ceil(0.10 * len(dataset))

    train_ds, val_ds = random_split(
        dataset,
        lengths=(len(dataset) - n_validate, n_validate),
        generator=simulator._la._BACKEND._rng
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if 'n_latent' not in kwargs:
        kwargs['n_latent'] = len(simulator.theta_id)

    mdvae = MDVAE(
        n_data=data.shape[-1],
        **kwargs,
    )
    if standardize:
        nn.Sequential # standardize data and make a sequential (subtract mean and divide by std)

    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(mdvae.parameters(), lr=lr, weight_decay=weight_decay)

    print(f'MSE between all mu and data: {loss_f(*dataset[:]).numpy().round(4)}')

    pbar = tqdm.tqdm(total=n_epochs * len(train_loader), ncols=100)
    prr = lambda x: x.to('cpu').data.numpy().round(4)
    try:
        for epoch in range(n_epochs):
            for i, (x, y) in enumerate(train_loader):
                x_hat, mean, log_var = mdvae.forward(x)
                reconstruct = loss_f(x_hat, y)
                KL_div = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = reconstruct + KL_div
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                pbar.update()
                if i % 50 == 0:
                    pbar.set_postfix(loss=prr(loss), KL_div=prr(KL_div),  mse=prr(reconstruct))
            with torch.no_grad():
                x_val, y_val = val_ds[:]
                x_val_hat, mean, log_var = mdvae.forward(x_val)
                reconstruct_val = loss_f(x_val_hat, y_val)
                KL_div_val = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss_val = reconstruct_val + KL_div_val
                print(f'loss: {prr(loss_val)}, KL_div: {prr(KL_div_val)}, MSE: {prr(reconstruct_val)}', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        pbar.close()
        return mdvae


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
    hdf = r"C:\python_projects\sbmfi\spiro_AB2.h5"
    did = 'mdvae'

    mdvae = train_mdv_encoder(hdf, sim, 'mdvae')

