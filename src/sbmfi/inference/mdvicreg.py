import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, random_split
from sbmfi.core.simulator import _BaseSimulator
from sbmfi.core.util import hdf_opener_and_closer
import math
import tqdm
import torch.distributed as dist
import torch.nn.functional as F

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


   # parser.add_argument("--arch", type=str, default="resnet50",
   #                      help='Architecture of the backbone encoder network')
   #  parser.add_argument("--mlp", default="8192-8192-8192",
   #                      help='Size and number of layers of the MLP expander head')
class VICReg(nn.Module):
    def __init__(
            self,
            n_data: int,
            n_latent=0.3,
            n_hidden=0.6,
            n_hidden_layers=1,
            bias=True,
            activation=nn.LeakyReLU(0.01),

            batch_size = 64,

            sim_coeff = 25.0,
            std_coeff = 25.0,
            cov_coeff = 1.0,
    ):
        super().__init__()
        if isinstance(n_latent, float):
            n_latent = math.ceil(n_latent * n_data)
        if isinstance(n_hidden, float):
            n_hidden = math.ceil(n_hidden * n_data)

        if n_latent > n_hidden:
            raise ValueError

        self.batch_size = batch_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

        #  ResNet(Bottleneck, [3, 4, 6, 3], **kwargs), 2048
        self.embedding = 2048

        endocer_layers = [nn.Linear(n_data, n_hidden, bias=bias), activation]
        for i in range(n_hidden_layers):
            endocer_layers.extend([nn.Linear(n_hidden, n_hidden, bias=bias), activation])

            # add batchnorm?
        endocer_layers.append(nn.Linear(n_hidden, n_latent))
        self.encoder = nn.Sequential(*endocer_layers)


    def forward(self, x, y):
        x = self.encoder(x)
        y = self.encoder(y)

        repr_loss = F.mse_loss(x, y)

        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)

        n_f = x.shape[-1]
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(n_f) + off_diagonal(cov_y).pow_(2).sum().div(n_f)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


from sbmfi.inference.mdvae import MDVAE_Dataset
def train_vicreg(
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
    dataset = MDVAE_Dataset(data, mu)

    n_validate = math.ceil(0.10 * len(dataset))

    train_ds, val_ds = random_split(
        dataset,
        lengths=(len(dataset) - n_validate, n_validate),
        generator=simulator._la._BACKEND._rng
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if 'n_latent' not in kwargs:
        kwargs['n_latent'] = len(simulator.theta_id)

    vicreg = VICReg(
        n_data=data.shape[-1],
        batch_size=batch_size,
        **kwargs,
    )
    if standardize:
        raise ValueError
        # nn.Sequential # standardize data and make a sequential (subtract mean and divide by std)

    optimizer = torch.optim.Adam(vicreg.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm.tqdm(total=n_epochs * len(train_loader), ncols=100)
    prr = lambda x: x.to('cpu').data.numpy().round(4)
    losses = []
    try:
        for epoch in range(n_epochs):
            for i, (x, y) in enumerate(train_loader):
                loss = vicreg.forward(x, y)
                losses.append(loss.detach())
                optimizer.zero_grad()
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                pbar.update()
                if i % 50 == 0:
                    pbar.set_postfix(loss=prr(loss))
                # with torch.no_grad():
                #     x_val, y_val = val_ds[:]
    except KeyboardInterrupt:
        pass
    finally:
        pbar.close()
        return vicreg, losses


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

    mdvae = train_vicreg(hdf, sim, 'mdvae')