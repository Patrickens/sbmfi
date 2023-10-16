import normflows
import torch
from normflows.flows.neural_spline.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from normflows.utils.splines import rational_quadratic_spline
# from sbi.neural_nets.flow import build_maf, build_nsf
from functools import partial
from pyknos.nflows.nn import nets
from pyknos.nflows import flows, transforms
from pyknos.nflows.transforms import Transform
from torch import Tensor, nn, relu, tanh, tensor, uint8
from typing import Optional
from sbi.utils.torchutils import create_alternating_binary_mask
from normflows.distributions.base import BaseDistribution, Uniform, UniformGaussian
from normflows.flows import CircularAutoregressiveRationalQuadraticSpline
from sbmfi.core.polytopia import FluxCoordinateMapper
from sbmfi.core.model import LabellingModel
from sbmfi.core.simulator import _BaseSimulator
from sbmfi.inference.priors import _BasePrior
import math
import numpy as np
import tqdm


class MFA_Flow(Transform):
    def __init__(
            self,
            fcm: FluxCoordinateMapper,
    ):
        if fcm._sampler.basis_coordinates not in ['spherical', 'semi_spherical']:
            raise ValueError

        self._cylinder = CylinderFlow(
            num_input_channels=len(fcm.theta_id) - 1,
        )
        self._fcm = fcm

    def forward(self, inputs, context=None):
        if self._fcm._nx > 0:
            pass

    def inverse(self, inputs, context=None):
        pass


def build_ball_flows(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
):
    from sbi.neural_nets.flow import build_nsf
    conditioner = partial(
        nets.ResidualNet,
        hidden_features=hidden_features,
        context_features=y_numel,
        num_blocks=num_blocks,
        activation=relu,
        dropout_probability=dropout_probability,
        use_batch_norm=use_batch_norm,
    )
    def mask_in_layer(i):
        return create_alternating_binary_mask(features=x_numel, even=(i % 2 == 0))
    transform_list = []
    for i in range(num_transforms):
        block = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(i) if x_numel > 1 else tensor([1], dtype=uint8),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]
        # Add LU transform only for high D x. Permutation makes sense only for more than
        # one feature.
        if x_numel > 1:
            block.append(
                transforms.LULinear(x_numel, identity_init=True),
            )
        transform_list += block

class MFA_Uniform(BaseDistribution):
    """
    Multivariate uniform distribution
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          low: Lower bound of uniform distribution
          high: Upper bound of uniform distribution
        """
        super().__init__()
        self.low = torch.as_tensor(low)
        self.high = torch.as_tensor(high)
        self.log_prob_val = - torch.sum(torch.log(self.high - self.low))

    def forward(self, num_samples=1, context=None):
        eps = torch.rand(
            (num_samples,) + self.low.shape, dtype=self.low.dtype, device=self.low.device
        )
        z = self.low + (self.high - self.low) * eps
        log_p = self.log_prob_val * torch.ones(num_samples, device=self.low.device)
        return z, log_p

    def log_prob(self, z, context=None):
        log_p = self.log_prob_val * torch.ones(z.shape[0], device=z.device)
        out_range = torch.logical_or(z < self.low, z > self.high)
        ind_inf = torch.any(torch.reshape(out_range, (z.shape[0], -1)), dim=-1)
        log_p[ind_inf] = -np.inf
        return log_p


def fuckin_aboot(
        simulator: _BaseSimulator = None,
        fcm: FluxCoordinateMapper = None,
        # prior_flow just makes a normalizing flow that matches samples from a prior
        #   thus not needing to fuck around with context = conditioning on data
        prior_flow=True,
        autoregressive=True,
        num_bins=10,
        dropout_probability=0.0,
        num_transforms = 3,
):
    if not prior_flow and (simulator is None):
        raise ValueError('need model to determine context features brøh')

    if simulator is not None:
        fcm = simulator._model.flux_coordinate_mapper

    if not ((fcm._sampler.basis_coordinates == 'cylinder') and ((fcm._bound is None) or (fcm.logit_xch_fluxes))):
        raise ValueError('needs to have cylinder base_coordinates and a tail_bound or logit ya schmuckington')

    n_theta = len(fcm.theta_id)

    base_bounds = np.ones((n_theta, 2), dtype=np.double)
    base_bounds[0, :] = math.pi
    base_bounds[:, 0] *= -1
    if fcm._nx > 0:
        base_bounds[-fcm._nx:, :] = torch.as_tensor(fcm._rho_bounds)

    base = MFA_Uniform(low=base_bounds[:, 0], high=base_bounds[:, 1])
    ncc = None if prior_flow else len(simulator.data_id)

    transforms = []
    for i in range(num_transforms):
        transforms.extend([
            CircularAutoregressiveRationalQuadraticSpline(
                num_input_channels=n_theta,
                num_blocks=4,
                num_hidden_channels=20,
                ind_circ=[0],
                num_context_channels=ncc,
                num_bins=num_bins,
                tail_bound=fcm._bound if fcm._bound else 1.0,
                activation=nn.ReLU,
                dropout_probability=dropout_probability,
                permute_mask=True,
                init_identity=True,
            ),
            normflows.flows.LULinearPermute(n_theta),
        ])
    flow = normflows.NormalizingFlow(q0=base, flows=transforms)
    return flow

def train_prior_flow(
        prior: _BasePrior,
        max_iter = 20,
        train_batch = 256,
):
    pbar = tqdm.tqdm(total=max_iter, ncols=200, desc='loss')
    prior_flow = fuckin_aboot(fcm=prior._fcm)
    optimizer = torch.optim.Adam(prior_flow.parameters(), lr=1e-3, weight_decay=1e-5)
    print(abs(prior_flow.sample(20)[0][:, 0].detach()) > 1.0)
    # for i in range(max_iter):
    #     pbar.update(i)
    #     x = prior.sample((train_batch, ))
    #     loss = prior_flow.forward_kld(x)
    #     print(loss)
    #     if ~(torch.isnan(loss) | torch.isinf(loss)):
    #         loss.backward()
    #         optimizer.step()
    #
    #     loss_np = loss.to('cpu').data.numpy()
    #     pbar.set_postfix(loss=loss_np)
    pass


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    from sbmfi.core.polytopia import sample_polytope
    from sbmfi.inference.priors import UniformNetPrior

    model, kwargs = spiro(
        backend='torch', v2_reversible=True, v5_reversible=False, build_simulator=False, which_measurements=None
    )
    fcm = FluxCoordinateMapper(
        model,
        basis_coordinates='cylinder'

    )


    up = UniformNetPrior(fcm, cache_size=1000)
    # draws = up.sample((30,))
    # flow = fuckin_aboot(fcm=fcm)
    train_prior_flow(up)







