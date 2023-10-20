import types

import torch
from torch import nn
import numpy as np

from normflows.flows.base import Flow
from normflows.core import ConditionalNormalizingFlow
from normflows.flows.neural_spline.coupling import PiecewiseRationalQuadraticCoupling
from normflows.flows.neural_spline.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from normflows.nets.resnet import ResidualNet
from normflows.utils.masks import create_alternating_binary_mask
from normflows.utils.nn import PeriodicFeaturesElementwise
from normflows.utils.splines import DEFAULT_MIN_DERIVATIVE
import inspect


class CircularCoupledRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer with circular coordinates
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        ind_circ,
        num_context_channels=None,
        num_bins=8,
        tail_bound=3.0,
        activation=nn.ReLU,
        dropout_probability=0.0,
        use_batch_norm=False,
        reverse_mask=False,
        mask=None,
        init_identity=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          ind_circ (Iterable): Indices of the circular coordinates
          num_bins (int): Number of bins
          tail_bound (float or Iterable): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          reverse_mask (bool): Flag whether the reverse mask should be used
          mask (torch tensor): Mask to be used, alternating masked generated is None
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        if mask is None:
            mask = create_alternating_binary_mask(num_input_channels, even=reverse_mask)
        features_vector = torch.arange(num_input_channels)
        identity_features = features_vector.masked_select(mask <= 0)
        ind_circ = torch.tensor(ind_circ)
        ind_circ_id = []
        for i, id in enumerate(identity_features):
            if id in ind_circ:
                ind_circ_id += [i]

        if torch.is_tensor(tail_bound):
            scale_pf = np.pi / tail_bound[ind_circ_id]
        else:
            scale_pf = np.pi / tail_bound

        def transform_net_create_fn(in_features, out_features):
            if len(ind_circ_id) > 0:
                pf = PeriodicFeaturesElementwise(in_features, ind_circ_id, scale_pf)
            else:
                pf = None
            net = ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=num_context_channels,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
                preprocessing=pf,
            )
            if init_identity:
                torch.nn.init.constant_(net.final_layer.weight, 0.0)
                torch.nn.init.constant_(
                    net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
                )
            return net

        tails = [
            "circular" if i in ind_circ else "linear" for i in range(num_input_channels)
        ]

        self.prqct = PiecewiseRationalQuadraticCoupling(
            mask=mask,
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            apply_unconditional_transform=True,
        )

    def forward(self, z, context=None):
        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)


class CircularAutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [sources](https://github.com/bayesiains/nsf)
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        ind_circ,
        num_context_channels=None,
        num_bins=8,
        tail_bound=3,
        activation=nn.ReLU,
        dropout_probability=0.0,
        use_batch_norm=True,
        permute_mask=True,
        init_identity=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          ind_circ (Iterable): Indices of the circular coordinates
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tail_bound (int): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        tails = [
            "circular" if i in ind_circ else "linear" for i in range(num_input_channels)
        ]

        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=num_context_channels,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            init_identity=init_identity,
        )

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context=context)
        return z, log_det.view(-1)


def _embedinator(fn):
    def inner(*args, **kwargs):
        self = args[0]
        context = kwargs.get('context', None)
        if context is None and len(args) > 2:
            context = args[-1]
            args = args[:-1]
        if (context is not None) and (self._embnet is not None):
            context = self._embnet(context)
            kwargs['context'] = context
        return fn(*args, **kwargs)
    return inner


def _class_embedinator(decorator):
    def decorate(cls):
        for attr in dir(cls): # there's propably a better way to do this
            cls_attr = getattr(cls, attr)
            if isinstance(cls_attr, types.FunctionType):
                if 'context' in inspect.getfullargspec(cls_attr).args:
                    setattr(cls, attr, decorator(cls_attr))
        return cls
    return decorate


@_class_embedinator(_embedinator)
class EmbeddingConditionalNormalizingFlow(ConditionalNormalizingFlow):
    """
    Conditional normalizing flow model, providing condition,
    which is also called context, to both the base distribution
    and the flow layers
    """
    def __init__(self, q0, flows, p=None, embedding_net=None):
        super().__init__(q0, flows, p)
        self._embnet = embedding_net
        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )


if __name__ == "__main__":
    from normflows.distributions.base import BaseDistribution, Uniform, UniformGaussian
    from normflows.flows import Permute, LULinearPermute

    transform = LULinearPermute(num_channels=5)
    base = Uniform(shape=5, low=-1.0, high=1.0)
    aa = EmbeddingConditionalNormalizingFlow(base, [transform])
    aa.sample(20)
