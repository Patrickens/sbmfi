import torch
from torch import Tensor, nn, relu, tanh, tensor, uint8, functional
from sbmfi.core.simulator import _BaseSimulator
import math

class MDV_Encoder(nn.Module):
    # denoising auto-encoder; pass in noisy features, decode to noise-free features; use latent variables for inference
    def __init__(
            self,
            n_data: int,
            n_latent = 0.3,
            n_hidden=0.6,
            n_hidden_layers = 2,
            bias = True,
            tied = False,
    ):
        super().__init__()
        if isinstance(n_latent, float):
            n_latent = math.ceil(n_latent * n_data)
        if isinstance(n_hidden, float):
            n_hidden = math.ceil(n_hidden * n_data)

        self._bias = bias

        n_0 = n_latent if n_hidden_layers == 0 else n_hidden
        if not tied:
            encoder = [nn.Linear(n_data, n_0, bias=bias), nn.ReLU()]
            decoder = [nn.Linear(n_0, n_data, bias=bias), nn.ReLU()]
            for i in range(n_hidden_layers):
                n_i = n_latent if i == n_hidden_layers - 1 else n_hidden
                encoder.extend([nn.Linear(n_hidden, n_i, bias=bias), nn.ReLU()])
                decoder.extend([nn.Linear(n_i, n_hidden, bias=bias), nn.ReLU()])
            self.encoder = nn.Sequential(*encoder)
            self.decoder = nn.Sequential(*decoder[:-1][::-1])

        else:
            raise NotImplementedError

            self._weight_0 = nn.Parameter(torch.randn(n_0, n_data))
            self._weights = [self._weight_0]
            if bias:
                self._bias_0 = nn.Parameter(torch.randn(n_0))
                self._bias_n = nn.Parameter(torch.randn(n_data))
                self._encoder_biases = [self._bias_0]
                self._decoder_biases = [self._bias_0]

            if n_hidden_layers > 0:
                for i in range(1, n_hidden_layers):
                    n_i = n_hidden

                    weights_i = nn.Parameter(torch.randn(n_0, n_data))

                    # setattr(
                    #     self, f'_weights_{i}',
                    #     nn.Parameter(torch.randn(n_0, n_data))
                    # )
    def tied_encoder(self, input):
        temp = input
        for i, weight_i in enumerate(self._weights):
            bias = None
            if self._bias:
                bias = self._biases[i]
            temp = nn.functional.linear(temp, weight_i, bias)
        return temp

    def tied_decoder(self, input):
        temp = input
        for i, weight_i in enumerate(self._weights):
            bias = None
            if self._bias:
                bias = self._biases[i]
            temp = nn.functional.linear(temp, weight_i, bias)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_mdv_encoder(
        hdf,
        simulator,
        denoising=True,
):
    pass


if __name__ == "__main__":
    a = MDV_Encoder(50)
    print(a)