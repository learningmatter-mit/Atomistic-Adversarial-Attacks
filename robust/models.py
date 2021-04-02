from typing import List

import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Dense, Softplus
from nff.utils.scatter import compute_grad


class NnRegressor(nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        num_layers=4,
        layer_dim=1024,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Dense(
                    input_dim,
                    layer_dim,
                    activation=Softplus(),
                    dropout_rate=dropout_rate,
                ),
                *[
                    Dense(
                        layer_dim,
                        layer_dim,
                        activation=Softplus(),
                        dropout_rate=dropout_rate,
                    )
                ]
                * (num_layers - 2),
                Dense(layer_dim, output_dim),
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        forces = -compute_grad(x, out)
        return x, out, forces


class NnEnsemble(nn.Module):
    def __init__(self, networks: List[nn.Module]):
        super().__init__()
        self.networks = nn.ModuleList(networks)

    def forward(self, x):
        results = [net.forward(x) for net in self.networks]

        energies = ch.stack([en for x_, en, f in results], dim=-1)
        forces = ch.stack([f for x_, en, f in results], dim=-1)

        return x, energies, forces

    def mean(self, x):
        return x.mean(dim=-1)

    def var(self, x):
        return x.var(dim=-1)
