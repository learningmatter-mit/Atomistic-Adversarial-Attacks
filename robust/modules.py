import numpy as np
from functools import partial

import torch as ch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.init import xavier_uniform_, constant_


zeros_initializer = partial(constant_, val=0.0)


class Dense(nn.Linear):
    """Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        dropout_rate=0.0,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init

        super().__init__(in_features, out_features, bias)

        self.activation = activation

        if dropout_rate > 0 and dropout_rate < 1:
            self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:


        Returns:
            ch.Tensor: Output of the dense layer.
        """
        y = super().forward(inputs)

        if hasattr(self, "dropout"):
            y = self.dropout(y)

        if self.activation:
            y = self.activation(y)

        return y


class Softplus(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inp):
        return F.softplus(inp) - np.log(2.0)
