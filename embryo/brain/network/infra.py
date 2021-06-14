'''Network infrastructure

Created by Minhui Li on April 19, 2021
'''


from typing import Callable, Optional, Sequence

import torch
from torch import nn


class MLP(nn.Module):
    '''Multi-Layer Perceptron
    '''

    def __init__(
        self,
        input_units: int,
        layer_units: Sequence[int],
        activations: Optional[Sequence[Callable]] = None,
        normalizations: Optional[Sequence[Callable]] = None,
    ):
        '''Initialization method

        Args:
            input_units: Number of input neurons
            layer_units: Number of neurons in each layer
            activations: Activation of each layer
            normalizations: Normalization of each layer
        '''

        super().__init__()
        self.num_layers = len(layer_units)

        layers = [nn.Linear(input_units, layer_units[0])]
        for i in range(1, self.num_layers):
            layers.append(nn.Linear(layer_units[i - 1], layer_units[i]))
        self.layers = nn.ModuleList(layers)

        if activations:
            if isinstance(activations, (list, tuple)):
                self.activations = activations
            else:
                self.activations = [activations] * self.num_layers
        else:
            self.activations = [None] * self.num_layers

        if normalizations:
            if callable(normalizations):
                normalizations = [normalizations] * self.num_layers
            self.normalizations = nn.ModuleList(normalizations)
        else:
            self.normalizations = [None] * self.num_layers

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if self.normalizations[i]:
                x = self.normalizations[i](x)
            if self.activations[i]:
                x = self.activations[i](x)
        return x