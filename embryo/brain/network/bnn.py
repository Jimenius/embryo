'''

Created by Minhui Li on April 12, 2021
'''


import torch
from torch import nn


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)


class Layer(nn.Module):
    '''An operation layer of the Ensemble MLP.
    '''

    def __init__(
        self,
        ensemble_size: int,
        in_feature: int,
        out_feature: int,
        bias: bool = False,
    ) -> None:
        '''Initialization method

        Args:
            ensemble_size: number of networks in the ensemble
            in_feature: number of input neurons
            out_feature: number of output neurons
            bias: whether to add a bias
        '''
        
        # To be consistent with PyTorch default initializer
        k = torch.sqrt(1. / in_feature)
        weight_data = torch.rand((ensemble_size, in_feature, out_feature)) \
            * 2 * k - k
        self.weight = nn.Parameter(
            weight_data,
            requires_grad=True,
        )

        if bias:
            bias_data = torch.rand((ensemble_size, out_feature)) \
                * 2 * k - k
            self.bias = nn.Parameter(
                bias_data,
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x


class EnsembleMLP(nn.Module):
    '''
    '''

    def __init__(self, ensemble_size):
        self.l1 = Layer(7, 1, 1)
        self.l2 = Layer(7, 1, 1)
        self.lout = Layer(7, 1, 1)

    def forward(self, x):
        x = swish(self.l1(x))
        x = swish(self.l2(x))
        x = self.lout(x)
        return x