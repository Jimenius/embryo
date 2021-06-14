'''Application networks

Created by Minhui Li on April 19, 2021
'''


import torch
from torch import nn
import torch.nn.functional as F

from embryo.brain.network import NETWORK_REGISTRY
from embryo.brain.network.infra import MLP


__all__ = [
    'QNet',
    'Actor',
    'Critic',
]


@NETWORK_REGISTRY.register()
class QNet(nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        # self.D1 = nn.Linear(4, 64)
        # self.D2 = nn.Linear(64, 64)
        # self.Q = nn.Linear(64, 2)
        self.D1 = nn.Linear(6, 128)
        self.Q = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.D1(x))
        # x = F.relu(self.D2(x))
        x = self.Q(x)
        return x


@NETWORK_REGISTRY.register()
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.mu = MLP(
            input_units=obs_dim[0],
            layer_units=(128, 128, action_dim[0]),
            activations=[F.relu] * 2 + [None],
        )
        self.sigma = nn.Parameter(torch.zeros(action_dim[0], 1))

    def forward(self, x):
        mu = self.mu(x)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = torch.exp(self.sigma.view(shape) + torch.zeros_like(mu))
        return mu, sigma


@NETWORK_REGISTRY.register()
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.Q = MLP(
            input_units=obs_dim[0] + action_dim[0],
            layer_units=(128, 128, 1),
            activations=[F.relu] * 2 + [None],
        )

    def forward(self, s, a):
        x = torch.cat((s, a), dim=-1)
        x = self.Q(x)
        return x