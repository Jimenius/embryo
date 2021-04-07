import torch
from torch import nn
import torch.nn.functional as F

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

class ProcessNet(nn.Module):

    def __init__(self, i):
        super().__init__()
        self.D1 = nn.Linear(i, 128)
        self.D2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.D1(x))
        x = F.relu(self.D2(x))
        return x

class Actor(nn.Module):
    
    def __init__(self, pre):
        super().__init__()
        self.pre = pre
        self.mu = nn.Linear(128, 1)
        # self.sigma = nn.Linear(128, 1)
        self.sigma = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        x = self.pre(x)
        mu = self.mu(x)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = torch.exp(self.sigma.view(shape) + torch.zeros_like(mu))
        # sigma = torch.exp(self.sigma)
        return mu, sigma

class Critic(nn.Module):
    
    def __init__(self, pre):
        super().__init__()
        self.pre = pre
        self.Q = nn.Linear(128, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=-1)
        x = self.pre(x)
        x = self.Q(x)
        return x