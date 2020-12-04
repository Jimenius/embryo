from torch import nn
import torch.nn.functional as F

class QNet(nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        self.D1 = nn.Linear(4, 64)
        self.D2 = nn.Linear(64, 64)
        self.Q = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.D1(x))
        x = F.relu(self.D2(x))
        x = self.Q(x)
        return x