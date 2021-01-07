'''Deep Q Network

Created by Minhui Li on September 30, 2019
'''


import os
from typing import Optional, Union

import numpy as np
import torch

from embryo.brain.central.base import Central
from embryo.brain.memory import SequentialMemory
from embryo.brain.central.models import QNet
from embryo.ion import Ion
from embryo.utils.explorations import Epsilon_Greedy
from embryo.utils.torch_utils import to_tensor
from . import CENTRAL_REGISTRY


@CENTRAL_REGISTRY.register()
class DQNCentral(Central):
    '''Deep Q-Learning agent

    Reference:
        Mnih et al, Playing Atari with Deep Reinforcement Learning
        Mnih et al, Human-level Control through Deep Reinforcement Learning
        Hasselt et al, Deep Reinforcement Learning with Double Q-Learning
        Wang et al, Dueling Network Architectures for Deep Reinforcement Learning
    '''

    def __init__(
        self,
        double: bool = False,
        device: Union[torch.device, str] = 'cpu',
        network = Optional[torch.nn.Module],
        target_update_frequency: Union[int, float] = 1,
    ) -> None:

        # Initialize parameters
        super().__init__(**kwargs)
        self.device = device
        self.double = double
        if 0 < update < 1:
            self.target_update_frequency = target_update_frequency # Soft update
        elif update >= 1:
            self.target_update = int(target_update_frequency) # Hard update
        else:
            raise ValueError(
                'Target update should be greater than 0. ',
                '(0, 1) for soft update, [1, inf) for hard update.'
            )

        self.gradient_step = 0 # Gradient step counter
        self.QNet = QNet()
        self.QTargetNet = QNet()
        self.QTargetNet.eval()
        self._hard_update_target()
        self.optimizer = torch.optim.Adam(self.QNet.parameters())

    def _hard_update_target(self):
        self.QTargetNet.load_state_dict(self.QNet.state_dict())

    def _soft_update_target(self):
        for main, target in zip(self.QNet.parameters(), self.QTargetNet.parameters()):
            target.data.copy_(
                target.data * (1. - self.target_update_frequency) + \
                    main.data * self.target_update_frequency
            )

    def set_explore_rate(
        self,
        explore_rate: float = 1.
    ) -> None:
        '''Set exploration rate

        In DQN, it is Epsilon.

        Args:
            explore_rate: Epsilon in Epsilon-Greedy policy.
        '''

        self.explore_rate = explore_rate

    def train(
        self,
    ) -> None:
        self.QNet.train()

    def eval(
        self,
    ) -> None:
        self.QNet.eval()

    def learn(
        self,
        batch: Ion,
    ) -> Dict[str, float]:
        '''
        '''

        batch.to(ctype='TORCH', device=device)
        q = self.control(batch).logits
        batch_size = q.size(0)
        q = q[np.arange(batch_size), batch.action]
        loss = torch.nn.MSELoss()(q, batch.returns)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.QNet.parameters(), max_norm=1.)
        self.optim.step()
        self.gradient_step += 1

        # Update target network
        if self.target_update_frequency < 1:
            self._soft_update_target()
        elif self.gradient_step % self.target_update_frequency:
            self._hard_update_target()

        return {"loss": loss.item()}

    def load(self, timestamp):
        pass

    def save(self, timestamp):
        pass

    def control(
        self,
        batch: Ion,
    ) -> Ion:
        '''Control method
        '''

        batch = batch.to(ctype='TORCH', device=self.device)
        observation: Union[np.ndarray, torch.Tensor] = batch.observation
        Q: torch.Tensor = self.QNet(observation)
        action, _ = Q.max(dim=-1)
        return Ion(logits=Q, action=action)