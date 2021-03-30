'''Deep Q Network

Created by Minhui Li on September 30, 2019
'''


import os
from typing import Dict, Optional, Union

import numpy as np
import torch

from embryo.brain.central import CENTRAL_REGISTRY
from embryo.brain.central.base import Central
from embryo.brain.central.models import QNet
from embryo.ion import Ion
from embryo.utils.explorations import Epsilon_Greedy


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
        network = Optional[torch.nn.Module],
        target_update_frequency: Union[int, float] = 1,
        **kwargs
    ) -> None:
        '''Initialization method

        Args:
            network:
            target_update_frequency:
        '''

        # Initialize parameters
        super().__init__(**kwargs)
        if 0 < target_update_frequency < 1:
            self.target_update_frequency = target_update_frequency # Soft update
        elif target_update_frequency >= 1:
            self.target_update_frequency = int(target_update_frequency) # Hard update
        else:
            raise ValueError(
                'Target update should be greater than 0. ',
                '(0, 1) for soft update, [1, inf) for hard update.'
            )

        self.explore_rate = 0.
        self.gradient_step = 0 # Gradient step counter
        self.QNet = QNet().to(self.device)
        self.QTargetNet = QNet().to(self.device)
        self.QTargetNet.eval()
        self._hard_update_target()
        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=1e-3)

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

    def get_target_value(
        self,
        batch: Ion,
    ) -> Ion:
        '''
        '''

        batch.to(
            ctype='torch',
            device=self.device,
        )
        observation: torch.Tensor = batch.observation.to(torch.float)
        with torch.no_grad():
            q_target = self.QTargetNet(observation)
        return Ion(value=q_target)

    def update(
        self,
        batch: Ion,
    ) -> Ion:
        '''Perform a backward gradient step.

        Args:
            batch: Data, should contain keys 'observation' and 'returns'.

        Returns:
            losses
        '''

        q = self.control(batch).value
        batch_size = q.size(0)
        q = q[torch.arange(batch_size), batch.action]
        loss = torch.nn.MSELoss()(q, batch.returns)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.QNet.parameters(), max_norm=1.)
        self.optimizer.step()
        self.gradient_step += 1

        # Update target network
        if self.target_update_frequency < 1:
            self._soft_update_target()
        elif self.gradient_step % self.target_update_frequency:
            self._hard_update_target()

        return Ion(loss=loss.item())

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

        batch.to(
            ctype='torch',
            device=self.device,
        )
        observation: torch.Tensor = batch.observation.to(dtype=torch.float)
        Q: torch.Tensor = self.QNet(observation)
        action = Epsilon_Greedy(value=Q, epsilon=self.explore_rate)
        return Ion(value=Q, action=action)