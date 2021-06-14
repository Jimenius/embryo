'''Deep Q Network

Created by Minhui Li on September 30, 2019
'''


from copy import deepcopy
from typing import Dict, Optional, Union

from gym.spaces import Space
import numpy as np
import torch
from yacs.config import CfgNode

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
        config: CfgNode,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        '''Initialization method

        Args:
            config: Configurations
            observation_space: Observation space
            action_space: Action space
        '''

        # Initialize parameters
        super().__init__(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        freq = self.config.TARGET_UPDATE_FREQUENCY
        if 0 < freq < 1:
            self.target_update_frequency = freq # Soft update
        elif freq >= 1:
            self.target_update_frequency = int(freq) # Hard update
        else:
            raise ValueError(
                'Target update should be greater than 0. ',
                '(0, 1) for soft update, [1, inf) for hard update.'
            )

        self.explore_rate = 0.

        net_cfg = self.config.NETWORK
        q = net_cfg.Q
        self.QNet = NETWORK_REGISTRY.get(q.NAME)(
            self.observation_dim,
            self.action_dim,
        ).to(self.device)
        self.QTargetNet = deepcopy(self.QNet)
        self.QTargetNet.eval()
        self.optimizer = torch.optim.Adam(
            self.QNet.parameters(),
            lr=q.LEARNING_RATE,
        )

    def _hard_update_target(self):
        '''Hard update target networks by direct parameter assignment.
        '''

        self.QTargetNet.load_state_dict(self.QNet.state_dict())

    def _soft_update_target(self):
        '''Soft update target network by interpolation.
        '''

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
        self.training = True

    def eval(
        self,
    ) -> None:
        self.QNet.eval()
        self.training = False

    def get_target_value(
        self,
        batch: Ion,
    ) -> Ion:
        '''Compute target values of the observations.

        Args:
            batch: Input data, should contain keys 'observation'
        '''

        batch.to(
            ctype='torch',
            device=self.device,
        )
        observation: torch.Tensor = batch.observation.to(torch.float)
        with torch.no_grad():
            q_target = self.QTargetNet(observation).max(dim=-1)
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