'''Soft Actor Critic

Created by Minhui Li on March 17, 2021
'''


from copy import deepcopy
from typing import Dict, Optional, Union

from gym.spaces import Space
import numpy as np
import torch
from torch.distributions import Distribution, Independent
from yacs.config import CfgNode

from embryo.brain.central import CENTRAL_REGISTRY
from embryo.brain.central.base import Central
from embryo.brain.network import NETWORK_REGISTRY
from embryo.ion import Ion


@CENTRAL_REGISTRY.register()
class SACCentral(Central):
    '''Soft Actor Critic agent

    Reference:
        Haarnoja et al, Soft Actor Critic 
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

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha.detach())
        alpha_lr = self.config.ALPHA_LEARNING_RATE
        self.alphaOptimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Networks
        net_cfg = self.config.NETWORK
        actor = net_cfg.ACTOR
        self.Actor: torch.nn.Module = NETWORK_REGISTRY.get(actor.NAME)(
            self.observation_dim,
            self.action_dim,
        ).to(self.device)
        self.Actor_optimizer = torch.optim.Adam(
            self.Actor.parameters(),
            lr=actor.LEARNING_RATE,
        )
        critic = net_cfg.CRITIC
        self.Critic1 = NETWORK_REGISTRY.get(critic.NAME)(
            self.observation_dim,
            self.action_dim,
        ).to(self.device)
        self.CriticTarget1 = deepcopy(self.Critic1)
        self.CriticOptimizer1 = torch.optim.Adam(
            self.Critic1.parameters(),
            lr=critic.LEARNING_RATE,
        )
        self.Critic2 = NETWORK_REGISTRY.get(critic.NAME)(
            self.observation_dim,
            self.action_dim,
        ).to(self.device)
        self.CriticTarget2 = deepcopy(self.Critic2)
        self.CriticOptimizer2 = torch.optim.Adam(
            self.Critic2.parameters(),
            lr=critic.LEARNING_RATE,
        )
        self.CriticTarget1.eval()
        self.CriticTarget2.eval()

        self.action_prior = getattr(torch.distributions, self.config.ACTION_PRIOR)
        self.action_scale = torch.tensor(
            (self.action_range[1] - self.action_range[0]) / 2.,
            device=self.device,
        )
        self.action_bias = torch.tensor(
            (self.action_range[1] + self.action_range[0]) / 2.,
            device=self.device,
        )
        self.target_entropy = -np.prod(self.action_dim)

        # A small number to avoid mathematical error caused by 0.
        self._eps = np.finfo(np.float32).eps.item()

    def _hard_update_target(self):
        '''Hard update target networks by direct parameter assignment.
        '''

        self.CriticTarget1.load_state_dict(self.Critic1.state_dict())
        self.CriticTarget2.load_state_dict(self.Critic2.state_dict())

    def _soft_update_target(self):
        '''Soft update target network by interpolation.
        '''

        for main1, target1, main2, target2 in \
            zip(
                self.Critic1.parameters(),
                self.CriticTarget1.parameters(),
                self.Critic2.parameters(),
                self.CriticTarget2.parameters()
            ):
            target1.data.copy_(
                target1.data * (1. - self.target_update_frequency) + \
                    main1.data * self.target_update_frequency
            )
            target2.data.copy_(
                target2.data * (1. - self.target_update_frequency) + \
                    main2.data * self.target_update_frequency
            )

    def _update_critic(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        returns: torch.Tensor,
        idx: int = 1,
    ) -> torch.Tensor:
        '''Update each critic network.

        Args:
            observation: Observation tensor
            action: Action tensor
            returns: Return tensor
            idx: index of the critic

        Returns:
            Mean Square Error loss
        '''

        if idx == 1:
            critic = self.Critic1
            optimizer = self.CriticOptimizer1
        elif idx == 2:
            critic = self.Critic2
            optimizer = self.CriticOptimizer2
        else:
            raise ValueError(
                ''
            )

        Q = critic(observation, action).flatten()
        loss = torch.nn.MSELoss()(Q, returns.flatten())
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.)
        optimizer.step()

        return loss

    def train(
        self,
    ) -> None:
        self.Actor.train()
        self.Critic1.train()
        self.Critic2.train()
        self.training = True

    def eval(
        self,
    ) -> None:
        self.Actor.eval()
        self.Critic1.eval()
        self.Critic2.eval()
        self.training = False

    def get_target_value(
        self,
        batch: Ion,
    ) -> Ion:
        '''Compute target values of the observations.

        Args:
            batch: Input data, should contain keys 'observation'
        '''

        with torch.no_grad():
            control = self.control(batch=batch)
            observation: torch.Tensor = batch.observation
            action: torch.Tensor = control.action
            log_prob: torch.Tensor = control.log_prob
            q1 = self.CriticTarget1(observation, action)
            q2 = self.CriticTarget2(observation, action)
            q_target = torch.min(q1, q2) - self.alpha * log_prob
            q_target = q_target.squeeze(-1)

        return Ion(value=q_target)

    def update(
        self,
        batch: Ion,
        repeat: int = 0,
    ) -> Ion:
        '''Perform a backward gradient step.

        Args:
            batch: Data, should contain keys 'observation' and 'returns'.

        Returns:
            losses
        '''
        
        control = self.control(batch)
        observation: torch.Tensor = batch.observation
        action: torch.Tensor = batch.action
        returns: torch.Tensor = batch.returns

        critic1_loss = self._update_critic(
            observation=observation,
            action=action,
            returns=returns,
            idx=1,
        )
        critic2_loss = self._update_critic(
            observation=observation,
            action=action,
            returns=returns,
            idx=2,
        )

        action: torch.Tensor = control.action
        log_prob: torch.Tensor = control.log_prob
        q1 = self.Critic1(observation, action).flatten()
        q2 = self.Critic2(observation, action).flatten()

        actor_loss = torch.mean(
            self.alpha * log_prob.flatten() - \
            torch.min(q1, q2)
        )
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.)
        self.Actor_optimizer.step()

        log_prob = log_prob.detach() + self.target_entropy
        alpha_loss = -(self.log_alpha * log_prob).mean()
        self.alphaOptimizer.zero_grad()
        alpha_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.log_alpha, max_norm=1.)
        self.alphaOptimizer.step()
        self.alpha = torch.exp(self.log_alpha.detach())

        self.gradient_step += 1
        self._soft_update_target()

        losses = Ion(
            actor_loss=actor_loss.item(),
            alpha_loss=alpha_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
        )

        return losses

    def load(self, timestamp):
        pass

    def save(self, timestamp):
        pass

    def control(
        self,
        batch: Ion,
    ) -> Ion:
        '''Control method

        Args:
            batch: Input data, should contain keys 'observation'.
        '''

        batch.to(
            ctype='torch',
            device=self.device,
            dtype=torch.float,
        )
        observation: torch.Tensor = batch.observation
        action_parameter = self.Actor(observation)
        dist = Independent(self.action_prior(*action_parameter), 1)
        if self.training:
            # Random sample an action from the prior distribution when training.
            x = dist.rsample()
        else:
            # When testing, take the mean value instead of random sampling.
            x = action_parameter[0]
        log_prob = dist.log_prob(x).unsqueeze(-1)

        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        # Equation 21 in the paper
        det = self.action_scale * (1. - y.pow(2))
        log_prob = log_prob - torch.log(det + self._eps).sum(-1, keepdim=True)

        return Ion(
            action=action,
            log_prob=log_prob,
            parameters=action_parameter,
        )