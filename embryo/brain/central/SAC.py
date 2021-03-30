'''Soft Actor Critic

Created by Minhui Li on March 17, 2021
'''


from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.distributions import Distribution, Independent, Normal

from embryo.brain.central import CENTRAL_REGISTRY
from embryo.brain.central.base import Central
from embryo.brain.central.models import Actor, Critic, ProcessNet
from embryo.ion import Ion


@CENTRAL_REGISTRY.register()
class SACCentral(Central):
    '''Soft Actor Critic agent

    Reference:
        Haarnoja et al, Soft Actor Critic 
    '''

    def __init__(
        self,
        network: Optional[torch.nn.Module],
        target_update_frequency: Union[int, float] = 1,
        action_prior: Distribution = Normal,
        **kwargs
    ) -> None:
        '''Initialization method

        Args:
            network:
            target_update_frequency:
            action_prior:
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

        self.gradient_step = 0 # Gradient step counter

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha.detach())
        self.alphaOptimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        pre = ProcessNet(2)
        self.Actor = Actor(pre).to(self.device)
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=3e-4)
        pre = ProcessNet(3)
        self.Critic1 = Critic(pre).to(self.device)
        self.CriticTarget1 = Critic(pre).to(self.device)
        self.CriticOptimizer1 = torch.optim.Adam(self.Critic1.parameters(), lr=3e-4)
        pre = ProcessNet(3)
        self.Critic2 = Critic(pre).to(self.device)
        self.CriticTarget2 = Critic(pre).to(self.device)
        self.CriticOptimizer2 = torch.optim.Adam(self.Critic2.parameters(), lr=3e-4)
        self._hard_update_target()
        self.CriticTarget1.eval()
        self.CriticTarget2.eval()

        self.action_prior = action_prior
        a_low = -1.
        a_high = 1.
        self.action_scale = (a_high - a_low) / 2.
        self.action_bias = (a_high + a_low) / 2.

        # A small number to avoid mathematical error caused by 0.
        self._eps = np.finfo(np.float32).eps.item()

    def _hard_update_target(self):
        '''
        '''

        self.CriticTarget1.load_state_dict(self.Critic1.state_dict())
        self.CriticTarget2.load_state_dict(self.Critic2.state_dict())

    def _soft_update_target(self):
        '''
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
        '''
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
        # loss = (Q - returns).pow(2).mean()
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

    def eval(
        self,
    ) -> None:
        self.Actor.eval()
        self.Critic1.eval()
        self.Critic2.eval()

    def get_target_value(
        self,
        batch: Ion,
    ) -> Ion:
        '''

        Args:
            batch: Input data, should contain keys 'observation'
        '''

        with torch.no_grad():
            control = self.control(batch=batch)
            observation: torch.Tensor = batch.observation
            action: torch.Tensor = control.action
            log_prob: torch.Tensor = control.log_prob
            # q_target = torch.min(
            #     self.CriticTarget1(observation, action),
            #     self.CriticTarget2(observation, action),
            # ) - self.alpha * log_prob
            q1 = self.CriticTarget1(observation, action)
            q2 = self.CriticTarget2(observation, action)
            q_target = torch.min(q1, q2) - self.alpha * log_prob
            print(self.gradient_step * 5, q1.mean().item(), q2.mean().item(), q_target.mean().item())
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
        ###
        print(self.gradient_step * 5, returns.mean().item(), log_prob.mean().item(), q1.mean().item())
        ###

        actor_loss = torch.mean(
            self.alpha * log_prob.flatten() - \
            torch.min(q1, q2)
        )
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.)
        self.Actor_optimizer.step()

        log_prob = log_prob.detach() - 1
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
        x = dist.rsample()
        log_prob = dist.log_prob(x).unsqueeze(-1)
        if True:
            y = torch.tanh(x)
            action = y * self.action_scale + self.action_bias
            det = self.action_scale * (1. - y.pow(2))
            # Equation 21 in the paper
            log_prob = log_prob - torch.log(det + self._eps).sum(-1, keepdim=True)
        else:
            action = x

        return Ion(action=action, log_prob=log_prob)