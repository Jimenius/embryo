'''Soft Actor Critic

Created by Minhui Li on March 17, 2021
'''


from copy import deepcopy
from typing import Dict, Optional, Union

import numpy as np
import torch

from embryo.brain.central import CENTRAL_REGISTRY
from embryo.brain.central.SAC import SACCentral
from embryo.brain.central.models import Actor, Critic, ProcessNet
from embryo.ion import Ion


@CENTRAL_REGISTRY.register()
class REDQCentral(SACCentral):
    '''Randomized Ensembled Double Q-Learning agent

    Reference:
        Chen et al, Randomized Ensembled Double Q-Learning
    '''

    aggregation_mode_collection = (
        'min',
        'average',
        'mix',
    )

    def __init__(
        self,
        target_update_frequency: Union[int, float] = 1,
        ensemble_size: int = 2,
        subset_size: int = 2,
        critic_update_frequency: int = 1,
        target_mode: str = 'min',
        prediction_mode: str = 'average',
        **kwargs
    ) -> None:
        '''Initialization method

        Args:
            network:
            target_update_frequency:
            ensemble_size:
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

        # Set up alpha and actor network
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha.detach())
        self.alphaOptimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        pre = ProcessNet(2)
        self.Actor = Actor(pre).to(self.device)
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=3e-4)

        # Set up critic networks
        if subset_size > ensemble_size:
            raise ValueError(
                'Subset size should be smaller than ensemble size, ',
                'but got subset size:{} ensemble size:{}.'.format(
                    subset_size,
                    ensemble_size,
                )
            )
        self.ensemble_size = ensemble_size
        self.subset_size = subset_size
        self.Critic = [None] * ensemble_size
        self.CriticTarget = [None] * ensemble_size
        self.CriticOptimizer = [None] * ensemble_size
        for i in range(ensemble_size):
            pre = ProcessNet(3)
            self.Critic[i] = Critic(pre).to(self.device)
            self.CriticTarget[i] = deepcopy(self.Critic[i])
            self.CriticTarget[i].eval()
            self.CriticOptimizer[i] = \
                torch.optim.Adam(self.Critic[i].parameters(), lr=3e-4)
        self.critic_update_frequency = critic_update_frequency

        if target_mode in self.aggregation_mode_collection:
            self.target_mode = target_mode
        else:
            raise ValueError(
                'Undefined aggregation mode {}'.format(target_mode),
            )

        if prediction_mode in self.aggregation_mode_collection:
            self.prediction_mode = prediction_mode
        else:
            raise ValueError(
                'Undefined aggregation mode {}'.format(prediction_mode),
            )

        self.training = True

        a_low = -1.
        a_high = 1.
        self.action_scale = (a_high - a_low) / 2.
        self.action_bias = (a_high + a_low) / 2.

        # A small number to avoid mathematical error caused by 0.
        self._eps = np.finfo(np.float32).eps.item()

    def _hard_update_target(self):
        '''
        '''

        for i in range(self.ensemble_size):
            self.CriticTarget[i].load_state_dict(self.Critic[i].state_dict())

    def _soft_update_target(self):
        '''
        '''

        for i in range(self.ensemble_size):
            for main, target in zip(
                self.Critic[i].parameters(),
                self.CriticTarget[i].parameters(),
            ):
                target.data.copy_(
                    target.data * (1. - self.target_update_frequency) + \
                        main.data * self.target_update_frequency
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

        critic = self.Critic[idx]
        optimizer = self.CriticOptimizer[idx]
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
        for i in range(self.ensemble_size):
            self.Critic[i].train()
        self.training = True

    def eval(
        self,
    ) -> None:
        self.Actor.eval()
        for i in range(self.ensemble_size):
            self.Critic[i].eval()
        self.training = False

    def get_target_value(
        self,
        batch: Ion,
    ) -> Ion:
        '''Compute target values of the observations.

        Args:
            batch: Input data, should contain keys 'observation'

        Returns:
            of shape (batch_size,)
        
        Raises:
            NotImplementedError:
        '''

        with torch.no_grad():
            control = self.control(batch=batch)
            observation: torch.Tensor = batch.observation
            action: torch.Tensor = control.action
            log_prob: torch.Tensor = control.log_prob
            indice = np.random.choice(
                self.ensemble_size,
                self.subset_size,
                replace=False,
            )
            
            if self.target_mode == 'min':
                q = [None] * self.subset_size
                for i, idx in enumerate(indice):
                    q[i] = self.CriticTarget[idx](observation, action)
                q = torch.stack(q, dim=-1)
                q, _ = torch.min(q, dim=-1)
            elif self.target_mode == 'average':
                q = [None] * self.ensemble_size
                for i in range(self.ensemble_size):
                    q[i] = self.Critic[i](observation, action).flatten()
                q = torch.stack(q, dim=-1)
                q = torch.mean(q, dim=-1)
            elif self.target_mode == 'mix':
                raise NotImplementedError
            else:
                raise NotImplementedError

            q_target = q - self.alpha * log_prob
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
        returns: torch.Tensor = batch.returns.flatten()

        critic_losses = np.zeros(
            (
                self.critic_update_frequency,
                self.ensemble_size,
            ),
            dtype=np.float32,
        )
        for step in range(self.critic_update_frequency):
            for i in range(self.ensemble_size):
                loss = self._update_critic(
                    observation=observation,
                    action=action,
                    returns=returns,
                    idx=i,
                )
                critic_losses[step, i] = loss.item()

            self._soft_update_target()

        action: torch.Tensor = control.action
        log_prob: torch.Tensor = control.log_prob

        indice = np.random.choice(
            self.ensemble_size,
            self.subset_size,
            replace=False,
        )
            
        if self.target_mode == 'min':
            q = [None] * self.subset_size
            for i, idx in enumerate(indice):
                q[i] = self.Critic[idx](observation, action).flatten()
            q = torch.stack(q, dim=-1)
            q, _ = torch.min(q, dim=-1)
        elif self.target_mode == 'average':
            q = [None] * self.ensemble_size
            for i in range(self.ensemble_size):
                q[i] = self.Critic[i](observation, action).flatten()
            q = torch.stack(q, dim=-1)
            q = torch.mean(q, dim=-1)
        elif self.target_mode == 'mix':
            raise NotImplementedError
        else:
            raise NotImplementedError

        actor_loss = torch.mean(
            self.alpha * log_prob.flatten() - q
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

        losses = Ion(
            actor_loss=actor_loss.item(),
            alpha_loss=alpha_loss.item(),
            critic_loss=np.mean(critic_losses),
        )

        return losses

    def load(self, timestamp):
        pass

    def save(self, timestamp):
        pass