'''Deep Q Network

Created by Minhui Li on September 30, 2019
'''


import os
import torch
from typing import Optional, Union

from embryo.brain.central.base import Central
from embryo.brain.memory import SequentialMemory
from embryo.brain.central.models import QNet
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
        capacity: int = 100,
        max_step: int = 0,
        double: bool = False,
        epsilon: float = 1.,
        epsilon_decay_type: str = 'Exponential',
        epsilon_decay: float = 0.9,
        epsilon_end: float = 0.01,
        network = Optional[torch.nn.Module],
        batch_size: int = 32,
        update: Union[int, float] = 1,
        tb_writer: Optional = None,
        **kwargs
    ) -> None:

        # Initialize parameters
        super().__init__(**kwargs)
        self.memory = SequentialMemory(capacity=capacity)
        self.max_step = max_step
        self.double = double
        self.explore_rate = epsilon
        self.explore_decay = epsilon_decay
        self.explore_decay_type = epsilon_decay_type.upper()
        self.explore_rate_min = epsilon_end
        self.batch_size = batch_size
        if 0 < update < 1:
            self.target_update = update # Soft update
        elif update >= 1:
            self.target_update = int(update) # Hard update
        else:
            raise ValueError('Target update should be greater than 0. (0, 1) for soft update, [1, inf] for hard update.')

        self.QNet = QNet()
        self.QTargetNet = QNet()
        self.QTargetNet.eval()
        self._target_hard_update()
        self.optimizer = torch.optim.Adam(self.QNet.parameters())
        self.loss = torch.nn.MSELoss()

    def _target_hard_update(self):
        self.QTargetNet.load_state_dict(self.QNet.state_dict())

    def _train_net(self):
        batch = self.memory.sample(num=self.batch_size) # Sample batch from memory
        s, a, r, ns, mask = zip(*batch) # Reorganize items
        x = to_tensor(s)
        a = to_tensor(a)
        ns = to_tensor(ns)
        with torch.no_grad():
            if self.double: # Double Q-Learning
                # nQ = self.QTargetNet.predict(ns)[np.arange(self.batch_size), np.argmax(self.QNet.predict(ns), axis = -1)] * mask # Q'(s', argmax(Q(s')))
                pass
            else:
                # nQ = np.amax(self.QTargetNet.predict(ns), axis = 1) * mask # max(Q(s', a'))
                nQ = self.QTargetNet(ns).max(dim=1)
            nQ = self.gamma * nQ + r

        Q = self.QNet(ns).gather(dim=1, index=a.unsqueeze(1))

        self.optimizer.zero_grad()
        loss = self.loss(Q, nQ)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.QNet.parameters(), max_norm=1.)
        self.optimizer.step()

        return loss.item()

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

    def learn(
        self,
        max_epoch: int = 0,
        evaluate: bool = False,
        plot: bool = False
    ) -> None:
        '''Training method
        
        Inputs:
        max_epoch: Maximum learning epochs
        evaluate: boolean
            Whether to evaluate agent after a epoch of training
        logger: Logger
            Evaluation logger
        plot: boolean
            Whether to plot a figure after training
        '''

        global_step = 0
        for epoch in range(max_epoch):
            state = self.reset()
            for _ in range(self.max_step):
                with torch.no_grad():
                    input_tensor = to_tensor(state).unsqueeze(0)
                    q = self.QNet(input_tensor).flatten()
                action = Epsilon_Greedy(q, self.explore_rate)
                next_state, reward, terminal, _ = self.step(action)
                self.memory.add((state, action, reward, next_state, not terminal)) # Append relavant information in the queue
                if terminal:
                    break
                state = next_state # Transit to the next state
                if len(self.memory) >= self.batch_size: # Memory is large enough for training network
                    loss = self._train_net()
                    global_step += 1
                    if global_step 
                    # Update target network
                    if self.target_update >= 1:
                        # Target hard update
                        if global_step % self.target_update == 0:
                            self._target_hard_update()
                        else:
                            pass # Placeholder for soft update

            # Evaluating current performance
            if evaluate:
                _ = self.render(num_episode = 1, vis = False, intv = 0, logger = logger)

            # Shrink exploration rate 
            if self.explore_rate > self.explore_rate_min:
                if self.explore_decay_type == 'EXPONENTIAL':
                    self.explore_rate *= self.explore_decay
                elif self.explore_decay_type == 'LINEAR':
                    self.explore_rate -= self.explore_decay
                else:
                    raise ValueError('Unsupported decay type.')

    def load_brain(self, timestamp):
        if self.backend == 'TENSORFLOW':
            pass
        elif self.backend == 'TORCH':
            checkpoint = torch.load('models/DQN/QNet{}.pth'.format(timestamp))
            self.QNet.load_state_dict(checkpoint.state_dict())
        else:
            raise ValueError('Backend not supported.')

    def save_brain(self, timestamp):
        torch.save(self.QNet, 'models/DQN/QNet{}.pth'.format(timestamp))

    def control(self, observation):
        '''Control method
        '''
        
        self.QNet.eval()
        state = torch.as_tensor(observation)
        with torch.no_grad():
            Q = self.QNet(state.unsqueeze(0))
        return Q.argmax(dim=-1)


class DoubleDQNAgent(DQNAgent):
    pass