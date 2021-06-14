from abc import ABC, abstractmethod
import logging
from time import sleep
from typing import Callable, Optional, Union

from gym.spaces import Box, Discrete, Space, Tuple
import numpy as np
import torch
from yacs.config import CfgNode

from embryo.brain.memory.base import Memory
from embryo.ion import Ion


class Central(ABC):
    '''General agent super class
    '''

    def __init__(
        self,
        config: CfgNode,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        '''Initial method
        
        Args:
            config: Configurations
            observation_space: Observation space
            action_space: Action space
        '''

        super().__init__()
        self.config = config

        device: str = config.DEVICE
        if 'cuda' in device and not torch.cuda.is_available():
            device = 'cpu'
            logging.warning(
                'Attempt to use CUDA but CUDA is not available.'
                'Device is changed to CPU.'
            )
        self.device = device

        obs = observation_space
        acs = action_space[0]
        self.observation_dim = obs.shape[1:]
        if isinstance(acs, Discrete):
            self.action_dim = acs.n
        elif isinstance(acs, Box):
            self.action_dim = acs.shape
            self.action_range = (acs.low, acs.high)
        else:
            raise AttributeError(
                'Can\'t fetch action dimension.',
            )

        # Gradient step counter
        self.gradient_step = 0
        self.training = True

    @abstractmethod
    def get_target_value(
        self,
        batch: Ion,
    ) -> Ion:
        '''
        '''

        raise NotImplementedError # To be completed by subclasses

    @abstractmethod
    def update(self):
        '''
        '''

        raise NotImplementedError # To be completed by subclasses

    @abstractmethod
    def load(self):
        '''
        '''

        raise NotImplementedError # To be completed by subclasses

    @abstractmethod
    def save(self):
        '''
        '''

        raise NotImplementedError # To be completed by subclasses
    
    @abstractmethod
    def control(self, observation):
        '''
        '''

        raise NotImplementedError # To be completed by subclasses


def compute_episodic_return(
) -> None:
    '''Compute returns in an episode with General Advantage Estimation. 

    Args:

    Reference:
        Schulman et al, High-dimensional Continuous Control Using Generalized Advantage Estimation
    '''

    pass

def compute_nstep_return(
    memory: Memory,
    indice: np.ndarray,
    target_fn: Callable[[Ion], Ion],
    n: int = 1,
    gamma: float = 1.,
) -> Ion:
    '''Compute n-step returns of memory[indice]

    Usually used for off-policy algorithms.

    Args:
        memory: Replay memory
        indice: Indice sampled from replay memory
        target_fn: Function to compute target value
        n: Step number
        gamma: Discount factor
    '''

    if n < 1:
        raise ValueError(
            'Step should be at least 1.'
        )

    batch: Ion = memory[indice]
    batch.to(ctype='numpy')
    temp_batch: Ion = batch.copy()
    batch_size = len(indice)
    finish_step = np.full(batch_size, fill_value=n, dtype=int)
    gamma_factor = 1.
    returns_est = np.zeros(batch_size, dtype=np.float32)

    for i in range(n):
        # 1. done & finish_step >= i: R = R + \gamma ^ i * r, finish = i
        # 2. done & finish_step < i: R = R, finish = finish
        # 3. ~done & .next = -1 & finish_step >= i: R = R, finish = i
        # 4. ~done & .next = -1 & finish_step < i: R = R, finish = finish
        # 5. Otherwise (~done & .next >= 0): R = R + \gamma ^ i * r, finish = finish, go to next

        done = temp_batch.done
        finished = finish_step < i
        step = temp_batch.next >= 0
        # Condition to go to the next step
        next_cond = ~done & step
        # Condition to add reward to the return estimation
        reward_cond = done & ~finished | next_cond
        # Condition to finish calculation
        finish_cond = ~finished & (~(done | step) | done)

        temp_reward = gamma_factor * temp_batch.reward
        returns_est[reward_cond] += temp_reward[reward_cond]
        gamma_factor *= gamma
        temp_batch[next_cond] = memory[temp_batch.next[next_cond]]
        finish_step[finish_cond] = i
    
    # R = R + \gamma ^ n * Q_target, if not done
    if not temp_batch.done.all():
        observations = Ion(
            observation=temp_batch.observation[~temp_batch.done],
        )
        q_est = target_fn(observations).to(ctype='numpy').value
        returns_est[~temp_batch.done] += gamma ** finish_step[~temp_batch.done] * q_est

    # observations  =Ion(
    #     observation=temp_batch.onext,
    # )
    # q_est = target_fn(observations).to(ctype='numpy').value
    # returns_est = temp_batch.reward + gamma * q_est.flatten() * temp_batch.done

    # Write to result
    batch.update(returns=returns_est)
    return batch


if __name__ == '__main__':
    from embryo.brain.memory.linked import LinkedMemory
    m = LinkedMemory(max_size=15)
    prev = [-1, -1]
    for i in range(20):
        prev[0] = m.add(observation=i,reward=1,done=i % 5 == 0,prev=prev[0])
        prev[1] = m.add(observation=i * 2,reward=2,done=i % 3 == 0,prev=prev[1])
    m.done[7] = False
    print('prev', m.prev)
    print('obs', m.observation)
    print('reward', m.reward)
    print('done', m.done)
    print('next', m.next)
    target_fn = lambda ion: Ion(value=ion.observation+1)
    indice = list(range(15))
    num_step = 3
    print(m[indice])
    b = compute_nstep_return(memory=m, indice=indice, target_fn=target_fn, n=num_step, gamma = 0.1)
    print('returns', b.returns)