from abc import ABC, abstractmethod
from time import sleep
from typing import Callable, Union

import numpy as np
import torch

from embryo.brain.memory.base import Memory
from embryo.ion import Ion


class Central(ABC):
    '''General agent super class
    '''

    def __init__(
        self,
        device: Union[torch.device, str] = 'cpu',
        gamma: float = 1.,
    ) -> None:
        '''
        Args:
        env: Target environment
        gamma: Discount factor
        brain: str
            Brain index
        '''

        super().__init__()
        self.device = device
        # self.env = env
        # self.model = env.env # Internal environment model alias
        # self.reset = self.env.reset # Reset environment alias
        # self.step = self.env.step # Step method alias

        # obs = env.observation_space
        # acs = env.action_space
        # if hasattr(obs, 'n'):
        #     self.state_dim = obs.n
        # else:
        #     self.state_dim = obs.shape

        # if hasattr(acs, 'n'):
        #     self.action_dim = acs.n
        # else:
        #     self.action_dim = acs.shape
        #     self.action_range = (acs.low, acs.high)

        # Parameters
        self.gamma = gamma

        # # Trained model
        # self.brain = brain

    @abstractmethod
    def get_target_value(
        self,
        batch: Ion,
    ) -> Ion:
        raise NotImplementedError # To be completed by subclasses

    @abstractmethod
    def update(self):
        raise NotImplementedError # To be completed by subclasses

    @abstractmethod
    def load(self):
        raise NotImplementedError # To be completed by subclasses

    @abstractmethod
    def save(self):
        raise NotImplementedError # To be completed by subclasses
    
    @abstractmethod
    def control(self, observation):
        raise NotImplementedError # To be completed by subclasses

    def render(self, num_episode = 1, vis = False, intv = 1, logger = None):
        '''Evaluate and visualize the result

        Args:
        num_episode: int
            Number of render episodes
        vis: boolean
            Action values
        intv: float
            Time interval for env.render()
        logger: logging.Logger
            logger
            
        Returns:
        float
            Average cumulative rewards achieved in multiple episodes
        '''
        
        avg_reward = 0. # Average reward of episodes
        for episode in range(num_episode):
            cumulative_reward = 0. # Accumulate rewards of steps in an episode
            terminal = False
            observation = self.env.reset()
            while not terminal:
                if vis:
                    self.env.render()
                    sleep(intv)
                try:
                    action = self.control(observation)
                except NotImplementedError:
                    action = self.env.action_space.sample()
                observation, reward, terminal, _ = self.env.step(action)
                cumulative_reward += reward
                
            avg_reward += cumulative_reward
            if vis:
                self.env.render()
                logtxt = 'Episode {} ends with cumulative reward {}.'.format(episode, cumulative_reward)
                try:
                    logger(logtxt)
                except:
                    print(logtxt)

        if num_episode > 0: # Avoid divided by 0
            avg_reward /= num_episode
            logtxt = 'The agent achieves an average reward of {} in {} episodes.'.format(avg_reward, num_episode)
            try:
                logger(logtxt)
            except:
                print(logtxt)
        self.env.close()
        return avg_reward


def compute_episodic_return(
    memory: Memory,
    indice: np.ndarray,
    target_fn: Callable[[Ion], Ion],
    gamma: float = 1.,
    adv_lambda: float = 1.
) -> None:
    '''Compute returns in an episode with General Advantage Estimation. 

    Args:

    Reference:
    '''

    # batch = memory[indice]
    # returns = np.roll(v_s_, 1)
    # factor = (1.0 - done) * gamma
    # delta = rew + v_s_ * factor - returns
    # factor *= adv_lambda

    # for i in range(len(reward) - 1, -1, -1):
    #     gae = delta[i] + m[i] * gae
    #     returns[i] += gae
    # return batch
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
    # returns_est: np.ndarray = temp_batch.reward.copy()
    # for i in range(1, n):
    #     # 1. done: R = R + \gamma ^ i * r, finish = i
    #     # 2. .next = -1: finish = i
    #     # 3. Otherwise: R = R + \gamma ^ i * r, go to next
    #     gamma_factor *= gamma
    #     # changing_indice = finish_step > i
    #     # finish_indice = changing_indice and (temp_batch.done or temp_batch.next < 0)
    #     finish_indice = finish_step > i and (temp_batch.done or temp_batch.next < 0)
    #     temp_r = gamma_factor * temp_batch.reward
    #     next_indice = np.where(not finish_indice)
    #     temp_batch[~finish_indice] = memory[next_indice]
    #     finish_step[finish_indice] = i
    #     returns_est += temp_r

    returns_est = np.zeros(batch_size, dtype=np.float32)
    for i in range(n):
        clip_cond = temp_batch.next < 0
        finish_cond = temp_batch.done | clip_cond
        temp_reward = gamma_factor * temp_batch.reward
        returns_est[~clip_cond] += temp_reward[~clip_cond]
        gamma_factor *= gamma
        temp_batch[~finish_cond] = memory[temp_batch.next[~finish_cond]]
    
    # R = R + \gamma ^ n * max(Q_target), if not done
    if not temp_batch.done.all():
        observations = Ion(
            observation=temp_batch.observation[~temp_batch.done],
        )
        q_est = target_fn(observations).to(ctype='numpy').value
        returns_est[~temp_batch.done] += gamma ** finish_step[~temp_batch.done] * q_est.max(axis=-1)
        # returns_est[~temp_batch.done] += gamma ** finish_step[~temp_batch.done] * q_est.flatten()


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
    print('obs', m.observation)
    print('reward', m.reward)
    print('done', m.done)
    print('next', m.next)
    target_fn = lambda ion: Ion(value=ion.observation+1)
    indice = [3, 7, 10]
    num_step = 3
    print(m[indice])
    b = compute_nstep_return(memory=m, indice=indice, target_fn=target_fn, n=num_step, gamma = 1.)
    print('returns', b.returns)