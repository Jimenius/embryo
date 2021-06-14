'''

Created by Minhui Li on April 12, 2021
'''
import numpy as np
import torch

from embryo.brain.memory import Memory
from embryo.ion import Ion


class PlanEnv:

    def __init__(
        self,
        network,
        memory: Memory,
        num_envs: int = 1,
    ) -> None:
        '''Initialization method

        Args:
            network:
            memory:
            num_envs: Number of virtual environments executed in parallel
        '''

        self.network = network
        self.memory = memory
        self.num_envs = num_envs

    def reset(
        self,
    ) -> np.ndarray:
        '''Reset the environment and provide the initial observation.

        Returns:
        '''

        batch = self.memory.sample(
            batch_size=self.num_envs,
        )
        obs = batch.observation.to(ctype='numpy')

        return obs 

    def step(
        self,
        batch: Ion,
    ) -> Ion:
        '''Take a step in the dream environment.

        Args:
            batch: , should have keys 'observation' and 'action'.

        Returns:
            : , 'observation' and 'reward'
        '''

        batch = batch.copy().to(
            ctype='torch',
            dtype=torch.float,
            device=self.network.device,
        )
        observation = batch.observation
        action = batch.action

        next_observation, reward, done, info = self.network(observation, action)

        return next_observation, reward, done, info

    def close(self):
        '''Close the environment.

        Do nothing, just to make it compatible with real environments. 
        '''

        pass