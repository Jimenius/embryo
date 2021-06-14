'''

Created by Minhui Li on December 8, 2020
'''


from abc import ABC, abstractmethod
import os
from typing import Any, Dict

import gym
import torch
from yacs.config import CfgNode

from embryo.brain.central.base import compute_episodic_return, compute_nstep_return
from embryo.brain.central import Central, CENTRAL_REGISTRY
from embryo.brain.memory import Memory, MEMORY_REGISTRY
from embryo.ion import Ion
from embryo.limbs import LIMBS_REGISTRY
from embryo.limbs.limbs import Limbs


##########
import numpy as np
from tianshou.exploration import OUNoise
ou = OUNoise(0., 1.2)
def noise(action):
    a = action + ou(action.shape)
    a = np.clip(a, -1., 1.)
    return a
##########


class Agent(ABC):
    '''Agent definition
    '''

    def __init__(
        self,
        config: CfgNode,
        train_env: gym.Env,
        test_env: gym.Env,
    ) -> None:
        '''Initialization method

        Args:
            config: Configuration
        '''
        
        self.config = config
        self.train_env = train_env
        self.test_env = test_env
        self._build()
        self.env_step = 0
        self.gradient_step = 0

    def _build(
        self,
    ):
        '''Parse configuration to setup the agent.
        '''

        self.central: Central = CENTRAL_REGISTRY.get(self.config.CENTRAL.NAME)(
            config=self.config.CENTRAL,
            observation_space=self.train_env.observation_space,
            action_space=self.train_env.action_space,
        )
        self.memory: Memory = MEMORY_REGISTRY.get(self.config.MEMORY.NAME)(
            self.config.MEMORY,
        )
        
        limbs_cfg = self.config.LIMBS
        limbs_class = LIMBS_REGISTRY.get(limbs_cfg.NAME)
        preprocess = limbs_cfg.PREPROCESS
        if preprocess:
            pass
        postprocess = limbs_cfg.POSTPROCESS
        if postprocess:
            pass
        ###########
        postprocess = noise
        ###########
        self.train_limbs = limbs_class(
            env=self.train_env,
            central=self.central,
            memory=self.memory,
            preprocess=preprocess,
            postprocess=postprocess,
        )
        self.test_limbs = limbs_class(
            env=self.test_env,
            central=self.central,
        )

        solution = self.config.SOLUTION
        self.batch_size = solution.BATCH_SIZE
        self.gamma = solution.GAMMA
        self.init_step = solution.INIT_STEP
        self.max_epoch = solution.MAX_EPOCH
        self.step_per_epoch = solution.STEP_PER_EPOCH
        self.step_per_interact = solution.STEP_PER_INTERACT
        self.test_num_episode = solution.TEST_NUM_EPISODE
        self.update_per_step = solution.UPDATE_PER_STEP
        return_step = solution.RETURN_STEP
        if return_step > 0:
            self.return_method = compute_nstep_return
            self.nstep = return_step
        elif return_step == 0:
            self.return_method = compute_episodic_return
        else:
            raise ValueError(
                'Undefined negative return step.'
            )

    def _init_train_step(
        self,
    ) -> Dict[str, Any]:
        '''
        '''

        if self.init_step > 0:
            result = self.train_limbs.interact(num_step = self.init_step)
        else:
            result = {}

    @abstractmethod
    def learn(
        self,
    ):
        '''Learn from environment
        '''

        raise NotImplementedError

    def perform(
        self,
    ):
        '''Evaluate performance
        '''

        pass

    def save(
        self,
        directory: str,
    ) -> None:
        '''Save necessary properties to be recovered
        '''

        torch.save(
            {
                'memory': self.memory,
            },
            os.path.join(directory, 'agent.pth')
        )