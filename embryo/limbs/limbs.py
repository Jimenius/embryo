'''

Created by Minhui Li on December 7, 2020
'''


import time
from typing import Callable, Dict, Optional, Union

import gym
import numpy as np

from embryo.brain.central.base import Central
from embryo.ion import Ion


class Limbs:
    '''Limbs connect the central policy and the environment.
    '''

    def __init__(
        self,
        env: gym.Env,
        central: Central,
        preprocess: Optional[Callable[Ion]] = None,
        postprocess: Optional[Callable[Ion]] = None,
    ) -> None:
        '''Initialization method

        Args:
            env: Environment
            brain:
            preprocess: observation preprocess function
        '''

        self.env = env
        self.env_number = len(self.env)
        self.central = central
        self.preprocess = preprocess
        self.postprocess = postprocess

        self.reset()

    def get_env_number(self) -> int:
        '''Env_num getter
        '''

        return self.env_number

    def reset(
        self,
    ) -> None:
        '''Reset associated attributes
        '''

        self.interaction = Ion()
        obs = self.env.reset()
        if self.preprocess:
            obs = self.preprocess(obs)
        self.interaction.update(obs)

    def interact(
        self,
        num_step: int,
        num_episode,
        render: float = 0.,
    ) -> Dict[str, Union[float, int]]:
        '''Interact with the environment.

        Args:
            num_step:
            num_episode:
            render: Visualization time gap
        '''

        step_counter: int = 0
        episode_count = None
        while True:
            action = self.central.control(self.interaction)
            if self.postprocess:
                action = self.postprocess(self.interaction.action)
            else:
                action = self.interaction.action
            obs_next, reward, done, info = self.env.step(action)
            step_counter += 1
            self.interaction.update(obs_next=obs_next, reward=reward, done=done, info=info)
            
            if render:
                self.env.render()
                time.sleep(render)

            if num_step:
                if step_counter >= num_step:
                    break
            elif num_episode:
                pass
            else:
                raise ValueError(
                    'Lack condition to break the loop. ',
                    'Number of steps: {}. '.format(num_step),
                    'Number of episodes {}'.format(num_episode),
                )

            if self.preprocess:
                haha = self.preprocess()
                self.interaction.update(haha)
            
        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "v/st": step_count / duration,
            "v/ep": episode_count / duration,
            "rew": np.mean(rewards),
            "rew_std": np.std(rewards),
            "len": step_count / episode_count,
        }