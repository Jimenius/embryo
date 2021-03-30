'''

Created by Minhui Li on December 7, 2020
'''


import time
from typing import Callable, Dict, Optional, Union

import gym
import numpy as np
import torch

from embryo.brain.central import Central
from embryo.brain.memory import Memory
from embryo.ion import Ion


class Limbs:
    '''Limbs connect the central policy and the environment.

    Limbs receive order from the central, take a step in the environment, and adds the interaction into the memory.
    '''

    def __init__(
        self,
        env: gym.Env,
        central: Central,
        memory: Optional[Memory] = None,
        preprocess: Optional[Callable[[Ion], None]] = None,
        postprocess: Optional[Callable[[Ion], None]] = None,
    ) -> None:
        '''Initialization method

        Args:
            env: Environment
            brain:
            preprocess: observation preprocess function
        '''

        self.env = env
        self.env_number = self.env.num_envs
        self.central = central
        self.memory = memory
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
        self.interaction.update(observation=obs)

    def interact(
        self,
        num_step: int = 0,
        num_episode: int = 0,
        render: float = 0.,
    ) -> Dict[str, Union[float, int]]:
        '''Interact with the environment.

        Args:
            num_step: Maximum number of step
            num_episode: Maximum number of episode
            render: Visualization time gap

        Raises:
            ValueError:
        '''

        step_counter: int = 0
        episode_counter = 0
        start_time = time.time()
        # Problematic in train
        reward_collection = []
        cumulative_reward = np.zeros(self.env_number)

        if num_episode < 0 or num_step < 0:
            raise ValueError(
                'Invalid negative number. ',
                'Number of steps: {}. '.format(num_step),
                'Number of episodes {}'.format(num_episode),
            )
        if num_episode + num_step == 0:
            raise ValueError(
                'Lack condition to break the loop. ',
                'Number of steps: {}. '.format(num_step),
                'Number of episodes {}'.format(num_episode),
            )

        while num_episode and episode_counter < num_episode \
            or num_step and step_counter < num_step:
            with torch.no_grad():
                action = self.central.control(self.interaction).to(ctype='numpy').action
            if self.postprocess:
                action = self.postprocess(action)
            obs_next, reward, done, info = self.env.step(action)
            step_counter += self.env_number
            episode_counter += sum(done)
            cumulative_reward += reward
            self.interaction.update(
                action=action,
                reward=reward,
                done=done,
                info=info,
            )

            # Record cumulative reward in an episode.
            if done.any():
                indice = np.where(done)
                for r in cumulative_reward[indice]:
                    reward_collection.append(r)
                cumulative_reward[indice] = 0.
            
            if render:
                self.env.render()
                time.sleep(render)

            if self.memory:
                self.memorize()

            if self.preprocess:
                obs_next = self.preprocess(obs_next)
            self.interaction = Ion(observation=obs_next)

        duration = max(time.time() - start_time, 1e-9)
        mean = np.mean(reward_collection) if reward_collection else 0.
        std = np.std(reward_collection) if reward_collection else 0.
        length = step_counter / episode_counter if episode_counter else step_counter

        return {
            "episode": episode_counter,
            "step": step_counter,
            "vstep": step_counter / duration,
            "vepisode": episode_counter / duration,
            "reward": mean,
            "reward_std": std,
            "length": length,
        }

    def memorize(
        self,
    ) -> None:
        '''Abstract method to add interactions to the memory.
        '''

        raise NotImplementedError


class OffPolicyLimbs(Limbs):
    '''
    '''

    def reset(
        self,
    ):
        '''
        '''

        super().reset()
        if self.memory:
            self.prev = np.full(self.env_number, fill_value=-1)

    def memorize(
        self,
    ) -> None:
        '''Add interactions to the memory.
        '''

        for i in range(self.env_number):
            ###
            info = self.interaction.info[i]
            if info:
                print(info)
                if isinstance(info, dict) and \
                    'TimeLimit.truncated' in info and \
                    info['TimeLimit.truncated']:
                    self.prev[i] = -1
                    continue
            ###
            self.prev[i] = self.memory.add(
                prev=self.prev[i],
                observation=self.interaction.observation[i],
                action=self.interaction.action[i],
                reward=self.interaction.reward[i],
                done=self.interaction.done[i],
                info=self.interaction.info[i],
            )

        self.prev[self.interaction.done] = -1