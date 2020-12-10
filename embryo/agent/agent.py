'''

Created by Minhui Li on December 8, 2020
'''


from abc import ABC, abstractmethod

from embryo.brain.central import Central
from embryo.brain.memory import Memory
from embryo.ion import Ion
from embryo.limbs import Limbs


class Agent(ABC):
    '''Agent definition
    '''

    def __init__(
        self,
        central: Central,
        memory: Memory,
        limbs: Limbs,
        writer,
        batch_size: int = 0,
        step_per_epoch: int = 0,
        interact_per_step: int = 0
    ) -> None:
        '''Initialization method

        Args:
            central: Training method
            memory: Replay memory
            limbs: Policy interaction
            writer:
            batch_size: Batch size
            step_per_epoch: Step per epoch
            interact_per_step: Interaction per step
        '''

        self.env_step = 0
        self.gradient_step = 0
        self.best_epoch = 0
        self.best_reward = -float('inf')

        self.central = central
        self.limbs = limbs
        self.memory = memory
        self.step_per_epoch = step_per_epoch
        self.interact_per_step = interact_per_step

    def __call__(
        self,
        max_epoch: int,
    ):
        '''

        Args:
            max_epoch: Maximum training epoch
        '''
        for epoch in range(max_epoch):
            for step in range(self.step_per_epoch):
                result: Ion = self.limbs.interact(num_step=self.interact_per_step)
                self.env_step += 1
                if writer:
                    pass
                losses = self.central.learn()
                self.gradient_step += 1
                for k in losses.keys():
                    pass
            result: Ion = test()


class OnPolicyAgent(Agent):
    '''On-policy agent
    '''

    pass


class OffPolicyAgent(Agent):
    '''Off-policy agent
    '''

    pass