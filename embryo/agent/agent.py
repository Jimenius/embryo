'''

Created by Minhui Li on December 8, 2020
'''


from abc import ABC, abstractmethod

from yacs import CfgNode

from embryo.brain.central import Central, CENTRAL_REGISTRY
from embryo.brain.memory import Memory, MEMORY_REGISTRY
from embryo.ion import Ion
from embryo.limbs import Limbs


class Agent(ABC):
    '''Agent definition
    '''

    def __init__(
        self,
        config: CfgNode,
        env,
    ) -> None:
        '''Initialization method

        Args:
            config: Configuration
        '''
        
        self.config = config
        self.build()

    def build(
        self,
    ):
        '''Parse configuration to setup the agent.
        '''

        self.central = CENTRAL_REGISTRY.get(self.config.CENTRAL.NAME)()
        self.memory = MEMORY_REGISTRY.get(self.config.MEMORY.NAME)(
            max_size=self.config.MEMORY.SIZE,
            stack_num=self.config.MEMORY.STACK,
        )
        self.limbs = Limbs(
            env=env,
            central=self.central,
            memory=self.memory,
        )

    def learn(
        self,
        env,
    ):
        '''
        '''

    def interact(
        self,
        env,
    ):
        '''
        '''