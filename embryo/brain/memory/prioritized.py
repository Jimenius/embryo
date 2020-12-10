'''Prioritized memory for experience replay

Created by Minhui Li on December 9, 2020
'''


from embryo.brain.memory import MEMORY_REGISTRY
from embryo.brain.memory.base import Memory


@MEMORY_REGISTRY.register()
class PrioritizedMemory(Memory):
    '''Sequential memory
    '''

    pass