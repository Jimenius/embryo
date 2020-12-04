'''Replay memory

Created by Minhui Li on March 22, 2020
'''


import random
from embryo.utils.structures import BinarySumTree


class SequentialMemory(object):
    '''Sequential Experience Replay Memory
    '''

    def __init__(
        self,
        capacity: int = 1
    ):
        self.memory = [None] * capacity
        self.capacity = capacity
        self.write_pointer = 0
        self.size = 0

    def add(
        self,
        element
    ) -> None:
        self.memory[self.write_pointer] = element
        self.write_pointer = (self.write_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(
        self,
        num: int = 0
    ):
        if num > self.size:
            raise ValueError("The number requested exceeds the memory size.")
        return random.sample(self.memory[:self.size], k=num)

    def __len__(self):
        return self.size


class PrioritizedMemory(SequentialMemory):
    '''Prioritized Experience Replay Memory.
    
    Inherited from Sequential Experience Replay Memory.

    Reference:
        Shaul et al, Prioritized Experience Replay
    '''

    def __init__(self, capacity=1):
        super(PrioritizedMemory, self).__init__(capacity=capacity)
        self.tree = BinarySumTree(node_num=2*capacity-1)

    def add(self, element):
        pass

    def sample(self, num=0):
        pass