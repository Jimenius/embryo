'''Linked memory for experience replay

Created by Minhui Li on January 7, 2021
'''


from typing import List, Optional, Tuple

import numpy as np
import torch

from embryo.brain.memory import MEMORY_REGISTRY
from embryo.brain.memory.base import Memory
from embryo.ion import Ion, extend_space


@MEMORY_REGISTRY.register()
class LinkedMemory(Memory):
    '''Linked memory

    Slots are linked by prev and next attributes.
    The linkage relationship is the same as that in an episode
    '''

    def __init__(
        self,
        max_size: int = 1,
    ) -> None:
        '''Initialization method

        Args:
            max_size: Maximum size of the memory, unlimited if 0
            stack_num: Number of stack for each element
        '''

        super().__init__(max_size=max_size)
        self.reset()

    def add(
        self,
        element: Optional[Ion] = None,
        **kwargs
    ) -> int:
        '''Add a batch of data into the memory

        Args:
            element: Explicit element in Ion form

        Returns:
            Index in the memory added.

        Raises:
            KeyError: 
        '''

        # All element added should contain the same keys.

        for k in ('element', 'next'):
            if k in kwargs:
                raise KeyError(
                    'Please choose another name for {} as it is reserved.'.format(k)
                )

        if element:
            element.update(kwargs)
        else:
            element = Ion(data_dict=kwargs)
        if element.is_empty():
            raise AttributeError(
                'Cannot add an empty element to the memory.'
            )

        if 'prev' not in element:
            raise KeyError(
                'Linked memory requires key \'prev\' for linkage.'
            )
        element.update({'next': -1})

        if self._data.is_empty():
            for k, v in element.items:
                self._data[k] = extend_space(value=v, extend_size=self.max_size)
            self._data.prev -= 1
            self._data.next -= 1
        else:
            # Check if there is an unexpected key
            for k in element:
                if k not in self._data:
                    raise KeyError(
                        'All element added should contain the same keys, ',
                        'but got a new key: {}.'.format(k)
                    )
            original_next_index = self._data.next[self._index]
            # If the slot has data originally,
            # set the prev of the original next to -1
            # as data replacement.
            if original_next_index >= 0:
                self._data.prev[original_next_index] = -1
            self._data[self._index] = element
        if element['prev'] >= 0:
            self._data.next[element['prev']] = self._index
        
        index_inserted = self._index
        if self.max_size > 0:
            self._size = min(self._size + 1, self.max_size)
            self._index = (self._index + 1) % self.max_size
        else:
            self._size = self._index = self._index + 1

        return index_inserted

    def sample(
        self,
        batch_size: int = 1,
    ) -> Tuple[Ion, np.ndarray]:
        '''Sample randomly a batch of size 'batch_size'

        Sample all valid elements in the memory if batch_size = 0.

        Args:
            batch_size: Batch size
        '''

        if batch_size > 0:
            indice = np.random.choice(self.size, batch_size)
        elif batch_size < 0:
            raise ValueError(
                'Undefined negative batch size ',
                '= {}.'.format(batch_size)
            )
        else:
            indice = np.concatenate([
                np.arange(self._index, self.size),
                np.arange(self._index),
            ])

        return self[indice], indice


if __name__ == '__main__':
    m = LinkedMemory(max_size=15)
    prev = [-1, -1]
    for i in range(20):
        prev[0] = m.add(obs=i,done=i % 4,prev=prev[0])
        prev[1] = m.add(obs=i * 2,done=i % 3,prev=prev[1])
    print(m.obs)
    print(m.done)
    print(m.prev)
    print(m.next)
    print(m)