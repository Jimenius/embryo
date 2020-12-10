'''Sequential memory for experience replay

Created by Minhui Li on December 9, 2020
'''


from typing import List, Optional, Tuple

import numpy as np
import torch

from embryo.brain.memory import MEMORY_REGISTRY
from embryo.brain.memory.base import Memory
from embryo.ion import Ion, extend_space


@MEMORY_REGISTRY.register()
class SequentialMemory(Memory):
    '''Sequential memory
    '''

    def __init__(
        self,
        max_size: int = 1,
        stack_num: int = 1,
    ) -> None:
        '''Initialization method

        Args:
            max_size: Maximum size of the memory, unlimited if 0
            stack_num: Number of stack for each element
        '''

        super().__init__(max_size=max_size)
        self.stack_num = stack_num

        self.reset()

    def reset(self) -> None:
        '''Reset buffer by setting current size and index to 0.
        '''
        
        super().reset()
        self._stack_valid_indice = [] if self.stack_num > 1 else None

    def add(
        self,
        element: Optional[Ion] = None,
        **kwargs
    ) -> None:
        '''Add a batch of data into the memory

        Args:
            element:
        '''

        # All element added should contain the same keys.

        if element:
            element.update(kwargs)
        else:
            element = Ion(data_dict=kwargs)
        if element.is_empty():
            raise AttributeError(
                'Cannot add an empty element to the memory.'
            )

        if self._data.is_empty():
            for k, v in element.items:
                self._data[k] = extend_space(value=v, extend_size=self.max_size)
        else:
            # Check if there is an unexpected key
            for k in element:
                if k not in self._data:
                    raise KeyError(
                        'All element added should contain the same keys, ',
                        'but got a new key: {}.'.format(k)
                    )
            self._data[self._index] = element

        if self._stack_valid_indice is not None and 'done' in self._data:
            valid = sum(
                self._data.done[i]
                for i in range(self._index - self.stack_num + 1, self._index)
            ) == 0 and self._size >= self.stack_num - 1
            if valid and self._index not in self._stack_valid_indice:
                self._stack_valid_indice.append(self._index)
            elif not valid and self._index in self._stack_valid_indice:
                self._stack_valid_indice.remove(self._index)
            # Also invalidate element away 
            index: int = (self._index + self.stack_num - 1) % self._maxsize
            if index in self._stack_valid_indice:
                self._stack_valid_indice.remove(index)

        if self.max_size > 0:
            self._size = min(self._size + 1, self.max_size)
            self._index = (self._index + 1) % self.max_size
        else:
            self._size = self._index = self._index + 1

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
            if self._stack_valid_indice is not None:
                valid_indice = self._stack_valid_indice
            else:
                valid_indice = self.size
            indice = np.random.choice(valid_indice, batch_size)
        elif batch_size < 0:
            raise ValueError(
                'Undefined negative batch size ',
                '= {}.'.format(batch_size)
            )
        else:
            # Batch size = 0, return all data in the memory
            if self._stack_valid_indice:
                indice = np.array(self._stack_valid_indice)
            else:
                indice = np.concatenate([
                    np.arange(self._index, self.size),
                    np.arange(self._index),
                ])

        return self[indice], indice


if __name__ == '__main__':
    m = SequentialMemory(max_size=15)
    for i in range(20):
        m.add(obs=i,done=i % 4,obsnext=i+1)
    print(m.obs)
    print(m.done)
    print(m.obsnext)