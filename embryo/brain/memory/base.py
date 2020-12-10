'''Base memory for experience replay

Created by Minhui Li on December 9, 2020
'''


from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from embryo.ion import Ion


class Memory(ABC):
    '''
    '''

    def __init__(
        self,
        max_size: int = 1,
    ) -> None:
        '''
        '''

        self.max_size = max_size

    def __getattr__(self, key: str) -> Any:
        '''
        '''

        if key in self._data:
            return self._data[key]
        else:
            raise AttributeError(
                'Attribute {} not found.'.format(key),
            )

    def reset(self) -> None:
        '''Reset the memory
        '''

        self._size = 0
        self._index = 0
        self._data = Ion()

    @abstractmethod
    def add(
        self,
        elements: Ion,
    ) -> None:
        '''Add an element into the memory.
        '''

        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        batch_size: int = 1,
    ) -> Ion:
        '''Sample randomly a batch of size 'batch_size'
        '''

        raise NotImplementedError
        