'''Base memory for experience replay

Created by Minhui Li on December 9, 2020
'''


from abc import ABC, abstractmethod
from typing import Any, Union

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

    def __getitem__(self, index: Union[int, str]) -> Any:
        '''
        '''

        if isinstance(index, str):
            return getattr(self._data, index)
        elif isinstance(index, int):
            element = Ion()
            for k in self._data:
                element[k] = self._data[k][index]
            return element
        else:
            raise TypeError(
                'Unsupported index type: {}'.format(type(index))
            )

    def __repr__(self) -> str:
        '''
        '''

        rep_str = self.__class__.__name__ + '('
        rep_str += 'max_size: {}'.format(self.max_size)
        rep_str += ', data: {}'.format(self._data)
        rep_str += ')'
        return rep_str

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
    ) -> int:
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
        