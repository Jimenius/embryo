'''Base memory for experience replay

Created by Minhui Li on December 9, 2020
'''


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import torch
from yacs.config import CfgNode

from embryo.ion import Ion


class Memory(ABC):
    '''
    '''

    def __init__(
        self,
        config: CfgNode,
    ) -> None:
        '''Initialization method

        Args:
            max_size: Maximum size of the memory, unlimited if 0
        '''

        self.max_size = config.SIZE

    def __getattr__(self, key: str) -> Any:
        '''
        '''

        if key in self._data:
            return self._data[key]
        else:
            raise AttributeError(
                'Attribute {} not found.'.format(key),
            )

    def __getitem__(
        self,
        index: Union[str, slice, int, np.integer, np.ndarray, List[int]]
    ) -> Any:
        '''Support self[index]
        '''

        return self._data[index]

    def __getstate__(self) -> Dict[str, Any]:
        '''Pickling interface.
        '''

        return self._data.__getstate__()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        '''Unpickling interface.
        '''

        self._data = Ion(**state)

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
        