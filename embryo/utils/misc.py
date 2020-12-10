from numbers import Number
from typing import List, Union

import torch
import numpy as np


class MovingAverage:
    '''Moving average implementation

    Automatically exclude infinity and NaN.
    '''

    def __init__(
        self,
        size: int = 100
    ) -> None:
        self.size = size
        self.cache: List[Union[Number, np.number]] = []
        self.banned = [np.inf, np.nan, -np.inf]

    def add(
        self,
        x: Union[Number, list, np.number, np.ndarray, torch.Tensor]
    ) -> np.number:
        '''Add a scalar into the class
        '''

        if isinstance(x, torch.Tensor):
            x = to_numpy(x.flatten())
        if isinstance(x, list) or isinstance(x, np.ndarray):
            for i in x:
                if i not in self.banned:
                    self.cache.append(i)
        elif x not in self.banned:
            self.cache.append(x)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size:]
        return self.get()

    def get(self) -> np.number:
        """Get the average."""
        if len(self.cache) == 0:
            return 0
        return np.mean(self.cache)

    def mean(self) -> np.number:
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self) -> np.number:
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0
        return np.std(self.cache)
