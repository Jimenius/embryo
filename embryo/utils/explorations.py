from typing import Union

import torch
import numpy as np


def Epsilon_Greedy(
    value: Union[np.ndarray, torch.Tensor],
    epsilon: float = 1.,
) -> Union[np.ndarray, torch.Tensor]:
    '''Epsilon Greedy action selection

    Select an action from a discrete action space based on values
    
    Args:
        value: Action values, shape = (batch_size, )
        epsilon: Exploration rate, Epsilon
    
    Returns:
        Action index

    Raises:
        ValueError,
        TypeError,
    '''

    dim = len(value.shape)
    if dim != 2:
        raise ValueError(
            'Value should have dimension 2, ',
            'but got dimension {}.'.format(dim),
        )
    batch_size = value.shape[0]
    nA = value.shape[1] # Number of actions
    if isinstance(value, np.ndarray):
        # Greedy action
        exploit = np.argmax(value, axis=1)
        # Randomly select an action
        explore = np.random.choice(nA, size=batch_size, replace=False)
        # Choose between greedy and random
        stack = np.stack((exploit, explore), axis=1)
        # Get indice from bernouli distribution,
        # with probability epsilon to be 1.
        indice = np.random.binomial(n=1, p=epsilon, size=batch_size)
        a = stack[np.arange(batch_size), indice]
    elif isinstance(value, torch.Tensor):
        # Greedy action
        exploit = torch.argmax(value, axis=1)
        # Randomly select an action
        explore = torch.randint(0, nA, size=(batch_size,))
        # Choose between greedy and random
        stack = torch.stack((exploit, explore), axis=1)
        # Get indice from bernouli distribution,
        # with probability epsilon to be 1.
        p = torch.full((batch_size,), fill_value=epsilon, dtype=float)
        indice = torch.bernoulli(p).to(dtype=int)
        a = stack[torch.arange(batch_size), indice]
    else:
        raise TypeError(
            'Unsupported type',
        )

    return a


if __name__ == '__main__':
    vn = np.array([[22,5,3,5,1,3,2], [4,8,9,1,2,4,5]])
    vt = torch.tensor([[6,3,6,7],[3,6,8,10]])
    nc = np.zeros((7,7))
    tc = torch.zeros((4,4))
    eps = 0.5
    for i in range(100000):
        an = Epsilon_Greedy(vn, eps)
        at = Epsilon_Greedy(vt, eps)
        nc[tuple(an)] += 1.
        tc[tuple(at)] += 1.
    nc /= 100000.
    tc /= 100000.
    print(nc.sum(axis=0))
    print(nc.sum(axis=1))
    print(tc.sum(dim=0))
    print(tc.sum(dim=1))
