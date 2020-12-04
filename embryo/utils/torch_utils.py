'''PyTorch utilities

Created by Minhui Li on September 23, 2020
Functions: to_tensor
'''

import numpy as np
import torch
from typing import Union


def to_tensor(
    x: Union[dict, list, tuple, np.ndarray, torch.Tensor],
    device: Union[str, torch.device] = 'cpu',
) -> Union[dict, list, torch.Tensor]:
    '''Convert data into torch tensors

    Args:
    x: Union[dict, list, tuple, np.ndarray, torch.Tensor]
        The data to be converted
    device: Union[str, torch.device]
        device to be sent to
    
    Returns:
        A tensor or a structure with tensor inside
    '''

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
        return x
    elif isinstance(x, torch.Tensor): 
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_tensor(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [to_tensor(e, device) for e in x]
    else:
        raise TypeError('Object type ({}) not supported'.format(type(x)))
