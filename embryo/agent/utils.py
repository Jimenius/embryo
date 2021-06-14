'''

Created by Minhui Li on April 12, 2021
'''


from typing import Any, Callable, Dict, Optional, Union

from tianshou.exploration import OUNoise

from embryo.brain.central.base import Central
from embryo.brain.memory.linked import LinkedMemory
from embryo.ion import Ion
from embryo.limbs.limbs import Limbs
from embryo.utils.registry import Registry


def test_episode(
    central: Central,
    limbs: Limbs,
    step: int = 0,
    num_episode: int = 1,
    test_fn: Optional[Callable[[Any], None]] = None,
) -> Union[Dict, Ion]:
    '''
    '''

    limbs.reset()
    central.eval()
    if test_fn:
        test_fn(step)
    result = limbs.interact(num_episode=num_episode)

    return result