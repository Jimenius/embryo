import torch
import numpy as np


def Epsilon_Greedy(
    value: torch.Tensor,
    epsilon: float = 1.
) -> int:
    '''Epsilon Greedy policy to select an action based on values
    
    Args:
        value: Action values
        epsilon: Exploration rate, Epsilon
    
    Returns:
        Action index
    '''

    nA = value.shape[0] # Number of actions
    exploit = np.argmax(value) # Greedy action
    explore = np.random.choice(nA) # Randomly select an action
    a = np.random.choice((exploit, explore), p = (1 - epsilon, epsilon)) # Choose between greedy and random
    return a
