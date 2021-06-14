from embryo.brain.central.base import Central
from embryo.utils.registry import Registry
CENTRAL_REGISTRY = Registry('Central')
from embryo.brain.central.DQN import DQNCentral
from embryo.brain.central.rand import RandomCentral
from embryo.brain.central.SAC import SACCentral