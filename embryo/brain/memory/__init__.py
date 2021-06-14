from embryo.brain.memory.base import Memory
from embryo.utils.registry import Registry
MEMORY_REGISTRY = Registry('memory')
from embryo.brain.memory.linked import LinkedMemory