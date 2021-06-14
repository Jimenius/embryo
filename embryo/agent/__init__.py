from embryo.utils.registry import Registry
AGENT_REGISTRY = Registry('Agent')
from embryo.agent.base import Agent
from embryo.agent.offpolicy import OffPolicyAgent