"""
Algorithms module for multi-agent reinforcement learning.
"""

# Import algorithm components
from .base_algorithm import BaseAlgorithm
from .communication import CommNet, TarMAC
from .meta_learning import MAML

# Import specific algorithms
from .policy_gradient import MADDPG, MAPPO
from .value_decomposition import QMIX, VDN

# Export all algorithms
__all__ = [
    # Base Algorithm
    "BaseAlgorithm",
    # Policy Gradient Algorithms
    "MAPPO",
    "MADDPG",
    # Value Decomposition Algorithms
    "QMIX",
    "VDN",
    # Communication Algorithms
    "CommNet",
    "TarMAC",
    # Meta-Learning Algorithms
    "MAML",
]
