"""
Algorithms module for multi-agent reinforcement learning.
"""

# Import algorithm components
from .base_algorithm import BaseAlgorithm

# Import specific algorithms
from .policy_gradient import MAPPO, MADDPG
from .value_decomposition import QMIX, VDN
from .communication import CommNet, TarMAC
from .meta_learning import MAML

# Export all algorithms
__all__ = [
    # Base Algorithm
    'BaseAlgorithm',
    
    # Policy Gradient Algorithms
    'MAPPO',
    'MADDPG',
    
    # Value Decomposition Algorithms
    'QMIX',
    'VDN',
    
    # Communication Algorithms
    'CommNet',
    'TarMAC',
    
    # Meta-Learning Algorithms
    'MAML'
] 