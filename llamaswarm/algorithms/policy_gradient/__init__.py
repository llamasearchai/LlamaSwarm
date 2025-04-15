"""
Policy gradient algorithms for multi-agent reinforcement learning.

These algorithms optimize policies directly using gradient-based methods.
"""

from .maddpg import MADDPG
from .mappo import MAPPO

__all__ = ["MAPPO", "MADDPG"]
