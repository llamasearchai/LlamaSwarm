"""
LlamaSwarm: A simulator for multi-agent reinforcement learning.

This package provides tools for implementing, training, and evaluating
multi-agent reinforcement learning algorithms in various environments.
"""

__version__ = "0.1.0"

from . import core
from . import algorithms
from . import environments
from . import utils
from . import visualization

__all__ = [
    "core",
    "algorithms",
    "environments",
    "utils",
    "visualization"
] 