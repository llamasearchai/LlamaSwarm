"""
Cooperative multi-agent environments.

These environments involve agents working together to achieve common goals.
"""

from .cooperative_navigation import CooperativeNavigation
from .predator_prey import PredatorPrey
from .resource_allocation import ResourceAllocation

__all__ = [
    'CooperativeNavigation',
    'PredatorPrey',
    'ResourceAllocation'
] 