"""
Environment module for multi-agent reinforcement learning.
"""

# Import basic environment components
from .base_env import MultiAgentEnv
from .competitive import CompetitiveForaging, TagChase, TerritoryControl

# Import specific environment types
from .cooperative import CooperativeNavigation, PredatorPrey, ResourceAllocation
from .mixed import SocialDilemma, TeamCompetition, TradeMarket

# Export all environments
__all__ = [
    # Base Environment
    "MultiAgentEnv",
    # Cooperative Environments
    "CooperativeNavigation",
    "PredatorPrey",
    "ResourceAllocation",
    # Competitive Environments
    "CompetitiveForaging",
    "TagChase",
    "TerritoryControl",
    # Mixed Environments
    "TradeMarket",
    "TeamCompetition",
    "SocialDilemma",
]
