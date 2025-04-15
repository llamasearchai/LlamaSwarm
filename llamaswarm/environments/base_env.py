"""
Base environment class for multi-agent reinforcement learning.
"""

from abc import ABC, abstractmethod

import gym
import numpy as np


class MultiAgentEnv(ABC):
    """
    Abstract base class for multi-agent environments.

    This class defines the interface for all multi-agent environments
    in LlamaSwarm. It is compatible with gym-like interfaces but extended
    for multiple agents.

    Parameters
    ----------
    n_agents : int
        Number of agents in the environment
    max_steps : int, optional
        Maximum number of steps per episode
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(self, n_agents, max_steps=1000, seed=None):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.current_step = 0

        # Set random seed if provided
        if seed is not None:
            self.seed(seed)

        # Define action and observation spaces for each agent
        self.action_spaces = self._make_action_spaces()
        self.observation_spaces = self._make_observation_spaces()

        # Initialize state for the environment
        self.state = None

    @abstractmethod
    def _make_action_spaces(self):
        """
        Define action spaces for each agent.

        Returns
        -------
        list of gym.Space
            List of action spaces for each agent
        """
        pass

    @abstractmethod
    def _make_observation_spaces(self):
        """
        Define observation spaces for each agent.

        Returns
        -------
        list of gym.Space
            List of observation spaces for each agent
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to an initial state.

        Returns
        -------
        list of numpy.ndarray
            Initial observations for each agent
        """
        self.current_step = 0

    @abstractmethod
    def step(self, actions):
        """
        Take a step in the environment with the given actions.

        Parameters
        ----------
        actions : list
            List of actions for each agent

        Returns
        -------
        tuple
            Tuple of (observations, rewards, dones, infos)
            - observations: list of observations for each agent
            - rewards: list of rewards for each agent
            - dones: list of done flags for each agent
            - infos: list of info dictionaries for each agent
        """
        self.current_step += 1

    @abstractmethod
    def render(self, mode="human"):
        """
        Render the environment.

        Parameters
        ----------
        mode : str, optional
            Rendering mode ('human', 'rgb_array', etc.)

        Returns
        -------
        object
            Rendering result (None for 'human' mode, numpy array for 'rgb_array' mode)
        """
        pass

    def seed(self, seed=None):
        """
        Set random seed for reproducibility.

        Parameters
        ----------
        seed : int, optional
            Random seed

        Returns
        -------
        list
            List of seeds used
        """
        np.random.seed(seed)
        return [seed]

    def close(self):
        """
        Clean up resources used by the environment.
        """
        pass

    def get_state(self):
        """
        Get the full state of the environment.

        Returns
        -------
        object
            Current environment state
        """
        return self.state

    def get_agent_obs(self, agent_id):
        """
        Get observation for a specific agent.

        Parameters
        ----------
        agent_id : int
            ID of the agent

        Returns
        -------
        numpy.ndarray
            Observation for the specified agent
        """
        observations = self.get_obs()
        if agent_id < len(observations):
            return observations[agent_id]
        else:
            raise ValueError(
                f"Agent ID {agent_id} out of range (max {len(observations)-1})"
            )

    @abstractmethod
    def get_obs(self):
        """
        Get observations for all agents.

        Returns
        -------
        list of numpy.ndarray
            List of observations for each agent
        """
        pass

    def get_agent_reward(self, agent_id, state, action):
        """
        Calculate reward for a specific agent.

        Parameters
        ----------
        agent_id : int
            ID of the agent
        state : object
            Current state
        action : object
            Action taken by the agent

        Returns
        -------
        float
            Reward for the specified agent
        """
        rewards = self.get_rewards(
            state, [action if i == agent_id else None for i in range(self.n_agents)]
        )
        return rewards[agent_id]

    @abstractmethod
    def get_rewards(self, state, actions):
        """
        Calculate rewards for all agents.

        Parameters
        ----------
        state : object
            Current state
        actions : list
            List of actions taken by each agent

        Returns
        -------
        list of float
            List of rewards for each agent
        """
        pass

    def get_env_info(self):
        """
        Get information about the environment.

        Returns
        -------
        dict
            Dictionary containing environment information
        """
        return {
            "n_agents": self.n_agents,
            "action_spaces": self.action_spaces,
            "observation_spaces": self.observation_spaces,
            "max_steps": self.max_steps,
            "state_shape": self.get_state_shape(),
            "obs_shapes": [space.shape for space in self.observation_spaces],
            "action_shapes": [
                space.shape if hasattr(space, "shape") else (1,)
                for space in self.action_spaces
            ],
        }

    def get_state_shape(self):
        """
        Get the shape of the state representation.

        Returns
        -------
        tuple
            Shape of the state
        """
        state = self.get_state()
        if hasattr(state, "shape"):
            return state.shape
        else:
            return (1,)  # Default for scalar or non-array states
