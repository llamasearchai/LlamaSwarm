"""
Base algorithm class for multi-agent reinforcement learning.
"""

import os
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseAlgorithm(ABC):
    """
    Abstract base class for multi-agent reinforcement learning algorithms.

    Parameters
    ----------
    n_agents : int
        Number of agents in the environment
    state_dim : int
        Dimension of the state space
    action_dim : int
        Dimension of the action space
    discrete : bool, optional
        Whether the action space is discrete or continuous
    lr : float, optional
        Learning rate for the optimizer
    gamma : float, optional
        Discount factor for future rewards
    device : str, optional
        Device to use for tensor operations ('cpu' or 'cuda')
    """

    def __init__(
        self,
        n_agents,
        state_dim,
        action_dim,
        discrete=False,
        lr=3e-4,
        gamma=0.99,
        device="cpu",
    ):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.lr = lr
        self.gamma = gamma
        self.device = torch.device(device)

        # Initialize agents and networks
        self.agents = self._init_agents()
        self.optimizer = self._init_optimizer()

        # Tracking metrics
        self.train_info = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "grad_norm": [],
        }

    @abstractmethod
    def _init_agents(self):
        """
        Initialize agents for the algorithm.

        Returns
        -------
        list or object
            Agent(s) for the algorithm
        """
        pass

    @abstractmethod
    def _init_optimizer(self):
        """
        Initialize optimizer(s) for training.

        Returns
        -------
        optimizer or dict
            Optimizer(s) for training
        """
        pass

    @abstractmethod
    def select_action(self, obs, explore=True):
        """
        Select actions for agents based on observations.

        Parameters
        ----------
        obs : list or numpy.ndarray
            Observations for each agent
        explore : bool, optional
            Whether to explore or exploit

        Returns
        -------
        list or numpy.ndarray
            Actions for each agent
        """
        pass

    @abstractmethod
    def update(self, batch):
        """
        Update algorithm parameters using a batch of experiences.

        Parameters
        ----------
        batch : dict
            Batch of experiences (states, actions, rewards, next_states, dones)

        Returns
        -------
        dict
            Dictionary of loss metrics
        """
        pass

    def process_batch(self, batch):
        """
        Process a batch of experiences before updating.

        Parameters
        ----------
        batch : dict
            Batch of experiences

        Returns
        -------
        dict
            Processed batch
        """
        # Default implementation assumes batch is already properly formatted
        return batch

    def compute_returns(self, rewards, dones, bootstrap_values=None):
        """
        Compute returns for a batch of episodes.

        Parameters
        ----------
        rewards : torch.Tensor
            Rewards for each step and agent
        dones : torch.Tensor
            Done flags for each step and agent
        bootstrap_values : torch.Tensor, optional
            Bootstrap values for incomplete episodes

        Returns
        -------
        torch.Tensor
            Returns for each step and agent
        """
        batch_size = rewards.shape[0]
        returns = torch.zeros_like(rewards)

        if bootstrap_values is not None:
            returns[:, -1] = bootstrap_values

        for t in reversed(range(rewards.shape[1] - 1)):
            returns[:, t] = rewards[:, t] + self.gamma * returns[:, t + 1] * (
                1 - dones[:, t]
            )

        return returns

    def save(self, path):
        """
        Save the algorithm's parameters to a file.

        Parameters
        ----------
        path : str
            Path to save the parameters
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            "algorithm_params": {
                "n_agents": self.n_agents,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "discrete": self.discrete,
                "lr": self.lr,
                "gamma": self.gamma,
            },
            "train_info": self.train_info,
        }

        # Add model-specific parameters
        save_dict.update(self._get_save_dict())

        # Save to file
        torch.save(save_dict, path)

    def load(self, path):
        """
        Load the algorithm's parameters from a file.

        Parameters
        ----------
        path : str
            Path to load the parameters from
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load algorithm parameters
        for key, value in checkpoint["algorithm_params"].items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Load training info
        if "train_info" in checkpoint:
            self.train_info = checkpoint["train_info"]

        # Load model-specific parameters
        self._set_load_dict(checkpoint)

    @abstractmethod
    def _get_save_dict(self):
        """
        Get algorithm-specific parameters for saving.

        Returns
        -------
        dict
            Dictionary of parameters to save
        """
        pass

    @abstractmethod
    def _set_load_dict(self, checkpoint):
        """
        Set algorithm-specific parameters from loaded checkpoint.

        Parameters
        ----------
        checkpoint : dict
            Loaded checkpoint
        """
        pass
