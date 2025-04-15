"""
Replay buffer implementation for experience storage and sampling.
"""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.

    Parameters
    ----------
    capacity : int
        Maximum size of the buffer
    batch_size : int
        Size of batches to sample
    state_dim : int
        Dimension of the state space
    action_dim : int
        Dimension of the action space
    n_agents : int
        Number of agents in the environment
    device : str, optional
        Device to use for tensor operations ('cpu', 'cuda')
    """

    def __init__(
        self, capacity, batch_size, state_dim, action_dim, n_agents, device="cpu"
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = torch.device(device)

        self.buffer = deque(maxlen=capacity)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, info=None):
        """
        Add an experience to the buffer.

        Parameters
        ----------
        state : numpy.ndarray
            Current state observation, shape (n_agents, state_dim)
        action : numpy.ndarray
            Action taken, shape (n_agents, action_dim)
        reward : numpy.ndarray
            Reward received, shape (n_agents,)
        next_state : numpy.ndarray
            Next state observation, shape (n_agents, state_dim)
        done : numpy.ndarray
            Done flags, shape (n_agents,)
        info : dict, optional
            Additional information from the environment
        """
        experience = (state, action, reward, next_state, done)
        if info is not None:
            experience = experience + (info,)

        self.buffer.append(experience)
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=None):
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int, optional
            Size of the batch to sample. If None, use self.batch_size

        Returns
        -------
        dict
            Dictionary containing batched experiences
        """
        if batch_size is None:
            batch_size = self.batch_size

        batch_size = min(batch_size, self.size)
        batch = random.sample(self.buffer, batch_size)

        # Separate experiences into components
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        batch_dict = {
            "states": states_tensor,
            "actions": actions_tensor,
            "rewards": rewards_tensor,
            "next_states": next_states_tensor,
            "dones": dones_tensor,
        }

        # Add info if available
        if len(batch[0]) > 5:
            infos = [exp[5] for exp in batch]
            batch_dict["infos"] = infos

        return batch_dict

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns
        -------
        int
            Current size of buffer
        """
        return self.size
