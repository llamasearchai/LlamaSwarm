"""
Base agent class for multi-agent reinforcement learning.
"""

import numpy as np
import torch


class Agent:
    """
    Base class for all agents in LlamaSwarm.
    
    Parameters
    ----------
    state_dim : int
        Dimension of the state space
    action_dim : int
        Dimension of the action space
    discrete : bool, optional
        Whether the action space is discrete or continuous
    hidden_dim : int, optional
        Dimension of hidden layers in neural networks
    learning_rate : float, optional
        Learning rate for optimization
    device : str, optional
        Device to use for tensor operations ('cpu', 'cuda')
    """
    
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        discrete=False, 
        hidden_dim=256, 
        learning_rate=1e-4,
        device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        self._build_networks()
    
    def _build_networks(self):
        """
        Build neural networks for the agent.
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _build_networks")
    
    def select_action(self, state, explore=True):
        """
        Select an action based on the current state.
        
        Parameters
        ----------
        state : numpy.ndarray
            Current state observation
        explore : bool, optional
            Whether to explore or exploit
            
        Returns
        -------
        numpy.ndarray
            Selected action
        """
        raise NotImplementedError("Subclasses must implement select_action")
    
    def update(self, batch):
        """
        Update the agent's parameters using a batch of experiences.
        
        Parameters
        ----------
        batch : dict
            Batch of experiences (states, actions, rewards, next_states, dones)
            
        Returns
        -------
        dict
            Dictionary of loss metrics
        """
        raise NotImplementedError("Subclasses must implement update")
    
    def save(self, path):
        """
        Save the agent's parameters to a file.
        
        Parameters
        ----------
        path : str
            Path to save the parameters
        """
        raise NotImplementedError("Subclasses must implement save")
    
    def load(self, path):
        """
        Load the agent's parameters from a file.
        
        Parameters
        ----------
        path : str
            Path to load the parameters from
        """
        raise NotImplementedError("Subclasses must implement load") 