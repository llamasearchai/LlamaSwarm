"""
Mathematical utility functions for multi-agent reinforcement learning.
"""

import numpy as np
import torch


def normalize(data, epsilon=1e-8):
    """
    Normalize data to have zero mean and unit variance.
    
    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Data to normalize
    epsilon : float, optional
        Small constant to avoid division by zero
        
    Returns
    -------
    numpy.ndarray or torch.Tensor
        Normalized data
    """
    if isinstance(data, torch.Tensor):
        mean = torch.mean(data)
        std = torch.std(data)
        return (data - mean) / (std + epsilon)
    else:
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + epsilon)


def standardize(data, min_val=None, max_val=None, epsilon=1e-8):
    """
    Standardize data to a specified range, default [0, 1].
    
    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Data to standardize
    min_val : float, optional
        Minimum value after standardization
    max_val : float, optional
        Maximum value after standardization
    epsilon : float, optional
        Small constant to avoid division by zero
        
    Returns
    -------
    numpy.ndarray or torch.Tensor
        Standardized data
    """
    if min_val is None:
        min_val = 0.0
    if max_val is None:
        max_val = 1.0
        
    if isinstance(data, torch.Tensor):
        data_min = torch.min(data)
        data_max = torch.max(data)
        normalized = (data - data_min) / (data_max - data_min + epsilon)
        return normalized * (max_val - min_val) + min_val
    else:
        data_min = np.min(data)
        data_max = np.max(data)
        normalized = (data - data_min) / (data_max - data_min + epsilon)
        return normalized * (max_val - min_val) + min_val


def discount_rewards(rewards, gamma, masks=None):
    """
    Calculate discounted cumulative rewards.
    
    Parameters
    ----------
    rewards : numpy.ndarray or torch.Tensor
        Rewards sequence
    gamma : float
        Discount factor
    masks : numpy.ndarray or torch.Tensor, optional
        Masks for valid timesteps (1 for valid, 0 for invalid)
        
    Returns
    -------
    numpy.ndarray or torch.Tensor
        Discounted cumulative rewards
    """
    if isinstance(rewards, torch.Tensor):
        returns = torch.zeros_like(rewards)
        running_returns = 0
        
        for t in reversed(range(len(rewards))):
            if masks is not None:
                running_returns = rewards[t] + gamma * running_returns * masks[t]
            else:
                running_returns = rewards[t] + gamma * running_returns
            returns[t] = running_returns
            
        return returns
    else:
        returns = np.zeros_like(rewards)
        running_returns = 0
        
        for t in reversed(range(len(rewards))):
            if masks is not None:
                running_returns = rewards[t] + gamma * running_returns * masks[t]
            else:
                running_returns = rewards[t] + gamma * running_returns
            returns[t] = running_returns
            
        return returns


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Parameters
    ----------
    rewards : numpy.ndarray or torch.Tensor
        Rewards for each step
    values : numpy.ndarray or torch.Tensor
        Value estimates for each step
    next_values : numpy.ndarray or torch.Tensor
        Value estimates for next steps
    dones : numpy.ndarray or torch.Tensor
        Done flags for each step
    gamma : float, optional
        Discount factor
    lambda_ : float, optional
        GAE lambda parameter
        
    Returns
    -------
    numpy.ndarray or torch.Tensor
        Advantage estimates
    """
    if isinstance(rewards, torch.Tensor):
        gae = torch.zeros_like(rewards)
        masks = 1.0 - dones
        deltas = rewards + gamma * next_values * masks - values
        
        running_gae = 0
        for t in reversed(range(len(rewards))):
            running_gae = deltas[t] + gamma * lambda_ * masks[t] * running_gae
            gae[t] = running_gae
            
        return gae
    else:
        gae = np.zeros_like(rewards)
        masks = 1.0 - dones
        deltas = rewards + gamma * next_values * masks - values
        
        running_gae = 0
        for t in reversed(range(len(rewards))):
            running_gae = deltas[t] + gamma * lambda_ * masks[t] * running_gae
            gae[t] = running_gae
            
        return gae


def explained_variance(y_pred, y_true):
    """
    Calculate explained variance.
    
    Parameters
    ----------
    y_pred : numpy.ndarray or torch.Tensor
        Predicted values
    y_true : numpy.ndarray or torch.Tensor
        True values
        
    Returns
    -------
    float
        Explained variance score
    """
    if isinstance(y_pred, torch.Tensor):
        var_y = torch.var(y_true)
        if var_y == 0:
            return 0.0
        return 1.0 - torch.var(y_true - y_pred) / var_y
    else:
        var_y = np.var(y_true)
        if var_y == 0:
            return 0.0
        return 1.0 - np.var(y_true - y_pred) / var_y 