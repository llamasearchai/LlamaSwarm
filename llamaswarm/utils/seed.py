"""
Utility for setting random seeds across different libraries for reproducibility.
"""

import random
import numpy as np
import torch
import os
import gym


def set_seed(seed, use_cuda=False):
    """
    Set random seeds for reproducibility across different libraries.
    
    Parameters
    ----------
    seed : int
        Random seed to use
    use_cuda : bool, optional
        Whether to set CUDA seeds as well
        
    Returns
    -------
    int
        The seed that was set
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA seed if requested and available
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Additional CUDA settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variables for potential OpenMP threads
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set Gym seed (may fail if Gym is not installed)
    try:
        gym.utils.seeding.np_random(seed)
    except:
        pass
    
    return seed


def get_random_seed():
    """
    Generate a random seed based on current time.
    
    Returns
    -------
    int
        A randomly generated seed
    """
    import time
    
    # Generate a seed based on current time
    t = int(time.time() * 1000.0)
    seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
    return seed & 0x7FFFFFFF  # Ensure it's a positive integer within int32 range 