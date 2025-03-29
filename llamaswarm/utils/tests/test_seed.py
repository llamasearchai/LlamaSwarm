"""
Unit tests for seed utility.
"""

import unittest
import random
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from llamaswarm.utils.seed import set_seed, get_random_seed


class TestSeed(unittest.TestCase):
    """Test cases for seed utility."""
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed ensures reproducible random numbers."""
        # Set a specific seed
        seed_value = 42
        set_seed(seed_value)
        
        # Get random numbers from different libraries
        random_values1 = [random.random() for _ in range(5)]
        numpy_values1 = np.random.rand(5).tolist()
        torch_values1 = torch.rand(5).tolist()
        
        # Reset seed to the same value
        set_seed(seed_value)
        
        # Get random numbers again
        random_values2 = [random.random() for _ in range(5)]
        numpy_values2 = np.random.rand(5).tolist()
        torch_values2 = torch.rand(5).tolist()
        
        # Check that the random numbers are the same
        self.assertEqual(random_values1, random_values2)
        np.testing.assert_array_almost_equal(numpy_values1, numpy_values2)
        torch.testing.assert_close(torch.tensor(torch_values1), torch.tensor(torch_values2))
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different random numbers."""
        # Set the first seed
        set_seed(42)
        
        # Get random numbers
        random_values1 = [random.random() for _ in range(5)]
        numpy_values1 = np.random.rand(5).tolist()
        torch_values1 = torch.rand(5).tolist()
        
        # Set a different seed
        set_seed(43)
        
        # Get random numbers again
        random_values2 = [random.random() for _ in range(5)]
        numpy_values2 = np.random.rand(5).tolist()
        torch_values2 = torch.rand(5).tolist()
        
        # Check that the random numbers are different
        self.assertNotEqual(random_values1, random_values2)
        self.assertFalse(np.allclose(numpy_values1, numpy_values2))
        self.assertFalse(torch.allclose(torch.tensor(torch_values1), torch.tensor(torch_values2)))
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.manual_seed_all')
    def test_cuda_seed_setting(self, mock_cuda_seed, mock_is_available):
        """Test that CUDA seeds are set when use_cuda is True."""
        # Call set_seed with use_cuda=True
        set_seed(42, use_cuda=True)
        
        # Check that CUDA seed was set
        mock_cuda_seed.assert_called_once_with(42)
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.cuda.manual_seed_all')
    def test_cuda_seed_not_available(self, mock_cuda_seed, mock_is_available):
        """Test behavior when CUDA is not available but use_cuda is True."""
        # Call set_seed with use_cuda=True
        set_seed(42, use_cuda=True)
        
        # Check that CUDA seed was not called
        mock_cuda_seed.assert_not_called()
    
    @patch('random.seed')
    @patch('numpy.random.seed')
    @patch('torch.manual_seed')
    def test_all_seed_functions_called(self, mock_torch_seed, mock_np_seed, mock_random_seed):
        """Test that all seed functions are called with the correct seed."""
        # Call set_seed
        set_seed(42)
        
        # Check that all seed functions were called with the correct seed
        mock_random_seed.assert_called_once_with(42)
        mock_np_seed.assert_called_once_with(42)
        mock_torch_seed.assert_called_once_with(42)
    
    @patch('time.time', return_value=1234567890.123456)
    def test_get_random_seed(self, mock_time):
        """Test that get_random_seed returns a positive integer based on time."""
        # Get a random seed
        seed = get_random_seed()
        
        # Check that the seed is a positive integer within int32 range
        self.assertIsInstance(seed, int)
        self.assertGreater(seed, 0)
        self.assertLess(seed, 2**31 - 1)  # int32 max value
        
        # Check that the seed is based on the mocked time
        expected_seed = int(1234567890.123456 * 1000) % (2**31 - 1)
        self.assertEqual(seed, expected_seed)


if __name__ == '__main__':
    unittest.main() 