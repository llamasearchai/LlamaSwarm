"""
Unit tests for math utility functions.
"""

import unittest
import numpy as np
import torch

from llamaswarm.utils.math_utils import (
    normalize, 
    standardize, 
    discount_rewards, 
    compute_gae, 
    explained_variance
)


class TestMathUtils(unittest.TestCase):
    """Test cases for math utility functions."""
    
    def test_normalize_numpy(self):
        """Test normalize function with numpy arrays."""
        # Test with 1D array
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize(data)
        
        self.assertEqual(normalized.shape, data.shape)
        self.assertAlmostEqual(normalized.mean(), 0.0, places=6)
        self.assertAlmostEqual(normalized.std(), 1.0, places=6)
        
        # Test with 2D array
        data_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized_2d = normalize(data_2d)
        
        self.assertEqual(normalized_2d.shape, data_2d.shape)
        self.assertAlmostEqual(normalized_2d.mean(), 0.0, places=6)
        self.assertAlmostEqual(normalized_2d.std(), 1.0, places=6)
    
    def test_normalize_torch(self):
        """Test normalize function with torch tensors."""
        # Test with 1D tensor
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize(data)
        
        self.assertEqual(normalized.shape, data.shape)
        self.assertAlmostEqual(normalized.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(normalized.std().item(), 1.0, places=6)
        
        # Test with 2D tensor
        data_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized_2d = normalize(data_2d)
        
        self.assertEqual(normalized_2d.shape, data_2d.shape)
        self.assertAlmostEqual(normalized_2d.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(normalized_2d.std().item(), 1.0, places=6)
    
    def test_standardize_numpy(self):
        """Test standardize function with numpy arrays."""
        # Test with default range [0, 1]
        data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        standardized = standardize(data)
        
        self.assertEqual(standardized.shape, data.shape)
        self.assertAlmostEqual(standardized.min(), 0.0, places=6)
        self.assertAlmostEqual(standardized.max(), 1.0, places=6)
        
        # Test with custom range [-1, 1]
        standardized_custom = standardize(data, min_val=-1.0, max_val=1.0)
        
        self.assertEqual(standardized_custom.shape, data.shape)
        self.assertAlmostEqual(standardized_custom.min(), -1.0, places=6)
        self.assertAlmostEqual(standardized_custom.max(), 1.0, places=6)
    
    def test_standardize_torch(self):
        """Test standardize function with torch tensors."""
        # Test with default range [0, 1]
        data = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        standardized = standardize(data)
        
        self.assertEqual(standardized.shape, data.shape)
        self.assertAlmostEqual(standardized.min().item(), 0.0, places=6)
        self.assertAlmostEqual(standardized.max().item(), 1.0, places=6)
        
        # Test with custom range [-1, 1]
        standardized_custom = standardize(data, min_val=-1.0, max_val=1.0)
        
        self.assertEqual(standardized_custom.shape, data.shape)
        self.assertAlmostEqual(standardized_custom.min().item(), -1.0, places=6)
        self.assertAlmostEqual(standardized_custom.max().item(), 1.0, places=6)
    
    def test_discount_rewards_numpy(self):
        """Test discount_rewards function with numpy arrays."""
        rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        gamma = 0.99
        
        discounted = discount_rewards(rewards, gamma)
        
        self.assertEqual(discounted.shape, rewards.shape)
        
        # Manually calculate expected values
        expected = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + gamma * running_add
            expected[t] = running_add
            
        np.testing.assert_array_almost_equal(discounted, expected)
        
        # Test with masks
        masks = np.array([1.0, 1.0, 1.0, 0.0, 1.0])  # Episode break at index 3
        discounted_masked = discount_rewards(rewards, gamma, masks)
        
        # Manually calculate expected values with masks
        expected_masked = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + gamma * running_add * masks[t]
            expected_masked[t] = running_add
            
        np.testing.assert_array_almost_equal(discounted_masked, expected_masked)
    
    def test_discount_rewards_torch(self):
        """Test discount_rewards function with torch tensors."""
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        gamma = 0.99
        
        discounted = discount_rewards(rewards, gamma)
        
        self.assertEqual(discounted.shape, rewards.shape)
        
        # Manually calculate expected values
        expected = torch.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + gamma * running_add
            expected[t] = running_add
            
        torch.testing.assert_close(discounted, expected)
        
        # Test with masks
        masks = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0])  # Episode break at index 3
        discounted_masked = discount_rewards(rewards, gamma, masks)
        
        # Manually calculate expected values with masks
        expected_masked = torch.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + gamma * running_add * masks[t]
            expected_masked[t] = running_add
            
        torch.testing.assert_close(discounted_masked, expected_masked)
    
    def test_compute_gae_numpy(self):
        """Test compute_gae function with numpy arrays."""
        rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        values = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        next_values = np.array([0.5, 0.5, 0.5, 0.5, 0.0])  # Last next_value is 0
        dones = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # Episode done at index 3
        gamma = 0.99
        lambda_ = 0.95
        
        gae = compute_gae(rewards, values, next_values, dones, gamma, lambda_)
        
        self.assertEqual(gae.shape, rewards.shape)
        
        # Manually calculate expected values
        expected = np.zeros_like(rewards)
        masks = 1.0 - dones
        deltas = rewards + gamma * next_values * masks - values
        
        running_gae = 0
        for t in reversed(range(len(rewards))):
            running_gae = deltas[t] + gamma * lambda_ * masks[t] * running_gae
            expected[t] = running_gae
            
        np.testing.assert_array_almost_equal(gae, expected)
    
    def test_compute_gae_torch(self):
        """Test compute_gae function with torch tensors."""
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        next_values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0])  # Last next_value is 0
        dones = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])  # Episode done at index 3
        gamma = 0.99
        lambda_ = 0.95
        
        gae = compute_gae(rewards, values, next_values, dones, gamma, lambda_)
        
        self.assertEqual(gae.shape, rewards.shape)
        
        # Manually calculate expected values
        expected = torch.zeros_like(rewards)
        masks = 1.0 - dones
        deltas = rewards + gamma * next_values * masks - values
        
        running_gae = 0
        for t in reversed(range(len(rewards))):
            running_gae = deltas[t] + gamma * lambda_ * masks[t] * running_gae
            expected[t] = running_gae
            
        torch.testing.assert_close(gae, expected)
    
    def test_explained_variance_numpy(self):
        """Test explained_variance function with numpy arrays."""
        # Perfect prediction
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        ev = explained_variance(y_pred, y_true)
        self.assertAlmostEqual(ev, 1.0, places=6)
        
        # No correlation
        y_pred_random = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        ev_random = explained_variance(y_pred_random, y_true)
        self.assertLess(ev_random, 1.0)
        
        # Handle zero variance case
        y_const = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        ev_const = explained_variance(y_const, y_const)
        self.assertEqual(ev_const, 0.0)
    
    def test_explained_variance_torch(self):
        """Test explained_variance function with torch tensors."""
        # Perfect prediction
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        ev = explained_variance(y_pred, y_true)
        self.assertAlmostEqual(ev, 1.0, places=6)
        
        # No correlation
        y_pred_random = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0])
        ev_random = explained_variance(y_pred_random, y_true)
        self.assertLess(ev_random, 1.0)
        
        # Handle zero variance case
        y_const = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
        ev_const = explained_variance(y_const, y_const)
        self.assertEqual(ev_const, 0.0)


if __name__ == '__main__':
    unittest.main() 