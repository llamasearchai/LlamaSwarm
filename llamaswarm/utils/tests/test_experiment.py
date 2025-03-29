"""
Unit tests for experiment manager utility.
"""

import unittest
import os
import shutil
import json
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock

from llamaswarm.utils.experiment import ExperimentManager


class TestExperimentManager(unittest.TestCase):
    """Test cases for experiment manager utility."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for experiments
        self.test_dir = tempfile.mkdtemp()
        self.experiment_name = "test_experiment"
        self.base_config = {
            "algorithm": "MAPPO",
            "environment": "CooperativeNavigation",
            "num_agents": 3,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 64
        }
        
        # Create the experiment manager
        self.experiment_manager = ExperimentManager(
            base_dir=self.test_dir,
            experiment_name=self.experiment_name,
            config=self.base_config
        )
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test experiment manager initialization."""
        # Check that experiment directory was created
        experiment_dir = os.path.join(self.test_dir, self.experiment_name)
        self.assertTrue(os.path.exists(experiment_dir))
        
        # Check that experiment info was saved
        info_file = os.path.join(experiment_dir, "experiment_info.json")
        self.assertTrue(os.path.exists(info_file))
        
        # Check that info file contains expected data
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        self.assertEqual(info["experiment_name"], self.experiment_name)
        self.assertIn("creation_time", info)
        
        # Check that base config was saved
        config_file = os.path.join(experiment_dir, "base_config.json")
        self.assertTrue(os.path.exists(config_file))
        
        # Check that config file contains expected data
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.assertEqual(config, self.base_config)
    
    def test_load_config_json(self):
        """Test loading configuration from a JSON file."""
        # Create a temporary config file
        config_data = {
            "algorithm": "PPO",
            "environment": "GridWorld",
            "num_agents": 5
        }
        
        config_file = os.path.join(self.test_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load the config
        self.experiment_manager.load_config(config_file)
        
        # Check that config was updated
        self.assertEqual(self.experiment_manager.config["algorithm"], "PPO")
        self.assertEqual(self.experiment_manager.config["environment"], "GridWorld")
        self.assertEqual(self.experiment_manager.config["num_agents"], 5)
        
        # Check that original config values are preserved
        self.assertEqual(self.experiment_manager.config["learning_rate"], 0.001)
        self.assertEqual(self.experiment_manager.config["gamma"], 0.99)
        self.assertEqual(self.experiment_manager.config["batch_size"], 64)
    
    def test_save_config(self):
        """Test saving configuration to a file."""
        # Create a file path for saving
        save_path = os.path.join(self.test_dir, "saved_config.json")
        
        # Save the config
        self.experiment_manager.save_config(save_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Check that file contains expected data
        with open(save_path, 'r') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config, self.base_config)
    
    def test_update_config(self):
        """Test updating the base configuration."""
        # Update config
        update_data = {
            "learning_rate": 0.0005,
            "new_param": "value"
        }
        
        self.experiment_manager.update_config(update_data)
        
        # Check that config was updated
        self.assertEqual(self.experiment_manager.config["learning_rate"], 0.0005)
        self.assertEqual(self.experiment_manager.config["new_param"], "value")
        
        # Check that original config values are preserved
        self.assertEqual(self.experiment_manager.config["algorithm"], "MAPPO")
        self.assertEqual(self.experiment_manager.config["environment"], "CooperativeNavigation")
        self.assertEqual(self.experiment_manager.config["num_agents"], 3)
        self.assertEqual(self.experiment_manager.config["gamma"], 0.99)
        self.assertEqual(self.experiment_manager.config["batch_size"], 64)
    
    def test_create_run(self):
        """Test creating a new run."""
        # Create a run
        run_config = {"learning_rate": 0.0005}
        run_id = self.experiment_manager.create_run(run_config)
        
        # Check that run directory was created
        run_dir = os.path.join(self.test_dir, self.experiment_name, "runs", run_id)
        self.assertTrue(os.path.exists(run_dir))
        
        # Check that run config was saved
        config_file = os.path.join(run_dir, "config.json")
        self.assertTrue(os.path.exists(config_file))
        
        # Check that config file contains expected data
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check that run config has updated and original values
        self.assertEqual(config["learning_rate"], 0.0005)
        self.assertEqual(config["algorithm"], "MAPPO")
        self.assertEqual(config["environment"], "CooperativeNavigation")
        self.assertEqual(config["num_agents"], 3)
        self.assertEqual(config["gamma"], 0.99)
        self.assertEqual(config["batch_size"], 64)
    
    @patch("llamaswarm.utils.experiment.ExperimentManager.run_experiment")
    def test_grid_search(self, mock_run_experiment):
        """Test performing a grid search over parameters."""
        # Mock the run_experiment method to return some results
        mock_run_experiment.return_value = {"reward": 10.0, "success_rate": 0.8}
        
        # Define grid search parameters
        param_grid = {
            "learning_rate": [0.001, 0.0005],
            "batch_size": [32, 64]
        }
        
        # Run grid search
        results = self.experiment_manager.grid_search(param_grid)
        
        # Check that run_experiment was called the correct number of times
        self.assertEqual(mock_run_experiment.call_count, 4)  # 2x2 parameter combinations
        
        # Check that results have the expected format
        self.assertEqual(len(results), 4)
        
        # Check that each result has the expected keys
        for result in results:
            self.assertIn("run_id", result)
            self.assertIn("params", result)
            self.assertIn("learning_rate", result["params"])
            self.assertIn("batch_size", result["params"])
            self.assertIn("results", result)
            self.assertEqual(result["results"]["reward"], 10.0)
            self.assertEqual(result["results"]["success_rate"], 0.8)
    
    @patch("pandas.DataFrame.to_csv")
    def test_save_summary(self, mock_to_csv):
        """Test saving a summary of all runs."""
        # Create some run data
        self.experiment_manager.runs = [
            {
                "run_id": "run_1",
                "params": {"learning_rate": 0.001, "batch_size": 32},
                "results": {"reward": 10.0, "success_rate": 0.8}
            },
            {
                "run_id": "run_2",
                "params": {"learning_rate": 0.0005, "batch_size": 64},
                "results": {"reward": 12.0, "success_rate": 0.9}
            }
        ]
        
        # Save summary
        summary_path = os.path.join(self.test_dir, "summary.csv")
        self.experiment_manager.save_summary(summary_path)
        
        # Check that to_csv was called
        mock_to_csv.assert_called_once_with(summary_path)
    
    def test_get_best_run(self):
        """Test retrieving the best run based on a metric."""
        # Create some run data
        self.experiment_manager.runs = [
            {
                "run_id": "run_1",
                "params": {"learning_rate": 0.001, "batch_size": 32},
                "results": {"reward": 10.0, "success_rate": 0.8}
            },
            {
                "run_id": "run_2",
                "params": {"learning_rate": 0.0005, "batch_size": 64},
                "results": {"reward": 12.0, "success_rate": 0.7}
            },
            {
                "run_id": "run_3",
                "params": {"learning_rate": 0.0001, "batch_size": 128},
                "results": {"reward": 11.0, "success_rate": 0.9}
            }
        ]
        
        # Get best run by reward (higher is better)
        best_run_reward = self.experiment_manager.get_best_run("reward", higher_is_better=True)
        self.assertEqual(best_run_reward["run_id"], "run_2")
        self.assertEqual(best_run_reward["results"]["reward"], 12.0)
        
        # Get best run by success_rate (higher is better)
        best_run_success = self.experiment_manager.get_best_run("success_rate", higher_is_better=True)
        self.assertEqual(best_run_success["run_id"], "run_3")
        self.assertEqual(best_run_success["results"]["success_rate"], 0.9)
        
        # Test with higher_is_better=False
        worst_run_reward = self.experiment_manager.get_best_run("reward", higher_is_better=False)
        self.assertEqual(worst_run_reward["run_id"], "run_1")
        self.assertEqual(worst_run_reward["results"]["reward"], 10.0)
    
    def test_get_run_summary(self):
        """Test retrieving a summary of all runs."""
        # Create some run data
        self.experiment_manager.runs = [
            {
                "run_id": "run_1",
                "params": {"learning_rate": 0.001, "batch_size": 32},
                "results": {"reward": 10.0, "success_rate": 0.8}
            },
            {
                "run_id": "run_2",
                "params": {"learning_rate": 0.0005, "batch_size": 64},
                "results": {"reward": 12.0, "success_rate": 0.7}
            }
        ]
        
        # Get summary
        summary = self.experiment_manager.get_run_summary()
        
        # Check that summary has the expected format
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 2)
        
        # Check that summary contains expected columns
        expected_columns = ["run_id", "learning_rate", "batch_size", "reward", "success_rate"]
        for col in expected_columns:
            self.assertIn(col, summary.columns)
        
        # Check that summary contains expected data
        self.assertEqual(summary.iloc[0]["run_id"], "run_1")
        self.assertEqual(summary.iloc[0]["learning_rate"], 0.001)
        self.assertEqual(summary.iloc[0]["batch_size"], 32)
        self.assertEqual(summary.iloc[0]["reward"], 10.0)
        self.assertEqual(summary.iloc[0]["success_rate"], 0.8)
        
        self.assertEqual(summary.iloc[1]["run_id"], "run_2")
        self.assertEqual(summary.iloc[1]["learning_rate"], 0.0005)
        self.assertEqual(summary.iloc[1]["batch_size"], 64)
        self.assertEqual(summary.iloc[1]["reward"], 12.0)
        self.assertEqual(summary.iloc[1]["success_rate"], 0.7)


if __name__ == '__main__':
    unittest.main() 