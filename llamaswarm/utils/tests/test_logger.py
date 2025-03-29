"""
Unit tests for logger utility.
"""

import unittest
import os
import shutil
import json
import pandas as pd
import tempfile
from unittest.mock import patch

from llamaswarm.utils.logger import Logger


class TestLogger(unittest.TestCase):
    """Test cases for logger utility."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.experiment_name = "test_experiment"
        self.log_dir = os.path.join(self.test_dir, "logs")
        
        # Create the logger
        self.logger = Logger(
            log_dir=self.log_dir,
            experiment_name=self.experiment_name,
            use_tensorboard=False,
            use_csv=True,
            use_json=True
        )
    
    def tearDown(self):
        """Clean up after each test."""
        # Close the logger
        self.logger.close()
        
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test logger initialization."""
        # Check that log directories were created
        experiment_dir = os.path.join(self.log_dir, self.experiment_name)
        self.assertTrue(os.path.exists(experiment_dir))
        
        # Check that experiment info was saved
        info_file = os.path.join(experiment_dir, "experiment_info.json")
        self.assertTrue(os.path.exists(info_file))
        
        # Check that info file contains expected data
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        self.assertEqual(info["experiment_name"], self.experiment_name)
        self.assertIn("start_time", info)
    
    def test_log_scalar(self):
        """Test logging scalar values."""
        # Log some values
        self.logger.log("test_metric", 1.0, step=1)
        self.logger.log("test_metric", 2.0, step=2)
        self.logger.log("test_metric", 3.0, step=3)
        
        # Check that values were stored correctly
        self.assertEqual(len(self.logger.metrics["test_metric"]), 3)
        self.assertEqual(self.logger.metrics["test_metric"][0]["value"], 1.0)
        self.assertEqual(self.logger.metrics["test_metric"][1]["value"], 2.0)
        self.assertEqual(self.logger.metrics["test_metric"][2]["value"], 3.0)
        
        # Check that steps were stored correctly
        self.assertEqual(self.logger.metrics["test_metric"][0]["step"], 1)
        self.assertEqual(self.logger.metrics["test_metric"][1]["step"], 2)
        self.assertEqual(self.logger.metrics["test_metric"][2]["step"], 3)
    
    def test_log_dict(self):
        """Test logging multiple metrics from a dictionary."""
        # Log a dictionary of values
        metrics_dict = {
            "metric1": 1.0,
            "metric2": 2.0,
            "metric3": 3.0
        }
        
        self.logger.log_dict(metrics_dict, step=5)
        
        # Check that all metrics were stored correctly
        self.assertEqual(len(self.logger.metrics["metric1"]), 1)
        self.assertEqual(len(self.logger.metrics["metric2"]), 1)
        self.assertEqual(len(self.logger.metrics["metric3"]), 1)
        
        self.assertEqual(self.logger.metrics["metric1"][0]["value"], 1.0)
        self.assertEqual(self.logger.metrics["metric2"][0]["value"], 2.0)
        self.assertEqual(self.logger.metrics["metric3"][0]["value"], 3.0)
        
        # Check that steps were stored correctly
        self.assertEqual(self.logger.metrics["metric1"][0]["step"], 5)
        self.assertEqual(self.logger.metrics["metric2"][0]["step"], 5)
        self.assertEqual(self.logger.metrics["metric3"][0]["step"], 5)
    
    def test_log_episode_summary(self):
        """Test logging episode summaries."""
        # Create some episode data
        episode_rewards = [1.0, 2.0, 3.0]
        episode_steps = 100
        episode_metrics = {
            "loss": 0.5,
            "value_loss": 0.3,
            "policy_loss": 0.2
        }
        
        # Log the episode summary
        self.logger.log_episode_summary(
            episode=1,
            rewards=episode_rewards,
            steps=episode_steps,
            metrics=episode_metrics
        )
        
        # Check that episode metrics were stored correctly
        self.assertEqual(len(self.logger.metrics["episode_reward"]), 1)
        self.assertEqual(len(self.logger.metrics["episode_step"]), 1)
        self.assertEqual(len(self.logger.metrics["loss"]), 1)
        self.assertEqual(len(self.logger.metrics["value_loss"]), 1)
        self.assertEqual(len(self.logger.metrics["policy_loss"]), 1)
        
        # Check values
        self.assertEqual(self.logger.metrics["episode_reward"][0]["value"], sum(episode_rewards))
        self.assertEqual(self.logger.metrics["episode_step"][0]["value"], episode_steps)
        self.assertEqual(self.logger.metrics["loss"][0]["value"], 0.5)
        self.assertEqual(self.logger.metrics["value_loss"][0]["value"], 0.3)
        self.assertEqual(self.logger.metrics["policy_loss"][0]["value"], 0.2)
        
        # Check steps
        self.assertEqual(self.logger.metrics["episode_reward"][0]["step"], 1)
        self.assertEqual(self.logger.metrics["episode_step"][0]["step"], 1)
    
    def test_save_metrics(self):
        """Test saving metrics to disk."""
        # Log some metrics
        self.logger.log("metric1", 1.0, step=1)
        self.logger.log("metric1", 2.0, step=2)
        self.logger.log("metric2", 3.0, step=1)
        
        # Save metrics
        self.logger.save_metrics()
        
        # Check that files were created
        experiment_dir = os.path.join(self.log_dir, self.experiment_name)
        csv_file = os.path.join(experiment_dir, "metrics.csv")
        json_file = os.path.join(experiment_dir, "metrics.json")
        
        self.assertTrue(os.path.exists(csv_file))
        self.assertTrue(os.path.exists(json_file))
        
        # Check CSV content
        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 3)  # Three metrics logged in total
        self.assertIn("metric", df.columns)
        self.assertIn("value", df.columns)
        self.assertIn("step", df.columns)
        self.assertIn("timestamp", df.columns)
        
        # Check JSON content
        with open(json_file, 'r') as f:
            metrics_json = json.load(f)
        
        self.assertIn("metric1", metrics_json)
        self.assertIn("metric2", metrics_json)
        self.assertEqual(len(metrics_json["metric1"]), 2)
        self.assertEqual(len(metrics_json["metric2"]), 1)
    
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.figure")
    def test_plot_metric(self, mock_figure, mock_savefig):
        """Test plotting a specific metric."""
        # Log some metrics
        self.logger.log("test_metric", 1.0, step=1)
        self.logger.log("test_metric", 2.0, step=2)
        self.logger.log("test_metric", 3.0, step=3)
        
        # Plot the metric
        self.logger.plot_metric("test_metric", save=True)
        
        # Check that figure and savefig were called
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once()
    
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.figure")
    def test_plot_all_metrics(self, mock_figure, mock_savefig):
        """Test plotting all metrics."""
        # Log some metrics
        self.logger.log("metric1", 1.0, step=1)
        self.logger.log("metric1", 2.0, step=2)
        self.logger.log("metric2", 3.0, step=1)
        self.logger.log("metric2", 4.0, step=2)
        
        # Plot all metrics
        self.logger.plot_all_metrics(save=True)
        
        # Check that figure and savefig were called
        self.assertEqual(mock_figure.call_count, 2)  # Two different metrics
        self.assertEqual(mock_savefig.call_count, 2)
    
    def test_close(self):
        """Test closing the logger."""
        # Log some metrics
        self.logger.log("metric1", 1.0, step=1)
        
        # Close the logger
        self.logger.close()
        
        # Check that metrics were saved
        experiment_dir = os.path.join(self.log_dir, self.experiment_name)
        csv_file = os.path.join(experiment_dir, "metrics.csv")
        json_file = os.path.join(experiment_dir, "metrics.json")
        
        self.assertTrue(os.path.exists(csv_file))
        self.assertTrue(os.path.exists(json_file))
        
        # Check that end_time was added to experiment info
        info_file = os.path.join(experiment_dir, "experiment_info.json")
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        self.assertIn("end_time", info)


if __name__ == '__main__':
    unittest.main() 