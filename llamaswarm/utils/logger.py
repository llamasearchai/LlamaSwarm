"""
Logger utility for tracking experiment metrics.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger for tracking metrics during experiments.
    
    Parameters
    ----------
    log_dir : str
        Directory to save logs
    experiment_name : str, optional
        Name of the experiment
    use_tensorboard : bool, optional
        Whether to use TensorBoard for logging
    use_csv : bool, optional
        Whether to save metrics as CSV files
    use_json : bool, optional
        Whether to save metrics as JSON files
    """
    
    def __init__(
        self,
        log_dir,
        experiment_name=None,
        use_tensorboard=True,
        use_csv=True,
        use_json=True
    ):
        # Set experiment name
        if experiment_name is None:
            self.experiment_name = time.strftime("experiment_%Y%m%d_%H%M%S")
        else:
            self.experiment_name = experiment_name
        
        # Create log directory
        self.log_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard if requested
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        
        # Set other logging options
        self.use_csv = use_csv
        self.use_json = use_json
        
        # Initialize metrics storage
        self.metrics = defaultdict(list)
        self.step_metrics = defaultdict(dict)
        self.latest_metrics = {}
        
        # Create subdirectories
        self.plot_dir = os.path.join(self.log_dir, 'plots')
        self.csv_dir = os.path.join(self.log_dir, 'csv')
        self.json_dir = os.path.join(self.log_dir, 'json')
        
        os.makedirs(self.plot_dir, exist_ok=True)
        if use_csv:
            os.makedirs(self.csv_dir, exist_ok=True)
        if use_json:
            os.makedirs(self.json_dir, exist_ok=True)
        
        # Log initial info
        self._log_start_info()
    
    def _log_start_info(self):
        """Log initial information about the experiment."""
        info = {
            'experiment_name': self.experiment_name,
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'log_dir': self.log_dir,
        }
        
        with open(os.path.join(self.log_dir, 'experiment_info.json'), 'w') as f:
            json.dump(info, f, indent=4)
    
    def log(self, key, value, step=None):
        """
        Log a scalar metric.
        
        Parameters
        ----------
        key : str
            Name of the metric
        value : float or int
            Value of the metric
        step : int, optional
            Step or iteration number
        """
        # Convert values if needed
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        # Store metric
        self.metrics[key].append(value)
        self.latest_metrics[key] = value
        
        # Store step-specific metric if step is provided
        if step is not None:
            self.step_metrics[key][step] = value
            
            # Log to TensorBoard if enabled
            if self.use_tensorboard:
                self.tb_writer.add_scalar(key, value, step)
    
    def log_dict(self, metrics_dict, step=None, prefix=None):
        """
        Log multiple metrics from a dictionary.
        
        Parameters
        ----------
        metrics_dict : dict
            Dictionary of metrics
        step : int, optional
            Step or iteration number
        prefix : str, optional
            Prefix to add to metric names
        """
        for key, value in metrics_dict.items():
            if prefix:
                full_key = f"{prefix}/{key}"
            else:
                full_key = key
            self.log(full_key, value, step)
    
    def log_episode_summary(self, episode, episode_reward, episode_length, losses=None, eval_reward=None):
        """
        Log episode summary metrics.
        
        Parameters
        ----------
        episode : int
            Episode number
        episode_reward : float or list
            Episode reward(s)
        episode_length : int
            Episode length
        losses : dict, optional
            Dictionary of loss values
        eval_reward : float, optional
            Evaluation reward
        """
        # Log episode reward(s)
        if isinstance(episode_reward, (list, np.ndarray)):
            # Multiple agents, log mean and per-agent rewards
            mean_reward = np.mean(episode_reward)
            self.log('episode/mean_reward', mean_reward, episode)
            
            for i, reward in enumerate(episode_reward):
                self.log(f'episode/agent_{i}/reward', reward, episode)
        else:
            # Single reward
            self.log('episode/reward', episode_reward, episode)
        
        # Log episode length
        self.log('episode/length', episode_length, episode)
        
        # Log losses if provided
        if losses is not None:
            for loss_name, loss_value in losses.items():
                self.log(f'loss/{loss_name}', loss_value, episode)
        
        # Log evaluation reward if provided
        if eval_reward is not None:
            if isinstance(eval_reward, (list, np.ndarray)):
                # Multiple agents, log mean and per-agent rewards
                mean_eval = np.mean(eval_reward)
                self.log('eval/mean_reward', mean_eval, episode)
                
                for i, reward in enumerate(eval_reward):
                    self.log(f'eval/agent_{i}/reward', reward, episode)
            else:
                # Single reward
                self.log('eval/reward', eval_reward, episode)
    
    def save_metrics(self):
        """Save metrics to disk."""
        # Save to CSV if enabled
        if self.use_csv:
            metrics_df = pd.DataFrame(self.metrics)
            metrics_df.to_csv(os.path.join(self.csv_dir, 'metrics.csv'), index=False)
            
            # Save step metrics
            for key, step_dict in self.step_metrics.items():
                if step_dict:
                    step_df = pd.DataFrame({'step': list(step_dict.keys()), 
                                           'value': list(step_dict.values())})
                    step_df.to_csv(os.path.join(self.csv_dir, f'{key}.csv'), index=False)
        
        # Save to JSON if enabled
        if self.use_json:
            # Convert numpy values to Python native types
            clean_metrics = {}
            for key, values in self.metrics.items():
                clean_metrics[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                     for v in values]
            
            with open(os.path.join(self.json_dir, 'metrics.json'), 'w') as f:
                json.dump(clean_metrics, f, indent=2)
    
    def plot_metric(self, key, title=None, xlabel='Steps', ylabel=None, window=1, save=True):
        """
        Plot a specific metric.
        
        Parameters
        ----------
        key : str
            Name of the metric to plot
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        window : int, optional
            Window size for smoothing
        save : bool, optional
            Whether to save the plot
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if key not in self.metrics:
            print(f"Warning: Metric '{key}' not found in logged metrics")
            return None
        
        values = self.metrics[key]
        steps = list(range(len(values)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw data with low alpha
        ax.plot(steps, values, alpha=0.3, label='Raw')
        
        # Plot smoothed data if window > 1
        if window > 1 and len(values) > window:
            smoothed = np.convolve(values, np.ones(window) / window, mode='valid')
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, label=f'Smoothed (window={window})')
            ax.legend()
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ylabel else key)
        ax.set_title(title if title else f'{key} over time')
        ax.grid(True, alpha=0.3)
        
        # Save if requested
        if save:
            filename = f"{key.replace('/', '_')}.png"
            fig.savefig(os.path.join(self.plot_dir, filename), dpi=100, bbox_inches='tight')
        
        return fig
    
    def plot_all_metrics(self, window=1, save=True):
        """
        Plot all tracked metrics.
        
        Parameters
        ----------
        window : int, optional
            Window size for smoothing
        save : bool, optional
            Whether to save the plots
            
        Returns
        -------
        list
            List of generated figures
        """
        figures = []
        
        # Group metrics by category (using the first part of the key)
        categories = defaultdict(list)
        for key in self.metrics.keys():
            if '/' in key:
                category, name = key.split('/', 1)
                categories[category].append(key)
            else:
                categories['uncategorized'].append(key)
        
        # Plot each category on a separate figure
        for category, keys in categories.items():
            if len(keys) == 1:
                # Single metric in category, plot normally
                fig = self.plot_metric(keys[0], window=window, save=save)
                if fig:
                    figures.append(fig)
            else:
                # Multiple metrics in category, plot on the same axes
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for key in keys:
                    values = self.metrics[key]
                    steps = list(range(len(values)))
                    
                    # Plot raw data with low alpha if smoothing
                    if window > 1 and len(values) > window:
                        ax.plot(steps, values, alpha=0.15)
                        
                        # Plot smoothed data
                        smoothed = np.convolve(values, np.ones(window) / window, mode='valid')
                        smooth_steps = steps[window-1:]
                        ax.plot(smooth_steps, smoothed, label=key.split('/')[-1])
                    else:
                        ax.plot(steps, values, label=key.split('/')[-1])
                
                # Set labels and title
                ax.set_xlabel('Steps')
                ax.set_title(f'{category.capitalize()} Metrics')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Save if requested
                if save:
                    filename = f"{category}_metrics.png"
                    fig.savefig(os.path.join(self.plot_dir, filename), dpi=100, bbox_inches='tight')
                
                figures.append(fig)
        
        return figures
    
    def close(self):
        """Close the logger and save all data."""
        self.save_metrics()
        
        if self.use_tensorboard:
            self.tb_writer.close()
        
        # Log end info
        end_info = {
            'experiment_name': self.experiment_name,
            'end_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'metrics_summary': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 
                                   'min': float(np.min(v)), 'max': float(np.max(v))} 
                               for k, v in self.metrics.items() if v}
        }
        
        with open(os.path.join(self.log_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(end_info, f, indent=4)
        
        # Generate summary plots
        self.plot_all_metrics()
    
    def __del__(self):
        """Ensure resources are closed when the logger is deleted."""
        try:
            self.close()
        except:
            pass 