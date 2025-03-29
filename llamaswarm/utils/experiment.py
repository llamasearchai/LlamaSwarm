"""
Experiment manager utility for handling experiment configurations and runs.
"""

import os
import json
import yaml
import time
import shutil
import copy
from collections import defaultdict
import itertools
import pandas as pd
import numpy as np

from .logger import Logger
from .seed import set_seed, get_random_seed


class ExperimentManager:
    """
    Manager for handling experiment configurations and runs.
    
    Parameters
    ----------
    base_dir : str
        Base directory for all experiments
    experiment_name : str, optional
        Name of the experiment group
    config : dict, optional
        Base configuration for experiments
    """
    
    def __init__(self, base_dir, experiment_name=None, config=None):
        # Set experiment name
        if experiment_name is None:
            self.experiment_name = time.strftime("experiment_group_%Y%m%d_%H%M%S")
        else:
            self.experiment_name = experiment_name
        
        # Create experiment directory
        self.base_dir = os.path.join(base_dir, self.experiment_name)
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize config
        self.base_config = config or {}
        
        # Track runs
        self.runs = []
        self.run_results = {}
        
        # Save initial info
        self._save_experiment_info()
    
    def _save_experiment_info(self):
        """Save basic experiment information."""
        info = {
            'experiment_name': self.experiment_name,
            'created_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'base_dir': self.base_dir,
        }
        
        with open(os.path.join(self.base_dir, 'experiment_info.json'), 'w') as f:
            json.dump(info, f, indent=4)
    
    def load_config(self, config_path):
        """
        Load configuration from a JSON or YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
            
        Returns
        -------
        dict
            Loaded configuration
        """
        extension = os.path.splitext(config_path)[1].lower()
        
        with open(config_path, 'r') as f:
            if extension == '.json':
                config = json.load(f)
            elif extension in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file extension: {extension}")
        
        self.base_config = config
        
        # Save the loaded config
        config_filename = os.path.basename(config_path)
        shutil.copy(config_path, os.path.join(self.base_dir, config_filename))
        
        return config
    
    def save_config(self, config=None, filename='config.json'):
        """
        Save configuration to a file.
        
        Parameters
        ----------
        config : dict, optional
            Configuration to save. If None, uses base_config
        filename : str, optional
            Filename to save the configuration to
            
        Returns
        -------
        str
            Path to the saved configuration file
        """
        config = config or self.base_config
        save_path = os.path.join(self.base_dir, filename)
        
        extension = os.path.splitext(filename)[1].lower()
        
        with open(save_path, 'w') as f:
            if extension == '.json':
                json.dump(config, f, indent=4)
            elif extension in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False)
            else:
                # Default to JSON
                json.dump(config, f, indent=4)
        
        return save_path
    
    def update_config(self, updates):
        """
        Update base configuration with new values.
        
        Parameters
        ----------
        updates : dict
            Dictionary of configuration updates
            
        Returns
        -------
        dict
            Updated configuration
        """
        # Create a deep copy to avoid modifying the original
        updated_config = copy.deepcopy(self.base_config)
        
        # Apply updates
        for key, value in updates.items():
            if isinstance(value, dict) and key in updated_config and isinstance(updated_config[key], dict):
                # Recursively update nested dictionaries
                updated_config[key].update(value)
            else:
                # Direct update for regular values
                updated_config[key] = value
        
        self.base_config = updated_config
        return updated_config
    
    def create_run(self, run_name=None, config_updates=None, seed=None):
        """
        Create a new run with specified configuration.
        
        Parameters
        ----------
        run_name : str, optional
            Name for this run
        config_updates : dict, optional
            Updates to apply to the base configuration
        seed : int, optional
            Random seed for this run
            
        Returns
        -------
        tuple
            Tuple of (run_id, run_config, run_dir, logger)
        """
        # Generate run name if not provided
        if run_name is None:
            run_id = len(self.runs)
            run_name = f"run_{run_id}"
        else:
            run_id = len(self.runs)
        
        # Create run directory
        run_dir = os.path.join(self.base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Apply configuration updates
        run_config = copy.deepcopy(self.base_config)
        if config_updates:
            for key, value in config_updates.items():
                if isinstance(value, dict) and key in run_config and isinstance(run_config[key], dict):
                    run_config[key].update(value)
                else:
                    run_config[key] = value
        
        # Set seed if provided or generate one
        if seed is None:
            seed = get_random_seed()
        run_config['seed'] = seed
        
        # Save run configuration
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(run_config, f, indent=4)
        
        # Create logger for this run
        logger = Logger(run_dir, experiment_name=run_name)
        
        # Track this run
        run_info = {
            'id': run_id,
            'name': run_name,
            'dir': run_dir,
            'config': run_config,
            'seed': seed,
            'logger': logger,
            'status': 'created',
            'start_time': None,
            'end_time': None,
            'results': {}
        }
        self.runs.append(run_info)
        
        return run_id, run_config, run_dir, logger
    
    def run_experiment(self, run_fn, run_id=None, config=None):
        """
        Execute an experiment run.
        
        Parameters
        ----------
        run_fn : callable
            Function to execute the run. Should accept (config, logger) as arguments
            and return results
        run_id : int, optional
            ID of the run to execute. If None, creates a new run
        config : dict, optional
            Configuration for a new run if run_id is None
            
        Returns
        -------
        dict
            Results from the run
        """
        if run_id is None:
            # Create a new run
            run_id, run_config, run_dir, logger = self.create_run(config_updates=config)
        else:
            # Use existing run
            if run_id >= len(self.runs):
                raise ValueError(f"Run ID {run_id} not found")
            run_info = self.runs[run_id]
            run_config = run_info['config']
            logger = run_info['logger']
        
        # Update run status
        self.runs[run_id]['status'] = 'running'
        self.runs[run_id]['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Set random seed
        set_seed(run_config['seed'])
        
        try:
            # Execute the run
            results = run_fn(run_config, logger)
            
            # Update run status
            self.runs[run_id]['status'] = 'completed'
            self.runs[run_id]['results'] = results
            self.runs[run_id]['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Store results
            self.run_results[run_id] = results
            
            # Save results
            with open(os.path.join(self.runs[run_id]['dir'], 'results.json'), 'w') as f:
                json.dump(results, f, indent=4)
                
            return results
            
        except Exception as e:
            # Update run status
            self.runs[run_id]['status'] = 'failed'
            self.runs[run_id]['error'] = str(e)
            self.runs[run_id]['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Log error
            logger.log('error', str(e))
            
            # Re-raise exception
            raise e
        
        finally:
            # Close logger
            logger.close()
    
    def grid_search(self, param_grid, run_fn, num_seeds=1, base_seed=None):
        """
        Perform a grid search over specified parameters.
        
        Parameters
        ----------
        param_grid : dict
            Dictionary where keys are parameter names and values are lists of values to try
        run_fn : callable
            Function to execute each run
        num_seeds : int, optional
            Number of random seeds to use for each parameter combination
        base_seed : int, optional
            Base seed for generating run seeds
            
        Returns
        -------
        dict
            Dictionary of results for all runs
        """
        # Create parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
        
        # Create seeds
        if base_seed is not None:
            np.random.seed(base_seed)
        seeds = np.random.randint(0, 2**31 - 1, size=num_seeds)
        
        # Create a subdirectory for grid search
        grid_dir = os.path.join(self.base_dir, 'grid_search')
        os.makedirs(grid_dir, exist_ok=True)
        
        # Save grid search configuration
        grid_config = {
            'param_grid': param_grid,
            'num_seeds': num_seeds,
            'base_seed': base_seed,
            'seeds': seeds.tolist(),
            'total_runs': len(param_values) * num_seeds,
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(grid_dir, 'grid_config.json'), 'w') as f:
            json.dump(grid_config, f, indent=4)
        
        # Execute runs
        results = []
        for i, values in enumerate(param_values):
            # Create parameter dictionary
            param_dict = {name: values[j] for j, name in enumerate(param_names)}
            
            # Run for each seed
            for seed_idx, seed in enumerate(seeds):
                # Create a descriptive run name
                param_str = '_'.join([f"{name}={value}" for name, value in param_dict.items()])
                run_name = f"grid_run_{i}_{seed_idx}_{param_str}"
                
                # Create the run
                run_id, run_config, run_dir, logger = self.create_run(
                    run_name=run_name,
                    config_updates=param_dict,
                    seed=int(seed)
                )
                
                # Execute the run
                try:
                    run_results = self.run_experiment(run_fn, run_id=run_id)
                    
                    # Store results with parameters
                    run_results.update({
                        'params': param_dict,
                        'run_id': run_id,
                        'seed': int(seed)
                    })
                    results.append(run_results)
                    
                except Exception as e:
                    print(f"Error in run {run_name}: {e}")
        
        # Save all results
        grid_config['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(grid_dir, 'grid_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Create a summary DataFrame
        try:
            summary_data = []
            for result in results:
                data = {'run_id': result['run_id'], 'seed': result['seed']}
                data.update(result['params'])
                
                # Add important metrics
                for key, value in result.items():
                    if key not in ['params', 'run_id', 'seed'] and not isinstance(value, dict):
                        data[key] = value
                
                summary_data.append(data)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(grid_dir, 'grid_summary.csv'), index=False)
            
        except Exception as e:
            print(f"Error creating summary DataFrame: {e}")
        
        return results
    
    def get_best_run(self, metric, maximize=True):
        """
        Get the best run according to a specific metric.
        
        Parameters
        ----------
        metric : str
            Name of the metric to optimize
        maximize : bool, optional
            Whether to maximize or minimize the metric
            
        Returns
        -------
        dict
            Information about the best run
        """
        if not self.run_results:
            return None
        
        best_run_id = None
        best_value = float('-inf') if maximize else float('inf')
        
        for run_id, results in self.run_results.items():
            if metric in results:
                value = results[metric]
                if maximize and value > best_value:
                    best_value = value
                    best_run_id = run_id
                elif not maximize and value < best_value:
                    best_value = value
                    best_run_id = run_id
        
        if best_run_id is None:
            return None
        
        return {
            'run_id': best_run_id,
            'value': best_value,
            'config': self.runs[best_run_id]['config'],
            'results': self.run_results[best_run_id]
        }
    
    def get_run_summary(self):
        """
        Get a summary of all runs.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing run information
        """
        summary_data = []
        
        for run in self.runs:
            data = {
                'run_id': run['id'],
                'name': run['name'],
                'status': run['status'],
                'seed': run['seed'],
                'start_time': run['start_time'],
                'end_time': run['end_time']
            }
            
            # Add any results (if available)
            if run['id'] in self.run_results:
                for key, value in self.run_results[run['id']].items():
                    if not isinstance(value, dict):
                        data[key] = value
            
            summary_data.append(data)
        
        return pd.DataFrame(summary_data)
    
    def save_summary(self, filename='experiment_summary.csv'):
        """
        Save a summary of all runs to a CSV file.
        
        Parameters
        ----------
        filename : str, optional
            Filename to save the summary to
            
        Returns
        -------
        str
            Path to the saved summary file
        """
        summary_df = self.get_run_summary()
        save_path = os.path.join(self.base_dir, filename)
        summary_df.to_csv(save_path, index=False)
        return save_path 