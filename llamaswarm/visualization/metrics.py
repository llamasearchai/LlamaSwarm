"""
Metrics visualization tools for plotting training and evaluation metrics.
"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class MetricsPlotter:
    """
    Tool for plotting training and evaluation metrics.

    Parameters
    ----------
    save_dir : str, optional
        Directory to save plots
    figsize : tuple, optional
        Figure size for plots
    style : str, optional
        Matplotlib style to use
    dpi : int, optional
        DPI for saved figures
    """

    def __init__(
        self, save_dir="./plots", figsize=(12, 8), style="seaborn-whitegrid", dpi=150
    ):
        self.save_dir = save_dir
        self.figsize = figsize
        self.style = style
        self.dpi = dpi

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Set plot style
        plt.style.use(style)
        sns.set_context("paper")

    def plot_training_curve(
        self,
        metrics,
        metric_name,
        title=None,
        ylabel=None,
        window=None,
        save_name=None,
        show=True,
    ):
        """
        Plot a training curve for a specific metric.

        Parameters
        ----------
        metrics : dict or list
            Dictionary containing metrics or list of values
        metric_name : str
            Name of the metric to plot
        title : str, optional
            Plot title
        ylabel : str, optional
            Y-axis label
        window : int, optional
            Window size for smoothing
        save_name : str, optional
            Filename to save the plot
        show : bool, optional
            Whether to display the plot

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        plt.figure(figsize=self.figsize)

        # Extract data
        if isinstance(metrics, dict):
            if metric_name in metrics:
                data = np.array(metrics[metric_name])
            else:
                raise ValueError(f"Metric {metric_name} not found in metrics")
        else:
            data = np.array(metrics)

        # Create x-axis values
        x = np.arange(len(data))

        # Apply smoothing if specified
        if window is not None and window > 1:
            smooth_data = self._smooth(data, window)
            plt.plot(x, data, alpha=0.3, label="Raw")
            plt.plot(x, smooth_data, label=f"Smoothed (window={window})")
            plt.legend()
        else:
            plt.plot(x, data)

        # Set labels and title
        plt.xlabel("Steps")
        plt.ylabel(ylabel if ylabel else metric_name)
        plt.title(title if title else f"{metric_name} over time")

        # Add grid
        plt.grid(True, alpha=0.3)

        # Save plot if specified
        if save_name:
            filename = (
                f"{save_name}.png" if not save_name.endswith(".png") else save_name
            )
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches="tight"
            )

        # Show or close figure
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def plot_multi_agent_rewards(
        self,
        rewards,
        agent_names=None,
        title=None,
        window=None,
        save_name=None,
        show=True,
    ):
        """
        Plot rewards for multiple agents.

        Parameters
        ----------
        rewards : numpy.ndarray or list
            Rewards for each agent, shape (n_agents, n_steps)
        agent_names : list, optional
            Names of agents
        title : str, optional
            Plot title
        window : int, optional
            Window size for smoothing
        save_name : str, optional
            Filename to save the plot
        show : bool, optional
            Whether to display the plot

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        plt.figure(figsize=self.figsize)

        rewards = np.array(rewards)
        n_agents = rewards.shape[0]

        # Generate agent names if not provided
        if agent_names is None:
            agent_names = [f"Agent {i+1}" for i in range(n_agents)]

        # Plot rewards for each agent
        for i in range(n_agents):
            data = rewards[i]
            x = np.arange(len(data))

            # Apply smoothing if specified
            if window is not None and window > 1:
                smooth_data = self._smooth(data, window)
                plt.plot(x, smooth_data, label=agent_names[i])
            else:
                plt.plot(x, data, label=agent_names[i])

        # Set labels and title
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(title if title else "Agent Rewards")
        plt.legend()

        # Add grid
        plt.grid(True, alpha=0.3)

        # Save plot if specified
        if save_name:
            filename = (
                f"{save_name}.png" if not save_name.endswith(".png") else save_name
            )
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches="tight"
            )

        # Show or close figure
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def plot_loss_components(
        self, metrics, components, title=None, window=None, save_name=None, show=True
    ):
        """
        Plot multiple loss components on the same figure.

        Parameters
        ----------
        metrics : dict
            Dictionary containing loss metrics
        components : list
            List of component names to plot
        title : str, optional
            Plot title
        window : int, optional
            Window size for smoothing
        save_name : str, optional
            Filename to save the plot
        show : bool, optional
            Whether to display the plot

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        plt.figure(figsize=self.figsize)

        for component in components:
            if component in metrics:
                data = np.array(metrics[component])
                x = np.arange(len(data))

                # Apply smoothing if specified
                if window is not None and window > 1:
                    smooth_data = self._smooth(data, window)
                    plt.plot(x, smooth_data, label=component)
                else:
                    plt.plot(x, data, label=component)
            else:
                print(f"Warning: Component {component} not found in metrics")

        # Set labels and title
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(title if title else "Loss Components")
        plt.legend()

        # Add grid
        plt.grid(True, alpha=0.3)

        # Save plot if specified
        if save_name:
            filename = (
                f"{save_name}.png" if not save_name.endswith(".png") else save_name
            )
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches="tight"
            )

        # Show or close figure
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def plot_evaluation_metrics(
        self, metrics, include_std=True, title=None, save_name=None, show=True
    ):
        """
        Plot evaluation metrics with error bars.

        Parameters
        ----------
        metrics : dict
            Dictionary of evaluation metrics
        include_std : bool, optional
            Whether to include standard deviation as error bars
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save the plot
        show : bool, optional
            Whether to display the plot

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        plt.figure(figsize=self.figsize)

        # Convert metrics to dataframe for easier plotting
        metrics_df = pd.DataFrame(metrics)

        # Calculate mean and std if data is multi-dimensional
        if len(metrics_df.shape) > 1 and metrics_df.shape[1] > 1:
            means = metrics_df.mean(axis=1)
            stds = metrics_df.std(axis=1) if include_std else None
        else:
            means = metrics_df
            stds = None

        # Plot means
        x = np.arange(len(means))
        plt.plot(x, means, marker="o")

        # Add error bars if stds are available
        if stds is not None:
            plt.fill_between(x, means - stds, means + stds, alpha=0.3)

        # Set labels and title
        plt.xlabel("Episodes")
        plt.ylabel("Value")
        plt.title(title if title else "Evaluation Metrics")

        # Add grid
        plt.grid(True, alpha=0.3)

        # Save plot if specified
        if save_name:
            filename = (
                f"{save_name}.png" if not save_name.endswith(".png") else save_name
            )
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches="tight"
            )

        # Show or close figure
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def plot_agent_comparison(
        self,
        metrics_dict,
        metric_name,
        agent_names=None,
        title=None,
        save_name=None,
        show=True,
        bar=False,
    ):
        """
        Compare a specific metric across multiple agents or algorithms.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary of metrics for each agent/algorithm
        metric_name : str
            Name of the metric to compare
        agent_names : list, optional
            Names of agents/algorithms
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save the plot
        show : bool, optional
            Whether to display the plot
        bar : bool, optional
            Whether to use a bar plot instead of line plot

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        plt.figure(figsize=self.figsize)

        # Generate agent names if not provided
        if agent_names is None:
            agent_names = list(metrics_dict.keys())

        # Prepare data for plotting
        data = []
        labels = []

        for name in agent_names:
            if name in metrics_dict and metric_name in metrics_dict[name]:
                data.append(metrics_dict[name][metric_name])
                labels.append(name)
            else:
                print(f"Warning: Metric {metric_name} for {name} not found")

        # Plot data
        if bar:
            # For bar plot, use the mean values
            mean_values = [np.mean(d) for d in data]
            std_values = [np.std(d) for d in data]

            plt.bar(labels, mean_values, yerr=std_values, capsize=10)
            plt.xticks(rotation=45, ha="right")
        else:
            # For line plot
            for i, (d, label) in enumerate(zip(data, labels)):
                x = np.arange(len(d))
                plt.plot(x, d, label=label)

            plt.legend()

        # Set labels and title
        plt.ylabel(metric_name)
        plt.title(title if title else f"Comparison of {metric_name}")

        # Add grid
        plt.grid(True, alpha=0.3)

        # Save plot if specified
        if save_name:
            filename = (
                f"{save_name}.png" if not save_name.endswith(".png") else save_name
            )
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches="tight"
            )

        # Show or close figure
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def load_metrics(self, path):
        """
        Load metrics from a pickle file.

        Parameters
        ----------
        path : str
            Path to the metrics file

        Returns
        -------
        dict
            Loaded metrics
        """
        with open(path, "rb") as f:
            metrics = pickle.load(f)
        return metrics

    def _smooth(self, data, window):
        """
        Apply moving average smoothing to data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data
        window : int
            Window size for smoothing

        Returns
        -------
        numpy.ndarray
            Smoothed data
        """
        return np.convolve(data, np.ones(window) / window, mode="same")

    def plot_learning_curve_grid(
        self,
        metrics_dict,
        metric_names,
        titles=None,
        agent_names=None,
        window=None,
        save_name=None,
        show=True,
    ):
        """
        Create a grid of learning curves for multiple metrics and agents.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary of metrics for each agent
        metric_names : list
            List of metric names to plot
        titles : list, optional
            List of titles for each plot
        agent_names : list, optional
            Names of agents
        window : int, optional
            Window size for smoothing
        save_name : str, optional
            Filename to save the plot
        show : bool, optional
            Whether to display the plot

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        # Generate agent names if not provided
        if agent_names is None:
            agent_names = list(metrics_dict.keys())

        # Generate titles if not provided
        if titles is None:
            titles = metric_names

        # Calculate grid dimensions
        n_metrics = len(metric_names)
        rows = int(np.ceil(n_metrics / 2))
        cols = min(2, n_metrics)

        # Create figure and axes
        fig, axes = plt.subplots(
            rows, cols, figsize=(self.figsize[0], rows * 5), sharex=True
        )
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each metric
        for i, (metric_name, title) in enumerate(zip(metric_names, titles)):
            if i < len(axes):
                ax = axes[i]

                # Plot data for each agent
                for name in agent_names:
                    if name in metrics_dict and metric_name in metrics_dict[name]:
                        data = np.array(metrics_dict[name][metric_name])
                        x = np.arange(len(data))

                        # Apply smoothing if specified
                        if window is not None and window > 1:
                            smooth_data = self._smooth(data, window)
                            ax.plot(x, smooth_data, label=name)
                        else:
                            ax.plot(x, data, label=name)
                    else:
                        print(f"Warning: Metric {metric_name} for {name} not found")

                # Set labels and title
                ax.set_xlabel("Steps")
                ax.set_ylabel(metric_name)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])

        # Save plot if specified
        if save_name:
            filename = (
                f"{save_name}.png" if not save_name.endswith(".png") else save_name
            )
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches="tight"
            )

        # Show or close figure
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return fig
