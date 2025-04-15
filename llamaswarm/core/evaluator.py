"""
Evaluator class for assessing trained agents in multi-agent environments.
"""

import os
import pickle
import time

import numpy as np
import torch
from tqdm import tqdm


class Evaluator:
    """
    Evaluator for assessing performance of trained agents.

    Parameters
    ----------
    env : Environment
        Multi-agent environment
    agents : list of Agent
        List of agents to evaluate
    n_episodes : int
        Number of episodes to evaluate for
    max_steps : int, optional
        Maximum number of steps per episode
    render : bool, optional
        Whether to render the environment during evaluation
    save_dir : str, optional
        Directory to save evaluation results
    device : str, optional
        Device to use for tensor operations ('cpu', 'cuda')
    """

    def __init__(
        self,
        env,
        agents,
        n_episodes,
        max_steps=1000,
        render=False,
        save_dir="./eval_results",
        device="cpu",
    ):
        self.env = env
        self.agents = agents
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.render = render
        self.save_dir = save_dir
        self.device = torch.device(device)

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Initialize metrics
        self.eval_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "agent_rewards": [[] for _ in range(len(agents))],
            "trajectories": [],
        }

    def evaluate(self, save_trajectories=False):
        """
        Evaluate agents for the specified number of episodes.

        Parameters
        ----------
        save_trajectories : bool, optional
            Whether to save full trajectories

        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        print(f"Starting evaluation for {self.n_episodes} episodes...")

        for episode in tqdm(range(self.n_episodes)):
            episode_rewards, episode_length, trajectory = self._evaluate_episode(
                save_trajectories
            )

            # Log episode metrics
            self.eval_metrics["episode_rewards"].append(episode_rewards)
            self.eval_metrics["episode_lengths"].append(episode_length)
            for i, reward in enumerate(episode_rewards):
                self.eval_metrics["agent_rewards"][i].append(reward)

            if save_trajectories:
                self.eval_metrics["trajectories"].append(trajectory)

        # Calculate summary statistics
        mean_rewards = np.mean(self.eval_metrics["episode_rewards"], axis=0)
        std_rewards = np.std(self.eval_metrics["episode_rewards"], axis=0)
        mean_length = np.mean(self.eval_metrics["episode_lengths"])
        std_length = np.std(self.eval_metrics["episode_lengths"])

        print(f"Evaluation completed.")
        print(f"Mean rewards: {mean_rewards}, Std: {std_rewards}")
        print(f"Mean episode length: {mean_length:.2f}, Std: {std_length:.2f}")

        # Save results
        self._save_results()

        return self.eval_metrics

    def _evaluate_episode(self, save_trajectory=False):
        """
        Evaluate agents for a single episode.

        Parameters
        ----------
        save_trajectory : bool
            Whether to save the full trajectory

        Returns
        -------
        tuple
            Tuple of (episode_rewards, episode_length, trajectory)
        """
        states = self.env.reset()
        episode_rewards = np.zeros(len(self.agents))
        done = False
        step = 0

        trajectory = [] if save_trajectory else None

        while not done and step < self.max_steps:
            if self.render:
                self.env.render()
                time.sleep(0.02)  # Small delay for visualization

            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(states[i], explore=False)
                actions.append(action)

            next_states, rewards, dones, infos = self.env.step(actions)

            if save_trajectory:
                trajectory.append(
                    {
                        "states": states,
                        "actions": actions,
                        "rewards": rewards,
                        "next_states": next_states,
                        "dones": dones,
                        "infos": infos,
                    }
                )

            states = next_states
            episode_rewards += rewards
            done = all(dones)
            step += 1

        return episode_rewards, step, trajectory

    def compare_agents(self, agent_names=None):
        """
        Compare the performance of different agents.

        Parameters
        ----------
        agent_names : list of str, optional
            Names of the agents for labeling

        Returns
        -------
        dict
            Comparison metrics by agent
        """
        if agent_names is None:
            agent_names = [f"Agent {i}" for i in range(len(self.agents))]

        comparison = {}
        for i, name in enumerate(agent_names):
            rewards = self.eval_metrics["agent_rewards"][i]
            comparison[name] = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "min_reward": np.min(rewards),
                "max_reward": np.max(rewards),
            }

        return comparison

    def _save_results(self):
        """
        Save evaluation results.
        """
        # Save metrics without trajectories
        metrics_copy = {
            k: v for k, v in self.eval_metrics.items() if k != "trajectories"
        }
        metrics_path = os.path.join(self.save_dir, "eval_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics_copy, f)

        # Save trajectories separately if they exist
        if self.eval_metrics["trajectories"]:
            trajectories_path = os.path.join(self.save_dir, "trajectories.pkl")
            with open(trajectories_path, "wb") as f:
                pickle.dump(self.eval_metrics["trajectories"], f)

    def load_agents(self, model_paths):
        """
        Load agents from saved models.

        Parameters
        ----------
        model_paths : list of str
            Paths to saved agent models
        """
        for i, path in enumerate(model_paths):
            if i < len(self.agents):
                self.agents[i].load(path)
            else:
                print(f"Warning: More model paths than agents. Ignoring {path}")

    def analyze_cooperation(self):
        """
        Analyze cooperation between agents in cooperative scenarios.

        Returns
        -------
        dict
            Cooperation metrics
        """
        if not self.eval_metrics["trajectories"]:
            print("No trajectories saved. Run evaluate with save_trajectories=True")
            return {}

        agent_rewards = np.array(self.eval_metrics["agent_rewards"])
        reward_correlation = np.corrcoef(agent_rewards)

        cooperation_metrics = {
            "reward_correlation": reward_correlation,
            "reward_variance": np.var(agent_rewards, axis=0),
            "collective_reward": np.sum(agent_rewards, axis=0),
        }

        return cooperation_metrics
