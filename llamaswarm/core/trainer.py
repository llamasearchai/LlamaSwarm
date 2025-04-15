"""
Trainer class for multi-agent reinforcement learning algorithms.
"""

import os
import pickle
import time

import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    """
    Trainer for multi-agent reinforcement learning algorithms.

    Parameters
    ----------
    env : Environment
        Multi-agent environment
    agents : list of Agent
        List of agents to train
    n_episodes : int
        Number of episodes to train for
    max_steps : int
        Maximum number of steps per episode
    log_interval : int, optional
        Interval for logging training progress
    eval_interval : int, optional
        Interval for evaluating agents
    save_dir : str, optional
        Directory to save trained models and results
    device : str, optional
        Device to use for tensor operations ('cpu', 'cuda')
    """

    def __init__(
        self,
        env,
        agents,
        n_episodes,
        max_steps,
        log_interval=10,
        eval_interval=100,
        save_dir="./results",
        device="cpu",
    ):
        self.env = env
        self.agents = agents
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.device = torch.device(device)

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Initialize metrics
        self.episode_rewards = []
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "agent_losses": [[] for _ in range(len(agents))],
            "eval_rewards": [],
            "eval_lengths": [],
        }

    def train(self, replay_buffer=None):
        """
        Train agents for the specified number of episodes.

        Parameters
        ----------
        replay_buffer : ReplayBuffer, optional
            Replay buffer for experience replay

        Returns
        -------
        dict
            Dictionary of training metrics
        """
        print(f"Starting training for {self.n_episodes} episodes...")

        for episode in tqdm(range(self.n_episodes)):
            episode_rewards, episode_length, losses = self._train_episode(replay_buffer)

            # Log episode metrics
            self.training_metrics["episode_rewards"].append(episode_rewards)
            self.training_metrics["episode_lengths"].append(episode_length)
            for i, loss in enumerate(losses):
                self.training_metrics["agent_losses"][i].append(loss)

            # Log progress
            if (episode + 1) % self.log_interval == 0:
                avg_reward = np.mean(
                    self.training_metrics["episode_rewards"][-self.log_interval :]
                )
                avg_length = np.mean(
                    self.training_metrics["episode_lengths"][-self.log_interval :]
                )
                print(
                    f"Episode {episode + 1}/{self.n_episodes} - Avg. Reward: {avg_reward:.2f}, Avg. Length: {avg_length:.2f}"
                )

            # Evaluate agents
            if (episode + 1) % self.eval_interval == 0:
                eval_rewards, eval_length = self._evaluate()
                self.training_metrics["eval_rewards"].append(eval_rewards)
                self.training_metrics["eval_lengths"].append(eval_length)
                print(
                    f"Evaluation - Avg. Reward: {np.mean(eval_rewards):.2f}, Avg. Length: {eval_length:.2f}"
                )

                # Save models and metrics
                self._save_checkpoint(episode + 1)

        print("Training completed.")
        self._save_results()
        return self.training_metrics

    def _train_episode(self, replay_buffer=None):
        """
        Train agents for a single episode.

        Parameters
        ----------
        replay_buffer : ReplayBuffer, optional
            Replay buffer for experience replay

        Returns
        -------
        tuple
            Tuple of (episode_rewards, episode_length, losses)
        """
        states = self.env.reset()
        episode_rewards = np.zeros(len(self.agents))
        losses = [[] for _ in range(len(self.agents))]
        done = False
        step = 0

        while not done and step < self.max_steps:
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(states[i], explore=True)
                actions.append(action)

            next_states, rewards, dones, infos = self.env.step(actions)

            # Store experience in replay buffer if provided
            if replay_buffer is not None:
                replay_buffer.add(states, actions, rewards, next_states, dones, infos)

                # Update agents using replay buffer
                if len(replay_buffer) >= replay_buffer.batch_size:
                    batch = replay_buffer.sample()
                    for i, agent in enumerate(self.agents):
                        loss = agent.update(batch)
                        losses[i].append(loss)
            else:
                # Direct update without replay buffer
                for i, agent in enumerate(self.agents):
                    experience = {
                        "states": states[i],
                        "actions": actions[i],
                        "rewards": rewards[i],
                        "next_states": next_states[i],
                        "dones": dones[i],
                    }
                    loss = agent.update(experience)
                    losses[i].append(loss)

            states = next_states
            episode_rewards += rewards
            done = all(dones)
            step += 1

        # Calculate average losses for each agent
        avg_losses = []
        for agent_losses in losses:
            if agent_losses:
                avg_loss = {
                    k: np.mean([l[k] for l in agent_losses]) for k in agent_losses[0]
                }
                avg_losses.append(avg_loss)
            else:
                avg_losses.append({})

        return episode_rewards, step, avg_losses

    def _evaluate(self, n_episodes=5):
        """
        Evaluate agents without exploration.

        Parameters
        ----------
        n_episodes : int, optional
            Number of episodes to evaluate for

        Returns
        -------
        tuple
            Tuple of (average_rewards, average_length)
        """
        eval_rewards = []
        eval_lengths = []

        for _ in range(n_episodes):
            states = self.env.reset()
            episode_rewards = np.zeros(len(self.agents))
            done = False
            step = 0

            while not done and step < self.max_steps:
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(states[i], explore=False)
                    actions.append(action)

                next_states, rewards, dones, _ = self.env.step(actions)

                states = next_states
                episode_rewards += rewards
                done = all(dones)
                step += 1

            eval_rewards.append(episode_rewards)
            eval_lengths.append(step)

        return np.mean(eval_rewards, axis=0), np.mean(eval_lengths)

    def _save_checkpoint(self, episode):
        """
        Save agent models and training metrics.

        Parameters
        ----------
        episode : int
            Current episode number
        """
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint_{episode}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Save agent models
        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(checkpoint_dir, f"agent_{i}")
            agent.save(agent_path)

        # Save training metrics
        metrics_path = os.path.join(checkpoint_dir, "metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(self.training_metrics, f)

    def _save_results(self):
        """
        Save final results.
        """
        results_path = os.path.join(self.save_dir, "final_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(self.training_metrics, f)

        # Save final agent models
        models_dir = os.path.join(self.save_dir, "final_models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(models_dir, f"agent_{i}")
            agent.save(agent_path)
