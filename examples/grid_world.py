#!/usr/bin/env python
"""
Example demonstrating a grid world environment with obstacles and goals.

In this environment, agents must navigate through a grid world with obstacles
to reach their goals while minimizing the number of steps taken.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llamaswarm.algorithms.value_based import DQN
from llamaswarm.core import Trainer
from llamaswarm.environments.single_agent import GridWorld
from llamaswarm.utils import Logger, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Grid world example")

    # Environment parameters
    parser.add_argument("--grid-size", type=int, default=10, help="Size of the grid")
    parser.add_argument(
        "--num-obstacles", type=int, default=15, help="Number of obstacles"
    )
    parser.add_argument(
        "--obstacle-penalty",
        type=float,
        default=-1.0,
        help="Penalty for hitting obstacles",
    )
    parser.add_argument(
        "--goal-reward", type=float, default=10.0, help="Reward for reaching the goal"
    )
    parser.add_argument(
        "--step-penalty", type=float, default=-0.1, help="Penalty for each step"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Maximum steps per episode"
    )

    # Training parameters
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval-interval", type=int, default=50, help="Evaluation interval"
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )

    # Algorithm parameters
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension of neural networks",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Starting epsilon for exploration",
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.1, help="Final epsilon for exploration"
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate"
    )
    parser.add_argument(
        "--target-update", type=int, default=10, help="Target network update frequency"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--buffer-size", type=int, default=10000, help="Replay buffer size"
    )

    # Output parameters
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument(
        "--render", action="store_true", help="Render during evaluation"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize training progress"
    )

    return parser.parse_args()


def visualize_policy(env, agent, episode, log_dir):
    """Visualize the learned policy by showing action probabilities."""
    # Create a grid to store the action values
    q_values = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Compute Q-values for each position in the grid
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            # Skip if this position has an obstacle
            if (i, j) in env.obstacles:
                continue

            # Create a state representation as if the agent were at position (i, j)
            state = env.get_state_representation((i, j))
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Get Q-values from the agent
            with torch.no_grad():
                q_value = agent.policy_net(state_tensor).cpu().numpy()[0]

            q_values[i, j] = q_value

    # Create a figure with subplots for each action
    fig, axes = plt.subplots(1, env.action_space.n, figsize=(16, 4))
    fig.suptitle(f"Action Values at Episode {episode}")

    action_names = ["Up", "Right", "Down", "Left"]

    for a in range(env.action_space.n):
        # Create a heatmap for this action
        im = axes[a].imshow(q_values[:, :, a], cmap="viridis")
        axes[a].set_title(f"Action: {action_names[a]}")

        # Mark obstacles
        for obs in env.obstacles:
            axes[a].plot(obs[1], obs[0], "rx", markersize=10)

        # Mark the goal
        axes[a].plot(env.goal[1], env.goal[0], "g*", markersize=15)

        # Add colorbar
        plt.colorbar(im, ax=axes[a])

    # Save the figure
    os.makedirs(os.path.join(log_dir, "visualizations"), exist_ok=True)
    plt.savefig(
        os.path.join(log_dir, "visualizations", f"policy_episode_{episode}.png")
    )
    plt.close()


def main():
    """Run the grid world example."""
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create logger
    logger = Logger(
        log_dir=args.log_dir,
        experiment_name="grid_world",
        use_tensorboard=True,
        use_csv=True,
        use_json=True,
    )

    # Create environment
    env = GridWorld(
        grid_size=args.grid_size,
        num_obstacles=args.num_obstacles,
        obstacle_penalty=args.obstacle_penalty,
        goal_reward=args.goal_reward,
        step_penalty=args.step_penalty,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create algorithm (DQN)
    algorithm = DQN(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
    )

    # Custom callback function for visualization
    def episode_callback(episode, stats):
        """Callback function called after each episode."""
        # Log metrics
        logger.log("episode/reward", stats["episode_reward"], episode)
        logger.log("episode/length", stats["episode_length"], episode)

        # Visualize policy at regular intervals
        if args.visualize and episode % 100 == 0:
            visualize_policy(env, algorithm, episode, logger.log_dir)

        # Print progress
        if episode % 50 == 0:
            print(
                f"Episode {episode}: Reward = {stats['episode_reward']:.2f}, Length = {stats['episode_length']}"
            )

    # Create trainer
    trainer = Trainer(
        env=env,
        algorithm=algorithm,
        max_episodes=args.num_episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        logger=logger,
        batch_size=args.batch_size,
        episode_callback=episode_callback,
    )

    # Train agent
    trainer.train()

    # Visualize final policy
    if args.visualize:
        visualize_policy(env, algorithm, args.num_episodes, logger.log_dir)

    # Evaluate trained agent
    mean_reward, std_reward = trainer.evaluate(
        num_episodes=args.num_eval_episodes, render=args.render
    )

    print(f"Evaluation results: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

    # Save the trained model
    save_path = os.path.join(logger.log_dir, "model")
    algorithm.save(save_path)
    print(f"Model saved to {save_path}")

    # Close the logger
    logger.close()

    # Generate a visualization of the agent's path
    if args.visualize:
        # Reset the environment
        state = env.reset()

        # Create a new figure
        plt.figure(figsize=(8, 8))
        plt.title("Agent Path in Grid World")

        # Plot the grid
        plt.xlim(0, env.grid_size - 1)
        plt.ylim(0, env.grid_size - 1)
        plt.grid(True)

        # Plot obstacles
        for obs in env.obstacles:
            plt.plot(obs[1], obs[0], "rs", markersize=10)

        # Plot the goal
        plt.plot(env.goal[1], env.goal[0], "g*", markersize=15)

        # Initialize agent position
        agent_pos = env.agent_pos
        positions = [agent_pos]

        # Run the agent until it reaches the goal or maximum steps
        done = False
        step = 0

        while not done and step < args.max_steps:
            # Select action using the trained policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = algorithm.select_action(state_tensor, epsilon=0)

            # Take the action
            next_state, reward, done, _ = env.step(action)

            # Update state and position
            state = next_state
            agent_pos = env.agent_pos
            positions.append(agent_pos)

            step += 1

        # Convert positions to array for plotting
        positions = np.array(positions)

        # Plot the path
        plt.plot(positions[:, 1], positions[:, 0], "b-", linewidth=2)
        plt.plot(positions[0, 1], positions[0, 0], "bo", markersize=10, label="Start")

        # Add legend
        plt.legend(["Goal", "Start", "Path", "Obstacles"])

        # Save the figure
        plt.savefig(os.path.join(logger.log_dir, "visualizations", "agent_path.png"))
        plt.close()


if __name__ == "__main__":
    main()
