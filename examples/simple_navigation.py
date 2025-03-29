#!/usr/bin/env python
"""
Simple example demonstrating the cooperative navigation environment.

In this environment, multiple agents must navigate to specific target locations
while avoiding collisions with each other.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Add parent directory to path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llamaswarm.algorithms.policy_gradient import MAPPO
from llamaswarm.core import Trainer
from llamaswarm.environments.cooperative import CooperativeNavigation
from llamaswarm.utils import set_seed, Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple navigation example")
    
    # Environment parameters
    parser.add_argument("--num-agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--grid-size", type=int, default=10, help="Size of the grid")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    
    # Training parameters
    parser.add_argument("--num-episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--num-eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    
    # Algorithm parameters
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of neural networks")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    
    # Output parameters
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    return parser.parse_args()


def main():
    """Run the example."""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create logger
    logger = Logger(
        log_dir=args.log_dir,
        experiment_name="simple_navigation",
        use_tensorboard=True,
        use_csv=True,
        use_json=True
    )
    
    # Create environment
    env = CooperativeNavigation(
        num_agents=args.num_agents,
        grid_size=args.grid_size,
        max_steps=args.max_steps
    )
    
    # Get observation and action dimensions
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    
    # Create algorithm
    algorithm = MAPPO(
        state_dim=obs_dim,
        action_dim=action_dim,
        num_agents=args.num_agents,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        algorithm=algorithm,
        max_episodes=args.num_episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        logger=logger
    )
    
    # Train agents
    trainer.train()
    
    # Evaluate trained agents
    mean_reward, std_reward = trainer.evaluate(
        num_episodes=args.num_eval_episodes,
        render=args.render
    )
    
    print(f"Evaluation results: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save the trained model
    save_path = os.path.join(args.log_dir, "simple_navigation", "model")
    os.makedirs(save_path, exist_ok=True)
    algorithm.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Close the logger
    logger.close()


if __name__ == "__main__":
    main() 