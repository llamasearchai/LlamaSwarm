#!/usr/bin/env python
"""
Example demonstrating the predator-prey competitive environment.

In this environment, predator agents try to catch prey agents, while prey
agents try to escape and survive as long as possible.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Add parent directory to path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llamaswarm.algorithms.multi_agent import MADDPG
from llamaswarm.core import Trainer
from llamaswarm.environments.competitive import PredatorPrey
from llamaswarm.utils import ExperimentManager, Logger, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predator-prey example")

    # Environment parameters
    parser.add_argument(
        "--num-predators", type=int, default=2, help="Number of predator agents"
    )
    parser.add_argument("--num-prey", type=int, default=4, help="Number of prey agents")
    parser.add_argument("--grid-size", type=int, default=15, help="Size of the grid")
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--sensor-range", type=float, default=5.0, help="Agent sensor range"
    )
    parser.add_argument(
        "--prey-speed", type=float, default=1.0, help="Prey movement speed"
    )
    parser.add_argument(
        "--predator-speed", type=float, default=1.2, help="Predator movement speed"
    )

    # Training parameters
    parser.add_argument(
        "--num-episodes", type=int, default=3000, help="Number of training episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval-interval", type=int, default=100, help="Evaluation interval"
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
        default=256,
        help="Hidden dimension of neural networks",
    )
    parser.add_argument(
        "--actor-lr", type=float, default=1e-4, help="Actor learning rate"
    )
    parser.add_argument(
        "--critic-lr", type=float, default=1e-3, help="Critic learning rate"
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--buffer-size", type=int, default=1000000, help="Replay buffer size"
    )
    parser.add_argument(
        "--update-freq", type=int, default=100, help="Policy update frequency"
    )

    # Output parameters
    parser.add_argument(
        "--exp-name", type=str, default="predator_prey", help="Experiment name"
    )
    parser.add_argument(
        "--exp-dir", type=str, default="experiments", help="Experiment directory"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render during evaluation"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    return parser.parse_args()


def run_experiment(config, logger):
    """Run a predator-prey experiment with given configuration."""
    # Set random seed for reproducibility
    set_seed(config["seed"])

    # Create environment
    env = PredatorPrey(
        num_predators=config["num_predators"],
        num_prey=config["num_prey"],
        grid_size=config["grid_size"],
        max_steps=config["max_steps"],
        sensor_range=config["sensor_range"],
        prey_speed=config["prey_speed"],
        predator_speed=config["predator_speed"],
    )

    # Get observation and action dimensions for each agent type
    predator_obs_dim = env.observation_space_predator.shape[0]
    prey_obs_dim = env.observation_space_prey.shape[0]

    predator_action_dim = env.action_space_predator.n
    prey_action_dim = env.action_space_prey.n

    # Create separate algorithms for predators and prey
    predator_algorithm = MADDPG(
        num_agents=config["num_predators"],
        state_dim=predator_obs_dim,
        action_dim=predator_action_dim,
        hidden_dim=config["hidden_dim"],
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        discrete_actions=True,
        buffer_size=config["buffer_size"],
    )

    prey_algorithm = MADDPG(
        num_agents=config["num_prey"],
        state_dim=prey_obs_dim,
        action_dim=prey_action_dim,
        hidden_dim=config["hidden_dim"],
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        discrete_actions=True,
        buffer_size=config["buffer_size"],
    )

    # Create trainer for competitive environment
    trainer = Trainer(
        env=env,
        algorithm={"predator": predator_algorithm, "prey": prey_algorithm},
        max_episodes=config["num_episodes"],
        max_steps=config["max_steps"],
        eval_interval=config["eval_interval"],
        num_eval_episodes=config["num_eval_episodes"],
        logger=logger,
        update_frequency=config["update_freq"],
        batch_size=config["batch_size"],
        verbose=config["verbose"],
    )

    # Train agents
    training_info = trainer.train()

    # Evaluate trained agents
    eval_info = trainer.evaluate(
        num_episodes=config["num_eval_episodes"], render=config["render"]
    )

    # Save the trained models
    predator_save_path = os.path.join(logger.log_dir, "predator_model")
    prey_save_path = os.path.join(logger.log_dir, "prey_model")

    os.makedirs(predator_save_path, exist_ok=True)
    os.makedirs(prey_save_path, exist_ok=True)

    predator_algorithm.save(predator_save_path)
    prey_algorithm.save(prey_save_path)

    if config["verbose"]:
        print(f"Predator model saved to {predator_save_path}")
        print(f"Prey model saved to {prey_save_path}")

    # Return results
    return {
        "training": training_info,
        "evaluation": eval_info,
        "predator_captures": eval_info.get("predator_captures", 0),
        "prey_escapes": eval_info.get("prey_escapes", 0),
        "average_episode_length": eval_info.get("average_episode_length", 0),
    }


def main():
    """Run the predator-prey example."""
    args = parse_args()

    # Create experiment manager
    exp_manager = ExperimentManager(
        base_dir=args.exp_dir,
        experiment_name=args.exp_name,
        config={
            "num_predators": args.num_predators,
            "num_prey": args.num_prey,
            "grid_size": args.grid_size,
            "max_steps": args.max_steps,
            "sensor_range": args.sensor_range,
            "prey_speed": args.prey_speed,
            "predator_speed": args.predator_speed,
            "num_episodes": args.num_episodes,
            "seed": args.seed,
            "eval_interval": args.eval_interval,
            "num_eval_episodes": args.num_eval_episodes,
            "hidden_dim": args.hidden_dim,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "gamma": args.gamma,
            "tau": args.tau,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "update_freq": args.update_freq,
            "render": args.render,
            "verbose": args.verbose,
        },
    )

    # Run the experiment
    results = exp_manager.run_experiment(run_experiment)

    # Display summary results
    if args.verbose:
        print("\nExperiment Results:")
        print(f"Predator captures: {results['predator_captures']}")
        print(f"Prey escapes: {results['prey_escapes']}")
        print(f"Average episode length: {results['average_episode_length']:.2f}")

    # Example of hyperparameter exploration
    if args.verbose:
        print("\nTo run a grid search over hyperparameters, use:")
        print("python predator_prey.py --exp-name predator_prey_grid_search --verbose")
        print(
            "  --grid-search-params prey_speed=0.8,1.0,1.2 predator_speed=1.0,1.2,1.4"
        )


if __name__ == "__main__":
    main()
