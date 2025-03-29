#!/usr/bin/env python
"""
Example demonstrating how to create a custom environment.

This example shows how to implement a simple custom environment
by extending the base environment class and using it with an agent.
"""

import argparse
import os
import sys
import numpy as np

# Add parent directory to path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from gymnasium import spaces
import torch

from llamaswarm.algorithms.policy_gradient import PPO
from llamaswarm.core import Trainer
from llamaswarm.environments import BaseEnv
from llamaswarm.utils import set_seed, Logger


class ResourceGatheringEnv(BaseEnv):
    """
    Custom environment for resource gathering.
    
    In this environment, agents collect resources of different types
    while managing energy levels. The goal is to maximize resource collection
    while maintaining sufficient energy.
    """
    
    def __init__(
        self,
        grid_size=10,
        num_resource_types=3,
        max_resources=5,
        max_energy=100,
        energy_decay_rate=1,
        resource_respawn_probability=0.05,
        max_steps=200,
        seed=None
    ):
        """
        Initialize the resource gathering environment.
        
        Parameters
        ----------
        grid_size : int
            Size of the grid world
        num_resource_types : int
            Number of different resource types
        max_resources : int
            Maximum number of each resource type
        max_energy : int
            Maximum energy level
        energy_decay_rate : float
            Rate at which energy decreases per step
        resource_respawn_probability : float
            Probability of a resource respawning
        max_steps : int
            Maximum steps per episode
        seed : int, optional
            Random seed
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.num_resource_types = num_resource_types
        self.max_resources = max_resources
        self.max_energy = max_energy
        self.energy_decay_rate = energy_decay_rate
        self.resource_respawn_probability = resource_respawn_probability
        self.max_steps = max_steps
        
        # Set random seed
        self.seed(seed)
        
        # Define action space (up, right, down, left, gather)
        self.action_space = spaces.Discrete(5)
        
        # Define observation space
        # [x, y, energy] + [resource_1_x, resource_1_y, ...] for all resources
        obs_dim = 3 + 2 * num_resource_types * max_resources
        self.observation_space = spaces.Box(
            low=0,
            high=max(grid_size, max_energy),
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize environment state
        self.reset()
    
    def seed(self, seed=None):
        """Set random seed."""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def reset(self):
        """Reset the environment to initial state."""
        # Reset agent position to random location
        self.agent_pos = (
            self.np_random.randint(0, self.grid_size),
            self.np_random.randint(0, self.grid_size)
        )
        
        # Reset energy
        self.energy = self.max_energy
        
        # Reset collected resources
        self.collected_resources = [0] * self.num_resource_types
        
        # Initialize resources
        self.resources = []
        for resource_type in range(self.num_resource_types):
            type_resources = []
            for _ in range(self.max_resources):
                pos = (
                    self.np_random.randint(0, self.grid_size),
                    self.np_random.randint(0, self.grid_size)
                )
                type_resources.append((pos, True))  # (position, is_active)
            self.resources.append(type_resources)
        
        # Reset step counter
        self.steps = 0
        
        # Get observation
        observation = self._get_observation()
        
        return observation
    
    def _get_observation(self):
        """Construct observation from environment state."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Agent position and energy
        obs[0] = self.agent_pos[0]
        obs[1] = self.agent_pos[1]
        obs[2] = self.energy
        
        # Resource positions
        idx = 3
        for resource_type in range(self.num_resource_types):
            for r_idx, (pos, active) in enumerate(self.resources[resource_type]):
                if active:
                    obs[idx] = pos[0]
                    obs[idx + 1] = pos[1]
                else:
                    obs[idx] = -1
                    obs[idx + 1] = -1
                idx += 2
        
        return obs
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Parameters
        ----------
        action : int
            Action to take (0=up, 1=right, 2=down, 3=left, 4=gather)
            
        Returns
        -------
        tuple
            (observation, reward, done, info)
        """
        # Increment step counter
        self.steps += 1
        
        # Process action
        reward = 0
        
        if action == 0:  # Up
            new_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
            self.agent_pos = new_pos
            reward -= 0.1  # Small penalty for movement
        elif action == 1:  # Right
            new_pos = (self.agent_pos[0], min(self.grid_size - 1, self.agent_pos[1] + 1))
            self.agent_pos = new_pos
            reward -= 0.1
        elif action == 2:  # Down
            new_pos = (min(self.grid_size - 1, self.agent_pos[0] + 1), self.agent_pos[1])
            self.agent_pos = new_pos
            reward -= 0.1
        elif action == 3:  # Left
            new_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
            self.agent_pos = new_pos
            reward -= 0.1
        elif action == 4:  # Gather
            # Check if there's a resource at the agent's position
            for resource_type in range(self.num_resource_types):
                for r_idx, (pos, active) in enumerate(self.resources[resource_type]):
                    if active and pos == self.agent_pos:
                        # Collect the resource
                        self.resources[resource_type][r_idx] = (pos, False)
                        self.collected_resources[resource_type] += 1
                        
                        # Give reward based on resource type
                        # Higher value for rarer resource types
                        reward += (resource_type + 1) * 2.0
                        
                        # Gathering costs more energy
                        self.energy -= 2 * self.energy_decay_rate
                        break
        
        # Decrease energy each step
        self.energy -= self.energy_decay_rate
        
        # Check for resource respawning
        for resource_type in range(self.num_resource_types):
            for r_idx, (pos, active) in enumerate(self.resources[resource_type]):
                if not active and self.np_random.random() < self.resource_respawn_probability:
                    # Respawn at a new position
                    new_pos = (
                        self.np_random.randint(0, self.grid_size),
                        self.np_random.randint(0, self.grid_size)
                    )
                    self.resources[resource_type][r_idx] = (new_pos, True)
        
        # Check if episode is done
        done = False
        
        # Done if out of energy
        if self.energy <= 0:
            done = True
            reward -= 10.0  # Penalty for running out of energy
        
        # Done if reached max steps
        if self.steps >= self.max_steps:
            done = True
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            "collected_resources": self.collected_resources,
            "energy": self.energy,
            "steps": self.steps
        }
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid.fill('.')
            
            # Mark resources
            for resource_type in range(self.num_resource_types):
                for pos, active in self.resources[resource_type]:
                    if active:
                        grid[pos] = str(resource_type + 1)
            
            # Mark agent
            grid[self.agent_pos] = 'A'
            
            # Print grid
            print("\n" + "-" * (self.grid_size * 2 + 1))
            for i in range(self.grid_size):
                print("|", end="")
                for j in range(self.grid_size):
                    print(f"{grid[i, j]} ", end="")
                print("|")
            print("-" * (self.grid_size * 2 + 1))
            
            # Print stats
            print(f"Energy: {self.energy}/{self.max_energy}")
            print(f"Resources collected: {self.collected_resources}")
            print(f"Steps: {self.steps}/{self.max_steps}")
            print()
        
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Custom environment example")
    
    # Environment parameters
    parser.add_argument("--grid-size", type=int, default=10, help="Size of the grid")
    parser.add_argument("--num-resource-types", type=int, default=3, help="Number of resource types")
    parser.add_argument("--max-resources", type=int, default=5, help="Maximum number of each resource")
    parser.add_argument("--max-energy", type=int, default=100, help="Maximum energy level")
    parser.add_argument("--energy-decay", type=float, default=1.0, help="Energy decay rate")
    parser.add_argument("--respawn-prob", type=float, default=0.05, help="Resource respawn probability")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    
    # Training parameters
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--num-eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    
    # Algorithm parameters
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of neural networks")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--epochs", type=int, default=4, help="Number of PPO epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    
    # Output parameters
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    return parser.parse_args()


def register_custom_env():
    """Register the custom environment with Gym."""
    # Register our custom environment
    gym.register(
        id='ResourceGathering-v0',
        entry_point='__main__:ResourceGatheringEnv',
        max_episode_steps=1000
    )


def main():
    """Run the custom environment example."""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Register the environment
    register_custom_env()
    
    # Create logger
    logger = Logger(
        log_dir=args.log_dir,
        experiment_name="resource_gathering",
        use_tensorboard=True,
        use_csv=True,
        use_json=True
    )
    
    # Create the custom environment
    env = ResourceGatheringEnv(
        grid_size=args.grid_size,
        num_resource_types=args.num_resource_types,
        max_resources=args.max_resources,
        max_energy=args.max_energy,
        energy_decay_rate=args.energy_decay,
        resource_respawn_probability=args.respawn_prob,
        max_steps=args.max_steps,
        seed=args.seed
    )
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create algorithm (PPO)
    algorithm = PPO(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        epochs=args.epochs
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
        batch_size=args.batch_size
    )
    
    # Train agent
    trainer.train()
    
    # Evaluate trained agent
    mean_reward, std_reward = trainer.evaluate(
        num_episodes=args.num_eval_episodes,
        render=args.render
    )
    
    print(f"Evaluation results: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save the trained model
    save_path = os.path.join(logger.log_dir, "model")
    algorithm.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Close the logger
    logger.close()
    
    # Example usage of the trained model
    print("\nRunning a sample episode with the trained model:")
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < args.max_steps:
        # Get action from the trained model
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = algorithm.select_action(state_tensor)
        
        # Take the action
        next_state, reward, done, info = env.step(action.item())
        
        # Render the environment
        if args.render:
            env.render()
        
        # Update state and counters
        state = next_state
        total_reward += reward
        steps += 1
    
    print(f"Episode finished after {steps} steps with reward {total_reward:.2f}")
    print(f"Collected resources: {env.collected_resources}")


if __name__ == "__main__":
    main() 