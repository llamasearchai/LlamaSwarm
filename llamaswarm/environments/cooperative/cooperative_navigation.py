"""
Cooperative navigation environment.

In this environment, agents must cooperate to navigate to specific landmarks
while avoiding collisions with each other.
"""

import numpy as np
import gym
from gym import spaces
import pygame

from ..base_env import MultiAgentEnv


class CooperativeNavigation(MultiAgentEnv):
    """
    Cooperative Navigation Environment.
    
    Agents need to navigate to specific landmarks while avoiding collisions.
    Each agent is rewarded based on how close it is to its landmark,
    with penalties for collisions with other agents.
    
    Parameters
    ----------
    n_agents : int
        Number of agents
    grid_size : tuple, optional
        Size of the grid (width, height)
    sight_radius : float, optional
        Observation radius for each agent
    collision_penalty : float, optional
        Penalty for agent collisions
    goal_reward : float, optional
        Reward for reaching the goal
    distance_factor : float, optional
        Factor for distance-based rewards
    max_steps : int, optional
        Maximum steps per episode
    seed : int, optional
        Random seed
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        n_agents,
        grid_size=(20, 20),
        sight_radius=5.0,
        collision_penalty=-1.0,
        goal_reward=10.0,
        distance_factor=-0.1,
        max_steps=100,
        seed=None
    ):
        self.grid_size = grid_size
        self.sight_radius = sight_radius
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.distance_factor = distance_factor
        
        # Rendering attributes
        self.screen = None
        self.screen_size = (800, 800)
        self.agent_colors = [
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Green
            (0, 0, 128),    # Navy
            (128, 128, 0)   # Olive
        ]
        self.landmark_color = (200, 200, 200)  # Light gray
        
        # Initialize base class
        super().__init__(n_agents, max_steps, seed)
        
        # Initialize agents and landmarks
        self.agent_positions = None
        self.agent_velocities = None
        self.landmarks = None
        self.agent_radius = 0.5
        self.landmark_radius = 0.5
        
    def _make_action_spaces(self):
        """
        Define action spaces for each agent.
        
        Each agent has a 2D continuous action space (dx, dy).
        """
        action_spaces = []
        for _ in range(self.n_agents):
            action_spaces.append(spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(2,), 
                dtype=np.float32
            ))
        return action_spaces
    
    def _make_observation_spaces(self):
        """
        Define observation spaces for each agent.
        
        Each agent observes:
        - Its own position (2D)
        - Its velocity (2D)
        - Vector to its target landmark (2D)
        - Relative positions of other agents within sight radius (2D * n_agents)
        """
        obs_dim = 2 + 2 + 2 + 2 * (self.n_agents - 1)
        observation_spaces = []
        for _ in range(self.n_agents):
            observation_spaces.append(spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(obs_dim,), 
                dtype=np.float32
            ))
        return observation_spaces
    
    def reset(self):
        """
        Reset the environment state.
        
        Returns
        -------
        list of numpy.ndarray
            Initial observations for each agent
        """
        super().reset()
        
        # Place agents randomly
        self.agent_positions = np.random.uniform(
            low=0, 
            high=np.array(self.grid_size), 
            size=(self.n_agents, 2)
        )
        
        # Initialize agent velocities to zero
        self.agent_velocities = np.zeros((self.n_agents, 2))
        
        # Place landmarks randomly, ensuring they don't overlap
        self.landmarks = np.random.uniform(
            low=0, 
            high=np.array(self.grid_size), 
            size=(self.n_agents, 2)
        )
        
        # Update state
        self.state = {
            'agent_positions': self.agent_positions.copy(),
            'agent_velocities': self.agent_velocities.copy(),
            'landmarks': self.landmarks.copy(),
            'step': 0
        }
        
        return self.get_obs()
    
    def step(self, actions):
        """
        Take a step in the environment.
        
        Parameters
        ----------
        actions : list of numpy.ndarray
            Actions for each agent
            
        Returns
        -------
        tuple
            Tuple of (observations, rewards, dones, infos)
        """
        super().step(actions)
        
        # Convert actions to numpy array if needed
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        # Update agent positions based on actions
        for i in range(self.n_agents):
            # Scale actions and update velocity
            action = actions[i]
            self.agent_velocities[i] = action
            
            # Update position
            self.agent_positions[i] += action
            
            # Clip positions to grid boundaries
            self.agent_positions[i] = np.clip(
                self.agent_positions[i], 
                0, 
                np.array(self.grid_size)
            )
        
        # Update state
        self.state = {
            'agent_positions': self.agent_positions.copy(),
            'agent_velocities': self.agent_velocities.copy(),
            'landmarks': self.landmarks.copy(),
            'step': self.current_step
        }
        
        # Calculate rewards
        rewards = self._compute_rewards()
        
        # Check if episode is done
        dones = [False] * self.n_agents
        if self.current_step >= self.max_steps:
            dones = [True] * self.n_agents
        
        # Check if all agents have reached their landmarks
        distances = np.linalg.norm(self.agent_positions - self.landmarks, axis=1)
        if np.all(distances < self.landmark_radius):
            dones = [True] * self.n_agents
        
        # Get observations
        observations = self.get_obs()
        
        # Create info dictionaries
        infos = [{} for _ in range(self.n_agents)]
        
        return observations, rewards, dones, infos
    
    def _compute_rewards(self):
        """
        Compute rewards for all agents.
        
        Returns
        -------
        numpy.ndarray
            Rewards for each agent
        """
        rewards = np.zeros(self.n_agents)
        
        # Calculate distances to landmarks
        landmark_distances = np.linalg.norm(
            self.agent_positions - self.landmarks, 
            axis=1
        )
        
        # Distance-based rewards
        rewards += self.distance_factor * landmark_distances
        
        # Goal rewards for reaching landmarks
        goal_reached = landmark_distances < self.landmark_radius
        rewards += goal_reached * self.goal_reward
        
        # Collision penalties
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                distance = np.linalg.norm(
                    self.agent_positions[i] - self.agent_positions[j]
                )
                if distance < 2 * self.agent_radius:
                    rewards[i] += self.collision_penalty
                    rewards[j] += self.collision_penalty
        
        return rewards
    
    def get_obs(self):
        """
        Get observations for all agents.
        
        Returns
        -------
        list of numpy.ndarray
            Observations for each agent
        """
        observations = []
        
        for i in range(self.n_agents):
            # Agent's own position and velocity
            own_pos = self.agent_positions[i]
            own_vel = self.agent_velocities[i]
            
            # Vector to target landmark
            landmark_vec = self.landmarks[i] - own_pos
            
            # Relative positions of other agents
            other_agents = []
            for j in range(self.n_agents):
                if i != j:
                    other_pos = self.agent_positions[j] - own_pos
                    # Check if within sight radius
                    if np.linalg.norm(other_pos) <= self.sight_radius:
                        other_agents.append(other_pos)
                    else:
                        other_agents.append(np.zeros(2))
            
            # Concatenate all observations
            obs = np.concatenate([
                own_pos, 
                own_vel, 
                landmark_vec, 
                *other_agents
            ])
            
            observations.append(obs)
        
        return observations
    
    def get_rewards(self, state, actions):
        """
        Calculate rewards for all agents based on state and actions.
        
        Parameters
        ----------
        state : dict
            Current state
        actions : list
            List of actions
            
        Returns
        -------
        numpy.ndarray
            Rewards for each agent
        """
        agent_positions = state['agent_positions']
        landmarks = state['landmarks']
        
        rewards = np.zeros(self.n_agents)
        
        # Calculate distances to landmarks
        landmark_distances = np.linalg.norm(
            agent_positions - landmarks, 
            axis=1
        )
        
        # Distance-based rewards
        rewards += self.distance_factor * landmark_distances
        
        # Goal rewards for reaching landmarks
        goal_reached = landmark_distances < self.landmark_radius
        rewards += goal_reached * self.goal_reward
        
        # Collision penalties
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                distance = np.linalg.norm(
                    agent_positions[i] - agent_positions[j]
                )
                if distance < 2 * self.agent_radius:
                    rewards[i] += self.collision_penalty
                    rewards[j] += self.collision_penalty
        
        return rewards
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Parameters
        ----------
        mode : str
            Rendering mode ('human' or 'rgb_array')
            
        Returns
        -------
        object
            None for 'human' mode, numpy array for 'rgb_array' mode
        """
        if self.screen is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption('Cooperative Navigation')
        
        if self.state is None:
            return None
        
        # Convert grid coordinates to screen coordinates
        scale_x = self.screen_size[0] / self.grid_size[0]
        scale_y = self.screen_size[1] / self.grid_size[1]
        
        def grid_to_screen(pos):
            return int(pos[0] * scale_x), int(pos[1] * scale_y)
        
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw landmarks
        for i, landmark in enumerate(self.landmarks):
            screen_pos = grid_to_screen(landmark)
            pygame.draw.circle(
                self.screen, 
                self.landmark_color, 
                screen_pos, 
                int(self.landmark_radius * scale_x)
            )
        
        # Draw agents
        for i, pos in enumerate(self.agent_positions):
            screen_pos = grid_to_screen(pos)
            color = self.agent_colors[i % len(self.agent_colors)]
            pygame.draw.circle(
                self.screen, 
                color, 
                screen_pos, 
                int(self.agent_radius * scale_x)
            )
            
            # Draw a line to the target landmark
            landmark_pos = grid_to_screen(self.landmarks[i])
            pygame.draw.line(
                self.screen, 
                color, 
                screen_pos, 
                landmark_pos, 
                1
            )
        
        if mode == 'human':
            pygame.event.pump()
            pygame.display.flip()
            return None
        elif mode == 'rgb_array':
            screen_data = pygame.surfarray.array3d(self.screen)
            return screen_data
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None 