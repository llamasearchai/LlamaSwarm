"""
Implementation of Multi-Agent Proximal Policy Optimization (MAPPO).

MAPPO extends PPO to the multi-agent setting with centralized training
and decentralized execution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from ..base_algorithm import BaseAlgorithm


class ActorNetwork(nn.Module):
    """
    Actor network for MAPPO.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input observations
    output_dim : int
        Dimension of action space
    hidden_dim : int, optional
        Dimension of hidden layers
    discrete : bool, optional
        Whether the action space is discrete or continuous
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, discrete=True):
        super(ActorNetwork, self).__init__()
        
        self.discrete = discrete
        
        # Shared network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers depend on action type
        if discrete:
            self.action_out = nn.Linear(hidden_dim, output_dim)
        else:
            self.mean = nn.Linear(hidden_dim, output_dim)
            self.log_std = nn.Parameter(torch.zeros(1, output_dim))
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Distribution
            Action distribution
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.discrete:
            action_probs = F.softmax(self.action_out(x), dim=-1)
            return Categorical(action_probs)
        else:
            action_mean = self.mean(x)
            action_std = self.log_std.exp().expand_as(action_mean)
            return Normal(action_mean, action_std)
    
    def get_log_prob(self, x, actions):
        """
        Get log probability of actions given observations.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        actions : torch.Tensor
            Actions tensor
            
        Returns
        -------
        torch.Tensor
            Log probabilities of actions
        """
        dist = self.forward(x)
        return dist.log_prob(actions)


class CriticNetwork(nn.Module):
    """
    Critic network for MAPPO.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input observations
    hidden_dim : int, optional
        Dimension of hidden layers
    """
    
    def __init__(self, input_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Value estimate
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_out(x)
        return value


class MAPPO(BaseAlgorithm):
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.
    
    MAPPO uses centralized critics and decentralized actors to train
    agents in cooperative environments.
    
    Parameters
    ----------
    n_agents : int
        Number of agents in the environment
    state_dim : int
        Dimension of the state space
    action_dim : int
        Dimension of the action space
    centralized_critic : bool, optional
        Whether to use a centralized critic
    discrete : bool, optional
        Whether the action space is discrete or continuous
    clip_param : float, optional
        PPO clipping parameter
    value_coef : float, optional
        Value loss coefficient
    entropy_coef : float, optional
        Entropy coefficient
    lr : float, optional
        Learning rate
    gamma : float, optional
        Discount factor
    gae_lambda : float, optional
        GAE lambda parameter
    device : str, optional
        Device to use for tensor operations
    """
    
    def __init__(
        self,
        n_agents,
        state_dim,
        action_dim,
        centralized_critic=True,
        discrete=True,
        clip_param=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        device='cpu'
    ):
        self.centralized_critic = centralized_critic
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        
        # Initialize base class
        super(MAPPO, self).__init__(
            n_agents,
            state_dim,
            action_dim,
            discrete,
            lr,
            gamma,
            device
        )
    
    def _init_agents(self):
        """
        Initialize actor and critic networks.
        
        Returns
        -------
        dict
            Dictionary containing actor and critic networks
        """
        actors = []
        
        # Initialize actors for each agent
        for _ in range(self.n_agents):
            actor = ActorNetwork(
                self.state_dim,
                self.action_dim,
                discrete=self.discrete
            ).to(self.device)
            actors.append(actor)
        
        # Initialize critic
        if self.centralized_critic:
            # Centralized critic observes all agents' states
            critic_input_dim = self.state_dim * self.n_agents
            critic = CriticNetwork(critic_input_dim).to(self.device)
        else:
            # Decentralized critics
            critic = [
                CriticNetwork(self.state_dim).to(self.device)
                for _ in range(self.n_agents)
            ]
        
        return {
            'actors': actors,
            'critic': critic
        }
    
    def _init_optimizer(self):
        """
        Initialize optimizers for actor and critic networks.
        
        Returns
        -------
        dict
            Dictionary containing optimizers
        """
        actor_optimizers = []
        
        # Initialize actor optimizers
        for actor in self.agents['actors']:
            optimizer = optim.Adam(actor.parameters(), lr=self.lr)
            actor_optimizers.append(optimizer)
        
        # Initialize critic optimizer
        if self.centralized_critic:
            critic_optimizer = optim.Adam(
                self.agents['critic'].parameters(),
                lr=self.lr
            )
        else:
            critic_optimizer = []
            for critic in self.agents['critic']:
                optimizer = optim.Adam(critic.parameters(), lr=self.lr)
                critic_optimizer.append(optimizer)
        
        return {
            'actor_optimizers': actor_optimizers,
            'critic_optimizer': critic_optimizer
        }
    
    def select_action(self, obs, explore=True):
        """
        Select actions for agents based on observations.
        
        Parameters
        ----------
        obs : list or numpy.ndarray
            Observations for each agent
        explore : bool, optional
            Whether to explore (sample) or exploit (use mean)
            
        Returns
        -------
        numpy.ndarray
            Actions for each agent
        """
        actions = []
        
        # Convert observations to tensors
        if isinstance(obs, list):
            obs = np.array(obs)
        
        with torch.no_grad():
            for i, actor in enumerate(self.agents['actors']):
                obs_tensor = torch.FloatTensor(obs[i]).to(self.device)
                
                # Get action distribution
                dist = actor(obs_tensor.unsqueeze(0))
                
                # Sample action or take mean based on explore flag
                if explore:
                    action = dist.sample()
                else:
                    if self.discrete:
                        action = torch.argmax(dist.probs)
                    else:
                        action = dist.mean
                
                actions.append(action.cpu().numpy().flatten())
        
        return np.array(actions)
    
    def update(self, batch):
        """
        Update algorithm parameters using a batch of experiences.
        
        Parameters
        ----------
        batch : dict
            Batch of experiences
            
        Returns
        -------
        dict
            Dictionary of loss metrics
        """
        # Process batch
        processed_batch = self.process_batch(batch)
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns(processed_batch)
        
        # Update actors and critics
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        for agent_idx in range(self.n_agents):
            # Get data for this agent
            obs = processed_batch['states'][:, agent_idx]
            actions = processed_batch['actions'][:, agent_idx]
            old_log_probs = processed_batch.get('log_probs', None)
            
            if old_log_probs is not None:
                old_log_probs = old_log_probs[:, agent_idx]
            
            # Actor update
            actor = self.agents['actors'][agent_idx]
            actor_optimizer = self.optimizer['actor_optimizers'][agent_idx]
            
            # Compute current log probabilities
            curr_log_probs = actor.get_log_prob(obs, actions)
            
            # If old log probs are not available, use current as old
            if old_log_probs is None:
                old_log_probs = curr_log_probs.detach()
            
            # Compute ratio of probabilities
            ratio = torch.exp(curr_log_probs - old_log_probs)
            
            # Clipped objective function
            surr1 = ratio * advantages[:, agent_idx]
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages[:, agent_idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy regularization
            dist = actor(obs)
            entropy = dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy
            
            # Combine losses
            total_actor_loss = actor_loss + entropy_loss
            
            # Update actor
            actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optimizer.step()
            
            # Critic update
            if self.centralized_critic:
                # For centralized critic, we need all agent observations
                critic_input = processed_batch['states'].reshape(processed_batch['states'].shape[0], -1)
                critic = self.agents['critic']
                critic_optimizer = self.optimizer['critic_optimizer']
            else:
                critic_input = obs
                critic = self.agents['critic'][agent_idx]
                critic_optimizer = self.optimizer['critic_optimizer'][agent_idx]
            
            # Compute value loss
            values = critic(critic_input).squeeze(-1)
            if self.centralized_critic:
                # For centralized critic, the target is the mean of all agent returns
                target_values = returns.mean(dim=1)
            else:
                target_values = returns[:, agent_idx]
            
            critic_loss = self.value_coef * F.mse_loss(values, target_values)
            
            # Update critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()
            
            # Record losses
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy.item())
        
        # Update training info
        self.train_info['policy_loss'].append(np.mean(actor_losses))
        self.train_info['value_loss'].append(np.mean(critic_losses))
        self.train_info['entropy'].append(np.mean(entropy_losses))
        self.train_info['loss'].append(
            np.mean(actor_losses) + np.mean(critic_losses) - np.mean(entropy_losses)
        )
        
        return {
            'policy_loss': np.mean(actor_losses),
            'value_loss': np.mean(critic_losses),
            'entropy': np.mean(entropy_losses),
            'loss': np.mean(actor_losses) + np.mean(critic_losses) - np.mean(entropy_losses)
        }
    
    def process_batch(self, batch):
        """
        Process a batch of experiences before updating.
        
        Parameters
        ----------
        batch : dict
            Batch of experiences
            
        Returns
        -------
        dict
            Processed batch with tensors
        """
        # Convert numpy arrays to tensors
        processed_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                processed_batch[key] = torch.FloatTensor(value).to(self.device)
            else:
                processed_batch[key] = value
        
        return processed_batch
    
    def _compute_advantages_and_returns(self, batch):
        """
        Compute advantages and returns using GAE.
        
        Parameters
        ----------
        batch : dict
            Batch of experiences
            
        Returns
        -------
        tuple
            Tuple of (advantages, returns)
        """
        states = batch['states']
        rewards = batch['rewards']
        dones = batch['dones']
        next_states = batch['next_states']
        
        # Compute values
        if self.centralized_critic:
            # For centralized critic, we need to flatten states
            critic_input = states.reshape(states.shape[0], -1)
            next_critic_input = next_states.reshape(next_states.shape[0], -1)
            
            with torch.no_grad():
                values = self.agents['critic'](critic_input).squeeze(-1)
                next_values = self.agents['critic'](next_critic_input).squeeze(-1)
            
            # Expand values for each agent
            values = values.unsqueeze(1).expand(-1, self.n_agents)
            next_values = next_values.unsqueeze(1).expand(-1, self.n_agents)
        else:
            values = torch.zeros((states.shape[0], self.n_agents), device=self.device)
            next_values = torch.zeros((states.shape[0], self.n_agents), device=self.device)
            
            with torch.no_grad():
                for i, critic in enumerate(self.agents['critic']):
                    values[:, i] = critic(states[:, i]).squeeze(-1)
                    next_values[:, i] = critic(next_states[:, i]).squeeze(-1)
        
        # Compute returns
        returns = rewards + self.gamma * next_values * (1 - dones)
        
        # Compute advantages using GAE
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)
        
        # GAE calculation
        lastgaelam = 0
        for t in reversed(range(deltas.shape[0])):
            lastgaelam = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * lastgaelam
            advantages[t] = lastgaelam
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _get_save_dict(self):
        """
        Get algorithm-specific parameters for saving.
        
        Returns
        -------
        dict
            Dictionary of parameters to save
        """
        save_dict = {
            'agents': {
                'actors_state_dict': [actor.state_dict() for actor in self.agents['actors']]
            },
            'hyperparams': {
                'clip_param': self.clip_param,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'gae_lambda': self.gae_lambda,
                'centralized_critic': self.centralized_critic
            }
        }
        
        # Save critic differently based on centralization
        if self.centralized_critic:
            save_dict['agents']['critic_state_dict'] = self.agents['critic'].state_dict()
        else:
            save_dict['agents']['critic_state_dict'] = [
                critic.state_dict() for critic in self.agents['critic']
            ]
        
        return save_dict
    
    def _set_load_dict(self, checkpoint):
        """
        Set algorithm-specific parameters from loaded checkpoint.
        
        Parameters
        ----------
        checkpoint : dict
            Loaded checkpoint
        """
        # Load hyperparameters
        if 'hyperparams' in checkpoint:
            for key, value in checkpoint['hyperparams'].items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Load actor and critic networks
        if 'agents' in checkpoint:
            # Load actors
            if 'actors_state_dict' in checkpoint['agents']:
                for i, state_dict in enumerate(checkpoint['agents']['actors_state_dict']):
                    if i < len(self.agents['actors']):
                        self.agents['actors'][i].load_state_dict(state_dict)
            
            # Load critic
            if 'critic_state_dict' in checkpoint['agents']:
                if self.centralized_critic:
                    self.agents['critic'].load_state_dict(
                        checkpoint['agents']['critic_state_dict']
                    )
                else:
                    for i, state_dict in enumerate(checkpoint['agents']['critic_state_dict']):
                        if i < len(self.agents['critic']):
                            self.agents['critic'][i].load_state_dict(state_dict) 