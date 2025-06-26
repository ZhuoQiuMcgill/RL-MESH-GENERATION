import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import os
from collections import defaultdict

from ..networks import MeshActor, MeshCritic, soft_update_target_network
from ..replay_buffer import MeshReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic agent for mesh generation.

    Implements the SAC algorithm with entropy regularization for
    stable and efficient learning in the mesh generation domain.
    """

    def __init__(self, config: Dict, device: torch.device = torch.device("cpu")):
        """
        Initialize SAC agent.

        Args:
            config: Configuration dictionary
            device: Device to run computations on
        """
        self.config = config
        self.device = device

        # Algorithm parameters (ensure proper types)
        self.lr = float(config['sac']['learning_rate'])
        self.gamma = float(config['sac']['discount_factor'])
        self.tau = float(config['sac']['tau'])
        self.batch_size = int(config['sac']['batch_size'])
        self.buffer_size = int(config['sac']['buffer_size'])
        self.gradient_steps = int(config['sac']['gradient_steps'])

        # Alpha (temperature) parameter configuration
        self.use_static_alpha = config['sac'].get('use_static_alpha', False)
        if self.use_static_alpha:
            self.alpha = float(config['sac'].get('static_alpha', 0.1))
            self.log_alpha = None
            self.alpha_optimizer = None
            print(f"Using static alpha: {self.alpha}")
        else:
            self.alpha = float(config['sac']['alpha'])
            # Automatic entropy tuning
            self.target_entropy = -3.0  # For 3D action space
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
            print(f"Using automatic alpha tuning, initial alpha: {self.alpha}")

        # Network parameters (ensure proper types)
        self.n_neighbors = int(config['environment']['n_neighbors'])
        self.n_fan_points = int(config['environment']['n_fan_points'])
        self.hidden_layers = config['networks']['actor_hidden_layers']

        # Initialize networks
        self._build_networks()

        # Initialize optimizers
        self._build_optimizers()

        # Initialize replay buffer
        self.replay_buffer = MeshReplayBuffer(self.buffer_size, device)

        # Training statistics
        self.training_stats = defaultdict(list)
        self.total_updates = 0

    def _build_networks(self):
        """Build actor and critic networks."""
        # Actor network
        self.actor = MeshActor(
            self.n_neighbors, self.n_fan_points,
            self.hidden_layers
        ).to(self.device)

        # Critic networks (double critic)
        self.critic = MeshCritic(
            self.n_neighbors, self.n_fan_points,
            self.hidden_layers
        ).to(self.device)

        # Target critic networks
        self.target_critic = MeshCritic(
            self.n_neighbors, self.n_fan_points,
            self.hidden_layers
        ).to(self.device)

        # Initialize target networks
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _build_optimizers(self):
        """Build optimizers for networks."""
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_action(self, state_dict: Dict, deterministic: bool = False) -> np.ndarray:
        """
        Select action based on current policy.

        Args:
            state_dict: Current state dictionary
            deterministic: Whether to use deterministic action

        Returns:
            Action array
        """
        with torch.no_grad():
            # Convert state dict to tensors if needed
            state_dict = self._ensure_tensor_state(state_dict)

            # Get action from actor
            action = self.actor.get_action(state_dict, deterministic)

            return action.cpu().numpy().flatten()

    def _ensure_tensor_state(self, state_dict: Dict) -> Dict:
        """Ensure all state components are tensors on correct device."""
        tensor_state = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                tensor_state[key] = value.to(self.device)
            elif isinstance(value, np.ndarray):
                tensor_state[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
            else:
                tensor_state[key] = torch.tensor([value], dtype=torch.float32, device=self.device)
        return tensor_state

    def store_transition(self, state_dict: Dict, action: np.ndarray, reward: float,
                         next_state_dict: Dict, done: bool):
        """
        Store transition in replay buffer.

        Args:
            state_dict: Current state dictionary
            action: Action taken
            reward: Reward received
            next_state_dict: Next state dictionary
            done: Whether episode ended
        """
        self.replay_buffer.add(state_dict, action, reward, next_state_dict, done)

    def update(self) -> Dict:
        """
        Update actor and critic networks.

        Returns:
            Dictionary of training statistics
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}

        stats = {}

        for _ in range(self.gradient_steps):
            # Sample batch from replay buffer
            batch = self.replay_buffer.sample(self.batch_size)

            # Update critic
            critic_loss = self._update_critic(batch)
            stats['critic_loss'] = critic_loss

            # Update actor
            actor_loss = self._update_actor(batch)
            stats['actor_loss'] = actor_loss

            # Update alpha (temperature parameter) only if not using static alpha
            if not self.use_static_alpha:
                alpha_loss = self._update_alpha(batch)
                stats['alpha_loss'] = alpha_loss
            else:
                # For static alpha, just record the current value
                stats['alpha_loss'] = 0.0

            stats['alpha'] = self.alpha

            # Update target networks
            soft_update_target_network(self.target_critic, self.critic, self.tau)

            self.total_updates += 1

        # Store training statistics
        for key, value in stats.items():
            self.training_stats[key].append(value)

        return stats

    def _update_critic(self, batch: Dict) -> float:
        """Update critic networks."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Compute target Q-values
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q

        # Get current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, batch: Dict) -> float:
        """Update actor network."""
        states = batch['states']

        # Sample actions from current policy
        actions, log_probs = self.actor.sample(states)

        # Compute Q-values for sampled actions
        q1, q2 = self.critic(states, actions)
        q_value = torch.min(q1, q2)

        # Actor loss (maximize Q-value and entropy)
        actor_loss = (self.alpha * log_probs - q_value).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, batch: Dict) -> float:
        """Update alpha (temperature parameter) - only called when not using static alpha."""
        if self.use_static_alpha:
            return 0.0

        states = batch['states']

        with torch.no_grad():
            _, log_probs = self.actor.sample(states)

        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()

        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update alpha value
        self.alpha = self.log_alpha.exp().item()

        return alpha_loss.item()

    def save(self, filepath: str):
        """
        Save agent state to file.

        Args:
            filepath: Path to save file
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_updates': self.total_updates,
            'configs': self.config,
            'use_static_alpha': self.use_static_alpha,
            'alpha': self.alpha
        }

        # Only save alpha-related parameters if using automatic tuning
        if not self.use_static_alpha:
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
            checkpoint['log_alpha'] = self.log_alpha

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """
        Load agent state from file.

        Args:
            filepath: Path to save file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']

        # Load alpha configuration
        saved_use_static = checkpoint.get('use_static_alpha', False)
        if saved_use_static and self.use_static_alpha:
            # Both using static alpha
            self.alpha = checkpoint.get('alpha', self.alpha)
        elif not saved_use_static and not self.use_static_alpha:
            # Both using automatic alpha tuning
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
        else:
            # Configuration mismatch - use current configuration
            print(f"Warning: Alpha configuration mismatch. Using current config: static={self.use_static_alpha}")

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return dict(self.training_stats)

    def reset_training_stats(self):
        """Reset training statistics."""
        self.training_stats = defaultdict(list)

    def set_training_mode(self, training: bool = True):
        """Set training mode for networks."""
        self.actor.train(training)
        self.critic.train(training)
        self.target_critic.train(training)

    def get_network_info(self) -> Dict:
        """Get information about network architectures."""

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'actor_parameters': count_parameters(self.actor),
            'critic_parameters': count_parameters(self.critic),
            'total_parameters': count_parameters(self.actor) + count_parameters(self.critic),
            'total_updates': self.total_updates,
            'use_static_alpha': self.use_static_alpha,
            'current_alpha': self.alpha
        }