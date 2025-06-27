import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import os
from collections import defaultdict

from ..networks import MeshActor, MeshCritic, SimpleMeshActor, SimpleMeshCritic, soft_update_target_network
from ..replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic agent for mesh generation.
    Modified to work with flat array observations matching original author's approach.
    """

    def __init__(self, config: Dict, device: torch.device = torch.device("cpu")):
        """
        Initialize SAC agent.
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
            self.target_entropy = float(config['sac'].get('target_entropy', -3.0))
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
            print(f"Using automatic alpha tuning, initial alpha: {self.alpha}, target entropy: {self.target_entropy}")

        # Environment parameters for network sizing
        self.neighbor_num = config['environment'].get('neighbor_num', 6)
        self.radius_num = config['environment'].get('radius_num', 3)

        # Calculate state dimension (flat array)
        self.state_dim = 2 * (self.neighbor_num + self.radius_num)
        self.action_dim = 3  # [rule_type, x, y]

        # Network architecture parameters
        self.hidden_layers = config['networks'].get('actor_hidden_layers', [128, 128, 128])

        # Initialize networks
        self._build_networks()

        # Initialize optimizers
        self._build_optimizers()

        # Initialize replay buffer (simplified for flat arrays)
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            state_shape=(self.state_dim,),
            action_dim=self.action_dim,
            device=device
        )

        # Training statistics
        self.training_stats = defaultdict(list)
        self.total_updates = 0

        print(f"SAC Agent initialized with state_dim={self.state_dim}, action_dim={self.action_dim}")

    def _build_networks(self):
        """Build actor and critic networks."""
        # Use simplified networks that match original author's approach
        use_simple = self.config.get('use_simple_networks', True)

        if use_simple:
            # Simple networks matching original approach
            hidden_dim = self.hidden_layers[0] if self.hidden_layers else 128
            num_layers = len(self.hidden_layers) if self.hidden_layers else 3

            self.actor = SimpleMeshActor(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(self.device)

            # Double critic for SAC
            self.critic1 = SimpleMeshCritic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(self.device)

            self.critic2 = SimpleMeshCritic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(self.device)

            # Target critics
            self.target_critic1 = SimpleMeshCritic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(self.device)

            self.target_critic2 = SimpleMeshCritic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(self.device)

        else:
            # Original complex networks
            self.actor = MeshActor(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_layers=self.hidden_layers
            ).to(self.device)

            self.critic = MeshCritic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_layers=self.hidden_layers
            ).to(self.device)

            # Target critic networks
            self.target_critic = MeshCritic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_layers=self.hidden_layers
            ).to(self.device)

        # Initialize target networks
        if use_simple:
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        else:
            self.target_critic.load_state_dict(self.critic.state_dict())

    def _build_optimizers(self):
        """Build optimizers for networks."""
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)

        use_simple = self.config.get('use_simple_networks', True)
        if use_simple:
            self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
            self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        else:
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_action(self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False) -> np.ndarray:
        """
        Select action based on current policy.

        Args:
            state: Current state (flat array)
            deterministic: Whether to use deterministic action

        Returns:
            Action array
        """
        with torch.no_grad():
            # Convert to tensor if needed
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)

            # Ensure correct shape
            if state.dim() == 1:
                state = state.unsqueeze(0)

            # Get action from actor
            action = self.actor.get_action(state, deterministic)

            return action.cpu().numpy().flatten()

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool):
        """
        Store transition in replay buffer.

        Args:
            state: Current state (flat array)
            action: Action taken
            reward: Reward received
            next_state: Next state (flat array)
            done: Whether episode ended
        """
        # Convert to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        self.replay_buffer.add(state_tensor, action_tensor, reward, next_state_tensor, done)

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
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            # Update critics
            critic_loss = self._update_critics(states, actions, rewards, next_states, dones)
            stats['critic_loss'] = critic_loss

            # Update actor
            actor_loss = self._update_actor(states)
            stats['actor_loss'] = actor_loss

            # Update alpha (temperature parameter) only if not using static alpha
            if not self.use_static_alpha:
                alpha_loss = self._update_alpha(states)
                stats['alpha_loss'] = alpha_loss
            else:
                stats['alpha_loss'] = 0.0

            stats['alpha'] = self.alpha

            # Update target networks
            use_simple = self.config.get('use_simple_networks', True)
            if use_simple:
                soft_update_target_network(self.target_critic1, self.critic1, self.tau)
                soft_update_target_network(self.target_critic2, self.critic2, self.tau)
            else:
                soft_update_target_network(self.target_critic, self.critic, self.tau)

            self.total_updates += 1

        # Store training statistics
        for key, value in stats.items():
            self.training_stats[key].append(value)

        return stats

    def _update_critics(self, states, actions, rewards, next_states, dones) -> float:
        """Update critic networks."""
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Compute target Q-values
            use_simple = self.config.get('use_simple_networks', True)
            if use_simple:
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            else:
                target_q1, target_q2 = self.target_critic(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs

            target_q = rewards + (1 - dones.float()) * self.gamma * target_q

        # Get current Q-values
        if use_simple:
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)

            # Compute critic losses
            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)

            # Update critics
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            critic_loss = critic1_loss.item() + critic2_loss.item()
        else:
            current_q1, current_q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            critic_loss = critic_loss.item()

        return critic_loss

    def _update_actor(self, states) -> float:
        """Update actor network."""
        # Sample actions from current policy
        actions, log_probs = self.actor.sample(states)

        # Compute Q-values for sampled actions
        use_simple = self.config.get('use_simple_networks', True)
        if use_simple:
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            q_value = torch.min(q1, q2)
        else:
            q1, q2 = self.critic(states, actions)
            q_value = torch.min(q1, q2)

        # Actor loss (maximize Q-value and entropy)
        actor_loss = (self.alpha * log_probs - q_value).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, states) -> float:
        """Update alpha (temperature parameter) - only called when not using static alpha."""
        if self.use_static_alpha:
            return 0.0

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
        use_simple = self.config.get('use_simple_networks', True)

        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'total_updates': self.total_updates,
            'config': self.config,
            'use_static_alpha': self.use_static_alpha,
            'alpha': self.alpha,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }

        if use_simple:
            checkpoint.update({
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'target_critic1_state_dict': self.target_critic1.state_dict(),
                'target_critic2_state_dict': self.target_critic2.state_dict(),
                'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            })
        else:
            checkpoint.update({
                'critic_state_dict': self.critic.state_dict(),
                'target_critic_state_dict': self.target_critic.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            })

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
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']

        use_simple = self.config.get('use_simple_networks', True)
        if use_simple:
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
            self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        else:
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

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

        use_simple = self.config.get('use_simple_networks', True)
        if use_simple:
            self.critic1.train(training)
            self.critic2.train(training)
            self.target_critic1.train(training)
            self.target_critic2.train(training)
        else:
            self.critic.train(training)
            self.target_critic.train(training)

    def get_network_info(self) -> Dict:
        """Get information about network architectures."""

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        use_simple = self.config.get('use_simple_networks', True)
        if use_simple:
            critic_params = count_parameters(self.critic1) + count_parameters(self.critic2)
        else:
            critic_params = count_parameters(self.critic)

        return {
            'actor_parameters': count_parameters(self.actor),
            'critic_parameters': critic_params,
            'total_parameters': count_parameters(self.actor) + critic_params,
            'total_updates': self.total_updates,
            'use_static_alpha': self.use_static_alpha,
            'current_alpha': self.alpha,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
