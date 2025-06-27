import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
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
        Initialize SAC agent with dimension validation.
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
        self._initialize_replay_buffer()

        # Training statistics
        self.training_stats = defaultdict(list)
        self.total_updates = 0

        print(f"SAC Agent initialized with state_dim={self.state_dim}, action_dim={self.action_dim}")

    def _initialize_replay_buffer(self):
        """Initialize replay buffer with correct dimensions."""
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            state_shape=(self.state_dim,),
            action_dim=self.action_dim,
            device=self.device
        )
        print(f"Replay buffer initialized with state_shape=({self.state_dim},)")

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

            # Double critic for SAC - using Critic directly since SimpleMeshCritic only creates single critic
            from ..networks import Critic
            self.critic1 = Critic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_layers=self.hidden_layers
            ).to(self.device)

            self.critic2 = Critic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_layers=self.hidden_layers
            ).to(self.device)

            # Target critics
            self.target_critic1 = Critic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_layers=self.hidden_layers
            ).to(self.device)

            self.target_critic2 = Critic(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_layers=self.hidden_layers
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

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state: Current state observation (flat array)
            deterministic: Whether to use deterministic action selection (for evaluation)

        Returns:
            Selected action as a 1D numpy array
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if deterministic:
                action_tensor = self.actor.get_action(state_tensor, deterministic=True)
            else:
                action_tensor, _ = self.actor.sample(state_tensor)

            return action_tensor.cpu().numpy().flatten()

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool):
        """
        Store transition in replay buffer with dimension validation.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Validate state dimensions
        if len(state) != self.state_dim:
            raise ValueError(f"State dimension mismatch. Expected {self.state_dim}, got {len(state)}")

        if len(next_state) != self.state_dim:
            raise ValueError(f"Next state dimension mismatch. Expected {self.state_dim}, got {len(next_state)}")

        # Convert to tensors
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)

        # Add to buffer
        self.replay_buffer.add(state_tensor, action_tensor, reward, next_state_tensor, done)

    def update_networks(self) -> Dict:
        """Update networks using SAC algorithm."""
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Update critics
        critic_loss1, critic_loss2 = self._update_critics(states, actions, rewards, next_states, dones)

        # Update actor
        actor_loss = self._update_actor(states)

        # Update target networks
        self._update_target_networks()

        # Update alpha if using automatic tuning
        alpha_loss = 0.0
        if not self.use_static_alpha:
            alpha_loss = self._update_alpha(states)

        self.total_updates += 1

        # Store training statistics
        stats = {
            'critic_loss1': critic_loss1.item(),
            'critic_loss2': critic_loss2.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            'alpha': self.alpha,
            'total_updates': self.total_updates
        }

        for key, value in stats.items():
            self.training_stats[key].append(value)

        return stats

    def update(self) -> Dict:
        """Update method expected by trainer (alias for update_networks)."""
        return self.update_networks()

    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks."""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)

            use_simple = self.config.get('use_simple_networks', True)
            if use_simple:
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            else:
                target_q1, target_q2 = self.target_critic(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs

            target_q = rewards + (1 - dones.float()) * self.gamma * target_q

        # Update critic networks
        if use_simple:
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)

            critic_loss1 = nn.MSELoss()(current_q1, target_q)
            critic_loss2 = nn.MSELoss()(current_q2, target_q)

            self.critic1_optimizer.zero_grad()
            critic_loss1.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic_loss2.backward()
            self.critic2_optimizer.step()

            return critic_loss1, critic_loss2
        else:
            current_q1, current_q2 = self.critic(states, actions)

            critic_loss1 = nn.MSELoss()(current_q1, target_q)
            critic_loss2 = nn.MSELoss()(current_q2, target_q)

            total_critic_loss = critic_loss1 + critic_loss2

            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            self.critic_optimizer.step()

            return critic_loss1, critic_loss2

    def _update_actor(self, states):
        """Update actor network."""
        actions, log_probs = self.actor.sample(states)

        use_simple = self.config.get('use_simple_networks', True)
        if use_simple:
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            q = torch.min(q1, q2)
        else:
            q1, q2 = self.critic(states, actions)
            q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss

    def _update_target_networks(self):
        """Soft update target networks."""
        use_simple = self.config.get('use_simple_networks', True)
        if use_simple:
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
        else:
            self._soft_update(self.target_critic, self.critic)

    def _soft_update(self, target_network, source_network):
        """Perform soft update of target network."""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def _update_alpha(self, states):
        """Update temperature parameter alpha."""
        if self.use_static_alpha:
            return 0.0

        actions, log_probs = self.actor.sample(states)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()
        return alpha_loss

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
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'use_static_alpha': self.use_static_alpha,
            'alpha': self.alpha
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

        if not self.use_static_alpha:
            checkpoint.update({
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            })

        torch.save(checkpoint, filepath)

    def load(self, filepath: str, force_reload: bool = False):
        """
        Load agent state from file with dimension validation.

        Args:
            filepath: Path to load file
            force_reload: If True, skip dimension validation and reinitialize networks
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Check dimension compatibility
        saved_state_dim = checkpoint.get('state_dim', None)
        saved_action_dim = checkpoint.get('action_dim', None)

        if saved_state_dim is not None and saved_action_dim is not None:
            if saved_state_dim != self.state_dim or saved_action_dim != self.action_dim:
                if not force_reload:
                    print(
                        f"Warning: Dimension mismatch detected. Saved: state_dim={saved_state_dim}, action_dim={saved_action_dim}")
                    print(f"Current: state_dim={self.state_dim}, action_dim={self.action_dim}")
                    print("Skipping model loading due to dimension mismatch. Use force_reload=True to override.")
                    return
                else:
                    print(f"Warning: Dimension mismatch detected, but force_reload=True")
                    print(f"Proceeding with loading, networks may be incompatible")

        # Load networks
        try:
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

            print(f"Successfully loaded model from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            if not force_reload:
                raise

    def clear_replay_buffer(self):
        """Clear replay buffer and reinitialize with correct dimensions."""
        self._initialize_replay_buffer()
        print("Replay buffer cleared and reinitialized")

    def clear_replay_buffer(self):
        """Clear replay buffer and reinitialize with correct dimensions."""
        self._initialize_replay_buffer()
        print("Replay buffer cleared and reinitialized")

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
