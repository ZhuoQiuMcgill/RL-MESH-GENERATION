import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import os
from collections import defaultdict

from .networks import MeshActor, MeshCritic, soft_update_target_network
from .replay_buffer import MeshReplayBuffer


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
        self.alpha = float(config['sac']['alpha'])
        self.batch_size = int(config['sac']['batch_size'])
        self.buffer_size = int(config['sac']['buffer_size'])
        self.gradient_steps = int(config['sac']['gradient_steps'])

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

        # Automatic entropy tuning
        self.target_entropy = -3.0  # For 3D action space
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

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
            if isinstance(value, np.ndarray):
                tensor_state[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
            elif isinstance(value, torch.Tensor):
                tensor_state[key] = value.to(self.device)
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

            # Update alpha (temperature parameter)
            alpha_loss = self._update_alpha(batch)
            stats['alpha_loss'] = alpha_loss
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
        """Update alpha (temperature parameter)."""
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
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'total_updates': self.total_updates,
            'configs': self.config
        }

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
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.total_updates = checkpoint['total_updates']

        # Update alpha value
        self.alpha = self.log_alpha.exp().item()

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
            'total_updates': self.total_updates
        }


class MeshSACTrainer:
    """
    High-level trainer for SAC agent in mesh generation.
    """

    def __init__(self, agent: SACAgent, env, config: Dict):
        """
        Initialize trainer.

        Args:
            agent: SAC agent
            env: Mesh environment
            config: Configuration dictionary
        """
        self.agent = agent
        self.env = env
        self.config = config

        # Training parameters (ensure proper types)
        self.total_timesteps = int(config['training']['total_timesteps'])
        self.eval_freq = int(config['training']['evaluation_freq'])
        self.log_interval = int(config['training']['log_interval'])
        self.save_freq = int(config['training']['save_freq'])

        # Paths
        self.models_dir = config['paths']['models_dir']
        self.logs_dir = config['paths']['logs_dir']

        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.evaluation_results = []

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def train(self) -> Dict:
        """
        Train the agent.

        Returns:
            Training results dictionary
        """
        print("Starting SAC training for mesh generation...")

        state_dict, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        timestep = 0
        episode = 0

        while timestep < self.total_timesteps:
            # Select action
            action = self.agent.select_action(state_dict, deterministic=False)

            # Take step in environment
            next_state_dict, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.store_transition(state_dict, action, reward, next_state_dict, done)

            # Update tracking
            episode_reward += reward
            episode_length += 1
            timestep += 1

            # Update agent
            if timestep > self.config['sac']['batch_size']:
                self.agent.update()

            # Handle episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                if episode % self.log_interval == 0:
                    self._log_progress(episode, timestep, episode_reward, episode_length)

                # Reset for next episode
                state_dict, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode += 1
            else:
                state_dict = next_state_dict

            # Evaluation
            if timestep % self.eval_freq == 0:
                eval_results = self._evaluate()
                self.evaluation_results.append(eval_results)

            # Save model
            if timestep % self.save_freq == 0:
                model_path = os.path.join(self.models_dir, f"model_{timestep}.pt")
                self.agent.save(model_path)

        # Final save
        final_model_path = os.path.join(self.models_dir, "final_model.pt")
        self.agent.save(final_model_path)

        print("Training completed!")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'evaluation_results': self.evaluation_results,
            'training_stats': self.agent.get_training_stats()
        }

    def _evaluate(self, num_episodes: int = 5) -> Dict:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Evaluation results
        """
        self.agent.set_training_mode(False)

        eval_rewards = []
        eval_lengths = []
        mesh_qualities = []

        for _ in range(num_episodes):
            state_dict, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            while True:
                action = self.agent.select_action(state_dict, deterministic=True)
                next_state_dict, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

                state_dict = next_state_dict

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

            # Get mesh quality metrics
            quality_metrics = self.env.get_mesh_quality_metrics()
            if quality_metrics:
                mesh_qualities.append(quality_metrics)

        self.agent.set_training_mode(True)

        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths)
        }

        if mesh_qualities:
            results['mean_mesh_quality'] = np.mean([q['mean_element_quality'] for q in mesh_qualities])

        return results

    def _log_progress(self, episode: int, timestep: int, reward: float, length: int):
        """Log training progress."""
        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(
            self.episode_rewards)
        avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(
            self.episode_lengths)

        print(f"Episode {episode}, Timestep {timestep}")
        print(f"  Reward: {reward:.2f} (avg: {avg_reward:.2f})")
        print(f"  Length: {length} (avg: {avg_length:.1f})")
        print(f"  Alpha: {self.agent.alpha:.4f}")
        print()

    def save_training_results(self, filepath: str):
        """Save training results to file."""
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'evaluation_results': self.evaluation_results,
            'training_stats': self.agent.get_training_stats(),
            'configs': self.config
        }

        torch.save(results, filepath)