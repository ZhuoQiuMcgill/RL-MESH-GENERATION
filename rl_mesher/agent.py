import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Tuple, Optional, List
import os
from collections import defaultdict

from .networks import MeshActor, MeshCritic, soft_update_target_network
from .replay_buffer import MeshReplayBuffer
from .utils.visualization import plot_mesh


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


class MeshSACTrainer:
    """
    Enhanced high-level trainer for SAC agent in mesh generation with comprehensive logging.
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
        self.figures_dir = config['paths']['figures_dir']

        # Training tracking with enhanced metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.episode_mesh_qualities = []
        self.episode_completion_rates = []
        self.evaluation_results = []

        # Additional tracking
        self.timestep_rewards = []
        self.training_losses = []
        self.best_reward = float('-inf')
        self.best_episode = 0

        # Timing and performance tracking
        self.training_start_time = None
        self.episode_start_time = None
        self.last_log_time = None
        self.episodes_per_hour = 0.0

        # Evaluation counter for mesh visualization
        self.eval_count = 0

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def train(self) -> Dict:
        """
        Enhanced training method with comprehensive monitoring.

        Returns:
            Training results dictionary
        """
        print("🚀 Starting SAC training for mesh generation...")
        print("=" * 70)
        self._print_training_setup()

        self.training_start_time = time.time()

        state_dict, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        timestep = 0
        episode = 0

        # Start first episode timer
        self.episode_start_time = time.time()

        while timestep < self.total_timesteps:
            # Select action
            action = self.agent.select_action(state_dict, deterministic=False)

            # Take step in environment
            next_state_dict, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.store_transition(state_dict, action, reward, next_state_dict, done)

            # Track timestep rewards
            self.timestep_rewards.append(reward)

            # Update tracking
            episode_reward += reward
            episode_length += 1
            timestep += 1

            # Update agent
            if timestep > self.config['sac']['batch_size']:
                training_stats = self.agent.update()
                if training_stats:
                    self.training_losses.append(training_stats)

            # Handle episode end
            if done:
                episode_time = time.time() - self.episode_start_time

                # Calculate mesh quality metrics
                mesh_quality_metrics = self.env.get_mesh_quality_metrics()
                mesh_quality = mesh_quality_metrics.get('mean_element_quality', 0.0) if mesh_quality_metrics else 0.0
                completion_rate = 1.0 if terminated else 0.0

                # Store episode data
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_times.append(episode_time)
                self.episode_mesh_qualities.append(mesh_quality)
                self.episode_completion_rates.append(completion_rate)

                # Update best episode tracking
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.best_episode = episode

                # Enhanced logging
                if episode % self.log_interval == 0:
                    self._log_progress(episode, timestep, episode_reward, episode_length,
                                       episode_time, mesh_quality, completion_rate)

                # Reset for next episode
                state_dict, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode += 1
                self.episode_start_time = time.time()
            else:
                state_dict = next_state_dict

            # Evaluation
            if timestep % self.eval_freq == 0:
                eval_results = self._evaluate(timestep)
                self.evaluation_results.append(eval_results)
                self._log_evaluation(timestep, eval_results)

            # Save model
            if timestep % self.save_freq == 0:
                model_path = os.path.join(self.models_dir, f"model_{timestep}.pt")
                self.agent.save(model_path)
                print(f"💾 Model saved: {model_path}")

        # Final save
        final_model_path = os.path.join(self.models_dir, "final_model.pt")
        self.agent.save(final_model_path)

        # Training completion summary
        self._print_training_summary()

        print("🎉 Training completed!")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'episode_mesh_qualities': self.episode_mesh_qualities,
            'episode_completion_rates': self.episode_completion_rates,
            'evaluation_results': self.evaluation_results,
            'training_stats': self.agent.get_training_stats(),
            'timestep_rewards': self.timestep_rewards,
            'training_losses': self.training_losses,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'total_training_time': time.time() - self.training_start_time
        }

    def _evaluate(self, timestep: int, num_episodes: int = 5) -> Dict:
        """
        Enhanced evaluation with detailed metrics collection and mesh visualization.

        Args:
            timestep: Current training timestep
            num_episodes: Number of episodes to evaluate

        Returns:
            Evaluation results
        """
        print(f"\n🔍 Starting evaluation ({num_episodes} episodes)...")
        eval_start_time = time.time()

        self.agent.set_training_mode(False)

        eval_rewards = []
        eval_lengths = []
        eval_times = []
        mesh_qualities = []
        completion_rates = []

        # Track the best episode for visualization
        best_episode_reward = float('-inf')
        best_episode_boundary = None
        best_episode_elements = None

        for episode in range(num_episodes):
            episode_start = time.time()
            state_dict, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            while True:
                action = self.agent.select_action(state_dict, deterministic=True)
                next_state_dict, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    completion_rates.append(1.0 if terminated else 0.0)

                    # Check if this is the best episode for visualization
                    if episode_reward > best_episode_reward:
                        best_episode_reward = episode_reward
                        best_episode_boundary, best_episode_elements = self.env.get_current_mesh()

                    break

                state_dict = next_state_dict

            episode_time = time.time() - episode_start

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_times.append(episode_time)

            # Get mesh quality metrics
            quality_metrics = self.env.get_mesh_quality_metrics()
            if quality_metrics:
                mesh_qualities.append(quality_metrics)

        self.agent.set_training_mode(True)

        eval_total_time = time.time() - eval_start_time

        # Create mesh visualization for the best evaluation episode
        self._save_evaluation_mesh_visualization(timestep, best_episode_boundary,
                                                 best_episode_elements, best_episode_reward)

        results = {
            'timestep': timestep,
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths),
            'mean_episode_time': np.mean(eval_times),
            'completion_rate': np.mean(completion_rates),
            'eval_total_time': eval_total_time
        }

        if mesh_qualities:
            # Calculate average mesh quality metrics
            quality_keys = mesh_qualities[0].keys()
            for key in quality_keys:
                values = [q[key] for q in mesh_qualities if key in q]
                if values:
                    results[f'mean_{key}'] = np.mean(values)
                    results[f'std_{key}'] = np.std(values)

        print(f"✅ Evaluation completed in {eval_total_time:.1f}s")
        self.eval_count += 1
        return results

    def _save_evaluation_mesh_visualization(self, timestep: int, boundary: np.ndarray,
                                            elements: List[np.ndarray], reward: float):
        """
        Save mesh visualization for the current evaluation.

        Args:
            timestep: Current training timestep
            boundary: Mesh boundary
            elements: Generated mesh elements
            reward: Episode reward for this mesh
        """
        try:
            # Create evaluation mesh visualization directory
            eval_mesh_dir = os.path.join(self.figures_dir, "evaluation_meshes")
            os.makedirs(eval_mesh_dir, exist_ok=True)

            # Generate filename with timestep and evaluation count
            mesh_filename = f"eval_{self.eval_count:03d}_timestep_{timestep:,}_reward_{reward:.2f}.png"
            mesh_path = os.path.join(eval_mesh_dir, mesh_filename)

            # Get domain info for title
            domain_info = ""
            if hasattr(self.env, 'domain_file'):
                domain_info = f" ({self.env.domain_file})"
            elif hasattr(self.env, 'get_current_domain_info'):
                domain_data = self.env.get_current_domain_info()
                domain_info = f" ({domain_data.get('domain_file', 'Unknown')})"

            # Create the title
            title = f"Evaluation Mesh - Timestep {timestep:,}{domain_info}\n"
            title += f"Reward: {reward:.2f} | Elements: {len(elements) if elements else 0}"

            # Plot and save the mesh
            plot_mesh(
                boundary=boundary,
                elements=elements,
                title=title,
                save_path=mesh_path,
                figsize=(10, 8),
                show_vertices=True,
                show_element_numbers=False
            )

            print(f"📸 Evaluation mesh saved: {mesh_filename}")

        except Exception as e:
            print(f"⚠️  Failed to save evaluation mesh visualization: {e}")

    def _log_progress(self, episode: int, timestep: int, reward: float, length: int,
                      episode_time: float, mesh_quality: float, completion_rate: float):
        """Enhanced progress logging with detailed information."""

        # Calculate statistics
        recent_episodes = min(100, len(self.episode_rewards))
        avg_reward = np.mean(self.episode_rewards[-recent_episodes:]) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths[-recent_episodes:]) if self.episode_lengths else 0
        avg_time = np.mean(self.episode_times[-recent_episodes:]) if self.episode_times else 0
        avg_quality = np.mean(self.episode_mesh_qualities[-recent_episodes:]) if self.episode_mesh_qualities else 0
        avg_completion = np.mean(
            self.episode_completion_rates[-recent_episodes:]) if self.episode_completion_rates else 0

        # Training time elapsed
        training_time = time.time() - self.training_start_time
        training_hours = training_time / 3600

        # Recent training statistics
        recent_alpha = self.agent.alpha if hasattr(self.agent, 'alpha') else 0

        # Progress percentage
        progress = (timestep / self.total_timesteps) * 100

        # Episodes per hour calculation
        if training_hours > 0:
            self.episodes_per_hour = episode / training_hours

        print(f"\n{'=' * 70}")
        print(f"📊 EPISODE {episode:,} | TIMESTEP {timestep:,} ({progress:.1f}%)")
        print(f"{'=' * 70}")

        # Current episode info
        print(f"🎯 Current Episode:")
        print(f"   Reward: {reward:8.2f} | Length: {length:4d} steps | Time: {episode_time:6.2f}s")
        print(f"   Quality: {mesh_quality:.3f} | Completed: {'✅' if completion_rate > 0 else '❌'}")

        # Recent performance (last 100 episodes)
        print(f"\n📈 Recent Performance (last {recent_episodes} episodes):")
        print(f"   Avg Reward: {avg_reward:8.2f} | Avg Length: {avg_length:6.1f} steps")
        print(f"   Avg Time: {avg_time:8.2f}s | Avg Quality: {avg_quality:.3f}")
        print(f"   Completion Rate: {avg_completion:.1%}")

        # Best performance tracking
        print(f"\n🏆 Best Performance:")
        print(f"   Best Reward: {self.best_reward:8.2f} (Episode {self.best_episode})")

        # Training progress
        print(f"\n⏱️  Training Progress:")
        print(f"   Elapsed Time: {training_hours:.2f} hours")
        print(f"   Episodes/Hour: {self.episodes_per_hour:.1f}")
        print(f"   Timesteps/Hour: {timestep / training_hours if training_hours > 0 else 0:,.0f}")

        # Agent statistics
        print(f"\n🤖 Agent Status:")
        alpha_status = f"Alpha: {recent_alpha:.4f}"
        if self.agent.use_static_alpha:
            alpha_status += " (static)"
        else:
            alpha_status += " (auto)"
        print(f"   {alpha_status}")
        print(f"   Replay Buffer Size: {len(self.agent.replay_buffer):,}")
        print(f"   Total Updates: {self.agent.total_updates:,}")

        # Memory usage if available
        if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
                memory_cached = torch.cuda.memory_reserved() / 1024 ** 3  # GB
                print(f"   GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
            except:
                pass

        print(f"{'=' * 70}")

    def _print_training_setup(self):
        """Print training configuration and setup information."""
        print(f"🚀 Training Configuration:")
        print(f"   Total Timesteps: {self.total_timesteps:,}")
        print(f"   Evaluation Frequency: {self.eval_freq:,}")
        print(f"   Log Interval: {self.log_interval}")
        print(f"   Save Frequency: {self.save_freq:,}")
        print(f"   Batch Size: {self.config['sac']['batch_size']}")
        print(f"   Learning Rate: {self.config['sac']['learning_rate']}")
        print(f"   Buffer Size: {self.config['sac']['buffer_size']:,}")

        # Alpha configuration info
        if self.agent.use_static_alpha:
            print(f"   Alpha: {self.agent.alpha} (static)")
        else:
            print(f"   Alpha: {self.agent.alpha} (automatic tuning)")
        print()

    def _log_evaluation(self, timestep: int, eval_results: Dict):
        """Log evaluation results."""
        print(f"\n🔍 EVALUATION at timestep {timestep:,}")
        print(f"   Mean Reward: {eval_results['mean_reward']:8.2f} ± {eval_results['std_reward']:.2f}")
        print(f"   Mean Length: {eval_results['mean_length']:8.1f} ± {eval_results['std_length']:.1f}")
        if 'mean_mean_element_quality' in eval_results:
            print(f"   Mean Quality: {eval_results['mean_mean_element_quality']:.3f}")
        print(f"   Completion Rate: {eval_results['completion_rate']:.1%}")

    def _print_training_summary(self):
        """Print comprehensive training summary."""
        total_training_time = time.time() - self.training_start_time
        total_hours = total_training_time / 3600

        print(f"\n{'=' * 70}")
        print(f"🎉 TRAINING COMPLETED!")
        print(f"{'=' * 70}")

        if self.episode_rewards:
            print(f"📊 Final Statistics:")
            print(f"   Total Episodes: {len(self.episode_rewards):,}")
            print(f"   Total Training Time: {total_hours:.2f} hours")
            print(f"   Final Reward: {self.episode_rewards[-1]:.2f}")
            print(f"   Best Reward: {self.best_reward:.2f} (Episode {self.best_episode})")
            print(f"   Mean Episode Time: {np.mean(self.episode_times):.2f}s")
            if self.episode_mesh_qualities:
                print(f"   Final Quality: {self.episode_mesh_qualities[-1]:.3f}")
                print(f"   Mean Quality: {np.mean(self.episode_mesh_qualities):.3f}")
            print(f"   Overall Completion Rate: {np.mean(self.episode_completion_rates):.1%}")
            print(f"   Episodes per Hour: {self.episodes_per_hour:.1f}")

            # Performance trends
            if len(self.episode_rewards) >= 200:
                early_rewards = np.mean(self.episode_rewards[:100])
                late_rewards = np.mean(self.episode_rewards[-100:])
                improvement = late_rewards - early_rewards

                print(f"\n📈 Learning Progress:")
                print(f"   Early Performance: {early_rewards:.2f}")
                print(f"   Late Performance: {late_rewards:.2f}")
                print(
                    f"   Improvement: {improvement:+.2f} ({(improvement / abs(early_rewards) * 100) if early_rewards != 0 else 0:+.1f}%)")

        print(f"{'=' * 70}")

        # Evaluation visualization summary
        if self.eval_count > 0:
            eval_mesh_dir = os.path.join(self.figures_dir, "evaluation_meshes")
            print(f"📸 Evaluation mesh visualizations saved to: {eval_mesh_dir}")
            print(f"   Total evaluation meshes: {self.eval_count}")

    def save_training_results(self, filepath: str):
        """Save comprehensive training results to file."""
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'episode_mesh_qualities': self.episode_mesh_qualities,
            'episode_completion_rates': self.episode_completion_rates,
            'evaluation_results': self.evaluation_results,
            'training_stats': self.agent.get_training_stats(),
            'timestep_rewards': self.timestep_rewards,
            'training_losses': self.training_losses,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'total_training_time': time.time() - self.training_start_time if self.training_start_time else 0,
            'configs': self.config
        }

        torch.save(results, filepath)
        print(f"📊 Training results saved to: {filepath}")

        # Also save a human-readable summary
        summary_path = filepath.replace('.pt', '_summary.txt')
        self._save_text_summary(summary_path, results)

    def _save_text_summary(self, filepath: str, results: Dict):
        """Save human-readable training summary."""
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MESH GENERATION TRAINING SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            if results['episode_rewards']:
                f.write(f"Training Episodes: {len(results['episode_rewards']):,}\n")
                f.write(f"Total Training Time: {results['total_training_time'] / 3600:.2f} hours\n")
                f.write(f"Final Reward: {results['episode_rewards'][-1]:.2f}\n")
                f.write(f"Best Reward: {results['best_reward']:.2f} (Episode {results['best_episode']})\n")
                f.write(f"Mean Episode Time: {np.mean(results['episode_times']):.2f}s\n")

                if results['episode_mesh_qualities']:
                    f.write(f"Final Mesh Quality: {results['episode_mesh_qualities'][-1]:.3f}\n")
                    f.write(f"Mean Mesh Quality: {np.mean(results['episode_mesh_qualities']):.3f}\n")

                if results['episode_completion_rates']:
                    f.write(f"Completion Rate: {np.mean(results['episode_completion_rates']):.1%}\n")

                f.write(f"\nPerformance Improvement:\n")
                if len(results['episode_rewards']) >= 200:
                    early_perf = np.mean(results['episode_rewards'][:100])
                    late_perf = np.mean(results['episode_rewards'][-100:])
                    improvement = late_perf - early_perf
                    f.write(f"  Early Performance: {early_perf:.2f}\n")
                    f.write(f"  Late Performance: {late_perf:.2f}\n")
                    f.write(f"  Improvement: {improvement:+.2f}\n")

            f.write(f"\nConfiguration:\n")
            f.write(f"  Algorithm: SAC (Soft Actor-Critic)\n")
            f.write(f"  Learning Rate: {results['configs']['sac']['learning_rate']}\n")
            f.write(f"  Batch Size: {results['configs']['sac']['batch_size']}\n")
            f.write(f"  Buffer Size: {results['configs']['sac']['buffer_size']:,}\n")

            # Alpha configuration
            if results['configs']['sac'].get('use_static_alpha', False):
                f.write(f"  Alpha: {results['configs']['sac'].get('static_alpha', 0.1)} (static)\n")
            else:
                f.write(f"  Alpha: {results['configs']['sac']['alpha']} (automatic tuning)\n")

        print(f"📄 Text summary saved to: {filepath}")
