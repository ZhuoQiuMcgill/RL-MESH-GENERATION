import torch
import numpy as np
import time
from typing import Dict, Tuple, Optional, List
import os
from datetime import datetime

from .sac_agent import SACAgent
from ..utils.visualization import plot_mesh


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
        print("Starting SAC training for mesh generation...")
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
            next_state_dict, reward, terminated, truncated, info = self.env.step(action, timestep)
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
                print(f"Model saved: {model_path}")

        # Final save
        final_model_path = os.path.join(self.models_dir, "final_model.pt")
        self.agent.save(final_model_path)

        # Training completion summary
        self._print_training_summary()

        print("Training completed!")

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
        Enhanced evaluation with detailed, step-by-step logging to debug
        premature termination. This is the complete function.
        """
        print(f"\n--- Starting Evaluation at Timestep {timestep:,} ({num_episodes} episodes) ---")
        eval_start_time = time.time()

        self.agent.set_training_mode(False)

        eval_rewards = []
        eval_lengths = []
        eval_times = []
        mesh_qualities = []
        completion_rates = []

        best_episode_reward = float('-inf')
        best_episode_boundary = None
        best_episode_elements = None

        for episode in range(num_episodes):
            print(f"  --- Running Eval Episode {episode + 1}/{num_episodes} ---")
            episode_start = time.time()
            state_dict, _ = self.env.reset()
            initial_boundary_size = len(self.env.current_boundary)
            print(f"    [Step 00] Env reset. Initial boundary size: {initial_boundary_size}")

            episode_reward = 0.0
            episode_length = 0

            while True:
                step_num = episode_length + 1
                try:
                    boundary_size_before = len(self.env.current_boundary)

                    action = self.agent.select_action(state_dict, deterministic=True)

                    # During evaluation, global_timestep is not relevant, so we pass None.
                    next_state_dict, reward, terminated, truncated, info = self.env.step(action, None)

                    boundary_size_after = info.get('boundary_vertices', 0)
                    is_valid_element = info.get('is_valid_element', False)



                    episode_reward += reward
                    episode_length += 1

                    if terminated or truncated:
                        # print(f"  --- Eval Episode {episode + 1} Ended at Step {step_num} ---")
                        # print(f"    Reason: {'Terminated' if terminated else 'Truncated'}")
                        # print(f"    Final Info Dict: {info}")
                        completion_rates.append(1.0 if terminated else 0.0)

                        if episode_reward > best_episode_reward:
                            best_episode_reward = episode_reward
                            best_episode_boundary, best_episode_elements = self.env.get_current_mesh()
                        break

                    state_dict = next_state_dict

                except Exception as e:
                    print(f"    !!!!!! CRASH DETECTED IN EVAL EPISODE {episode + 1} AT STEP {step_num} !!!!!!")
                    print(f"      Error Type: {type(e).__name__}")
                    print(f"      Error Message: {e}")
                    import traceback
                    traceback.print_exc()
                    terminated = True  # Force stop this episode
                    break

            episode_time = time.time() - episode_start
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_times.append(episode_time)
            quality_metrics = self.env.get_mesh_quality_metrics()
            if quality_metrics:
                mesh_qualities.append(quality_metrics)

        self.agent.set_training_mode(True)
        eval_total_time = time.time() - eval_start_time

        if best_episode_boundary is not None:
            self._save_evaluation_mesh_visualization(timestep, best_episode_boundary,
                                                     best_episode_elements, best_episode_reward)

        results = {
            'timestep': timestep,
            'mean_reward': np.mean(eval_rewards) if eval_rewards else 0.0,
            'std_reward': np.std(eval_rewards) if eval_rewards else 0.0,
            'min_reward': np.min(eval_rewards) if eval_rewards else 0.0,
            'max_reward': np.max(eval_rewards) if eval_rewards else 0.0,
            'mean_length': np.mean(eval_lengths) if eval_lengths else 0.0,
            'std_length': np.std(eval_lengths) if eval_lengths else 0.0,
            'mean_episode_time': np.mean(eval_times) if eval_times else 0.0,
            'completion_rate': np.mean(completion_rates) if completion_rates else 0.0,
            'eval_total_time': eval_total_time
        }

        if mesh_qualities and any(mq for mq in mesh_qualities):
            # Gracefully find the keys from the first valid quality dict
            quality_keys = next((item.keys() for item in mesh_qualities if item), [])
            for key in quality_keys:
                values = [q[key] for q in mesh_qualities if q and key in q]
                if values:
                    results[f'mean_{key}'] = np.mean(values)
                    results[f'std_{key}'] = np.std(values)

        print(f"--- Evaluation at Timestep {timestep:,} Finished in {eval_total_time:.1f}s ---")
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

            print(f"Evaluation mesh saved: {mesh_filename}")

        except Exception as e:
            print(f"Failed to save evaluation mesh visualization: {e}")

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
        print(f"EPISODE {episode:,} | TIMESTEP {timestep:,} ({progress:.1f}%)")
        print(f"{'=' * 70}")

        # Current episode info
        print(f"Current Episode:")
        print(f"   Reward: {reward:8.2f} | Length: {length:4d} steps | Time: {episode_time:6.2f}s")
        print(f"   Quality: {mesh_quality:.3f} | Completed: {'Yes' if completion_rate > 0 else 'No'}")

        # Recent performance (last 100 episodes)
        print(f"\nRecent Performance (last {recent_episodes} episodes):")
        print(f"   Avg Reward: {avg_reward:8.2f} | Avg Length: {avg_length:6.1f} steps")
        print(f"   Avg Time: {avg_time:8.2f}s | Avg Quality: {avg_quality:.3f}")
        print(f"   Completion Rate: {avg_completion:.1%}")

        # Best performance tracking
        print(f"\nBest Performance:")
        print(f"   Best Reward: {self.best_reward:8.2f} (Episode {self.best_episode})")

        # Training progress
        print(f"\nTraining Progress:")
        print(f"   Elapsed Time: {training_hours:.2f} hours")
        print(f"   Episodes/Hour: {self.episodes_per_hour:.1f}")
        print(f"   Timesteps/Hour: {timestep / training_hours if training_hours > 0 else 0:,.0f}")

        # Agent statistics
        print(f"\nAgent Status:")
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
        print(f"Training Configuration:")
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
        print(f"\nEVALUATION at timestep {timestep:,}")
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
        print(f"TRAINING COMPLETED!")
        print(f"{'=' * 70}")

        if self.episode_rewards:
            print(f"Final Statistics:")
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

                print(f"\nLearning Progress:")
                print(f"   Early Performance: {early_rewards:.2f}")
                print(f"   Late Performance: {late_rewards:.2f}")
                print(
                    f"   Improvement: {improvement:+.2f} ({(improvement / abs(early_rewards) * 100) if early_rewards != 0 else 0:+.1f}%)")

        print(f"{'=' * 70}")

        # Evaluation visualization summary
        if self.eval_count > 0:
            eval_mesh_dir = os.path.join(self.figures_dir, "evaluation_meshes")
            print(f"Evaluation mesh visualizations saved to: {eval_mesh_dir}")
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
        print(f"Training results saved to: {filepath}")

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

        print(f"Text summary saved to: {filepath}")
