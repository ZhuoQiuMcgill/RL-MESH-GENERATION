#!/usr/bin/env python3
"""
Enhanced training logger with multiple backends and advanced logging capabilities.

This module provides comprehensive logging functionality for RL training with support
for multiple backends like TensorBoard, CSV files, and JSON logs.
"""

import os
import csv
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np
from pathlib import Path
import psutil


class TrainingLogger:
    """
    Comprehensive training logger with multiple output formats and backends.
    """

    def __init__(self, log_dir: str, experiment_name: str, config: Dict = None):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            config: Training configuration dictionary
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.config = config or {}

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging backends
        self.csv_file = None
        self.json_file = None
        self.tensorboard_writer = None

        # Setup file logging
        self.setup_file_logging()

        # Setup CSV logging
        self.setup_csv_logging()

        # Setup TensorBoard (optional)
        self.setup_tensorboard()

        # Episode tracking
        self.episode_data = []
        self.step_data = []
        self.evaluation_data = []

        # Performance tracking
        self.start_time = time.time()
        self.last_log_time = time.time()

        print(f"📊 Training logger initialized:")
        print(f"   Log directory: {self.log_dir}")
        print(f"   Experiment: {experiment_name}")

    def setup_file_logging(self):
        """Setup file-based logging."""
        log_file = self.log_dir / f"{self.experiment_name}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(f"rl_mesh_{self.experiment_name}")

    def setup_csv_logging(self):
        """Setup CSV logging for tabular data."""
        csv_file = self.log_dir / f"{self.experiment_name}_episodes.csv"

        self.csv_file = open(csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write CSV header
        header = [
            'episode', 'timestep', 'reward', 'length', 'episode_time',
            'mesh_quality', 'completion_rate', 'cumulative_reward',
            'mean_reward_100', 'total_time', 'episodes_per_hour'
        ]
        self.csv_writer.writerow(header)

        print(f"📈 CSV logging enabled: {csv_file}")

    def setup_tensorboard(self):
        """Setup TensorBoard logging (optional)."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.log_dir / "tensorboard"
            self.tensorboard_writer = SummaryWriter(tb_dir)
            print(f"📊 TensorBoard logging enabled: {tb_dir}")

        except ImportError:
            print("⚠️  TensorBoard not available (install tensorboard for enhanced logging)")
            self.tensorboard_writer = None

    def log_episode(self, episode: int, timestep: int, episode_data: Dict):
        """
        Log episode completion data.

        Args:
            episode: Episode number
            timestep: Current timestep
            episode_data: Dictionary containing episode metrics
        """
        current_time = time.time()
        total_time = current_time - self.start_time

        # Extract episode metrics
        reward = episode_data.get('reward', 0.0)
        length = episode_data.get('length', 0)
        episode_time = episode_data.get('episode_time', 0.0)
        mesh_quality = episode_data.get('mesh_quality', 0.0)
        completion_rate = episode_data.get('completion_rate', 0.0)

        # Store episode data
        episode_record = {
            'episode': episode,
            'timestep': timestep,
            'reward': reward,
            'length': length,
            'episode_time': episode_time,
            'mesh_quality': mesh_quality,
            'completion_rate': completion_rate,
            'timestamp': current_time,
            'total_time': total_time
        }

        self.episode_data.append(episode_record)

        # Calculate running statistics
        cumulative_reward = sum(ep['reward'] for ep in self.episode_data)

        # Mean reward over last 100 episodes
        recent_episodes = self.episode_data[-100:] if len(self.episode_data) >= 100 else self.episode_data
        mean_reward_100 = np.mean([ep['reward'] for ep in recent_episodes])

        # Episodes per hour
        if len(self.episode_data) > 1:
            time_diff = current_time - self.episode_data[0]['timestamp']
            episodes_per_hour = len(self.episode_data) / (time_diff / 3600) if time_diff > 0 else 0
        else:
            episodes_per_hour = 0

        # Log to CSV
        csv_row = [
            episode, timestep, reward, length, episode_time,
            mesh_quality, completion_rate, cumulative_reward,
            mean_reward_100, total_time, episodes_per_hour
        ]
        self.csv_writer.writerow(csv_row)
        self.csv_file.flush()

        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Episode/Reward', reward, episode)
            self.tensorboard_writer.add_scalar('Episode/Length', length, episode)
            self.tensorboard_writer.add_scalar('Episode/Time', episode_time, episode)
            self.tensorboard_writer.add_scalar('Episode/MeshQuality', mesh_quality, episode)
            self.tensorboard_writer.add_scalar('Episode/CompletionRate', completion_rate, episode)
            self.tensorboard_writer.add_scalar('Episode/MeanReward100', mean_reward_100, episode)
            self.tensorboard_writer.add_scalar('Episode/EpisodesPerHour', episodes_per_hour, episode)

        # Detailed logging every N episodes
        if episode % 10 == 0:
            self.logger.info(
                f"Episode {episode:5d} | "
                f"Reward: {reward:8.2f} | "
                f"Length: {length:4d} | "
                f"Time: {episode_time:6.2f}s | "
                f"Quality: {mesh_quality:.3f} | "
                f"Avg100: {mean_reward_100:7.2f} | "
                f"Rate: {episodes_per_hour:5.1f}/h"
            )

        # Performance milestone logging
        if len(self.episode_data) >= 2:
            if reward > max(ep['reward'] for ep in self.episode_data[:-1]):
                self.logger.info(f"🏆 NEW BEST REWARD: {reward:.2f} (Episode {episode})")

        self.last_log_time = current_time

    def log_training_step(self, step: int, training_metrics: Dict):
        """
        Log training step metrics (losses, alpha, etc.).

        Args:
            step: Training step number
            training_metrics: Dictionary of training metrics
        """
        # Store step data
        step_record = {
            'step': step,
            'timestamp': time.time(),
            **training_metrics
        }
        self.step_data.append(step_record)

        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in training_metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f'Training/{key}', value, step)

        # Detailed logging every N steps
        if step % 1000 == 0 and training_metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in training_metrics.items()
                                      if isinstance(v, (int, float))])
            self.logger.info(f"Step {step:6d} | {metrics_str}")

    def log_evaluation(self, timestep: int, eval_results: Dict):
        """
        Log evaluation results.

        Args:
            timestep: Current timestep
            eval_results: Dictionary of evaluation results
        """
        eval_record = {
            'timestep': timestep,
            'timestamp': time.time(),
            **eval_results
        }
        self.evaluation_data.append(eval_record)

        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f'Evaluation/{key}', value, timestep)

        # Log important evaluation metrics
        mean_reward = eval_results.get('mean_reward', 0)
        completion_rate = eval_results.get('completion_rate', 0)

        self.logger.info(
            f"🔍 EVALUATION at step {timestep:,} | "
            f"Mean Reward: {mean_reward:7.2f} | "
            f"Completion: {completion_rate:.1%}"
        )

        # Check for best evaluation performance
        if len(self.evaluation_data) >= 2:
            previous_best = max(eval['mean_reward'] for eval in self.evaluation_data[:-1])
            if mean_reward > previous_best:
                self.logger.info(f"🎯 NEW BEST EVALUATION: {mean_reward:.2f}")

    def log_system_info(self):
        """Log system resource usage and information."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            system_info = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024 ** 3),
                'disk_free_gb': disk.free / (1024 ** 3)
            }

            # GPU information if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                    gpu_cached = torch.cuda.memory_reserved() / (1024 ** 3)
                    system_info.update({
                        'gpu_memory_gb': gpu_memory,
                        'gpu_cached_gb': gpu_cached
                    })
            except ImportError:
                pass

            # Log to TensorBoard
            if self.tensorboard_writer:
                step = len(self.episode_data)
                for key, value in system_info.items():
                    self.tensorboard_writer.add_scalar(f'System/{key}', value, step)

            self.logger.info(
                f"💻 System: CPU {cpu_percent:.1f}% | "
                f"RAM {memory.percent:.1f}% | "
                f"Disk {disk.free / (1024 ** 3):.1f}GB free"
            )

        except Exception as e:
            self.logger.warning(f"Could not log system info: {e}")

    def save_checkpoint_metadata(self, checkpoint_path: str, episode: int,
                                 metrics: Dict, notes: str = ""):
        """
        Save metadata for model checkpoints.

        Args:
            checkpoint_path: Path to saved checkpoint
            episode: Episode number at checkpoint
            metrics: Performance metrics at checkpoint
            notes: Additional notes
        """
        checkpoint_info = {
            'checkpoint_path': checkpoint_path,
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'notes': notes,
            'total_training_time': time.time() - self.start_time
        }

        # Save to JSON file
        checkpoints_file = self.log_dir / f"{self.experiment_name}_checkpoints.json"

        if checkpoints_file.exists():
            with open(checkpoints_file, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []

        checkpoints.append(checkpoint_info)

        with open(checkpoints_file, 'w') as f:
            json.dump(checkpoints, f, indent=2)

        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path} (Episode {episode})")

    def generate_training_summary(self) -> Dict:
        """Generate comprehensive training summary."""
        if not self.episode_data:
            return {}

        total_time = time.time() - self.start_time
        rewards = [ep['reward'] for ep in self.episode_data]
        lengths = [ep['length'] for ep in self.episode_data]
        times = [ep['episode_time'] for ep in self.episode_data]

        summary = {
            'experiment_name': self.experiment_name,
            'total_episodes': len(self.episode_data),
            'total_training_time_hours': total_time / 3600,
            'final_reward': rewards[-1],
            'best_reward': max(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_episode_length': np.mean(lengths),
            'mean_episode_time': np.mean(times),
            'episodes_per_hour': len(self.episode_data) / (total_time / 3600),
            'completion_timestamp': datetime.now().isoformat()
        }

        # Performance improvement analysis
        if len(rewards) >= 200:
            early_performance = np.mean(rewards[:100])
            recent_performance = np.mean(rewards[-100:])
            improvement = recent_performance - early_performance
            summary.update({
                'early_performance': early_performance,
                'recent_performance': recent_performance,
                'performance_improvement': improvement,
                'improvement_percentage': (improvement / abs(early_performance)) * 100 if early_performance != 0 else 0
            })

        # Evaluation summary
        if self.evaluation_data:
            eval_rewards = [eval_data['mean_reward'] for eval_data in self.evaluation_data]
            summary.update({
                'num_evaluations': len(self.evaluation_data),
                'best_evaluation_reward': max(eval_rewards),
                'final_evaluation_reward': eval_rewards[-1]
            })

        return summary

    def save_final_summary(self):
        """Save final training summary to JSON file."""
        summary = self.generate_training_summary()

        if summary:
            summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"📋 Final summary saved: {summary_file}")

            # Print summary to console
            print("\n" + "=" * 70)
            print("🎉 TRAINING SUMMARY")
            print("=" * 70)
            print(f"Experiment: {summary['experiment_name']}")
            print(f"Episodes: {summary['total_episodes']:,}")
            print(f"Training Time: {summary['total_training_time_hours']:.2f} hours")
            print(f"Final Reward: {summary['final_reward']:.2f}")
            print(f"Best Reward: {summary['best_reward']:.2f}")
            print(f"Mean Reward: {summary['mean_reward']:.2f}")

            if 'performance_improvement' in summary:
                print(
                    f"Performance Improvement: {summary['performance_improvement']:+.2f} ({summary['improvement_percentage']:+.1f}%)")

            print("=" * 70)

    def close(self):
        """Close all logging resources."""
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()

        # Close TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        # Save final summary
        self.save_final_summary()

        self.logger.info("📊 Training logger closed")


class PerformanceProfiler:
    """
    Performance profiling utility for training optimization.
    """

    def __init__(self, log_interval: int = 100):
        """
        Initialize performance profiler.

        Args:
            log_interval: Interval for logging performance metrics
        """
        self.log_interval = log_interval
        self.timers = {}
        self.counters = {}
        self.step_count = 0

    def start_timer(self, name: str):
        """Start a named timer."""
        self.timers[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time."""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]

            # Track timing statistics
            if f"{name}_times" not in self.counters:
                self.counters[f"{name}_times"] = []
            self.counters[f"{name}_times"].append(elapsed)

            return elapsed
        return 0.0

    def increment_counter(self, name: str, value: Union[int, float] = 1):
        """Increment a named counter."""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value

    def log_step_performance(self, logger: TrainingLogger):
        """Log performance metrics for this step."""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            # Calculate timing statistics
            perf_metrics = {}

            for key, times in self.counters.items():
                if key.endswith('_times') and isinstance(times, list) and times:
                    base_name = key[:-6]  # Remove '_times' suffix
                    perf_metrics[f"{base_name}_mean_time"] = np.mean(times)
                    perf_metrics[f"{base_name}_total_time"] = np.sum(times)

                    # Clear times to avoid memory buildup
                    times.clear()

            # Add counter values
            for key, value in self.counters.items():
                if not key.endswith('_times'):
                    perf_metrics[key] = value

            # Log to training logger
            if perf_metrics:
                logger.log_training_step(self.step_count, {'performance': perf_metrics})

    def get_summary(self) -> Dict:
        """Get performance summary."""
        summary = {
            'total_steps': self.step_count,
            'counters': {k: v for k, v in self.counters.items() if not isinstance(v, list)},
            'timing_summary': {}
        }

        # Calculate timing summaries
        for key, times in self.counters.items():
            if key.endswith('_times') and isinstance(times, list) and times:
                base_name = key[:-6]
                summary['timing_summary'][base_name] = {
                    'mean': np.mean(times),
                    'total': np.sum(times),
                    'count': len(times)
                }

        return summary


# Context manager for easy performance timing
class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self.profiler.start_timer(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.name)
