#!/usr/bin/env python3
"""
Real-time training monitor for RL-Mesh-Generation.

This script monitors training progress in real-time by reading saved training results
and displaying comprehensive statistics and visualizations.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List
from datetime import datetime, timedelta
import json

# Add project root to path
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)


class TrainingMonitor:
    """Real-time training monitor with live plotting capabilities."""

    def __init__(self, results_path: str, refresh_interval: int = 30):
        """
        Initialize training monitor.

        Args:
            results_path: Path to training results file
            refresh_interval: Refresh interval in seconds
        """
        self.results_path = results_path
        self.refresh_interval = refresh_interval

        # Data storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.evaluation_results = []
        self.last_update = None

        # Monitoring state
        self.is_training_active = True
        self.total_episodes = 0
        self.last_episode_count = 0

        # Setup plots
        self.setup_plots()

        print(f"🔍 Monitoring training at: {results_path}")
        print(f"🔄 Refresh interval: {refresh_interval}s")

    def setup_plots(self):
        """Setup real-time plots."""
        plt.ion()  # Enable interactive mode

        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time Training Monitor', fontsize=16, fontweight='bold')

        # Initialize empty plots
        self.reward_line, = self.axes[0, 0].plot([], [], 'b-', alpha=0.7, label='Episode Rewards')
        self.reward_avg_line, = self.axes[0, 0].plot([], [], 'r-', linewidth=2, label='Moving Average')
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)

        self.length_line, = self.axes[0, 1].plot([], [], 'g-', alpha=0.7, label='Episode Lengths')
        self.length_avg_line, = self.axes[0, 1].plot([], [], 'orange', linewidth=2, label='Moving Average')
        self.axes[0, 1].set_title('Episode Lengths')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Steps')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)

        self.time_line, = self.axes[1, 0].plot([], [], 'purple', alpha=0.7, label='Episode Times')
        self.axes[1, 0].set_title('Episode Execution Times')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Time (seconds)')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True, alpha=0.3)

        # Statistics display
        self.axes[1, 1].axis('off')
        self.stats_text = self.axes[1, 1].text(0.1, 0.9, "", transform=self.axes[1, 1].transAxes,
                                               fontfamily='monospace', fontsize=10,
                                               verticalalignment='top')

        plt.tight_layout()

    def load_training_data(self) -> bool:
        """
        Load training data from file.

        Returns:
            True if data was successfully loaded and updated
        """
        try:
            if not os.path.exists(self.results_path):
                return False

            # Check file modification time
            file_mtime = os.path.getmtime(self.results_path)
            if self.last_update and file_mtime <= self.last_update:
                return False  # No new data

            # Load data
            data = torch.load(self.results_path, map_location='cpu')

            # Update data if new episodes are available
            new_episode_count = len(data.get('episode_rewards', []))
            if new_episode_count <= self.last_episode_count:
                return False  # No new episodes

            self.episode_rewards = data.get('episode_rewards', [])
            self.episode_lengths = data.get('episode_lengths', [])
            self.episode_times = data.get('episode_times', [])
            self.evaluation_results = data.get('evaluation_results', [])

            self.total_episodes = len(self.episode_rewards)
            self.last_episode_count = self.total_episodes
            self.last_update = file_mtime

            return True

        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False

    def update_plots(self):
        """Update real-time plots with latest data."""
        if not self.episode_rewards:
            return

        episodes = np.arange(len(self.episode_rewards))

        # Update reward plot
        self.reward_line.set_data(episodes, self.episode_rewards)

        # Calculate moving average
        window_size = min(100, len(self.episode_rewards) // 10) if len(self.episode_rewards) > 10 else 1
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size) / window_size, mode='valid')
            avg_episodes = episodes[window_size - 1:]
            self.reward_avg_line.set_data(avg_episodes, moving_avg)

        # Update axes limits
        self.axes[0, 0].relim()
        self.axes[0, 0].autoscale_view()

        # Update length plot
        if self.episode_lengths:
            self.length_line.set_data(episodes, self.episode_lengths)

            if len(self.episode_lengths) >= window_size:
                length_avg = np.convolve(self.episode_lengths, np.ones(window_size) / window_size, mode='valid')
                self.length_avg_line.set_data(avg_episodes, length_avg)

            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()

        # Update time plot
        if self.episode_times:
            self.time_line.set_data(episodes, self.episode_times)
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()

        # Update statistics
        self.update_statistics_display()

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_statistics_display(self):
        """Update statistics text display."""
        if not self.episode_rewards:
            return

        # Calculate statistics
        current_reward = self.episode_rewards[-1]
        best_reward = max(self.episode_rewards)
        mean_reward = np.mean(self.episode_rewards)
        recent_mean = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else mean_reward

        mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        mean_time = np.mean(self.episode_times) if self.episode_times else 0

        # Performance trend
        if len(self.episode_rewards) >= 200:
            early_performance = np.mean(self.episode_rewards[:100])
            recent_performance = np.mean(self.episode_rewards[-100:])
            improvement = recent_performance - early_performance
            trend = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
        else:
            improvement = 0
            trend = "📊"

        # Episodes per hour estimate
        if len(self.episode_times) > 10:
            episodes_per_hour = 3600 / np.mean(self.episode_times[-10:])
        else:
            episodes_per_hour = 0

        # Format statistics text
        stats_text = f"""
TRAINING STATISTICS

📊 Episodes: {self.total_episodes:,}
🎯 Current Reward: {current_reward:.2f}
🏆 Best Reward: {best_reward:.2f}
📈 Mean Reward: {mean_reward:.2f}
🔄 Recent Mean (100): {recent_mean:.2f}

⏱️  Mean Episode Time: {mean_time:.2f}s
📏 Mean Episode Length: {mean_length:.1f}
⚡ Episodes/Hour: {episodes_per_hour:.1f}

{trend} Performance Trend: {improvement:+.2f}

🔄 Last Update: {datetime.now().strftime('%H:%M:%S')}
"""

        if self.evaluation_results:
            eval_rewards = [r['mean_reward'] for r in self.evaluation_results]
            stats_text += f"\n🔍 Evaluations: {len(self.evaluation_results)}"
            stats_text += f"\n📊 Best Eval: {max(eval_rewards):.2f}"
            stats_text += f"\n📊 Recent Eval: {eval_rewards[-1]:.2f}"

        # Check if training is still active
        if self.last_update:
            time_since_update = time.time() - self.last_update
            if time_since_update > 300:  # 5 minutes
                stats_text += f"\n⚠️  Training may have stopped\n   ({time_since_update / 60:.1f} min ago)"
                self.is_training_active = False
            else:
                self.is_training_active = True

        self.stats_text.set_text(stats_text)

    def run(self):
        """Run the monitoring loop."""
        print("🚀 Starting real-time monitoring...")
        print("Press Ctrl+C to stop monitoring")

        try:
            while True:
                # Load and update data
                if self.load_training_data():
                    self.update_plots()
                    print(f"📊 Updated: Episode {self.total_episodes}, Reward: {self.episode_rewards[-1]:.2f}")

                # Wait for next update
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
        finally:
            plt.ioff()
            print("👋 Monitor closed")

    def generate_report(self, output_path: str = None):
        """Generate a comprehensive training report."""
        if not self.episode_rewards:
            print("❌ No training data available for report generation")
            return

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"training_report_{timestamp}.txt"

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RL-MESH-GENERATION TRAINING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.results_path}\n\n")

            # Training summary
            f.write("TRAINING SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Episodes: {self.total_episodes:,}\n")
            f.write(f"Current Reward: {self.episode_rewards[-1]:.3f}\n")
            f.write(f"Best Reward: {max(self.episode_rewards):.3f}\n")
            f.write(f"Mean Reward: {np.mean(self.episode_rewards):.3f}\n")
            f.write(f"Reward Std: {np.std(self.episode_rewards):.3f}\n")

            if self.episode_lengths:
                f.write(f"Mean Episode Length: {np.mean(self.episode_lengths):.1f} steps\n")
                f.write(f"Length Std: {np.std(self.episode_lengths):.1f} steps\n")

            if self.episode_times:
                f.write(f"Mean Episode Time: {np.mean(self.episode_times):.2f}s\n")
                f.write(f"Total Training Time: {sum(self.episode_times) / 3600:.2f} hours\n")

            # Performance analysis
            if len(self.episode_rewards) >= 200:
                f.write(f"\nPERFORMANCE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                early_perf = np.mean(self.episode_rewards[:100])
                recent_perf = np.mean(self.episode_rewards[-100:])
                improvement = recent_perf - early_perf
                f.write(f"Early Performance (first 100): {early_perf:.3f}\n")
                f.write(f"Recent Performance (last 100): {recent_perf:.3f}\n")
                f.write(f"Improvement: {improvement:+.3f} ({improvement / abs(early_perf) * 100:+.1f}%)\n")

            # Evaluation summary
            if self.evaluation_results:
                f.write(f"\nEVALUATION SUMMARY\n")
                f.write("-" * 40 + "\n")
                eval_rewards = [r['mean_reward'] for r in self.evaluation_results]
                f.write(f"Total Evaluations: {len(self.evaluation_results)}\n")
                f.write(f"Best Evaluation: {max(eval_rewards):.3f}\n")
                f.write(f"Final Evaluation: {eval_rewards[-1]:.3f}\n")
                f.write(f"Evaluation Mean: {np.mean(eval_rewards):.3f}\n")

        print(f"📄 Training report saved to: {output_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Real-time training monitor for RL-Mesh-Generation")

    parser.add_argument(
        "results_path",
        type=str,
        help="Path to training results file (training_results.pt)"
    )

    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )

    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate training report and exit"
    )

    parser.add_argument(
        "--report-output",
        type=str,
        default=None,
        help="Output path for training report"
    )

    return parser.parse_args()


def main():
    """Main monitoring function."""
    args = parse_arguments()

    # Check if results file exists
    if not os.path.exists(args.results_path):
        print(f"❌ Training results file not found: {args.results_path}")
        print("   Make sure training is running and has saved results")
        return

    # Create monitor
    monitor = TrainingMonitor(args.results_path, args.refresh_interval)

    # Load initial data
    if not monitor.load_training_data():
        print(f"❌ Could not load training data from: {args.results_path}")
        return

    print(f"✅ Loaded training data: {monitor.total_episodes} episodes")

    if args.generate_report:
        # Generate report only
        monitor.generate_report(args.report_output)
    else:
        # Run real-time monitoring
        monitor.run()


if __name__ == "__main__":
    main()
