#!/usr/bin/env python3
"""
Training comparison and analysis tool for RL-Mesh-Generation.

This script allows comparison of multiple training runs, analysis of hyperparameter
effects, and generation of comparative reports and visualizations.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)


class TrainingComparator:
    """
    Tool for comparing multiple training runs and generating analysis reports.
    """

    def __init__(self):
        """Initialize training comparator."""
        self.experiments = {}
        self.comparison_data = None

        # Set plotting style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'library') else 'default')
        sns.set_palette("husl")

    def load_experiment(self, name: str, results_path: str, config_path: str = None):
        """
        Load a training experiment for comparison.

        Args:
            name: Name identifier for this experiment
            results_path: Path to training results file
            config_path: Optional path to configuration file
        """
        try:
            # Load training results
            results = torch.load(results_path, map_location='cpu')

            # Load configuration if provided
            config = None
            if config_path and os.path.exists(config_path):
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

            # Extract key metrics
            experiment_data = {
                'name': name,
                'results_path': results_path,
                'config_path': config_path,
                'config': config,
                'episode_rewards': results.get('episode_rewards', []),
                'episode_lengths': results.get('episode_lengths', []),
                'episode_times': results.get('episode_times', []),
                'episode_mesh_qualities': results.get('episode_mesh_qualities', []),
                'episode_completion_rates': results.get('episode_completion_rates', []),
                'evaluation_results': results.get('evaluation_results', []),
                'training_stats': results.get('training_stats', {}),
                'best_reward': results.get('best_reward', 0),
                'total_training_time': results.get('total_training_time', 0)
            }

            # Calculate summary statistics
            if experiment_data['episode_rewards']:
                rewards = experiment_data['episode_rewards']
                experiment_data['summary'] = {
                    'total_episodes': len(rewards),
                    'final_reward': rewards[-1],
                    'best_reward': max(rewards),
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'convergence_episode': self._find_convergence_point(rewards),
                    'sample_efficiency': self._calculate_sample_efficiency(rewards)
                }

                # Performance improvement
                if len(rewards) >= 200:
                    early_perf = np.mean(rewards[:100])
                    late_perf = np.mean(rewards[-100:])
                    experiment_data['summary']['improvement'] = late_perf - early_perf
                    experiment_data['summary']['improvement_pct'] = (late_perf - early_perf) / abs(early_perf) * 100

            self.experiments[name] = experiment_data
            print(f"✅ Loaded experiment '{name}': {len(experiment_data['episode_rewards'])} episodes")

        except Exception as e:
            print(f"❌ Error loading experiment '{name}': {e}")

    def _find_convergence_point(self, rewards: List[float], window: int = 100,
                                threshold: float = 0.01) -> int:
        """
        Find approximate convergence point in training.

        Args:
            rewards: List of episode rewards
            window: Window size for moving average
            threshold: Threshold for considering convergence

        Returns:
            Episode number where convergence is detected
        """
        if len(rewards) < window * 2:
            return len(rewards)

        # Calculate moving average
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')

        # Find where the moving average stabilizes
        for i in range(window, len(moving_avg) - window):
            recent_var = np.var(moving_avg[i:i + window])
            if recent_var < threshold:
                return i + window

        return len(rewards)

    def _calculate_sample_efficiency(self, rewards: List[float], target_percentile: float = 0.8) -> int:
        """
        Calculate sample efficiency (episodes to reach target performance).

        Args:
            rewards: List of episode rewards
            target_percentile: Target performance percentile

        Returns:
            Number of episodes to reach target
        """
        if not rewards:
            return 0

        target_reward = np.percentile(rewards, target_percentile * 100)

        # Find first episode that reaches target
        for i, reward in enumerate(rewards):
            if reward >= target_reward:
                return i + 1

        return len(rewards)

    def compare_learning_curves(self, save_path: str = None, max_episodes: int = None):
        """
        Generate comparison plot of learning curves.

        Args:
            save_path: Path to save the plot
            max_episodes: Maximum episodes to plot (for alignment)
        """
        if len(self.experiments) < 2:
            print("❌ Need at least 2 experiments for comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Raw learning curves
        ax = axes[0, 0]
        for name, exp in self.experiments.items():
            rewards = exp['episode_rewards']
            if not rewards:
                continue

            episodes = np.arange(len(rewards))
            if max_episodes:
                episodes = episodes[:max_episodes]
                rewards = rewards[:max_episodes]

            ax.plot(episodes, rewards, alpha=0.3, linewidth=0.8, label=f'{name} (raw)')

            # Moving average
            window = min(100, len(rewards) // 10)
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
                avg_episodes = episodes[window - 1:len(moving_avg) + window - 1]
                ax.plot(avg_episodes, moving_avg, linewidth=2, label=f'{name} (avg)')

        ax.set_title('Learning Curves Comparison')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Episode lengths
        ax = axes[0, 1]
        for name, exp in self.experiments.items():
            lengths = exp['episode_lengths']
            if not lengths:
                continue

            episodes = np.arange(len(lengths))
            if max_episodes:
                episodes = episodes[:max_episodes]
                lengths = lengths[:max_episodes]

            # Moving average only
            window = min(100, len(lengths) // 10)
            if len(lengths) >= window:
                moving_avg = np.convolve(lengths, np.ones(window) / window, mode='valid')
                avg_episodes = episodes[window - 1:len(moving_avg) + window - 1]
                ax.plot(avg_episodes, moving_avg, linewidth=2, label=name)

        ax.set_title('Episode Lengths Comparison')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Mesh quality (if available)
        ax = axes[1, 0]
        has_quality_data = False
        for name, exp in self.experiments.items():
            qualities = exp['episode_mesh_qualities']
            if not qualities:
                continue

            has_quality_data = True
            episodes = np.arange(len(qualities))
            if max_episodes:
                episodes = episodes[:max_episodes]
                qualities = qualities[:max_episodes]

            # Moving average
            window = min(100, len(qualities) // 10)
            if len(qualities) >= window:
                moving_avg = np.convolve(qualities, np.ones(window) / window, mode='valid')
                avg_episodes = episodes[window - 1:len(moving_avg) + window - 1]
                ax.plot(avg_episodes, moving_avg, linewidth=2, label=name)

        if has_quality_data:
            ax.set_title('Mesh Quality Comparison')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Quality Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No mesh quality data available',
                    transform=ax.transAxes, ha='center', va='center')

        # Plot 4: Completion rate
        ax = axes[1, 1]
        has_completion_data = False
        for name, exp in self.experiments.items():
            completion_rates = exp['episode_completion_rates']
            if not completion_rates:
                continue

            has_completion_data = True
            episodes = np.arange(len(completion_rates))
            if max_episodes:
                episodes = episodes[:max_episodes]
                completion_rates = completion_rates[:max_episodes]

            # Moving average
            window = min(100, len(completion_rates) // 10)
            if len(completion_rates) >= window:
                moving_avg = np.convolve([r * 100 for r in completion_rates],
                                         np.ones(window) / window, mode='valid')
                avg_episodes = episodes[window - 1:len(moving_avg) + window - 1]
                ax.plot(avg_episodes, moving_avg, linewidth=2, label=name)

        if has_completion_data:
            ax.set_title('Completion Rate Comparison')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Completion Rate (%)')
            ax.set_ylim(0, 100)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No completion rate data available',
                    transform=ax.transAxes, ha='center', va='center')

        plt.suptitle('Training Experiments Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Learning curves comparison saved to: {save_path}")

        plt.show()

    def compare_performance_distributions(self, save_path: str = None):
        """
        Compare performance distributions across experiments.

        Args:
            save_path: Path to save the plot
        """
        if len(self.experiments) < 2:
            print("❌ Need at least 2 experiments for comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Collect data for all experiments
        all_rewards = {}
        all_lengths = {}
        all_times = {}
        all_qualities = {}

        for name, exp in self.experiments.items():
            if exp['episode_rewards']:
                all_rewards[name] = exp['episode_rewards']
            if exp['episode_lengths']:
                all_lengths[name] = exp['episode_lengths']
            if exp['episode_times']:
                all_times[name] = exp['episode_times']
            if exp['episode_mesh_qualities']:
                all_qualities[name] = exp['episode_mesh_qualities']

        # Plot 1: Reward distributions
        ax = axes[0, 0]
        if all_rewards:
            data_list = [rewards for rewards in all_rewards.values()]
            labels = list(all_rewards.keys())
            ax.boxplot(data_list, labels=labels)
            ax.set_title('Reward Distributions')
            ax.set_ylabel('Reward')
            ax.tick_params(axis='x', rotation=45)

        # Plot 2: Episode length distributions
        ax = axes[0, 1]
        if all_lengths:
            data_list = [lengths for lengths in all_lengths.values()]
            labels = list(all_lengths.keys())
            ax.boxplot(data_list, labels=labels)
            ax.set_title('Episode Length Distributions')
            ax.set_ylabel('Steps')
            ax.tick_params(axis='x', rotation=45)

        # Plot 3: Episode time distributions
        ax = axes[1, 0]
        if all_times:
            data_list = [times for times in all_times.values()]
            labels = list(all_times.keys())
            ax.boxplot(data_list, labels=labels)
            ax.set_title('Episode Time Distributions')
            ax.set_ylabel('Time (seconds)')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No timing data available',
                    transform=ax.transAxes, ha='center', va='center')

        # Plot 4: Mesh quality distributions
        ax = axes[1, 1]
        if all_qualities:
            data_list = [qualities for qualities in all_qualities.values()]
            labels = list(all_qualities.keys())
            ax.boxplot(data_list, labels=labels)
            ax.set_title('Mesh Quality Distributions')
            ax.set_ylabel('Quality Score')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No quality data available',
                    transform=ax.transAxes, ha='center', va='center')

        plt.suptitle('Performance Distributions Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance distributions saved to: {save_path}")

        plt.show()

    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table of key metrics.

        Returns:
            DataFrame with comparison metrics
        """
        if not self.experiments:
            return pd.DataFrame()

        comparison_data = []

        for name, exp in self.experiments.items():
            summary = exp.get('summary', {})
            config = exp.get('config', {})

            row = {
                'Experiment': name,
                'Total Episodes': summary.get('total_episodes', 0),
                'Final Reward': summary.get('final_reward', 0),
                'Best Reward': summary.get('best_reward', 0),
                'Mean Reward': summary.get('mean_reward', 0),
                'Std Reward': summary.get('std_reward', 0),
                'Convergence Episode': summary.get('convergence_episode', 0),
                'Sample Efficiency': summary.get('sample_efficiency', 0),
                'Improvement': summary.get('improvement', 0),
                'Improvement %': summary.get('improvement_pct', 0),
                'Training Time (h)': exp.get('total_training_time', 0) / 3600
            }

            # Add hyperparameter information
            if config:
                sac_config = config.get('sac', {})
                row.update({
                    'Learning Rate': sac_config.get('learning_rate', 'N/A'),
                    'Batch Size': sac_config.get('batch_size', 'N/A'),
                    'Buffer Size': sac_config.get('buffer_size', 'N/A'),
                    'Alpha (initial)': sac_config.get('alpha', 'N/A')
                })

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        self.comparison_data = df
        return df

    def print_comparison_table(self):
        """Print formatted comparison table."""
        df = self.generate_comparison_table()
        if df.empty:
            print("❌ No experiments to compare")
            return

        print("\n" + "=" * 120)
        print("🔍 TRAINING EXPERIMENTS COMPARISON")
        print("=" * 120)

        # Key performance metrics
        print("\n📊 PERFORMANCE METRICS:")
        performance_cols = ['Experiment', 'Final Reward', 'Best Reward', 'Mean Reward',
                            'Convergence Episode', 'Sample Efficiency', 'Improvement %']
        perf_df = df[performance_cols]
        print(perf_df.to_string(index=False, float_format='%.3f'))

        # Training efficiency
        print("\n⏱️  TRAINING EFFICIENCY:")
        efficiency_cols = ['Experiment', 'Total Episodes', 'Training Time (h)',
                           'Convergence Episode', 'Sample Efficiency']
        eff_df = df[efficiency_cols]
        print(eff_df.to_string(index=False, float_format='%.2f'))

        # Hyperparameters (if available)
        hyperparam_cols = ['Experiment', 'Learning Rate', 'Batch Size', 'Buffer Size', 'Alpha (initial)']
        if all(col in df.columns for col in hyperparam_cols):
            print("\n🔧 HYPERPARAMETERS:")
            hyper_df = df[hyperparam_cols]
            print(hyper_df.to_string(index=False))

        print("=" * 120)

        # Best performer analysis
        best_final = df.loc[df['Final Reward'].idxmax(), 'Experiment']
        best_mean = df.loc[df['Mean Reward'].idxmax(), 'Experiment']
        most_efficient = df.loc[df['Sample Efficiency'].idxmin(), 'Experiment']

        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Best Final Reward: {best_final}")
        print(f"   Best Mean Reward: {best_mean}")
        print(f"   Most Sample Efficient: {most_efficient}")

    def save_comparison_report(self, output_dir: str):
        """
        Save comprehensive comparison report.

        Args:
            output_dir: Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate comparison table
        df = self.generate_comparison_table()

        # Save CSV
        csv_path = output_path / "comparison_table.csv"
        df.to_csv(csv_path, index=False)

        # Save detailed JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(self.experiments),
            'comparison_table': df.to_dict('records'),
            'detailed_results': {}
        }

        for name, exp in self.experiments.items():
            report['detailed_results'][name] = {
                'summary': exp.get('summary', {}),
                'config': exp.get('config', {}),
                'results_path': exp['results_path'],
                'config_path': exp['config_path']
            }

        json_path = output_path / "comparison_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate plots
        self.compare_learning_curves(save_path=output_path / "learning_curves_comparison.png")
        self.compare_performance_distributions(save_path=output_path / "performance_distributions.png")

        print(f"\n📊 Comparison report saved to: {output_path}")
        print(f"   CSV table: {csv_path}")
        print(f"   JSON report: {json_path}")
        print(f"   Plots: learning_curves_comparison.png, performance_distributions.png")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare multiple RL-Mesh-Generation training runs")

    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="List of experiment names and paths: name1:results_path1 [name2:results_path2 ...]"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Output directory for comparison report"
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum episodes to compare (for alignment)"
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively"
    )

    return parser.parse_args()


def main():
    """Main comparison function."""
    args = parse_arguments()

    print("🔍 RL-Mesh-Generation Training Comparison Tool")
    print("=" * 60)

    # Initialize comparator
    comparator = TrainingComparator()

    # Load experiments
    for exp_spec in args.experiments:
        if ':' not in exp_spec:
            print(f"❌ Invalid experiment specification: {exp_spec}")
            print("   Use format: name:results_path")
            continue

        name, results_path = exp_spec.split(':', 1)

        # Try to find config file
        config_path = None
        results_dir = os.path.dirname(results_path)
        possible_configs = [
            os.path.join(results_dir, "config.yaml"),
            os.path.join(results_dir, "configs.yaml"),
            os.path.join(results_dir, f"{name}_config.yaml")
        ]

        for config_file in possible_configs:
            if os.path.exists(config_file):
                config_path = config_file
                break

        comparator.load_experiment(name, results_path, config_path)

    if len(comparator.experiments) < 2:
        print("❌ Need at least 2 experiments for comparison")
        return

    print(f"\n✅ Loaded {len(comparator.experiments)} experiments for comparison")

    # Print comparison table
    comparator.print_comparison_table()

    # Generate plots
    if args.show_plots:
        print("\n📊 Generating comparison plots...")
        comparator.compare_learning_curves(max_episodes=args.max_episodes)
        comparator.compare_performance_distributions()

    # Save comprehensive report
    print(f"\n💾 Saving comparison report to: {args.output_dir}")
    comparator.save_comparison_report(args.output_dir)

    print("\n🎉 Comparison completed!")


if __name__ == "__main__":
    main()
