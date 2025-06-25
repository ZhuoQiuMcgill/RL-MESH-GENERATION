#!/usr/bin/env python3
"""
Training script for RL-based mesh generation.

This script trains a SAC agent to perform automatic quadrilateral mesh generation
using reinforcement learning.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime

# Add project root to path
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

from rl_mesher.environment import MeshEnv, MultiDomainMeshEnv
from rl_mesher.agent import SACAgent, MeshSACTrainer
from rl_mesher.utils.visualization import plot_learning_curve, plot_training_progress


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SAC agent for mesh generation")

    parser.add_argument(
        "--configs",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run training on"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--multi-domain",
        action="store_true",
        help="Use multiple domains for training"
    )

    parser.add_argument(
        "--domain-files",
        nargs="+",
        default=["T1.txt", "T2.txt"],
        help="Domain files to use for multi-domain training"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    return device


def setup_directories(config: dict, experiment_name: str = None):
    """Setup result directories."""
    if experiment_name:
        # Create experiment-specific directories
        base_dir = os.path.join("results", experiment_name)
        config['paths']['models_dir'] = os.path.join(base_dir, "models")
        config['paths']['logs_dir'] = os.path.join(base_dir, "logs")
        config['paths']['figures_dir'] = os.path.join(base_dir, "figures")

    # Create directories
    for path_key in ['models_dir', 'logs_dir', 'figures_dir']:
        path = config['paths'][path_key]
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def setup_environment(config: dict, multi_domain: bool = False,
                      domain_files: list = None) -> MeshEnv:
    """Setup training environment."""
    if multi_domain and domain_files:
        print(f"Setting up multi-domain environment with domains: {domain_files}")
        env = MultiDomainMeshEnv(config, domain_files)
    else:
        print(f"Setting up single-domain environment with domain: {config['domain']['training_domain']}")
        env = MeshEnv(config)

    return env


def setup_agent(config: dict, device: torch.device) -> SACAgent:
    """Setup SAC agent."""
    agent = SACAgent(config, device)

    print("SAC Agent initialized:")
    network_info = agent.get_network_info()
    print(f"  Actor parameters: {network_info['actor_parameters']:,}")
    print(f"  Critic parameters: {network_info['critic_parameters']:,}")
    print(f"  Total parameters: {network_info['total_parameters']:,}")

    return agent


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Set random seed to: {seed}")


def save_config(config: dict, save_path: str):
    """Save configuration to file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_training_summary(results: dict, config: dict, args,
                            experiment_name: str) -> dict:
    """Create comprehensive training summary."""
    summary = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'configs': config,
        'args': vars(args),
        'results': {
            'total_episodes': len(results['episode_rewards']),
            'final_reward': results['episode_rewards'][-1] if results['episode_rewards'] else 0,
            'best_reward': max(results['episode_rewards']) if results['episode_rewards'] else 0,
            'mean_reward_last_100': np.mean(results['episode_rewards'][-100:]) if len(
                results['episode_rewards']) >= 100 else np.mean(results['episode_rewards']),
            'mean_episode_length': np.mean(results['episode_lengths']) if results['episode_lengths'] else 0,
        }
    }

    if results['evaluation_results']:
        eval_rewards = [r['mean_reward'] for r in results['evaluation_results']]
        summary['results']['best_eval_reward'] = max(eval_rewards)
        summary['results']['final_eval_reward'] = eval_rewards[-1]

    return summary


def main():
    """Main training function."""
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Override configs with command line arguments
    if args.seed is not None:
        config['training']['seed'] = args.seed
        set_random_seeds(args.seed)
    elif 'seed' in config['training']:
        set_random_seeds(config['training']['seed'])

    # Setup device
    device = setup_device(args.device)

    # Create experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"mesh_sac_{timestamp}"

    print(f"Starting experiment: {experiment_name}")

    # Setup directories
    setup_directories(config, experiment_name)

    # Save configuration
    config_save_path = os.path.join(config['paths']['logs_dir'], "configs.yaml")
    save_config(config, config_save_path)

    # Setup environment
    env = setup_environment(config, args.multi_domain, args.domain_files)

    # Setup agent
    agent = setup_agent(config, device)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        agent.load(args.resume)

    # Setup trainer
    trainer = MeshSACTrainer(agent, env, config)

    # Start training
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    try:
        results = trainer.train()

        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 50)

        # Save training results
        results_path = os.path.join(config['paths']['logs_dir'], "training_results.pt")
        trainer.save_training_results(results_path)

        # Create and save training summary
        summary = create_training_summary(results, config, args, experiment_name)
        summary_path = os.path.join(config['paths']['logs_dir'], "training_summary.yaml")
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

        # Generate training plots
        print("Generating training plots...")

        # Learning curve
        if results['episode_rewards']:
            learning_curve_path = os.path.join(config['paths']['figures_dir'], "learning_curve.png")
            plot_learning_curve(
                results['episode_rewards'],
                results['episode_lengths'],
                title=f"Training Results - {experiment_name}",
                save_path=learning_curve_path
            )

        # Training progress
        if results['training_stats']:
            progress_path = os.path.join(config['paths']['figures_dir'], "training_progress.png")
            plot_training_progress(
                results['training_stats'],
                save_path=progress_path
            )

        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"  Total Episodes: {len(results['episode_rewards'])}")
        if results['episode_rewards']:
            print(f"  Final Reward: {results['episode_rewards'][-1]:.2f}")
            print(f"  Best Reward: {max(results['episode_rewards']):.2f}")
            print(f"  Mean Reward (last 100): {np.mean(results['episode_rewards'][-100:]):.2f}")

        if results['evaluation_results']:
            eval_rewards = [r['mean_reward'] for r in results['evaluation_results']]
            print(f"  Best Evaluation Reward: {max(eval_rewards):.2f}")

        print(f"\nResults saved to: {config['paths']['logs_dir']}")
        print(f"Models saved to: {config['paths']['models_dir']}")
        print(f"Figures saved to: {config['paths']['figures_dir']}")

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 50)

        # Save current progress
        interrupt_model_path = os.path.join(config['paths']['models_dir'], "interrupted_model.pt")
        agent.save(interrupt_model_path)
        print(f"Saved interrupted model to: {interrupt_model_path}")

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"TRAINING FAILED: {str(e)}")
        print("=" * 50)

        # Save current progress anyway
        error_model_path = os.path.join(config['paths']['models_dir'], "error_model.pt")
        agent.save(error_model_path)
        print(f"Saved model state to: {error_model_path}")
        raise


if __name__ == "__main__":
    main()