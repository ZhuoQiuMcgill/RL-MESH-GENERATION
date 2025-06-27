#!/usr/bin/env python3
"""
Enhanced training script for RL-based mesh generation with detailed logging.

This script trains a SAC agent to perform automatic quadrilateral mesh generation
using reinforcement learning with comprehensive progress tracking and logging.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import time
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
        "--config",
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

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with additional details"
    )

    parser.add_argument(
        "--log-frequency",
        type=int,
        default=None,
        help="Override log frequency from config"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps per episode from config"
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

    print(f"🖥️  Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.1f}GB")
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
        print(f"📁 Created directory: {path}")


def setup_environment(config: dict, domain_config) -> MeshEnv:
    """
    Setup training environment based on the domain configuration.
    It automatically detects single vs. multi-domain from the config type.
    """
    max_steps = config['environment'].get('max_steps', 1000)

    # If the domain_config is a list, we are in multi-domain mode
    if isinstance(domain_config, list):
        print(f"🌍 Setting up multi-domain environment with domains: {domain_config}")
        print(f"   Max steps per episode: {max_steps}")
        env = MultiDomainMeshEnv(config, domain_config)
    # Otherwise, it's single-domain mode
    else:
        print(f"🌍 Setting up single-domain environment with domain: {domain_config}")
        print(f"   Max steps per episode: {max_steps}")
        # We need to manually set the domain file in the config for MeshEnv to find it
        config['domain']['training_domain'] = domain_config
        env = MeshEnv(config)

    return env


def setup_agent(config: dict, device: torch.device) -> SACAgent:
    """Setup SAC agent."""
    agent = SACAgent(config, device)

    print("🤖 SAC Agent initialized:")
    network_info = agent.get_network_info()
    print(f"   Actor parameters: {network_info['actor_parameters']:,}")
    print(f"   Critic parameters: {network_info['critic_parameters']:,}")
    print(f"   Total parameters: {network_info['total_parameters']:,}")
    print(f"   Alpha mode: {'static' if network_info['use_static_alpha'] else 'automatic'}")
    print(f"   Current alpha: {network_info['current_alpha']:.4f}")

    return agent


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    seed = int(seed)  # Ensure seed is integer
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"🎲 Set random seed to: {seed}")


def save_config(config: dict, save_path: str):
    """Save configuration to file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_training_summary(results: dict, config: dict, args,
                            experiment_name: str) -> dict:
    """Create comprehensive training summary."""
    summary = {
        'experiment_info': {
            'name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'total_runtime_hours': results.get('total_training_time', 0) / 3600,
            'device': str(args.device),
            'multi_domain': args.multi_domain,
            'domain_files': args.domain_files if args.multi_domain else [config['domain']['training_domain']],
            'max_steps': config['environment'].get('max_steps', 1000)
        },
        'training_config': {
            'total_timesteps': config['training']['total_timesteps'],
            'batch_size': config['sac']['batch_size'],
            'learning_rate': config['sac']['learning_rate'],
            'buffer_size': config['sac']['buffer_size'],
            'evaluation_frequency': config['training']['evaluation_freq'],
            'alpha_mode': 'static' if config['sac'].get('use_static_alpha', False) else 'automatic',
            'alpha_value': config['sac'].get('static_alpha', config['sac']['alpha'])
        },
        'performance_summary': {
            'total_episodes': len(results['episode_rewards']) if results['episode_rewards'] else 0,
            'final_reward': results['episode_rewards'][-1] if results['episode_rewards'] else 0,
            'best_reward': results.get('best_reward', 0),
            'best_episode': results.get('best_episode', 0),
            'mean_episode_time': np.mean(results['episode_times']) if results.get('episode_times') else 0,
            'completion_rate': np.mean(results['episode_completion_rates']) if results.get(
                'episode_completion_rates') else 0,
        }
    }

    # Calculate performance improvement
    if results['episode_rewards'] and len(results['episode_rewards']) > 100:
        early_performance = np.mean(results['episode_rewards'][:100])
        late_performance = np.mean(results['episode_rewards'][-100:])
        improvement = late_performance - early_performance
        summary['performance_summary']['early_performance'] = early_performance
        summary['performance_summary']['late_performance'] = late_performance
        summary['performance_summary']['improvement'] = improvement
        summary['performance_summary']['improvement_percentage'] = (improvement / abs(
            early_performance)) * 100 if early_performance != 0 else 0

    # Evaluation results
    if results['evaluation_results']:
        eval_rewards = [r['mean_reward'] for r in results['evaluation_results']]
        summary['evaluation_summary'] = {
            'num_evaluations': len(results['evaluation_results']),
            'best_eval_reward': max(eval_rewards),
            'final_eval_reward': eval_rewards[-1],
            'eval_improvement': eval_rewards[-1] - eval_rewards[0] if len(eval_rewards) > 1 else 0
        }

    return summary


def print_training_header(args, config, experiment_name):
    """Print formatted training header that auto-detects training mode."""
    print("\n" + "=" * 80)
    print("🚀 RL-MESH-GENERATION TRAINING")
    print("=" * 80)
    print(f"📝 Experiment: {experiment_name}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Configuration: {args.config}")
    print(f"🎯 Algorithm: SAC (Soft Actor-Critic)")

    # --- MODIFICATION START ---
    # Auto-detect training mode from the config file
    training_domain_config = config['domain']['training_domain']
    if isinstance(training_domain_config, list):
        print(f"🌍 Training Mode: Multi-Domain ({len(training_domain_config)} domains)")
        print(f"   Domains: {', '.join(training_domain_config)}")
    else:
        print(f"🌍 Training Mode: Single-Domain ({training_domain_config})")
    # --- MODIFICATION END ---

    print(f"📊 Total Timesteps: {config['training']['total_timesteps']:,}")
    print(f"📈 Evaluation Frequency (Episodes): {config['training']['eval_freq_episode']:,}")
    print(f"💾 Save Frequency (Timesteps): {config['training']['save_freq']:,}")
    print(f"📏 Max Steps per Episode: {config['environment'].get('max_steps', 1000)}")

    if config['sac'].get('use_static_alpha', False):
        print(f"🌡️  Alpha (temperature): {config['sac'].get('static_alpha', 0.1)} (static)")
    else:
        print(f"🌡️  Alpha (temperature): {config['sac']['alpha']} (automatic tuning)")

    print("=" * 80)


def main():
    """Main training function, now driven by config file structure."""
    start_time = time.time()
    args = parse_arguments()

    print("🎯 Starting RL-Mesh-Generation Training")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")

    config = load_config(args.config)
    print(f"📋 Loaded configuration from: {args.config}")

    if args.log_frequency is not None:
        config['training']['log_interval'] = args.log_frequency
        print(f"🔄 Overriding log frequency to: {args.log_frequency}")

    if args.max_steps is not None:
        config['environment']['max_steps'] = args.max_steps
        print(f"🔄 Overriding max steps to: {args.max_steps}")

    if args.seed is not None:
        config['training']['seed'] = int(args.seed)
        set_random_seeds(int(args.seed))
    elif 'seed' in config['training']:
        set_random_seeds(int(config['training']['seed']))

    device = setup_device(args.device)

    # --- MODIFICATION START ---
    # Create experiment name based on training mode detected from config
    training_domain_config = config['domain']['training_domain']
    is_multi_domain = isinstance(training_domain_config, list)

    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_suffix = "multi_domain" if is_multi_domain else "single_domain"
        experiment_name = f"mesh_sac_{domain_suffix}_{timestamp}"

    print(f"🏷️  Experiment name: {experiment_name}")

    setup_directories(config, experiment_name)

    config_save_path = os.path.join(config['paths']['logs_dir'], "config.yaml")
    save_config(config, config_save_path)
    print(f"💾 Configuration saved to: {config_save_path}")

    # Pass the domain config directly to the setup function
    env = setup_environment(config, training_domain_config)
    # --- MODIFICATION END ---

    agent = setup_agent(config, device)

    if args.resume:
        print(f"🔄 Resuming training from: {args.resume}")
        agent.load(args.resume)

    trainer = MeshSACTrainer(agent, env, config)
    print_training_header(args, config, experiment_name)

    try:
        print("\n🚀 STARTING TRAINING")
        results = trainer.train()

        # ... (rest of the function remains the same)

        results_path = os.path.join(config['paths']['logs_dir'], "training_results.pt")
        trainer.save_training_results(results_path)
        summary = create_training_summary(results, config, args, experiment_name)
        summary_path = os.path.join(config['paths']['logs_dir'], "training_summary.yaml")
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        print(f"📊 Training summary saved to: {summary_path}")
        print("\n📈 Generating training plots...")
        if results['episode_rewards']:
            learning_curve_path = os.path.join(config['paths']['figures_dir'], "learning_curve.png")
            plot_learning_curve(
                results['episode_rewards'],
                results['episode_lengths'],
                title=f"Training Results - {experiment_name}",
                save_path=learning_curve_path
            )
            print(f"📈 Learning curve saved to: {learning_curve_path}")
        if results['training_stats']:
            progress_path = os.path.join(config['paths']['figures_dir'], "training_progress.png")
            plot_training_progress(
                results['training_stats'],
                save_path=progress_path
            )
            print(f"📊 Training progress saved to: {progress_path}")
        total_time = time.time() - start_time
        print(f"\n" + "=" * 80)
        print(f"🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"=" * 80)
        print(f"⏱️  Total Runtime: {total_time / 3600:.2f} hours")
        print(f"📊 Final Statistics:")
        print(f"   Total Episodes: {len(results['episode_rewards'])}")
        if results['episode_rewards']:
            print(f"   Final Reward: {results['episode_rewards'][-1]:.2f}")
            print(f"   Best Reward: {results.get('best_reward', 0):.2f} (Episode {results.get('best_episode', 0)})")
        if results.get('episode_completion_rates'):
            print(f"   Success Rate: {np.mean(results['episode_completion_rates']):.1%}")
        if results['evaluation_results']:
            eval_rewards = [r['mean_reward'] for r in results['evaluation_results']]
            if eval_rewards:
                print(f"   Best Evaluation Reward: {max(eval_rewards):.2f}")
                print(f"   Final Evaluation Reward: {eval_rewards[-1]:.2f}")
        print(f"\n📁 Results Location:")
        print(f"   Logs: {config['paths']['logs_dir']}")
        print(f"   Models: {config['paths']['models_dir']}")
        print(f"   Figures: {config['paths']['figures_dir']}")
        print(f"=" * 80)

    except KeyboardInterrupt:
        print(f"\n" + "=" * 80)
        print(f"⚠️  TRAINING INTERRUPTED BY USER")
        print(f"=" * 80)
        interrupt_model_path = os.path.join(config['paths']['models_dir'], "interrupted_model.pt")
        agent.save(interrupt_model_path)
        print(f"💾 Saved interrupted model to: {interrupt_model_path}")
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"❌ TRAINING FAILED: {str(e)}")
        print(f"=" * 80)
        error_model_path = os.path.join(config['paths']['models_dir'], "error_model.pt")
        try:
            agent.save(error_model_path)
            print(f"💾 Saved model state to: {error_model_path}")
        except Exception as save_error:
            print(f"⚠️  Could not save error model: {save_error}")
        raise
    finally:
        print(f"\n🏁 Training session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
