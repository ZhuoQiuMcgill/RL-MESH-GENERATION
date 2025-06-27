#!/usr/bin/env python3
"""
Evaluation script for trained mesh generation agents.

This script evaluates trained SAC agents on various test domains and
generates comprehensive performance reports.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import json

# Add project root to path
ROOT_DIR = os.path.getcwd()
sys.path.append(ROOT_DIR)

from rl_mesher.environment import MeshEnv
from rl_mesher.agent import SACAgent
from rl_mesher.utils.visualization import (
    plot_mesh, plot_quality_metrics, create_summary_report,
    plot_state_visualization, save_mesh_animation_frames
)
from rl_mesher.utils.geometry import calculate_element_quality, calculate_polygon_area


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained mesh generation agent")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file"
    )

    parser.add_argument(
        "--configs",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--test-domains",
        nargs="+",
        default=["T1.txt", "T2.txt", "T3.txt"],
        help="Test domain files to evaluate on"
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes per domain"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run evaluation on"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for evaluation results"
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy for evaluation"
    )

    parser.add_argument(
        "--save-meshes",
        action="store_true",
        help="Save generated mesh visualizations"
    )

    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="Save mesh generation animation frames"
    )

    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline methods (if available)"
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


def evaluate_single_domain(agent: SACAgent, env: MeshEnv, domain_file: str,
                           num_episodes: int, deterministic: bool = True,
                           save_meshes: bool = False, save_animation: bool = False,
                           output_dir: str = "") -> Dict:
    """
    Evaluate agent on a single domain with enhanced step-by-step logging
    to debug premature termination. This is the complete function.
    """
    print(f"Evaluating on domain: {domain_file}")

    # Set domain
    env.set_domain(domain_file)

    # Track results
    episode_rewards = []
    episode_lengths = []
    mesh_qualities = []
    completion_rates = []
    all_meshes = []

    # Animation tracking
    boundary_histories = []
    elements_histories = []

    for episode in range(num_episodes):
        print(f"\n--- Starting Standalone Eval Episode {episode + 1}/{num_episodes} ---")

        # Reset environment
        state_dict, _ = env.reset()
        initial_boundary_size = len(env.current_boundary)
        # print(f"  [Step 00] Environment reset. Initial boundary size: {initial_boundary_size}")

        # Episode tracking
        episode_reward = 0.0
        episode_length = 0
        completed = False

        if save_animation and episode == 0:
            boundary_history = [env.current_boundary.copy()]
            elements_history = [[]]

        while True:
            step_num = episode_length + 1
            try:
                # --- Pre-Step Logging ---
                boundary_size_before = len(env.current_boundary)

                # Select action
                action = agent.select_action(state_dict, deterministic=deterministic)

                # --- Take Step ---
                # Pass None for global_timestep as it's not relevant in standalone eval
                next_state_dict, reward, terminated, truncated, info = env.step(action, None)

                # --- Post-Step Logging ---
                boundary_size_after = info.get('boundary_vertices', 0)
                is_valid_element = info.get('is_valid_element', False)

                # print(f"  [Step {step_num:02d}] "
                #       f"Boundary: {boundary_size_before} -> {boundary_size_after}, "
                #       f"Valid: {is_valid_element}, "
                #       f"Reward: {reward:+.3f}, "
                #       f"Term: {terminated}, "
                #       f"Trunc: {truncated}")

                # Update tracking
                episode_reward += reward
                episode_length += 1

                # Animation tracking
                if save_animation and episode == 0:
                    boundary_history.append(env.current_boundary.copy())
                    elements_history.append(env.generated_elements.copy())

                # Check completion
                if terminated or truncated:
                    print(f"--- Episode {episode + 1} Ended at Step {step_num} ---")
                    print(f"  Reason: {'Terminated' if terminated else 'Truncated'}")
                    print(f"  Final Info: {info}")
                    completed = terminated
                    break

                state_dict = next_state_dict

            except Exception as e:
                print(f"!!!!!! CRASH DETECTED IN EPISODE {episode + 1} AT STEP {step_num} !!!!!!")
                print(f"  Error Type: {type(e).__name__}")
                print(f"  Error Message: {e}")
                import traceback
                traceback.print_exc()
                terminated = True
                break

        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        completion_rates.append(1.0 if completed else 0.0)

        final_boundary, final_elements = env.get_current_mesh()
        all_meshes.append((final_boundary, final_elements))
        quality_metrics = env.get_mesh_quality_metrics()
        mesh_qualities.append(quality_metrics)

        if save_animation and episode == 0:
            boundary_histories = boundary_history
            elements_histories = elements_history

    # Calculate aggregate statistics
    results = {
        'domain_file': domain_file,
        'num_episodes': num_episodes,
        'rewards': {
            'mean': np.mean(episode_rewards) if episode_rewards else 0,
            'std': np.std(episode_rewards) if episode_rewards else 0,
            'min': np.min(episode_rewards) if episode_rewards else 0,
            'max': np.max(episode_rewards) if episode_rewards else 0,
            'values': episode_rewards
        },
        'episode_lengths': {
            'mean': np.mean(episode_lengths) if episode_lengths else 0,
            'std': np.std(episode_lengths) if episode_lengths else 0,
            'min': np.min(episode_lengths) if episode_lengths else 0,
            'max': np.max(episode_lengths) if episode_lengths else 0,
            'values': episode_lengths
        },
        'completion_rate': np.mean(completion_rates) if completion_rates else 0,
        'mesh_quality': {}
    }

    if mesh_qualities and any(mq for mq in mesh_qualities):
        quality_keys = next((item.keys() for item in mesh_qualities if item), [])
        for key in quality_keys:
            values = [mq[key] for mq in mesh_qualities if mq and key in mq]
            if values:
                results['mesh_quality'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }

    # Save visualizations if requested
    if save_meshes and any(all_meshes):
        domain_name = os.path.splitext(domain_file)[0]
        mesh_dir = os.path.join(output_dir, "meshes", domain_name)
        os.makedirs(mesh_dir, exist_ok=True)

        if episode_rewards:
            best_episode = np.argmax(episode_rewards)
            best_boundary, best_elements = all_meshes[best_episode]
            mesh_path = os.path.join(mesh_dir, f"best_mesh_episode_{best_episode}.png")
            plot_mesh(best_boundary, best_elements,
                      title=f"Best Mesh - {domain_name} (Episode {best_episode})",
                      save_path=mesh_path)

            worst_episode = np.argmin(episode_rewards)
            worst_boundary, worst_elements = all_meshes[worst_episode]
            mesh_path = os.path.join(mesh_dir, f"worst_mesh_episode_{worst_episode}.png")
            plot_mesh(worst_boundary, worst_elements,
                      title=f"Worst Mesh - {domain_name} (Episode {worst_episode})",
                      save_path=mesh_path)

    # Save animation if requested
    if save_animation and boundary_histories:
        domain_name = os.path.splitext(domain_file)[0]
        anim_dir = os.path.join(output_dir, "animations", domain_name)
        save_mesh_animation_frames(boundary_histories, elements_histories, anim_dir)

    return results


def compare_with_baseline(domain_results: List[Dict], output_dir: str):
    """
    Compare results with baseline methods (placeholder for now).

    Args:
        domain_results: Results from all domains
        output_dir: Output directory
    """
    print("Baseline comparison not implemented yet.")
    # This would integrate with commercial meshing software
    # like Blossom-Quad or Pave for comparison
    pass


def generate_evaluation_report(all_results: List[Dict], args, config: Dict,
                               output_dir: str):
    """
    Generate comprehensive evaluation report.

    Args:
        all_results: Results from all domains
        args: Command line arguments
        config: Configuration dictionary
        output_dir: Output directory
    """
    print("Generating evaluation report...")

    # Create report data
    report = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model,
            'test_domains': args.test_domains,
            'num_episodes_per_domain': args.num_episodes,
            'deterministic': args.deterministic,
            'configs': config
        },
        'domain_results': all_results,
        'aggregate_results': {}
    }

    # Calculate aggregate statistics across all domains
    all_rewards = []
    all_lengths = []
    all_completion_rates = []
    all_mesh_qualities = {}

    for domain_result in all_results:
        all_rewards.extend(domain_result['rewards']['values'])
        all_lengths.extend(domain_result['episode_lengths']['values'])
        all_completion_rates.append(domain_result['completion_rate'])

        # Aggregate mesh quality metrics
        for metric, data in domain_result['mesh_quality'].items():
            if metric not in all_mesh_qualities:
                all_mesh_qualities[metric] = []
            all_mesh_qualities[metric].extend(data['values'])

    # Store aggregate results
    report['aggregate_results'] = {
        'overall_mean_reward': np.mean(all_rewards),
        'overall_std_reward': np.std(all_rewards),
        'overall_mean_length': np.mean(all_lengths),
        'overall_completion_rate': np.mean(all_completion_rates),
        'overall_mesh_quality': {}
    }

    for metric, values in all_mesh_qualities.items():
        report['aggregate_results']['overall_mesh_quality'][metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }

    # Save detailed report
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Save summary report
    summary_path = os.path.join(output_dir, "evaluation_summary.yaml")
    summary = {
        'timestamp': report['evaluation_info']['timestamp'],
        'num_domains': len(all_results),
        'total_episodes': len(all_rewards),
        'overall_performance': {
            'mean_reward': report['aggregate_results']['overall_mean_reward'],
            'completion_rate': report['aggregate_results']['overall_completion_rate'],
            'mean_episode_length': report['aggregate_results']['overall_mean_length']
        },
        'domain_performance': {
            domain_result['domain_file']: {
                'mean_reward': domain_result['rewards']['mean'],
                'completion_rate': domain_result['completion_rate']
            }
            for domain_result in all_results
        }
    }

    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    # Generate quality metrics visualization
    if all_mesh_qualities:
        quality_plot_path = os.path.join(output_dir, "quality_metrics.png")
        plot_quality_metrics(
            all_mesh_qualities,
            title="Mesh Quality Metrics Across All Domains",
            save_path=quality_plot_path
        )

    return report


def print_results_summary(all_results: List[Dict]):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    for result in all_results:
        domain_name = result['domain_file']
        print(f"\nDomain: {domain_name}")
        print(f"  Mean Reward: {result['rewards']['mean']:.2f} ± {result['rewards']['std']:.2f}")
        print(f"  Completion Rate: {result['completion_rate']:.1%}")
        print(f"  Mean Episode Length: {result['episode_lengths']['mean']:.1f}")

        if result['mesh_quality']:
            print("  Mesh Quality:")
            for metric, data in result['mesh_quality'].items():
                print(f"    {metric}: {data['mean']:.3f} ± {data['std']:.3f}")

    # Overall statistics
    all_rewards = []
    all_completion_rates = []

    for result in all_results:
        all_rewards.extend(result['rewards']['values'])
        all_completion_rates.append(result['completion_rate'])

    print(f"\nOverall Performance:")
    print(f"  Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean Completion Rate: {np.mean(all_completion_rates):.1%}")
    print(f"  Total Episodes: {len(all_rewards)}")


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Setup
    config = load_config(args.config)
    device = setup_device(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting evaluation with model: {args.model}")
    print(f"Test domains: {args.test_domains}")
    print(f"Episodes per domain: {args.num_episodes}")
    print(f"Deterministic policy: {args.deterministic}")

    # Load agent
    print("Loading trained agent...")
    agent = SACAgent(config, device)
    agent.load(args.model)
    agent.set_training_mode(False)  # Set to evaluation mode

    # Setup environment
    env = MeshEnv(config)

    # Evaluate on each domain
    all_results = []

    for domain_file in args.test_domains:
        try:
            domain_results = evaluate_single_domain(
                agent, env, domain_file, args.num_episodes,
                deterministic=args.deterministic,
                save_meshes=args.save_meshes,
                save_animation=args.save_animation,
                output_dir=args.output_dir
            )
            all_results.append(domain_results)

        except Exception as e:
            print(f"Error evaluating domain {domain_file}: {str(e)}")
            continue

    if not all_results:
        print("No successful evaluations completed!")
        return

    # Generate comprehensive report
    report = generate_evaluation_report(all_results, args, config, args.output_dir)

    # Compare with baseline if requested
    if args.compare_baseline:
        compare_with_baseline(all_results, args.output_dir)

    # Print summary
    print_results_summary(all_results)

    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
