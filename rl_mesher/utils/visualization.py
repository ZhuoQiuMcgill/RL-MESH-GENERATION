import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Optional
import os
from datetime import datetime


def plot_mesh(boundary: np.ndarray, elements: List[np.ndarray] = None,
              title: str = "Mesh Visualization", save_path: Optional[str] = None,
              figsize: tuple = (10, 8), show_vertices: bool = True,
              show_element_numbers: bool = False, original_boundary: np.ndarray = None) -> None:
    """
    Plot mesh boundary and generated quadrilateral elements with enhanced visualization.

    Args:
        boundary: Current boundary vertices [N, 2]
        elements: List of element vertices, each element is [4, 2] array
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        show_vertices: Whether to show vertex numbers
        show_element_numbers: Whether to show element numbers
        original_boundary: Original domain boundary to display for reference
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot original boundary first (if provided)
    if original_boundary is not None and len(original_boundary) > 0:
        # Close the original boundary for plotting
        original_closed = np.vstack([original_boundary, original_boundary[0]])
        ax.plot(original_closed[:, 0], original_closed[:, 1],
                'gray', linewidth=1, linestyle='--', alpha=0.7,
                label='Original Domain', zorder=1)

    # Plot generated elements with enhanced rendering
    if elements is not None and len(elements) > 0:
        for elem_idx, element in enumerate(elements):
            if element is not None and len(element) >= 3:
                # Ensure we have exactly 4 vertices for quadrilateral
                if len(element) == 4:
                    quad_vertices = element
                elif len(element) == 3:
                    # If somehow we get a triangle, duplicate last vertex to make it a quad
                    quad_vertices = np.vstack([element, element[-1]])
                else:
                    # Take first 4 vertices if more than 4
                    quad_vertices = element[:4]

                # Create quadrilateral patch for filled area
                try:
                    quad = patches.Polygon(quad_vertices,
                                           linewidth=2, edgecolor='blue',
                                           facecolor='lightblue', alpha=0.3, zorder=2)
                    ax.add_patch(quad)

                    # Plot element edges explicitly to ensure all edges are visible
                    quad_closed = np.vstack([quad_vertices, quad_vertices[0]])
                    ax.plot(quad_closed[:, 0], quad_closed[:, 1],
                            'blue', linewidth=2, alpha=0.8, zorder=3)

                    # Show element numbers if requested
                    if show_element_numbers:
                        center = np.mean(quad_vertices, axis=0)
                        ax.text(center[0], center[1], str(elem_idx),
                                ha='center', va='center', fontsize=8,
                                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))

                except Exception as e:
                    # Fallback: just plot the vertices as points
                    ax.scatter(quad_vertices[:, 0], quad_vertices[:, 1],
                               c='blue', s=20, alpha=0.7, zorder=2)

    # Plot current boundary
    if boundary is not None and len(boundary) > 0:
        # Close the boundary for plotting
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
                'black', linewidth=3, label='Current Boundary', zorder=4)

        # Plot boundary vertices with numbers
        if show_vertices:
            for i, vertex in enumerate(boundary):
                ax.scatter(vertex[0], vertex[1], c='red', s=80, zorder=5)
                ax.text(vertex[0] + 0.02, vertex[1] + 0.02, str(i),
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='black', alpha=0.9))

    # Set equal aspect ratio and adjust limits
    ax.set_aspect('equal')

    # Calculate bounds including all elements
    all_x_coords = []
    all_y_coords = []

    # Add original boundary coordinates
    if original_boundary is not None and len(original_boundary) > 0:
        all_x_coords.extend(original_boundary[:, 0])
        all_y_coords.extend(original_boundary[:, 1])

    # Add current boundary coordinates
    if boundary is not None and len(boundary) > 0:
        all_x_coords.extend(boundary[:, 0])
        all_y_coords.extend(boundary[:, 1])

    # Add element coordinates
    if elements is not None and len(elements) > 0:
        for element in elements:
            if element is not None and len(element) > 0:
                all_x_coords.extend(element[:, 0])
                all_y_coords.extend(element[:, 1])

    # Set limits with padding
    if all_x_coords and all_y_coords:
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)

        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = max(x_range, y_range) * 0.1 if max(x_range, y_range) > 0 else 0.1

        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

    # Labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Add enhanced mesh statistics
    current_vertices = len(boundary) if boundary is not None else 0
    original_vertices = len(original_boundary) if original_boundary is not None else 0
    generated_elements = len(elements) if elements is not None else 0

    info_text = f"Original vertices: {original_vertices}\n"
    info_text += f"Current vertices: {current_vertices}\n"
    info_text += f"Generated elements: {generated_elements}"

    # Add element quality info if available
    if elements is not None and len(elements) > 0:
        total_area = 0
        valid_elements = 0
        for element in elements:
            if element is not None and len(element) >= 3:
                try:
                    # Calculate area using shoelace formula
                    if len(element) >= 3:
                        x = element[:, 0]
                        y = element[:, 1]
                        area = 0.5 * abs(sum(x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i]
                                             for i in range(len(x))))
                        total_area += area
                        valid_elements += 1
                except:
                    pass

        info_text += f"\nTotal meshed area: {total_area:.3f}"
        if valid_elements != generated_elements:
            info_text += f"\nValid elements: {valid_elements}/{generated_elements}"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'))

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close to prevent memory accumulation
        except Exception as e:
            print(f"Warning: Failed to save figure to {save_path}: {e}")
            plt.show()
    else:
        plt.show()


def plot_mesh_generation_progress(boundary_history: List[np.ndarray],
                                  elements_history: List[List[np.ndarray]],
                                  title: str = "Mesh Generation Progress",
                                  save_path: Optional[str] = None,
                                  max_steps: int = 6,
                                  original_boundary: np.ndarray = None) -> None:
    """
    Plot mesh generation progress showing multiple steps with enhanced visualization.

    Args:
        boundary_history: List of boundary states
        elements_history: List of element lists for each step
        title: Plot title
        save_path: Path to save figure
        max_steps: Maximum number of steps to show
        original_boundary: Original domain boundary
    """
    if not boundary_history:
        print("No boundary history to plot")
        return

    # Limit the number of steps to display
    n_steps = min(len(boundary_history), max_steps)

    # Calculate grid layout
    cols = min(3, n_steps)
    rows = (n_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_steps == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n_steps):
        ax = axes[i] if n_steps > 1 else axes[0]

        # Get current boundary and elements
        current_boundary = boundary_history[i]
        current_elements = elements_history[i] if i < len(elements_history) else []

        # Plot original boundary
        if original_boundary is not None and len(original_boundary) > 0:
            original_closed = np.vstack([original_boundary, original_boundary[0]])
            ax.plot(original_closed[:, 0], original_closed[:, 1],
                    'gray', linewidth=1, linestyle='--', alpha=0.5, label='Original')

        # Plot elements
        for elem_idx, element in enumerate(current_elements):
            if element is not None and len(element) >= 3:
                # Ensure quadrilateral
                if len(element) == 4:
                    quad_vertices = element
                elif len(element) == 3:
                    quad_vertices = np.vstack([element, element[-1]])
                else:
                    quad_vertices = element[:4]

                try:
                    quad = patches.Polygon(quad_vertices,
                                           linewidth=1, edgecolor='blue',
                                           facecolor='lightblue', alpha=0.4)
                    ax.add_patch(quad)
                except:
                    ax.scatter(quad_vertices[:, 0], quad_vertices[:, 1],
                               c='blue', s=10, alpha=0.7)

        # Plot current boundary
        if current_boundary is not None and len(current_boundary) > 0:
            boundary_closed = np.vstack([current_boundary, current_boundary[0]])
            ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
                    'black', linewidth=2, label='Boundary')

            # Plot boundary vertices
            ax.scatter(current_boundary[:, 0], current_boundary[:, 1],
                       c='red', s=30, zorder=5)

        ax.set_aspect('equal')
        ax.set_title(f'Step {i + 1}\nElements: {len(current_elements)}, Vertices: {len(current_boundary)}',
                     fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set consistent limits across all subplots
        if original_boundary is not None:
            x_min, x_max = original_boundary[:, 0].min(), original_boundary[:, 0].max()
            y_min, y_max = original_boundary[:, 1].min(), original_boundary[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            padding = max(x_range, y_range) * 0.1
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)

    # Hide empty subplots
    for i in range(n_steps, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save figure to {save_path}: {e}")
            plt.show()
    else:
        plt.show()


def plot_learning_curve(episode_rewards: List[float],
                        episode_lengths: List[int],
                        title: str = "Learning Curve",
                        save_path: Optional[str] = None,
                        window_size: int = 100) -> None:
    """
    Plot learning curve with episode rewards and lengths.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        title: Plot title
        save_path: Path to save figure
        window_size: Window size for moving average
    """
    plot_enhanced_learning_curve(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        title=title,
        save_path=save_path,
        window_size=window_size
    )


def plot_enhanced_learning_curve(episode_rewards: List[float],
                                 episode_lengths: List[int] = None,
                                 episode_completion_rates: List[float] = None,
                                 title: str = "Enhanced Learning Curve",
                                 save_path: Optional[str] = None,
                                 window_size: int = 100) -> None:
    """
    Enhanced learning curve with multiple metrics and statistical analysis.
    """
    if not episode_rewards:
        print("No episode rewards to plot")
        return

    # Determine number of subplots based on available data
    n_plots = 1  # Always have rewards
    if episode_lengths:
        n_plots += 1
    if episode_completion_rates:
        n_plots += 1

    # Create subplot layout
    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes = [axes]
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    episodes = np.arange(len(episode_rewards))
    current_axis = 0

    # 1. Episode Rewards
    ax = axes[current_axis]
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', linewidth=0.8, label='Episode Rewards')

    # Moving average for rewards
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
        avg_episodes = episodes[window_size - 1:]
        ax.plot(avg_episodes, moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add reward statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)

    stats_text = f'Mean: {mean_reward:.2f}\nStd: {std_reward:.2f}\nMax: {max_reward:.2f}\nMin: {min_reward:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    current_axis += 1

    # 2. Episode Lengths (if available)
    if episode_lengths and current_axis < len(axes):
        ax = axes[current_axis]
        ax.plot(episodes, episode_lengths, alpha=0.6, color='green', linewidth=1, label='Episode Lengths')

        # Moving average for lengths
        if len(episode_lengths) >= window_size:
            moving_avg = np.convolve(episode_lengths, np.ones(window_size) / window_size, mode='valid')
            avg_episodes = episodes[window_size - 1:]
            ax.plot(avg_episodes, moving_avg, color='darkgreen', linewidth=2,
                    label=f'Moving Avg ({window_size})')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add length statistics
        mean_length = np.mean(episode_lengths)
        stats_text = f'Mean: {mean_length:.1f}\nMax: {np.max(episode_lengths)}\nMin: {np.min(episode_lengths)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        current_axis += 1

    # 3. Completion Rates (if available)
    if episode_completion_rates and current_axis < len(axes):
        ax = axes[current_axis]
        # Convert to percentage and plot as bars
        completion_percentages = [r * 100 for r in episode_completion_rates]

        # Use moving average for completion rate
        if len(episode_completion_rates) >= window_size:
            moving_completion = np.convolve(completion_percentages, np.ones(window_size) / window_size, mode='valid')
            avg_episodes = episodes[window_size - 1:]
            ax.plot(avg_episodes, moving_completion, color='darkgreen', linewidth=2,
                    label=f'Completion Rate ({window_size} ep avg)')
        else:
            ax.plot(episodes, completion_percentages, color='darkgreen', linewidth=2, label='Completion Rate')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Completion Rate (%)')
        ax.set_title('Episode Completion Rate')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics
        mean_completion = np.mean(completion_percentages)
        stats_text = f'Mean: {mean_completion:.1f}%\nRecent: {np.mean(completion_percentages[-50:]) if len(completion_percentages) >= 50 else mean_completion:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save figure to {save_path}: {e}")
            plt.show()
    else:
        plt.show()


def plot_training_progress(training_stats: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive training progress including multiple metrics.

    Args:
        training_stats: Dictionary containing training statistics
        save_path: Path to save the figure
    """
    if not training_stats:
        print("No training statistics available for plotting")
        return

    # Extract available metrics
    available_metrics = []
    for key in ['actor_loss', 'critic_loss', 'alpha', 'q_loss', 'policy_loss']:
        if key in training_stats and training_stats[key]:
            available_metrics.append(key)

    if not available_metrics:
        print("No recognized training metrics found")
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(available_metrics):
        values = training_stats[metric]
        steps = np.arange(len(values))

        axes[i].plot(steps, values, linewidth=1.5, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Evolution')
        axes[i].set_xlabel('Training Step')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True, alpha=0.3)

        # Add moving average for noisy metrics
        if len(values) > 50 and metric in ['actor_loss', 'critic_loss', 'q_loss', 'policy_loss']:
            window = min(50, len(values) // 10)
            moving_avg = np.convolve(values, np.ones(window) / window, mode='valid')
            axes[i].plot(steps[window - 1:], moving_avg, color='red', linewidth=2,
                         label=f'Moving Avg ({window})', alpha=0.8)
            axes[i].legend()

        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}'
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Training Progress - Algorithm Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save figure to {save_path}: {e}")
            plt.show()
    else:
        plt.show()


def plot_evaluation_progress(evaluation_results: List[Dict], save_path: Optional[str] = None) -> None:
    """
    Plot evaluation progress during training.

    Args:
        evaluation_results: List of evaluation result dictionaries
        save_path: Path to save the plot
    """
    if not evaluation_results:
        print("No evaluation results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    eval_episodes = np.arange(len(evaluation_results))

    # Extract metrics
    mean_rewards = [r.get('mean_reward', 0) for r in evaluation_results]
    std_rewards = [r.get('std_reward', 0) for r in evaluation_results]
    mean_lengths = [r.get('mean_length', 0) for r in evaluation_results]
    completion_rates = [r.get('completion_rate', 0) for r in evaluation_results]

    # Plot mean reward with error bars
    ax = axes[0, 0]
    ax.errorbar(eval_episodes, mean_rewards, yerr=std_rewards,
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax.set_title('Evaluation Mean Reward')
    ax.set_xlabel('Evaluation #')
    ax.set_ylabel('Mean Reward')
    ax.grid(True, alpha=0.3)

    # Plot mean episode length
    ax = axes[0, 1]
    ax.plot(eval_episodes, mean_lengths, marker='s', linewidth=2, color='green')
    ax.set_title('Evaluation Mean Episode Length')
    ax.set_xlabel('Evaluation #')
    ax.set_ylabel('Mean Steps')
    ax.grid(True, alpha=0.3)

    # Plot completion rate
    ax = axes[1, 0]
    ax.plot(eval_episodes, [r * 100 for r in completion_rates], marker='^', linewidth=2, color='red')
    ax.set_title('Evaluation Completion Rate')
    ax.set_xlabel('Evaluation #')
    ax.set_ylabel('Completion Rate (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Plot reward distribution evolution
    ax = axes[1, 1]
    for i, result in enumerate(evaluation_results[::max(1, len(evaluation_results) // 5)]):
        # Show distribution for selected evaluations
        rewards = [result.get('mean_reward', 0)] * 5  # Simplified representation
        ax.hist(rewards, bins=10, alpha=0.3, label=f'Eval {i * max(1, len(evaluation_results) // 5)}')
    ax.set_title('Reward Distribution Evolution')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Evaluation Progress During Training', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save figure to {save_path}: {e}")
            plt.show()
    else:
        plt.show()


def plot_training_dashboard(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive training dashboard.

    Args:
        results: Training results dictionary
        save_path: Path to save the dashboard
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Episode rewards
    if 'episode_rewards' in results:
        axes[0, 0].plot(results['episode_rewards'], alpha=0.7, label='Episode Rewards')
        if len(results['episode_rewards']) > 10:
            # Moving average
            window = min(100, len(results['episode_rewards']) // 10)
            moving_avg = np.convolve(results['episode_rewards'],
                                     np.ones(window) / window, mode='valid')
            axes[0, 0].plot(range(window - 1, len(results['episode_rewards'])),
                            moving_avg, 'r-', linewidth=2, label='Moving Average')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Episode lengths
    if 'episode_lengths' in results:
        axes[0, 1].plot(results['episode_lengths'], 'g-', alpha=0.7, label='Episode Lengths')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Element counts per episode
    if 'elements_per_episode' in results:
        axes[0, 2].plot(results['elements_per_episode'], 'purple', alpha=0.7,
                        label='Elements Generated')
        axes[0, 2].set_title('Elements Generated per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Element Count')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Training loss (if available)
    if 'training_losses' in results:
        axes[1, 0].plot(results['training_losses'], 'orange', alpha=0.7, label='Training Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')

    # Quality metrics (if available)
    if 'quality_metrics' in results:
        quality_data = results['quality_metrics']
        if isinstance(quality_data, dict):
            for metric_name, values in quality_data.items():
                axes[1, 1].plot(values, alpha=0.7, label=metric_name)
        axes[1, 1].set_title('Quality Metrics')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Statistics summary
    axes[1, 2].axis('off')
    stats_text = "Training Statistics\n\n"

    if 'episode_rewards' in results and results['episode_rewards']:
        rewards = results['episode_rewards']
        stats_text += f"Total Episodes: {len(rewards)}\n"
        stats_text += f"Mean Reward: {np.mean(rewards):.2f}\n"
        stats_text += f"Best Reward: {np.max(rewards):.2f}\n"
        stats_text += f"Worst Reward: {np.min(rewards):.2f}\n"
        stats_text += f"Std Reward: {np.std(rewards):.2f}\n\n"

    if 'episode_lengths' in results and results['episode_lengths']:
        lengths = results['episode_lengths']
        stats_text += f"Mean Episode Length: {np.mean(lengths):.1f}\n"
        stats_text += f"Max Episode Length: {np.max(lengths)}\n\n"

    if 'elements_per_episode' in results and results['elements_per_episode']:
        elements = results['elements_per_episode']
        stats_text += f"Mean Elements/Episode: {np.mean(elements):.1f}\n"
        stats_text += f"Max Elements/Episode: {np.max(elements)}\n"

    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                    fontfamily='monospace', fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save dashboard to {save_path}: {e}")
            plt.show()
    else:
        plt.show()


def create_training_report(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive training report.

    Args:
        results: Training results dictionary
        save_path: Path to save the report
    """
    # This function could be expanded to create a multi-page PDF report
    # For now, it creates a comprehensive dashboard


def plot_training_dashboard(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive training dashboard with multiple visualizations.

    Args:
        results: Training results dictionary
        save_path: Path to save the dashboard
    """
    fig = plt.figure(figsize=(20, 16))

    # Create grid layout
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Reward evolution (main plot)
    ax1 = fig.add_subplot(gs[0, :])
    if 'episode_rewards' in results and results['episode_rewards']:
        episode_rewards = results['episode_rewards']
        episodes = np.arange(len(episode_rewards))

        ax1.plot(episodes, episode_rewards, alpha=0.4, color='blue', linewidth=0.8)

        # Moving average
        window = min(100, len(episode_rewards) // 10)
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
            ax1.plot(episodes[window - 1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')

        ax1.set_title('Reward Evolution Over Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # 2. Episode lengths
    ax2 = fig.add_subplot(gs[1, 0])
    if 'episode_lengths' in results and results['episode_lengths']:
        episode_lengths = results['episode_lengths']
        episodes = np.arange(len(episode_lengths))
        ax2.plot(episodes, episode_lengths, 'g-', alpha=0.7, label='Episode Lengths')
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Element counts per episode
    ax3 = fig.add_subplot(gs[1, 1])
    if 'elements_per_episode' in results and results['elements_per_episode']:
        ax3.plot(results['elements_per_episode'], 'purple', alpha=0.7,
                 label='Elements Generated')
        ax3.set_title('Elements Generated per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Element Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Training loss (if available)
    ax4 = fig.add_subplot(gs[1, 2])
    if 'training_losses' in results and results['training_losses']:
        ax4.plot(results['training_losses'], 'orange', alpha=0.7, label='Training Loss')
        ax4.set_title('Training Loss')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

    # 5. Quality metrics (if available)
    ax5 = fig.add_subplot(gs[2, 0])
    if 'quality_metrics' in results and results['quality_metrics']:
        quality_data = results['quality_metrics']
        if isinstance(quality_data, dict):
            for metric_name, values in quality_data.items():
                ax5.plot(values, alpha=0.7, label=metric_name)
        ax5.set_title('Quality Metrics')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Quality Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Completion rates
    ax6 = fig.add_subplot(gs[2, 1])
    if 'episode_completion_rates' in results and results['episode_completion_rates']:
        completion_rates = [r * 100 for r in results['episode_completion_rates']]
        ax6.plot(completion_rates, 'red', alpha=0.7, label='Completion Rate')
        ax6.set_title('Episode Completion Rate')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Completion Rate (%)')
        ax6.set_ylim(0, 100)
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    # 7. Statistics summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    stats_text = "Training Statistics\n\n"

    if 'episode_rewards' in results and results['episode_rewards']:
        rewards = results['episode_rewards']
        stats_text += f"Total Episodes: {len(rewards)}\n"
        stats_text += f"Mean Reward: {np.mean(rewards):.2f}\n"
        stats_text += f"Best Reward: {np.max(rewards):.2f}\n"
        stats_text += f"Worst Reward: {np.min(rewards):.2f}\n"
        stats_text += f"Std Reward: {np.std(rewards):.2f}\n\n"

    if 'episode_lengths' in results and results['episode_lengths']:
        lengths = results['episode_lengths']
        stats_text += f"Mean Episode Length: {np.mean(lengths):.1f}\n"
        stats_text += f"Max Episode Length: {np.max(lengths)}\n\n"

    if 'elements_per_episode' in results and results['elements_per_episode']:
        elements = results['elements_per_episode']
        stats_text += f"Mean Elements/Episode: {np.mean(elements):.1f}\n"
        stats_text += f"Max Elements/Episode: {np.max(elements)}\n"

    ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save dashboard to {save_path}: {e}")
            plt.show()
    else:
        plt.show()


def create_training_report(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive training report.

    Args:
        results: Training results dictionary
        save_path: Path to save the report
    """
    # This function could be expanded to create a multi-page PDF report
    # For now, it creates a comprehensive dashboard
    plot_training_dashboard(results, save_path)


def plot_state_visualization(state_dict: Dict, title: str = "Agent State Visualization",
                             save_path: Optional[str] = None) -> None:
    """
    Visualize the agent's state representation.

    Args:
        state_dict: State dictionary from environment
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract state components
    ref_vertex = state_dict.get('ref_vertex', None)
    left_neighbors = state_dict.get('left_neighbors', None)
    right_neighbors = state_dict.get('right_neighbors', None)
    fan_points = state_dict.get('fan_points', None)
    area_ratio = state_dict.get('area_ratio', None)

    # Convert tensors to numpy if needed
    def to_numpy(tensor):
        if hasattr(tensor, 'numpy'):
            return tensor.detach().cpu().numpy()
        elif hasattr(tensor, 'cpu'):
            return tensor.cpu().numpy()
        else:
            return np.array(tensor)

    # Plot 1: Reference vertex and neighbors
    ax = axes[0, 0]
    if ref_vertex is not None:
        ref_np = to_numpy(ref_vertex)
        ax.scatter(ref_np[0], ref_np[1], c='red', s=100, label='Reference Vertex', zorder=3)

    if left_neighbors is not None:
        left_np = to_numpy(left_neighbors)
        if left_np.ndim == 2:
            ax.scatter(left_np[:, 0], left_np[:, 1], c='blue', s=50, label='Left Neighbors', zorder=2)

    if right_neighbors is not None:
        right_np = to_numpy(right_neighbors)
        if right_np.ndim == 2:
            ax.scatter(right_np[:, 0], right_np[:, 1], c='green', s=50, label='Right Neighbors', zorder=2)

    ax.set_title('Reference Vertex and Neighbors')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: Fan points
    ax = axes[0, 1]
    if fan_points is not None:
        fan_np = to_numpy(fan_points)
        if fan_np.ndim == 2:
            ax.scatter(fan_np[:, 0], fan_np[:, 1], c='orange', s=50, label='Fan Points')

            # Connect fan points to show the fan structure
            if ref_vertex is not None:
                ref_np = to_numpy(ref_vertex)
                for point in fan_np:
                    ax.plot([ref_np[0], point[0]], [ref_np[1], point[1]], 'orange', alpha=0.5, linewidth=1)
                ax.scatter(ref_np[0], ref_np[1], c='red', s=100, label='Reference Vertex', zorder=3)

    ax.set_title('Fan Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 3: Area ratio
    ax = axes[1, 0]
    if area_ratio is not None:
        ratio_val = to_numpy(area_ratio)
        if np.isscalar(ratio_val):
            ratio_val = [ratio_val]

        ax.bar(['Area Ratio'], ratio_val, color='purple', alpha=0.7)
        ax.set_title('Current Area Ratio')
        ax.set_ylabel('Ratio')
        ax.grid(True, alpha=0.3)

        # Add text annotation
        if len(ratio_val) > 0:
            ax.text(0, ratio_val[0] + 0.01, f'{ratio_val[0]:.3f}',
                    ha='center', va='bottom', fontweight='bold')

    # Plot 4: Combined view
    ax = axes[1, 1]
    if ref_vertex is not None:
        ref_np = to_numpy(ref_vertex)
        ax.scatter(ref_np[0], ref_np[1], c='red', s=100, label='Reference', zorder=4)

    if left_neighbors is not None:
        left_np = to_numpy(left_neighbors)
        if left_np.ndim == 2:
            ax.scatter(left_np[:, 0], left_np[:, 1], c='blue', s=30, alpha=0.7, zorder=2)

    if right_neighbors is not None:
        right_np = to_numpy(right_neighbors)
        if right_np.ndim == 2:
            ax.scatter(right_np[:, 0], right_np[:, 1], c='green', s=30, alpha=0.7, zorder=2)

    if fan_points is not None:
        fan_np = to_numpy(fan_points)
        if fan_np.ndim == 2:
            ax.scatter(fan_np[:, 0], fan_np[:, 1], c='orange', s=20, alpha=0.5, zorder=1)

    ax.set_title('Combined State View')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save figure to {save_path}: {e}")
            plt.show()
    else:
        plt.show()
