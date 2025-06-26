import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Optional
import os
import seaborn as sns
from datetime import datetime


def plot_mesh(boundary: np.ndarray, elements: List[np.ndarray] = None,
              title: str = "Mesh Visualization", save_path: Optional[str] = None,
              figsize: tuple = (10, 8), show_vertices: bool = True,
              show_element_numbers: bool = False) -> None:
    """
    Plot mesh boundary and generated quadrilateral elements.

    Args:
        boundary: Boundary vertices [N, 2]
        elements: List of element vertices, each element is [4, 2] array
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        show_vertices: Whether to show vertex numbers
        show_element_numbers: Whether to show element numbers
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot boundary
    if boundary is not None and len(boundary) > 0:
        # Close the boundary for plotting
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
                'k-', linewidth=2, label='Boundary', zorder=3)

        # Plot boundary vertices
        ax.scatter(boundary[:, 0], boundary[:, 1],
                   c='red', s=50, zorder=4, label='Boundary Vertices')

        # Show vertex numbers if requested
        if show_vertices:
            for i, (x, y) in enumerate(boundary):
                ax.annotate(f'{i}', (x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Plot generated elements
    if elements is not None and len(elements) > 0:
        for elem_idx, element in enumerate(elements):
            if element is not None and len(element) == 4:
                # Create quadrilateral patch
                quad = patches.Polygon(element, linewidth=1.5,
                                       edgecolor='blue', facecolor='lightblue',
                                       alpha=0.3, zorder=1)
                ax.add_patch(quad)

                # Plot element vertices
                ax.scatter(element[:, 0], element[:, 1],
                           c='blue', s=30, zorder=2, alpha=0.7)

                # Show element number if requested
                if show_element_numbers:
                    center = np.mean(element, axis=0)
                    ax.annotate(f'E{elem_idx}', center, ha='center', va='center',
                                fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))

    # Set equal aspect ratio and adjust limits
    ax.set_aspect('equal')

    # Set limits with some padding
    if boundary is not None and len(boundary) > 0:
        x_min, x_max = boundary[:, 0].min(), boundary[:, 0].max()
        y_min, y_max = boundary[:, 1].min(), boundary[:, 1].max()

        # Add elements to bounds calculation if they exist
        if elements is not None and len(elements) > 0:
            all_element_points = np.vstack([elem for elem in elements if elem is not None])
            if len(all_element_points) > 0:
                x_min = min(x_min, all_element_points[:, 0].min())
                x_max = max(x_max, all_element_points[:, 0].max())
                y_min = min(y_min, all_element_points[:, 1].min())
                y_max = max(y_max, all_element_points[:, 1].max())

        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = max(x_range, y_range) * 0.1

        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add mesh statistics
    info_text = f"Boundary vertices: {len(boundary) if boundary is not None else 0}\n"
    info_text += f"Generated elements: {len(elements) if elements is not None else 0}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to prevent memory accumulation
    else:
        plt.show()


def plot_mesh_generation_progress(boundary_history: List[np.ndarray],
                                  elements_history: List[List[np.ndarray]],
                                  title: str = "Mesh Generation Progress",
                                  save_path: Optional[str] = None,
                                  max_steps: int = 6) -> None:
    """
    Plot mesh generation progress showing multiple steps.

    Args:
        boundary_history: List of boundary states at each step
        elements_history: List of element lists at each step
        title: Plot title
        save_path: Path to save figure
        max_steps: Maximum number of steps to show
    """
    n_steps = min(len(boundary_history), len(elements_history), max_steps)

    if n_steps <= 1:
        # If only one step, use regular plot_mesh
        boundary = boundary_history[0] if boundary_history else None
        elements = elements_history[0] if elements_history else None
        plot_mesh(boundary, elements, title, save_path)
        return

    # Create subplot grid
    cols = min(3, n_steps)
    rows = (n_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for step in range(n_steps):
        ax = axes[step] if step < len(axes) else None
        if ax is None:
            continue

        boundary = boundary_history[step] if step < len(boundary_history) else None
        elements = elements_history[step] if step < len(elements_history) else []

        # Plot boundary
        if boundary is not None and len(boundary) > 0:
            boundary_closed = np.vstack([boundary, boundary[0]])
            ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
                    'k-', linewidth=2)
            ax.scatter(boundary[:, 0], boundary[:, 1], c='red', s=50, zorder=4)

        # Plot elements
        if elements is not None and len(elements) > 0:
            for element in elements:
                if element is not None and len(element) == 4:
                    quad = patches.Polygon(element, linewidth=1.5,
                                           edgecolor='blue', facecolor='lightblue',
                                           alpha=0.3)
                    ax.add_patch(quad)
                    ax.scatter(element[:, 0], element[:, 1], c='blue', s=30, alpha=0.7)

        ax.set_aspect('equal')
        ax.set_title(f'Step {step + 1}')
        ax.grid(True, alpha=0.3)

        # Set consistent limits across all subplots
        if boundary is not None and len(boundary) > 0:
            x_min, x_max = boundary[:, 0].min(), boundary[:, 0].max()
            y_min, y_max = boundary[:, 1].min(), boundary[:, 1].max()

            if elements is not None and len(elements) > 0:
                all_element_points = np.vstack([elem for elem in elements if elem is not None])
                if len(all_element_points) > 0:
                    x_min = min(x_min, all_element_points[:, 0].min())
                    x_max = max(x_max, all_element_points[:, 0].max())
                    y_min = min(y_min, all_element_points[:, 1].min())
                    y_max = max(y_max, all_element_points[:, 1].max())

            x_range = x_max - x_min
            y_range = y_max - y_min
            padding = max(x_range, y_range) * 0.1

            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)

    # Hide unused subplots
    for i in range(n_steps, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_mesh_animation_frames(boundary_history: List[np.ndarray],
                               elements_history: List[List[np.ndarray]],
                               output_dir: str) -> None:
    """
    Save individual frames for mesh generation animation.

    Args:
        boundary_history: List of boundary states
        elements_history: List of element lists
        output_dir: Directory to save frames
    """
    os.makedirs(output_dir, exist_ok=True)

    for step, (boundary, elements) in enumerate(zip(boundary_history, elements_history)):
        frame_path = os.path.join(output_dir, f"frame_{step:03d}.png")
        plot_mesh(boundary, elements,
                  title=f"Mesh Generation - Step {step + 1}",
                  save_path=frame_path,
                  show_vertices=False)

    print(f"💾 Animation frames saved to: {output_dir}")


def plot_quality_metrics(quality_data: Dict, title: str = "Mesh Quality Metrics",
                         save_path: Optional[str] = None) -> None:
    """
    Plot mesh quality metrics distribution.

    Args:
        quality_data: Dictionary with metric names as keys and lists of values
        title: Plot title
        save_path: Path to save figure
    """
    if not quality_data:
        print("No quality data to plot")
        return

    n_metrics = len(quality_data)
    if n_metrics == 0:
        return

    # Determine subplot layout
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if n_metrics == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, (metric_name, values) in enumerate(quality_data.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        if values and len(values) > 0:
            ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric_name.replace("_", " ").title()}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{metric_name.replace("_", " ").title()}')

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_summary_report(results: Dict, save_path: str) -> None:
    """
    Create a comprehensive summary report with multiple visualizations.

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
            ax.scatter(left_np[:, 0], left_np[:, 1], c='blue', s=50, label='Left', zorder=3)

    if right_neighbors is not None:
        right_np = to_numpy(right_neighbors)
        if right_np.ndim == 2:
            ax.scatter(right_np[:, 0], right_np[:, 1], c='green', s=50, label='Right', zorder=3)

    if fan_points is not None:
        fan_np = to_numpy(fan_points)
        if fan_np.ndim == 2:
            ax.scatter(fan_np[:, 0], fan_np[:, 1], c='orange', s=50, label='Fan', zorder=2)

    ax.set_title('Combined State View')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_enhanced_learning_curve(episode_rewards: List[float],
                                 episode_lengths: List[int],
                                 episode_times: List[float] = None,
                                 episode_qualities: List[float] = None,
                                 completion_rates: List[float] = None,
                                 title: str = "Enhanced Learning Curve",
                                 save_path: Optional[str] = None,
                                 window_size: int = 100) -> None:
    """
    Plot comprehensive learning curves with multiple metrics.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        episode_times: List of episode times
        episode_qualities: List of mesh qualities
        completion_rates: List of completion rates
        title: Plot title
        save_path: Path to save figure
        window_size: Moving average window size
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'library') else 'default')

    # Determine number of subplots
    n_metrics = 2  # rewards and lengths always present
    if episode_times:
        n_metrics += 1
    if episode_qualities:
        n_metrics += 1
    if completion_rates:
        n_metrics += 1

    # Create subplots
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    episodes = np.arange(len(episode_rewards))
    current_axis = 0

    # Plot rewards
    ax = axes[current_axis]
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', linewidth=0.8, label='Episode Rewards')

    # Moving average for rewards
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
        avg_episodes = episodes[window_size - 1:]
        ax.plot(avg_episodes, moving_avg, color='red', linewidth=2,
                label=f'Moving Average ({window_size} episodes)')

    # Add trend line
    if len(episodes) > 10:
        z = np.polyfit(episodes, episode_rewards, 1)
        p = np.poly1d(z)
        ax.plot(episodes, p(episodes), "--", color='green', alpha=0.8, linewidth=1.5, label='Trend')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add statistics text
    stats_text = f'Final: {episode_rewards[-1]:.2f}\nMax: {max(episode_rewards):.2f}\nMean: {np.mean(episode_rewards):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    current_axis += 1

    # Plot episode lengths
    ax = axes[current_axis]
    ax.plot(episodes, episode_lengths, alpha=0.3, color='green', linewidth=0.8, label='Episode Lengths')

    if len(episode_lengths) >= window_size:
        moving_avg_len = np.convolve(episode_lengths, np.ones(window_size) / window_size, mode='valid')
        ax.plot(avg_episodes, moving_avg_len, color='orange', linewidth=2,
                label=f'Moving Average ({window_size} episodes)')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Lengths')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add statistics
    stats_text = f'Final: {episode_lengths[-1]}\nMean: {np.mean(episode_lengths):.1f}\nStd: {np.std(episode_lengths):.1f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    current_axis += 1

    # Plot episode times if available
    if episode_times:
        ax = axes[current_axis]
        ax.plot(episodes, episode_times, alpha=0.3, color='purple', linewidth=0.8, label='Episode Times')

        if len(episode_times) >= window_size:
            moving_avg_time = np.convolve(episode_times, np.ones(window_size) / window_size, mode='valid')
            ax.plot(avg_episodes, moving_avg_time, color='darkviolet', linewidth=2,
                    label=f'Moving Average ({window_size} episodes)')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Episode Execution Times')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics
        stats_text = f'Mean: {np.mean(episode_times):.2f}s\nStd: {np.std(episode_times):.2f}s'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))

        current_axis += 1

    # Plot mesh qualities if available
    if episode_qualities:
        ax = axes[current_axis]
        ax.plot(episodes, episode_qualities, alpha=0.3, color='brown', linewidth=0.8, label='Mesh Quality')

        if len(episode_qualities) >= window_size:
            moving_avg_quality = np.convolve(episode_qualities, np.ones(window_size) / window_size, mode='valid')
            ax.plot(avg_episodes, moving_avg_quality, color='chocolate', linewidth=2,
                    label=f'Moving Average ({window_size} episodes)')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Quality Score')
        ax.set_title('Mesh Quality Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics
        stats_text = f'Final: {episode_qualities[-1]:.3f}\nMax: {max(episode_qualities):.3f}\nMean: {np.mean(episode_qualities):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='bisque', alpha=0.8))

        current_axis += 1

    # Plot completion rates if available
    if completion_rates:
        ax = axes[current_axis]
        # Convert to percentage and plot as bars
        completion_percentages = [r * 100 for r in completion_rates]

        # Use moving average for completion rate
        if len(completion_rates) >= window_size:
            moving_completion = np.convolve(completion_percentages, np.ones(window_size) / window_size, mode='valid')
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Enhanced learning curve saved to: {save_path}")

    plt.show()


def plot_training_dashboard(results: Dict, save_path: Optional[str] = None):
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

    # 2. Episode length distribution
    ax2 = fig.add_subplot(gs[1, 0])
    episode_lengths = results['episode_lengths']
    ax2.hist(episode_lengths, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Episode Length Distribution')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    # 3. Reward distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(episode_rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_title('Reward Distribution')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)

    # 4. Performance over time (if episode times available)
    ax4 = fig.add_subplot(gs[1, 2])
    if 'episode_times' in results and results['episode_times']:
        episode_times = results['episode_times']
        ax4.plot(episodes, episode_times, alpha=0.6, color='purple')
        ax4.set_title('Episode Execution Time')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Time (s)')
    else:
        # Plot episode length evolution instead
        ax4.plot(episodes, episode_lengths, alpha=0.6, color='green')
        ax4.set_title('Episode Length Evolution')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
    ax4.grid(True, alpha=0.3)

    # 5. Learning progress comparison (early vs late)
    ax5 = fig.add_subplot(gs[2, 0])
    if len(episode_rewards) >= 200:
        early_rewards = episode_rewards[:100]
        late_rewards = episode_rewards[-100:]

        ax5.boxplot([early_rewards, late_rewards], labels=['Early (first 100)', 'Late (last 100)'])
        ax5.set_title('Learning Progress Comparison')
        ax5.set_ylabel('Reward')
        ax5.grid(True, alpha=0.3)
    else:
        # Show reward trend instead
        if len(episodes) > 10:
            z = np.polyfit(episodes, episode_rewards, 1)
            p = np.poly1d(z)
            ax5.plot(episodes, episode_rewards, 'o', alpha=0.3, markersize=2)
            ax5.plot(episodes, p(episodes), "r--", linewidth=2, label=f'Trend (slope: {z[0]:.3f})')
            ax5.set_title('Reward Trend Analysis')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Reward')
            ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Mesh quality evolution (if available)
    ax6 = fig.add_subplot(gs[2, 1])
    if 'episode_mesh_qualities' in results and results['episode_mesh_qualities']:
        mesh_qualities = results['episode_mesh_qualities']
        ax6.plot(episodes, mesh_qualities, alpha=0.6, color='brown')
        ax6.set_title('Mesh Quality Evolution')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Quality Score')
    else:
        # Show cumulative reward instead
        cumulative_rewards = np.cumsum(episode_rewards)
        ax6.plot(episodes, cumulative_rewards, color='orange')
        ax6.set_title('Cumulative Reward')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Cumulative Reward')
    ax6.grid(True, alpha=0.3)

    # 7. Completion rate (if available)
    ax7 = fig.add_subplot(gs[2, 2])
    if 'episode_completion_rates' in results and results['episode_completion_rates']:
        completion_rates = results['episode_completion_rates']
        # Moving average of completion rate
        window = min(50, len(completion_rates) // 5)
        if len(completion_rates) >= window:
            moving_completion = np.convolve(completion_rates, np.ones(window) / window, mode='valid')
            ax7.plot(episodes[window - 1:], [r * 100 for r in moving_completion], color='darkgreen', linewidth=2)
        else:
            ax7.plot(episodes, [r * 100 for r in completion_rates], color='darkgreen')
        ax7.set_title('Completion Rate Evolution')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Completion Rate (%)')
        ax7.set_ylim(0, 100)
    else:
        # Show reward variance instead
        window = min(50, len(episode_rewards) // 5)
        if len(episode_rewards) >= window:
            rolling_var = []
            for i in range(window - 1, len(episode_rewards)):
                rolling_var.append(np.var(episode_rewards[i - window + 1:i + 1]))
            ax7.plot(episodes[window - 1:], rolling_var, color='red')
            ax7.set_title(f'Reward Variance (window={window})')
            ax7.set_xlabel('Episode')
            ax7.set_ylabel('Variance')
    ax7.grid(True, alpha=0.3)

    # 8. Training statistics summary
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')

    # Calculate summary statistics
    total_episodes = len(episode_rewards)
    final_reward = episode_rewards[-1] if episode_rewards else 0
    best_reward = max(episode_rewards) if episode_rewards else 0
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0

    # Create summary text
    summary_text = f"""
    TRAINING SUMMARY

    Total Episodes: {total_episodes:,}
    Final Reward: {final_reward:.2f}
    Best Reward: {best_reward:.2f}
    Mean Reward: {mean_reward:.2f}
    Mean Episode Length: {np.mean(episode_lengths):.1f} steps
    """

    if 'episode_times' in results and results['episode_times']:
        summary_text += f"    Mean Episode Time: {np.mean(results['episode_times']):.2f}s\n"

    if 'episode_mesh_qualities' in results and results['episode_mesh_qualities']:
        summary_text += f"    Mean Mesh Quality: {np.mean(results['episode_mesh_qualities']):.3f}\n"

    if 'episode_completion_rates' in results and results['episode_completion_rates']:
        summary_text += f"    Completion Rate: {np.mean(results['episode_completion_rates']):.1%}\n"

    # Performance improvement
    if len(episode_rewards) >= 100:
        early_perf = np.mean(episode_rewards[:50])
        late_perf = np.mean(episode_rewards[-50:])
        improvement = late_perf - early_perf
        summary_text += f"    Performance Improvement: {improvement:+.2f} ({improvement / abs(early_perf) * 100:+.1f}%)\n"

    ax8.text(0.1, 0.8, summary_text, transform=ax8.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax8.text(0.7, 0.2, f"Generated: {timestamp}", transform=ax8.transAxes,
             fontsize=10, style='italic')

    plt.suptitle('RL Mesh Generation Training Dashboard', fontsize=18, fontweight='bold', y=0.98)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training dashboard saved to: {save_path}")

    plt.show()


def plot_evaluation_progress(evaluation_results: List[Dict], save_path: Optional[str] = None):
    """
    Plot evaluation progress over training.

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
    mean_rewards = [r['mean_reward'] for r in evaluation_results]
    std_rewards = [r['std_reward'] for r in evaluation_results]
    mean_lengths = [r['mean_length'] for r in evaluation_results]
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
        rewards = [result['mean_reward']] * 5  # Simplified representation
        ax.hist(rewards, bins=10, alpha=0.3, label=f'Eval {i * max(1, len(evaluation_results) // 5)}')
    ax.set_title('Reward Distribution Evolution')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Evaluation Progress During Training', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Evaluation progress saved to: {save_path}")

    plt.show()


# Update the existing functions to use the enhanced versions
def plot_learning_curve(episode_rewards: List[float],
                        episode_lengths: List[int],
                        title: str = "Learning Curve",
                        save_path: Optional[str] = None,
                        window_size: int = 100) -> None:
    """
    Enhanced learning curve - delegates to the comprehensive version.
    """
    plot_enhanced_learning_curve(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        title=title,
        save_path=save_path,
        window_size=window_size
    )


def plot_training_progress(training_stats: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive training progress including multiple metrics.
    """
    if not training_stats:
        print("No training statistics available for plotting")
        return

    # Extract available metrics
    available_metrics = []
    for key in ['actor_loss', 'critic_loss', 'alpha']:
        if key in training_stats:
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

        axes[i].plot(steps, values, linewidth=1.5)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Evolution')
        axes[i].set_xlabel('Training Step')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True, alpha=0.3)

        # Add moving average for noisy metrics
        if len(values) > 50 and metric in ['actor_loss', 'critic_loss']:
            window = min(50, len(values) // 10)
            moving_avg = np.convolve(values, np.ones(window) / window, mode='valid')
            axes[i].plot(steps[window - 1:], moving_avg, color='red', linewidth=2,
                         label=f'Moving Avg ({window})', alpha=0.8)
            axes[i].legend()

    plt.suptitle('Training Progress - Algorithm Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training progress saved to: {save_path}")

    plt.show()
