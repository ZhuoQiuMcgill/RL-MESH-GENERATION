import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Optional
import os


def plot_mesh(boundary: np.ndarray, elements: List[np.ndarray],
              title: str = "Mesh Generation Result",
              save_path: Optional[str] = None,
              show_vertices: bool = True,
              show_element_numbers: bool = False) -> None:
    """
    Plot generated mesh with boundary and elements.

    Args:
        boundary: Boundary vertices
        elements: List of element vertices
        title: Plot title
        save_path: Path to save figure
        show_vertices: Whether to show vertex points
        show_element_numbers: Whether to show element numbers
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot boundary
    boundary_closed = np.vstack([boundary, boundary[0]])
    ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
            'k-', linewidth=2, label='Boundary')

    # Plot elements
    for i, element in enumerate(elements):
        # Close the element for plotting
        element_closed = np.vstack([element, element[0]])
        ax.plot(element_closed[:, 0], element_closed[:, 1],
                'b-', linewidth=1, alpha=0.7)

        # Fill element
        polygon = patches.Polygon(element, closed=True,
                                  facecolor='lightblue',
                                  edgecolor='blue',
                                  alpha=0.3, linewidth=1)
        ax.add_patch(polygon)

        # Add element numbers
        if show_element_numbers:
            centroid = np.mean(element, axis=0)
            ax.text(centroid[0], centroid[1], str(i),
                    ha='center', va='center', fontsize=8)

    # Plot vertices
    if show_vertices:
        ax.scatter(boundary[:, 0], boundary[:, 1],
                   c='red', s=30, zorder=5, label='Boundary Vertices')

        for elements_verts in elements:
            ax.scatter(elements_verts[:, 0], elements_verts[:, 1],
                       c='blue', s=15, zorder=4, alpha=0.7)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_learning_curve(episode_rewards: List[float],
                        episode_lengths: List[int],
                        title: str = "Learning Curve",
                        save_path: Optional[str] = None,
                        window_size: int = 100) -> None:
    """
    Plot learning curves for rewards and episode lengths.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        title: Plot title
        save_path: Path to save figure
        window_size: Moving average window size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    episodes = np.arange(len(episode_rewards))

    # Plot rewards
    ax1.plot(episodes, episode_rewards, alpha=0.3, color='blue')

    # Moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards,
                                 np.ones(window_size) / window_size,
                                 mode='valid')
        avg_episodes = episodes[window_size - 1:]
        ax1.plot(avg_episodes, moving_avg, color='red', linewidth=2,
                 label=f'Moving Average ({window_size} episodes)')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot episode lengths
    ax2.plot(episodes, episode_lengths, alpha=0.3, color='green')

    # Moving average for lengths
    if len(episode_lengths) >= window_size:
        moving_avg_len = np.convolve(episode_lengths,
                                     np.ones(window_size) / window_size,
                                     mode='valid')
        ax2.plot(avg_episodes, moving_avg_len, color='orange', linewidth=2,
                 label=f'Moving Average ({window_size} episodes)')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length (Steps)')
    ax2.set_title('Episode Lengths')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_quality_metrics(metrics_data: Dict[str, List[float]],
                         title: str = "Mesh Quality Metrics",
                         save_path: Optional[str] = None) -> None:
    """
    Plot quality metrics as box plots for comparison.

    Args:
        metrics_data: Dictionary with metric names as keys and values as lists
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    metric_names = list(metrics_data.keys())
    metric_values = [metrics_data[name] for name in metric_names]

    # Create box plot
    box_plot = ax.boxplot(metric_values, labels=metric_names, patch_artist=True)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(title)
    ax.set_ylabel('Quality Score')
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_training_progress(log_data: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive training progress including multiple metrics.

    Args:
        log_data: Dictionary containing training metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Reward progress
    if 'rewards' in log_data:
        axes[0, 0].plot(log_data['rewards'])
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

    # Actor loss
    if 'actor_loss' in log_data:
        axes[0, 1].plot(log_data['actor_loss'])
        axes[0, 1].set_title('Actor Loss')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

    # Critic loss
    if 'critic_loss' in log_data:
        axes[1, 0].plot(log_data['critic_loss'])
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)

    # Alpha (temperature) evolution
    if 'alpha' in log_data:
        axes[1, 1].plot(log_data['alpha'])
        axes[1, 1].set_title('Temperature Parameter (Alpha)')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Alpha')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_state_visualization(state_components: Dict, save_path: Optional[str] = None) -> None:
    """
    Visualize the state representation around reference vertex.

    Args:
        state_components: State components from get_state_components
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ref_vertex = state_components['ref_vertex']
    left_neighbors = state_components['left_neighbors']
    right_neighbors = state_components['right_neighbors']
    fan_points = state_components['fan_points']

    # Plot reference vertex
    ax.scatter(ref_vertex[0], ref_vertex[1], c='red', s=100,
               label='Reference Vertex', zorder=5)

    # Plot neighbors
    for i, (left, right) in enumerate(zip(left_neighbors, right_neighbors)):
        ax.scatter(left[0], left[1], c='blue', s=60,
                   label='Left Neighbors' if i == 0 else "", zorder=4)
        ax.scatter(right[0], right[1], c='green', s=60,
                   label='Right Neighbors' if i == 0 else "", zorder=4)

        # Draw lines to reference vertex
        ax.plot([ref_vertex[0], left[0]], [ref_vertex[1], left[1]],
                'b--', alpha=0.5)
        ax.plot([ref_vertex[0], right[0]], [ref_vertex[1], right[1]],
                'g--', alpha=0.5)

    # Plot fan observation points
    for i, fan_point in enumerate(fan_points):
        ax.scatter(fan_point[0], fan_point[1], c='orange', s=40,
                   label='Fan Points' if i == 0 else "", zorder=4)
        ax.plot([ref_vertex[0], fan_point[0]], [ref_vertex[1], fan_point[1]],
                'orange', alpha=0.3)

    # Draw observation radius
    circle = patches.Circle(ref_vertex, state_components['base_length'] * 6,
                            fill=False, linestyle=':', alpha=0.5, color='gray')
    ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('State Representation Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_action_space(ref_vertex: np.ndarray, reference_direction: np.ndarray,
                      alpha: float, base_length: float,
                      save_path: Optional[str] = None) -> None:
    """
    Visualize the action space (fan-shaped region).

    Args:
        ref_vertex: Reference vertex
        reference_direction: Reference direction vector
        alpha: Action radius factor
        base_length: Base length
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    radius = alpha * base_length

    # Plot reference vertex
    ax.scatter(ref_vertex[0], ref_vertex[1], c='red', s=100,
               label='Reference Vertex', zorder=5)

    # Plot reference direction
    ref_end = ref_vertex + reference_direction
    ax.arrow(ref_vertex[0], ref_vertex[1],
             reference_direction[0], reference_direction[1],
             head_width=0.1, head_length=0.1, fc='blue', ec='blue',
             label='Reference Direction')

    # Draw action space (fan-shaped area)
    # For simplicity, draw as a circle here
    circle = patches.Circle(ref_vertex, radius, fill=False,
                            linestyle='-', color='green', linewidth=2,
                            label=f'Action Space (r={radius:.2f})')
    ax.add_patch(circle)

    # Fill sector if we have specific angle constraints
    # This is a simplified visualization

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Action Space Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Set axis limits around the action space
    margin = radius * 0.2
    ax.set_xlim(ref_vertex[0] - radius - margin, ref_vertex[0] + radius + margin)
    ax.set_ylim(ref_vertex[1] - radius - margin, ref_vertex[1] + radius + margin)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def save_mesh_animation_frames(boundary_history: List[np.ndarray],
                               elements_history: List[List[np.ndarray]],
                               save_dir: str) -> None:
    """
    Save individual frames for mesh generation animation.

    Args:
        boundary_history: List of boundary states over time
        elements_history: List of element lists over time
        save_dir: Directory to save frames
    """
    os.makedirs(save_dir, exist_ok=True)

    for step, (boundary, elements) in enumerate(zip(boundary_history, elements_history)):
        save_path = os.path.join(save_dir, f"frame_{step:04d}.png")

        plot_mesh(boundary, elements,
                  title=f"Mesh Generation Step {step}",
                  save_path=save_path,
                  show_vertices=False)

        plt.close()  # Close figure to save memory


def create_summary_report(results: Dict, save_path: str) -> None:
    """
    Create a comprehensive summary report with multiple visualizations.

    Args:
        results: Dictionary containing all results and metrics
        save_path: Path to save the report
    """
    # Create a multi-page figure
    fig = plt.figure(figsize=(16, 20))

    # Page 1: Learning curves
    ax1 = plt.subplot(4, 2, 1)
    if 'episode_rewards' in results:
        plt.plot(results['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)

    # Page 2: Final mesh
    ax2 = plt.subplot(4, 2, 2)
    if 'final_boundary' in results and 'final_elements' in results:
        boundary = results['final_boundary']
        elements = results['final_elements']

        # Plot boundary
        boundary_closed = np.vstack([boundary, boundary[0]])
        plt.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'k-', linewidth=2)

        # Plot elements
        for element in elements:
            element_closed = np.vstack([element, element[0]])
            plt.plot(element_closed[:, 0], element_closed[:, 1], 'b-', alpha=0.7)

        plt.title('Final Mesh')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

    # Add more subplots for other metrics...

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()