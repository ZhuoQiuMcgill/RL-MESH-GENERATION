# Direct imports that won't cause circular dependencies
from .angle import euclidean_distance, get_interior_angle, is_angle_in_slice, normalize_coordinates_polar, decode_coordinate_polar, \
    normalize_coordinates_cartesian, decode_coordinate_cartesian, calculate_polygon_area, valid_element_angle, get_avg_interior_angle
from .segment import ray_segment_intersection, orientation, point_on_line_segment, line_segments_intersect, \
    segments_overlap_interior, point_to_segment_distance


# Lazy import modules that would cause circular dependencies
def _get_importer_module():
    """Lazy import importer module to avoid circular dependencies"""
    from .importer import MeshImporter, create_default_importer
    return MeshImporter, create_default_importer


def _get_checkpoint_manager_module():
    """Lazy import checkpoint_manager module"""
    from .checkpoint_manager import CheckpointManager, get_checkpoint_manager
    return CheckpointManager, get_checkpoint_manager


def _get_plotter_module():
    """Lazy import rl_ploter module"""
    from .rl_ploter import plot_reward_change, plot_training_metrics, plot_action_distribution, \
        plot_action_reward_distribution
    return plot_reward_change, plot_training_metrics, plot_action_distribution, plot_action_reward_distribution


# Implement lazy import through attribute access
def __getattr__(name):
    if name == 'MeshImporter':
        MeshImporter, _ = _get_importer_module()
        return MeshImporter
    elif name == 'create_default_importer':
        _, create_default_importer = _get_importer_module()
        return create_default_importer
    elif name == 'CheckpointManager':
        CheckpointManager, _ = _get_checkpoint_manager_module()
        return CheckpointManager
    elif name == 'get_checkpoint_manager':
        _, get_checkpoint_manager = _get_checkpoint_manager_module()
        return get_checkpoint_manager
    elif name == 'plot_reward_change':
        plot_reward_change, _, _, _ = _get_plotter_module()
        return plot_reward_change
    elif name == 'plot_training_metrics':
        _, plot_training_metrics, _, _ = _get_plotter_module()
        return plot_training_metrics
    elif name == 'plot_action_distribution':
        _, _, plot_action_distribution, _ = _get_plotter_module()
        return plot_action_distribution
    elif name == 'plot_action_reward_distribution':
        _, _, _, plot_action_reward_distribution = _get_plotter_module()
        return plot_action_reward_distribution
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Define module's public API (reference for explicit imports)
__all__ = [
    # Lazy imported items
    'MeshImporter',
    'create_default_importer',
    'CheckpointManager',
    'get_checkpoint_manager',
    'plot_reward_change',
    'plot_training_metrics',
    'plot_action_distribution',
    'plot_action_reward_distribution',

    # Directly imported items
    'euclidean_distance',
    'get_interior_angle',
    'is_angle_in_slice',
    'normalize_coordinates_polar',
    'normalize_coordinates_cartesian',
    'calculate_polygon_area',
    'ray_segment_intersection',
    'orientation',
    'point_on_line_segment',
    'line_segments_intersect',
    'segments_overlap_interior',
    'point_to_segment_distance',
    'decode_coordinate_polar',
    'decode_coordinate_cartesian',
    'valid_element_angle',
    'get_avg_interior_angle'
]

# Version information
__version__ = '1.2.0'

# Module author information
__author__ = 'ZhuoQiuMcgill'
