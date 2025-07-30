# 直接导入不会造成循环依赖的模块
from .angle import euclidean_distance, get_interior_angle, is_angle_in_slice, normalize_coordinates, decode_coordinate, \
    calculate_polygon_area, valid_element_angle, get_avg_interior_angle
from .segment import ray_segment_intersection, orientation, point_on_line_segment, line_segments_intersect, \
    segments_overlap_interior, point_to_segment_distance


# 延迟导入会造成循环依赖的模块
def _get_importer_module():
    """延迟导入importer模块以避免循环依赖"""
    from .importer import MeshImporter, create_default_importer
    return MeshImporter, create_default_importer


def _get_checkpoint_manager_module():
    """延迟导入checkpoint_manager模块"""
    from .checkpoint_manager import CheckpointManager, get_checkpoint_manager
    return CheckpointManager, get_checkpoint_manager


def _get_plotter_module():
    """延迟导入rl_ploter模块"""
    from .rl_ploter import plot_reward_change, plot_training_metrics, plot_action_distribution, \
        plot_action_reward_distribution
    return plot_reward_change, plot_training_metrics, plot_action_distribution, plot_action_reward_distribution


# 通过属性访问实现延迟导入
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


# 定义模块的公共API（用于显式导入时的参考）
__all__ = [
    # 延迟导入的项目
    'MeshImporter',
    'create_default_importer',
    'CheckpointManager',
    'get_checkpoint_manager',
    'plot_reward_change',
    'plot_training_metrics',
    'plot_action_distribution',
    'plot_action_reward_distribution',

    # 直接导入的项目
    'euclidean_distance',
    'get_interior_angle',
    'is_angle_in_slice',
    'normalize_coordinates',
    'calculate_polygon_area',
    'ray_segment_intersection',
    'orientation',
    'point_on_line_segment',
    'line_segments_intersect',
    'segments_overlap_interior',
    'point_to_segment_distance',
    'decode_coordinate',
    'valid_element_angle',
    'get_avg_interior_angle'
]

# 版本信息
__version__ = '1.2.0'

# 模块作者信息
__author__ = 'ZhuoQiuMcgill'
