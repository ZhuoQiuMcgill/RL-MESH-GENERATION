"""
工具模块

本模块提供用于网格生成的工具类和函数。
主要包含数据导入、文件处理、配置管理、可视化等功能。

主要类:
    MeshImporter: 网格数据导入器，用于从txt文件读取边界数据并创建网格

主要函数:
    create_default_importer: 创建默认的网格导入器实例
    plot_reward_change: 绘制奖励随训练步数的变化图
    plot_training_metrics: 绘制多个训练指标的图表

用法示例:
    from src.utils import MeshImporter, create_default_importer, plot_reward_change

    # 创建导入器
    importer = create_default_importer()

    # 或者手动指定数据目录
    importer = MeshImporter(data_root="/path/to/data")

    # 从文件创建网格
    mesh = importer.create_mesh_by_name("1")  # 读取 data/mesh/1.txt

    # 获取边界对象
    boundary = importer.load_boundary_by_name("simple_square")

    # 列出可用的网格文件
    available_meshes = importer.list_available_meshes()

    # 验证数据目录结构
    is_valid = importer.validate_data_structure()

    # 绘制训练奖励图
    plot_reward_change(
        timesteps=[100, 200, 300, 400],
        rewards=[0.1, 0.3, 0.5, 0.8],
        save_path="reward_plot.png"
    )
"""

# 直接导入不会造成循环依赖的模块
from .angle import euclidean_distance, get_interior_angle, is_angle_in_slice
from .segment import ray_segment_intersection, orientation, point_on_line_segment, line_segments_intersect, \
    segments_overlap_interior, point_to_segment_distance


# 延迟导入会造成循环依赖的模块
def _get_importer_module():
    """延迟导入importer模块以避免循环依赖"""
    from .importer import MeshImporter, create_default_importer
    return MeshImporter, create_default_importer


def _get_plotter_module():
    """延迟导入rl_ploter模块"""
    from .rl_ploter import plot_reward_change
    return plot_reward_change


# 通过属性访问实现延迟导入
def __getattr__(name):
    if name == 'MeshImporter':
        MeshImporter, _ = _get_importer_module()
        return MeshImporter
    elif name == 'create_default_importer':
        _, create_default_importer = _get_importer_module()
        return create_default_importer
    elif name == 'plot_reward_change':
        plot_reward_change, _, _ = _get_plotter_module()
        return plot_reward_change
    elif name == 'plot_training_metrics':
        _, plot_training_metrics, _ = _get_plotter_module()
        return plot_training_metrics
    elif name == 'create_training_summary_plot':
        _, _, create_training_summary_plot = _get_plotter_module()
        return create_training_summary_plot
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 定义模块的公共API（用于显式导入时的参考）
__all__ = [
    # 延迟导入的项目
    'MeshImporter',
    'create_default_importer',
    'plot_reward_change',
    'plot_training_metrics',
    'create_training_summary_plot',

    # 直接导入的项目
    'euclidean_distance',
    'get_interior_angle',
    'is_angle_in_slice',
    'ray_segment_intersection',
    'orientation',
    'point_on_line_segment',
    'line_segments_intersect',
    'segments_overlap_interior',
    'point_to_segment_distance'
]

# 版本信息
__version__ = '1.1.0'

# 模块作者信息
__author__ = 'ZhuoQiuMcgill'
