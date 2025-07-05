"""
工具模块

本模块提供用于网格生成的工具类和函数。
主要包含数据导入、文件处理、配置管理等功能。

主要类:
    MeshImporter: 网格数据导入器，用于从txt文件读取边界数据并创建网格

主要函数:
    create_default_importer: 创建默认的网格导入器实例

用法示例:
    from src.utils import MeshImporter, create_default_importer

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
"""

from .importer import MeshImporter, create_default_importer

# 定义模块的公共API
__all__ = [
    'MeshImporter',
    'create_default_importer'
]

# 版本信息
__version__ = '1.0.0'

# 模块作者信息
__author__ = 'ZhuoQiuMcgill'