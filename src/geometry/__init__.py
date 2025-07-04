"""
几何模块

本模块提供用于网格生成的几何相关类和功能。
主要包含边界(Boundary)和网格(Mesh)的定义与操作。

主要类:
    Boundary: 表示多边形边界，支持顶点操作、内部判断等功能
    Mesh: 表示网格结构，支持顶点和边的添加与管理

用法示例:
    from src.geometry import Boundary, Mesh

    # 创建边界
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    boundary = Boundary(vertices)

    # 创建网格
    mesh = Mesh(boundary)
"""

from .boundary import Boundary
from .mesh import Mesh

# 定义模块的公共API
__all__ = [
    'Boundary',
    'Mesh'
]

# 版本信息
__version__ = '1.0.0'

# 模块作者信息
__author__ = 'ZhuoQiuMcgill'
