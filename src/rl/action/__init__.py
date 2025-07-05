"""
动作模块

本模块提供网格生成中的三种基本动作类型。
每种动作类型都实现了从当前边界生成四边形元素的不同策略。

主要类:
    ActionType: 动作类型的抽象基类
    ActionType0: 不增加新顶点，直接连接四个边界点形成四边形
    ActionType1: 增加一个新顶点，形成四边形
    ActionType2: 增加两个新顶点，形成四边形

用法示例:
    from src.action import ActionType0, ActionType1, ActionType2

    # 创建动作实例
    action_type_0 = ActionType0()
    action_type_1 = ActionType1()
    action_type_2 = ActionType2()

    # 检查动作有效性
    if action_type_0.is_valid(boundary, reference_vertex_idx):
        # 执行动作（直接修改输入的mesh和boundary）
        mesh, boundary, element = action_type_0.execute(mesh, boundary, reference_vertex_idx)
"""

from .action import ActionType
from .type0 import ActionType0
from .type1 import ActionType1
from .type2 import ActionType2

# 定义模块的公共API
__all__ = [
    'ActionType',
    'ActionType0',
    'ActionType1',
    'ActionType2'
]

# 版本信息
__version__ = '1.0.0'

# 模块作者信息
__author__ = 'ZhuoQiuMcgill'