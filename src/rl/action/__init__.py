"""
动作模块

本模块提供网格生成中的四种基本动作类型。
每种动作类型都实现了从当前边界生成四边形元素的不同策略。

主要类:
    ActionType: 动作类型的抽象基类
    ActionType0Left: 不增加新顶点，连接左侧四个边界点形成四边形
    ActionType0Right: 不增加新顶点，连接右侧四个边界点形成四边形
    ActionType1: 增加一个新顶点，形成四边形
    ActionType2: 增加两个新顶点，形成四边形

用法示例:
    from src.rl.action import ActionType0Left, ActionType0Right, ActionType1, ActionType2

    # 创建动作实例
    action_type_0_left = ActionType0Left()
    action_type_0_right = ActionType0Right()
    action_type_1 = ActionType1()
    action_type_2 = ActionType2()

    # 检查动作有效性
    if action_type_0_left.is_valid(boundary, reference_vertex_idx):
        # 执行动作（直接修改输入的mesh和boundary）
        mesh, boundary, element = action_type_0_left.execute(mesh, boundary, reference_vertex_idx)

动作类型映射:
    - 0: ActionType0Left - 连接左侧相邻顶点
    - 1: ActionType0Right - 连接右侧相邻顶点
    - 2: ActionType1 - 添加一个新顶点
    - 3: ActionType2 - 添加两个新顶点
"""

from .action import ActionType
from .type0_left import ActionType0Left
from .type0_right import ActionType0Right
from .type1 import ActionType1

# 定义模块的公共API
__all__ = [
    'ActionType',
    'ActionType0Left',
    'ActionType0Right',
    'ActionType1',
]

# 版本信息
__version__ = '1.4.0'

# 模块作者信息
__author__ = 'ZhuoQiuMcgill'

# 动作类型映射字典，便于程序化访问
ACTION_TYPE_MAPPING = {
    0: ActionType0Left,
    1: ActionType0Right,
    2: ActionType1,
}

# 动作类型名称映射
ACTION_TYPE_NAMES = {
    0: "ActionType0Left",
    1: "ActionType0Right",
    2: "ActionType1",
}
