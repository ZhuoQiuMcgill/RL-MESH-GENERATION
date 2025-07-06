from .action import ActionType


class ActionType0(ActionType):
    """
    实现Type 0动作：不增加新顶点，直接连接边界上的四个点形成一个四边形。
    对应论文中的 Figure 5(a)。
    """

    def execute(self, mesh, boundary, reference_vertex_V0_idx):
        """
        执行Type 0动作的逻辑

        直接修改输入的mesh和boundary对象，避免深拷贝的开销。

        Args:
            mesh: 网格对象（会被直接修改）
            boundary: 边界对象（会被直接修改）
            reference_vertex_V0_idx: 参考顶点V0在边界中的索引

        Returns:
            list: 生成的四边形元素（四个顶点的列表）
        """
        # 使用新的封装函数获取顶点
        V0 = boundary.get_vertex_by_index(reference_vertex_V0_idx)
        V1 = boundary.get_vertex_by_index(reference_vertex_V0_idx + 1)
        V2 = boundary.get_vertex_by_index(reference_vertex_V0_idx + 2)
        V3 = boundary.get_vertex_by_index(reference_vertex_V0_idx - 1)

        # 创建四边形元素 (V0, V1, V2, V3)
        quadrilateral = [V0, V1, V2, V3]

        # 更新边界：移除被消耗的边界顶点V1和V2
        boundary.remove_vertex(V1)
        boundary.remove_vertex(V2)

        return quadrilateral

    def is_valid(self, boundary, reference_vertex_V0_idx):
        """
        检查Type 0动作的有效性
        """
        if boundary.size() < 4:
            return False

        # 获取构成四边形的四个顶点
        V0 = boundary.get_vertex_by_index(reference_vertex_V0_idx)
        V1 = boundary.get_vertex_by_index(reference_vertex_V0_idx + 1)
        V2 = boundary.get_vertex_by_index(reference_vertex_V0_idx + 2)
        V3 = boundary.get_vertex_by_index(reference_vertex_V0_idx - 1)

        # ActionType0通过移除V1和V2，将V0和V3连接起来，形成新的内部边。
        # 我们需要检查这条新形成的边 (V0, V3) 是否有效。
        new_internal_edge = (V0, V3)

        # 1. 检查新边是否与任何现有边界边相交（最关键的检查）
        #    注意：这里要排除与V0和V3相邻的边
        if boundary.edge_cross(new_internal_edge):
            return False

        # 2. 检查新边的中点是否在多边形内部，这是一个更严格的检查，可以防止
        #    在凹多边形中形成外部的边。
        mid_point = ((V0[0] + V3[0]) / 2, (V0[1] + V3[1]) / 2)
        if not boundary.vertex_inside_boundary(mid_point):
            return False

        return True
