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

        Args:
            boundary: 边界对象
            reference_vertex_V0_idx: 参考顶点V0在边界中的索引

        Returns:
            bool: 动作是否有效
        """
        if boundary.size() < 4:
            return False

        V0 = boundary.get_vertex_by_index(reference_vertex_V0_idx)
        V2 = boundary.get_vertex_by_index(reference_vertex_V0_idx + 2)

        diagonal_edge = (V0, V2)

        # 检查对角线边是否完整位于边界内部
        if not boundary.edge_inside_boundary(diagonal_edge):
            return False

        # 检查对角线边是否与边界的任何边相交
        if boundary.edge_cross(diagonal_edge):
            return False

        return True
