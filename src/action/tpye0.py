from action import ActionType
import copy


class ActionType0(ActionType):
    """
    实现Type 0动作：不增加新顶点，直接连接边界上的四个点形成一个四边形。
    对应论文中的 Figure 5(a)。
    """

    def execute(self, mesh, boundary, reference_vertex_V0_idx):
        """
        执行Type 0动作的逻辑

        Args:
            mesh: 网格对象
            boundary: 边界对象
            reference_vertex_V0_idx: 参考顶点V0在边界中的索引

        Returns:
            tuple: (更新后的mesh, 更新后的boundary, 生成的元素)
        """
        # 获取边界顶点列表
        vertices = boundary.get_vertices()
        boundary_size = len(vertices)

        # 获取参考顶点V0及其邻居点
        V0 = vertices[reference_vertex_V0_idx]
        V1 = vertices[(reference_vertex_V0_idx + 1) % boundary_size]
        V2 = vertices[(reference_vertex_V0_idx + 2) % boundary_size]
        V3 = vertices[(reference_vertex_V0_idx - 1) % boundary_size]

        # 创建新的网格和边界副本
        new_mesh = copy.deepcopy(mesh)
        new_boundary = copy.deepcopy(boundary)

        # 创建四边形元素 (V0, V1, V2, V3)
        quadrilateral = [V0, V1, V2, V3]

        # 【代码修正】
        # 不再向网格中添加内部对角线 V0-V2。
        # 新的内部边界 V0-V2 是隐式的，由新的边界定义。
        # V0, V1, V2, V3 四个点构成了一个四边形单元，
        # 它们的连接关系由 quadrilateral 列表定义。
        # new_mesh.add_edge(V0, V2) # <--- REMOVED

        # 更新边界：移除被消耗的边界顶点V1和V2
        new_boundary.remove_vertex(V1)
        new_boundary.remove_vertex(V2)

        return new_mesh, new_boundary, quadrilateral

    def is_valid(self, boundary, reference_vertex_V0_idx):
        """
        检查Type 0动作的有效性
        """
        vertices = boundary.get_vertices()
        boundary_size = len(vertices)

        if boundary_size < 4:
            return False

        V0 = vertices[reference_vertex_V0_idx]
        V2 = vertices[(reference_vertex_V0_idx + 2) % boundary_size]

        diagonal_edge = (V0, V2)

        # 检查对角线边是否完整位于边界内部
        if not boundary.edge_inside_boundary(diagonal_edge):
            return False

        # 检查对角线边是否与边界的任何边相交
        if boundary.edge_cross(diagonal_edge):
            return False

        return True
