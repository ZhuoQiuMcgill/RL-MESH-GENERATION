from .action import ActionType


class ActionType2(ActionType):
    """
    实现Type 2动作：增加两个新顶点V2和V3，形成一个四边形。
    对应论文中的 Figure 5(c)。
    """

    def execute(self, mesh, boundary, reference_vertex_V0_idx, new_vertex_V2_coords, new_vertex_V3_coords):
        """
        执行Type 2动作的逻辑

        直接修改输入的mesh和boundary对象，避免深拷贝的开销。

        Args:
            mesh: 网格对象（会被直接修改）
            boundary: 边界对象（会被直接修改）
            reference_vertex_V0_idx: 参考顶点V0在边界中的索引
            new_vertex_V2_coords: 新顶点V2的坐标
            new_vertex_V3_coords: 新顶点V3的坐标

        Returns:
            list: 生成的四边形元素（四个顶点的列表）
        """
        # 使用新的封装函数获取顶点
        V0 = boundary.get_vertex_by_index(reference_vertex_V0_idx)
        V1 = boundary.get_vertex_by_index(reference_vertex_V0_idx + 1)
        V2 = tuple(new_vertex_V2_coords)
        V3 = tuple(new_vertex_V3_coords)

        # 向网格中添加新顶点
        mesh.add_vertex(V2)
        mesh.add_vertex(V3)

        # 创建四边形元素
        quadrilateral = [V0, V1, V2, V3]

        # 在网格中添加新的边界边
        mesh.add_edge(V1, V2)
        mesh.add_edge(V2, V3)
        mesh.add_edge(V3, V0)

        # 更新边界：移除被消耗的边界顶点V1，并插入V2和V3
        boundary.remove_vertex(V1)

        boundary_vertices = boundary.get_vertices()
        v0_idx = -1
        for i, v in enumerate(boundary_vertices):
            if v == V0:
                v0_idx = i
                break

        if v0_idx != -1:
            # 在V0后面依次插入V3和V2，以匹配 [V0, V1, V2, V3] 的逆时针顺序
            # 论文图示是V0-V3-V2-V1，所以先插V3再插V2
            boundary.insert_vertex(V3, v0_idx + 1)
            boundary.insert_vertex(V2, v0_idx + 2)
        else:
            raise RuntimeError("Boundary update failed: V0 not found after removing V1.")

        return quadrilateral

    def is_valid(self, boundary, reference_vertex_V0_idx, new_vertex_V2_coords, new_vertex_V3_coords):
        """
        检查Type 2动作的有效性

        Args:
            boundary: 边界对象
            reference_vertex_V0_idx: 参考顶点V0在边界中的索引
            new_vertex_V2_coords: 新顶点V2的坐标
            new_vertex_V3_coords: 新顶点V3的坐标

        Returns:
            bool: 动作是否有效
        """
        if boundary.size() < 2:
            return False

        V0 = boundary.get_vertex_by_index(reference_vertex_V0_idx)
        V1 = boundary.get_vertex_by_index(reference_vertex_V0_idx + 1)
        V2 = tuple(new_vertex_V2_coords)
        V3 = tuple(new_vertex_V3_coords)

        if not boundary.vertex_inside_boundary(V2):
            return False
        if not boundary.vertex_inside_boundary(V3):
            return False

        edge_V1_V2 = (V1, V2)
        edge_V2_V3 = (V2, V3)
        edge_V3_V0 = (V3, V0)

        if not boundary.edge_inside_boundary(edge_V1_V2):
            return False
        if not boundary.edge_inside_boundary(edge_V2_V3):
            return False
        if not boundary.edge_inside_boundary(edge_V3_V0):
            return False

        if boundary.edge_cross(edge_V1_V2):
            return False
        if boundary.edge_cross(edge_V2_V3):
            return False
        if boundary.edge_cross(edge_V3_V0):
            return False

        return True
