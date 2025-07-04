from action import ActionType
import copy


class ActionType1(ActionType):
    """
    实现Type 1动作：增加一个新顶点V2，形成一个四边形。
    对应论文中的 Figure 5(b)。
    """

    def execute(self, mesh, boundary, reference_vertex_V0_idx, new_vertex_V2_coords):
        """
        执行Type 1动作的逻辑

        Args:
            mesh: 网格对象
            boundary: 边界对象
            reference_vertex_V0_idx: 参考顶点V0在边界中的索引
            new_vertex_V2_coords: 新顶点V2的坐标(x, y)

        Returns:
            tuple: (更新后的mesh, 更新后的boundary, 生成的元素)
        """
        # 获取边界顶点列表
        vertices = boundary.get_vertices()
        boundary_size = len(vertices)

        # 获取参考顶点V0及其邻居点
        V0 = vertices[reference_vertex_V0_idx]
        V1 = vertices[(reference_vertex_V0_idx + 1) % boundary_size]  # V0的右邻居
        V3 = vertices[(reference_vertex_V0_idx - 1) % boundary_size]  # V0的左邻居
        V2 = tuple(new_vertex_V2_coords)  # 新添加的顶点

        # 创建新的网格和边界副本
        new_mesh = copy.deepcopy(mesh)
        new_boundary = copy.deepcopy(boundary)

        # 在网格中添加新顶点V2
        new_mesh.add_vertex(V2)

        # 创建四边形元素 (V0, V1, V2, V3)
        quadrilateral = [V0, V1, V2, V3]

        # 在网格中添加新的边
        new_mesh.add_edge(V0, V2)  # 内部对角边
        new_mesh.add_edge(V1, V2)  # 新的边界边
        new_mesh.add_edge(V2, V3)  # 新的边界边

        # 更新边界：移除旧的边界边，添加新的边界边
        # 移除顶点V1
        new_boundary.remove_vertex(V1)

        # 在V0和V3之间插入新顶点V2
        # 找到V0在新边界中的位置
        boundary_vertices = new_boundary.get_vertices()
        v0_idx = None
        for i, v in enumerate(boundary_vertices):
            if v == V0:
                v0_idx = i
                break

        if v0_idx is not None:
            # 在V0后面插入V2
            new_boundary.insert_vertex(V2, v0_idx + 1)

        return new_mesh, new_boundary, quadrilateral

    def is_valid(self, boundary, reference_vertex_V0_idx, new_vertex_V2_coords):
        """
        检查Type 1动作的有效性

        Args:
            boundary: 边界对象
            reference_vertex_V0_idx: 参考顶点V0在边界中的索引
            new_vertex_V2_coords: 新顶点V2的坐标(x, y)

        Returns:
            bool: 如果动作有效返回True，否则返回False
        """
        vertices = boundary.get_vertices()
        boundary_size = len(vertices)

        # 检查边界是否有足够的顶点（至少需要3个顶点）
        if boundary_size < 3:
            return False

        # 获取相关顶点
        V0 = vertices[reference_vertex_V0_idx]
        V1 = vertices[(reference_vertex_V0_idx + 1) % boundary_size]
        V3 = vertices[(reference_vertex_V0_idx - 1) % boundary_size]
        V2 = tuple(new_vertex_V2_coords)

        # 1. 检查新顶点V2是否位于当前几何域的内部
        if not boundary.vertex_inside_boundary(V2):
            return False

        # 2. 检查新生成的边是否都在边界内部
        edge_V1_V2 = (V1, V2)
        edge_V2_V3 = (V2, V3)

        if not boundary.edge_inside_boundary(edge_V1_V2):
            return False
        if not boundary.edge_inside_boundary(edge_V2_V3):
            return False

        # 3. 检查新生成的边是否与边界的任何边相交
        if boundary.edge_cross(edge_V1_V2):
            return False
        if boundary.edge_cross(edge_V2_V3):
            return False

        return True
