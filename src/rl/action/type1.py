from .action import ActionType
from src.utils import get_interior_angle, euclidean_distance
import math


class ActionType1(ActionType):
    """
    实现Type 1动作：增加一个新顶点V2，形成一个四边形。
    对应论文中的 Figure 5(b)。
    """

    def get_element(self, boundary, reference_vertex_V0_idx, new_vertex_V2_coords):
        v0 = boundary.get_vertex_by_index(reference_vertex_V0_idx)
        v1 = boundary.get_vertex_by_index(reference_vertex_V0_idx - 1)
        v3 = boundary.get_vertex_by_index(reference_vertex_V0_idx + 1)
        v2 = tuple(new_vertex_V2_coords)
        return [v0, v3, v2, v1]

    def get_generated_angle(self, boundary, reference_vertex_V0_idx, new_vertex_V2_coords):
        v1 = boundary.get_vertex_by_index(reference_vertex_V0_idx - 1)
        v3 = boundary.get_vertex_by_index(reference_vertex_V0_idx + 1)
        v2 = tuple(new_vertex_V2_coords)
        angle_1 = [v2, v3, boundary.get_vertex_by_index(reference_vertex_V0_idx + 2)]
        angle_2 = [boundary.get_vertex_by_index(reference_vertex_V0_idx - 2), v1, v2]
        return angle_1, angle_2

    def execute(self, mesh, boundary, reference_vertex_V0_idx, new_vertex_V2_coords):
        """
        执行Type 1动作的逻辑

        直接修改输入的mesh和boundary对象，避免深拷贝的开销。

        Args:
            mesh: 网格对象（会被直接修改）
            boundary: 边界对象（会被直接修改）
            reference_vertex_V0_idx: 参考顶点V0在边界中的索引
            new_vertex_V2_coords: 新顶点V2的坐标

        Returns:
            list: 生成的四边形元素（四个顶点的列表）
        """

        quadrilateral = self.get_element(boundary, reference_vertex_V0_idx, new_vertex_V2_coords)
        v0, v3, v2, v1 = quadrilateral

        # 创建四边形元素
        try:
            # 向网格中添加新顶点
            mesh.add_vertex(v2)

            # 在网格中添加新的边界边
            mesh.add_edge(v1, v2)
            mesh.add_edge(v2, v3)

            # 移除V0（它变成内部点）
            boundary.remove_vertex(v0)
        except ValueError:
            return None

        v1_idx = boundary.get_vertex_index(v1)
        if v1_idx != -1:
            boundary.insert_vertex(v2, v1_idx + 1)
        else:
            raise RuntimeError("Boundary update failed: V3 not found after removing V0.")

        return quadrilateral

    def is_valid(self, boundary, reference_vertex_V0_idx, new_vertex_V2_coords):
        """
        检查Type 1动作的有效性

        Args:
            boundary: 边界对象
            reference_vertex_V0_idx: 参考顶点V0在边界中的索引
            new_vertex_V2_coords: 新顶点V2的坐标

        Returns:
            bool: 动作是否有效
        """
        if boundary.size() < 3:
            return False

        quadrilateral = self.get_element(boundary, reference_vertex_V0_idx, new_vertex_V2_coords)
        v0, v3, v2, v1 = quadrilateral
        if self.element_quality(quadrilateral) < self.QUALITY_THRESHOLD:
            return False

        if not boundary.vertex_inside_boundary(v2):
            return False

        edge_V1_V2 = (v1, v2)
        edge_V2_V3 = (v2, v3)

        if not boundary.edge_inside_boundary(edge_V1_V2):
            return False
        if not boundary.edge_inside_boundary(edge_V2_V3):
            return False

        if boundary.edge_cross(edge_V1_V2):
            return False
        if boundary.edge_cross(edge_V2_V3):
            return False

        return True

    def get_element_quality(self, boundary, reference_vertex_V0_idx, new_vertex_V2_coords):
        quadrilateral = self.get_element(boundary, reference_vertex_V0_idx, new_vertex_V2_coords)
        return self.element_quality(quadrilateral)

    def get_boundary_quality(self, boundary, reference_vertex_V0_idx, new_vertex_V2_coords, M_angle):
        a1, a2 = self.get_generated_angle(boundary, reference_vertex_V0_idx, new_vertex_V2_coords)
        angle1 = get_interior_angle(a1[0], a1[1], a1[2])
        angle2 = get_interior_angle(a2[0], a2[1], a2[2])

        quadrilateral = self.get_element(boundary, reference_vertex_V0_idx, new_vertex_V2_coords)
        [v0, v3, v2, v1] = quadrilateral

        d1 = euclidean_distance(v3, v0)
        d2 = euclidean_distance(v1, v0)

        ignore_edges = {(v1, v0), (v0, v3), (v3, v2), (v2, v1)}
        q_dist = boundary.get_closest_edge_distance(tuple(new_vertex_V2_coords), ignore_edges)

        if q_dist >= (d1 + d2) / 2:
            q_dist = 1
        return math.sqrt(min([angle1, angle2, M_angle]) / M_angle * q_dist) - 1
