from .action import ActionType
from src.utils import get_interior_angle, euclidean_distance
import math


class ActionType0Right(ActionType):
    """
    实现Type 0动作：不增加新顶点，直接连接边界上的四个点形成一个四边形。
    对应论文中的 Figure 5(a)。
    """

    def get_element(self, boundary, reference_vertex_idx, *coords):
        v0 = boundary.get_vertex_by_index(reference_vertex_idx)
        v1 = boundary.get_vertex_by_index(reference_vertex_idx + 1)
        v2 = boundary.get_vertex_by_index(reference_vertex_idx + 2)
        v3 = boundary.get_vertex_by_index(reference_vertex_idx - 1)
        return [v0, v1, v2, v3]

    def execute(self, mesh, boundary, reference_vertex_idx, *coords):
        """
        执行Type 0动作的逻辑

        直接修改输入的mesh和boundary对象，避免深拷贝的开销。

        Args:
            mesh: 网格对象（会被直接修改）
            boundary: 边界对象（会被直接修改）
            reference_vertex_idx: 参考顶点V0在边界中的索引

        Returns:
            list: 生成的四边形元素（四个顶点的列表）
        """

        quadrilateral = self.get_element(boundary, reference_vertex_idx)
        v0, v1, v2, v3 = quadrilateral
        try:
            mesh.add_edge(v2, v3)
        except ValueError:
            return None

        # 更新边界：移除被消耗的边界顶点V0和V1
        # 注意：移除V0和V1后，V2和V3会在边界上相邻
        boundary.remove_vertex(v0)
        boundary.remove_vertex(v1)

        return quadrilateral

    def is_valid(self, boundary, reference_vertex_idx, *coords):
        """
        检查Type 0动作的有效性
        """
        if boundary.size() < 4:
            return False

        # 获取构成四边形的四个顶点
        quadrilateral = self.get_element(boundary, reference_vertex_idx)

        if self.element_quality(quadrilateral) < self.QUALITY_THRESHOLD:
            return False

        v0, v1, v2, v3 = quadrilateral

        # 我们需要检查这条新形成的边 (V3, V3) 是否有效。
        new_internal_edge = (v2, v3)

        # 1. 检查新边是否与任何现有边界边相交（最关键的检查）
        #    注意：这里要排除与V0和V3相邻的边
        if boundary.edge_cross(new_internal_edge):
            return False

        # 2. 检查新边的中点是否在多边形内部，这是一个更严格的检查，可以防止
        #    在凹多边形中形成外部的边。
        mid_point = ((v2[0] + v3[0]) / 2, (v2[1] + v3[1]) / 2)
        if not boundary.vertex_inside_boundary(mid_point):
            return False

        return True

    def get_element_quality(self, boundary, reference_vertex_idx, *coords):
        quadrilateral = self.get_element(boundary, reference_vertex_idx)
        return self.element_quality(quadrilateral)

    def get_boundary_quality(self, boundary, reference_vertex_idx, *coords, M_angle):
        # locate the two boundary vertices that form the new edge
        v2 = boundary.get_vertex_by_index(reference_vertex_idx + 2)
        v3 = boundary.get_vertex_by_index(reference_vertex_idx - 1)
        angle_1 = [v3, v2, boundary.get_vertex_by_index(reference_vertex_idx + 3)]
        angle_2 = [boundary.get_vertex_by_index(reference_vertex_idx - 2), v3, v2]

        target_len = euclidean_distance(v2, v3)
        edge_lengths = target_len
        for i in range(1, 3):
            left_v1 = boundary.get_vertex_by_index(reference_vertex_idx + 1 + i)
            left_v2 = boundary.get_vertex_by_index(reference_vertex_idx + 1 + i + 1)
            right_v1 = boundary.get_vertex_by_index(reference_vertex_idx - i)
            right_v2 = boundary.get_vertex_by_index(reference_vertex_idx - i - 1)
            edge_lengths += euclidean_distance(left_v1, left_v2)
            edge_lengths += euclidean_distance(right_v1, right_v2)
        mean_dist = edge_lengths / 5.0

        # angle-quality term (saturated by M_angle)
        a1 = get_interior_angle(*angle_1)
        a2 = get_interior_angle(*angle_2)
        angle_quality = self.calculate_angle_quality(a1, a2, M_angle)

        # smoothness term
        smoothness = (
            min(mean_dist, target_len) / max(mean_dist, target_len)
            if max(mean_dist, target_len) > 0
            else 0.0
        )

        # final score (negative => penalty, 0 => perfect)
        quality = math.sqrt(angle_quality * smoothness)
        return quality
