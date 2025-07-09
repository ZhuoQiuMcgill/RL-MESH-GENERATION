from abc import ABC, abstractmethod
import math
from src.utils import euclidean_distance, get_interior_angle


class ActionType(ABC):
    """动作类型的抽象基类"""
    QUALITY_THRESHOLD = 0.3

    @abstractmethod
    def execute(self, mesh, boundary, **kwargs):
        """
        执行具体的几何操作来修改网格和边界。
        返回更新后的 (mesh, boundary, generated_element)。
        """
        pass

    @abstractmethod
    def is_valid(self, boundary, **kwargs):
        """
        检查动作在当前边界下是否有效。
        """
        pass

    @abstractmethod
    def get_element(self, boundary, **kwargs):
        pass

    @abstractmethod
    def get_generated_angle(self, boundary, **kwargs):
        pass

    @staticmethod
    def element_quality(element):
        if element is None or len(element) != 4:
            return 0.0

        # 计算边长
        edges = []
        for i in range(4):
            v1 = element[i]
            v2 = element[(i + 1) % 4]
            edge_length = euclidean_distance(v1, v2)
            edges.append(edge_length)

        # 计算对角线长度
        diag1 = euclidean_distance(element[0], element[2])
        diag2 = euclidean_distance(element[1], element[3])
        max_diagonal = max(diag1, diag2)

        # 计算边质量 q_edge
        min_edge = min(edges)
        q_edge = (math.sqrt(2) * min_edge) / max_diagonal if max_diagonal > 0 else 0

        # 计算内角
        angles = []
        for i in range(4):
            v_prev = element[(i - 1) % 4]
            v_curr = element[i]
            v_next = element[(i + 1) % 4]
            angle = get_interior_angle(v_prev, v_curr, v_next)
            angles.append(angle)

        # 计算角度质量 q_angle
        min_angle = min(angles)
        max_angle = max(angles)
        q_angle = min_angle / max_angle if max_angle > 0 else 0

        # 元素质量
        eta_e = math.sqrt(q_edge * q_angle)
        return min(1.0, max(0.0, eta_e))

    @abstractmethod
    def get_element_quality(self, boundary, **kwargs):
        pass

    @abstractmethod
    def get_boundary_quality(self, boundary, **kwargs):
        pass
