from abc import ABC, abstractmethod
from src.quality import *


class ActionType(ABC):
    """动作类型的抽象基类"""
    QUALITY_THRESHOLD = 0.08

    @abstractmethod
    def execute(self, mesh, boundary, reference_vertex_idx, *coords):
        """
        执行具体的几何操作来修改网格和边界。
        返回生成的元素。
        """
        pass

    @abstractmethod
    def is_valid(self, boundary, reference_vertex_idx, *coords):
        """
        检查动作在当前边界下是否有效。
        """
        pass

    @abstractmethod
    def get_element(self, boundary, reference_vertex_idx, *coords):
        pass

    @abstractmethod
    def get_generated_angle(self, boundary, reference_vertex_idx, *coords):
        pass

    @staticmethod
    def element_quality(element, gamma: float = 1.0) -> float:
        """
        Hybrid quality = robust * (clamped Scaled-Jacobian)**gamma
        :param element: iterable of 4 vertices
        :param gamma: Jacobian penalty exponent (>=1 makes penalty steeper)
        :return: quality in [0, 1]
        """
        sj = quality_s_jacobian(element)
        if sj <= 0.0:
            return 0.0  # flipped or collapsed

        sj = min(1.0, sj)  # clamp to [0,1]
        robust = quality_robust(element)

        q = robust * (sj ** gamma)  # smooth hybrid metric
        return max(0.0, min(1.0, q))

    @abstractmethod
    def get_element_quality(self, boundary, reference_vertex_idx, *coords):
        pass

    @abstractmethod
    def get_boundary_quality(self, boundary, reference_vertex_idx, *coords, M_angle):
        pass
