from abc import ABC, abstractmethod
import math
from src.utils import euclidean_distance, get_interior_angle
from src.rl.quality import *


class ActionType(ABC):
    """动作类型的抽象基类"""
    QUALITY_THRESHOLD = 0.01

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
    def element_quality(element, gamma: float = 1.0) -> float:
        """
        Hybrid quality = robust * (clamped Scaled-Jacobian)**gamma
        :param element: iterable of 4 vertices
        :param gamma: Jacobian penalty exponent (>=1 makes penalty steeper)
        :return: quality in [0, 1]
        """
        # sj = quality_s_jacobian(element)
        # if sj <= 0.0:
        #     return 0.0  # flipped or collapsed

        # sj = min(1.0, sj)  # clamp to [0,1]
        # robust = quality_robust(element)

        # q = robust * (sj ** gamma)  # smooth hybrid metric
        # return max(0.0, min(1.0, q))
        return quality_robust(element)

    @abstractmethod
    def get_element_quality(self, boundary, **kwargs):
        pass

    @abstractmethod
    def get_boundary_quality(self, boundary, **kwargs):
        pass
