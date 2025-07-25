from abc import ABC, abstractmethod
from src.quality import *


class ActionType(ABC):
    """动作类型的抽象基类"""
    QUALITY_THRESHOLD = 0.1

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

    @staticmethod
    def element_quality(element) -> float:
        """
        Hybrid quality = robust * (clamped Scaled-Jacobian)**gamma
        :param element: iterable of 4 vertices
        :return: quality in [0, 1]
        """
        return quality_hybrid(element)

    @staticmethod
    def calculate_angle_quality(angle1, angle2, M_angle,
                                pivot_ratio=0.7,  # pivot = pivot_ratio * M_angle
                                alpha=8.0,  # 控制 < pivot 区间的陡峭度 (alpha > 1  ⇒  f'' > 0)
                                beta=3.0):  # 控制 ≥ pivot 区间的平缓度 (beta > 1  ⇒  f'' < 0)

        # ---------------- basic checks ----------------
        if M_angle <= 0:
            raise ValueError("M_angle must be positive.")
        theta = min(angle1, angle2, M_angle)  # 饱和到阈值
        if theta <= 0:
            return 0.0  # Fast exit

        # ---------------- pre‑compute constants ----------------
        xp = pivot_ratio  # 0 < xp < 1
        x = theta / M_angle  # normalised angle ∈ [0, 1]

        # ---------------- piecewise curve ----------------
        if x < xp:  # 迅速下降区间，二阶导 > 0
            quality = 0.5 * (x / xp) ** alpha
        else:  # 缓慢下降区间，二阶导 < 0
            quality = 1.0 - 0.5 * ((1.0 - x) / (1.0 - xp)) ** beta

        # 数值护栏，确保 ∈ [0, 1]
        return max(0.0, min(1.0, quality))

    @abstractmethod
    def get_element_quality(self, boundary, reference_vertex_idx, *coords):
        pass

    @abstractmethod
    def get_boundary_quality(self, boundary, reference_vertex_idx, *coords, M_angle):
        pass
