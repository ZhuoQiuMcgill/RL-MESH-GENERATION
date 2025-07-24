from src.rl.config import load_config
import math
import numpy as np


class FanShape:
    def __init__(self, v_right, v_ref, v_left, base_L):
        """
        Args:
            v_right (np.ndarray | Tuple[float, float]): 右邻顶点
            v_ref   (np.ndarray | Tuple[float, float]): 参考顶点 V0
            v_left  (np.ndarray | Tuple[float, float]): 左邻顶点
            base_L  (float):     论文式 (2) 的平均边长
        """
        self.v_ref = np.asarray(v_ref, dtype=float)
        # 读取环境配置
        env_cfg = load_config().get("environment", {})
        self.beta = env_cfg.get("beta", 6)  # 放大因子
        self.g = env_cfg.get("g", 3)  # 扇形切片数

        self.fan_vertices = []  # g+1 条射线端点（定义 g 个扇形）
        self.radius = self.beta * base_L
        self._init_fan_shapes(np.asarray(v_right, dtype=float),
                              self.v_ref,
                              np.asarray(v_left, dtype=float))

    # ------------------------------------------------------------------ #
    # internal helpers (kept local, no external symbols are exported)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _angle(vec):
        """返回 0~2π 之间的极角"""
        return (math.atan2(vec[1], vec[0]) + 2 * math.pi) % (2 * math.pi)

    @staticmethod
    def _cw_diff(a, b):
        """顺时针方向上 a→b 的角度差 (0~2π)"""
        return (a - b) % (2 * math.pi)

    # ------------------------------------------------------------------ #
    #               1. 初始化 g 条扇形射线 self.fan_vertices
    # ------------------------------------------------------------------ #
    def _init_fan_shapes(self, v_right, v_ref, v_left):
        """
        构建 g+1 个极射线端点 (含两端)，用以定义 g 个顺时针扇形切片。

        其中：
            fan_vertices[0] 对应右邻方向，
            fan_vertices[-1] 对应左邻方向，
            中间射线按顺时针等角分布。
        """
        # 计算两邻顶点相对参考点的方向角
        angle_right = self._angle(v_right - v_ref)
        angle_left = self._angle(v_left - v_ref)

        # 顺时针方向的夹角 & 每切片角度
        total_angle = self._cw_diff(angle_right, angle_left)
        slice_angle = total_angle / self.g

        # 生成 g+1 条射线端点
        self.fan_vertices.clear()
        for i in range(self.g + 1):
            ang = (angle_right - i * slice_angle) % (2 * math.pi)
            dx, dy = math.cos(ang) * self.radius, math.sin(ang) * self.radius
            self.fan_vertices.append((v_ref[0] + dx, v_ref[1] + dy))

    # ------------------------------------------------------------------ #
    #               2. 根据扇形切片筛选边界顶点
    # ------------------------------------------------------------------ #
    def process(self, boundary_vertices):
        """
        Args:
            boundary_vertices (Iterable[Tuple[float, float]]):
                完整边界顶点（顺时针顺序）。

        Returns:
            List[Tuple[float, float] | None]:
                长度 == self.g，每个元素为对应切片内
                半径 <= beta * base_L 的最近顶点，
                若无符合则填 None。
        """
        result = []
        v_ref = self.v_ref
        two_pi = 2 * math.pi

        # 预计算 fan 射线角度
        fan_angles = [self._angle(np.asarray(v, dtype=float) - v_ref)
                      for v in self.fan_vertices]

        def _angle(vec):
            return (math.atan2(vec[1], vec[0]) + two_pi) % two_pi

        def _in_sector(angle_x, start, end):
            # 判断 angle_x 是否落在顺时针区间 start→end
            return self._cw_diff(start, angle_x) <= self._cw_diff(start, end) + 1e-12

        for i in range(self.g):
            start = fan_angles[i]
            end = fan_angles[i + 1]

            best_v, best_dist = None, None
            for v in boundary_vertices:
                # 跳过参考点自身
                if v == tuple(v_ref):
                    continue
                vec = np.asarray(v, dtype=float) - v_ref
                dist = np.linalg.norm(vec)
                # 半径限制
                if dist > self.radius:
                    continue
                ang = _angle(vec)
                if _in_sector(ang, start, end):
                    if best_dist is None or dist < best_dist:
                        best_v, best_dist = v, dist

            result.append(best_v if best_v is not None else None)

        return result
