import numpy as np
from typing import List, Tuple, Any


class Boundary:
    """顺时针排列顶点的封闭多边形边界"""

    def __init__(self, vertices: List[Tuple[float, float]]):
        """
        使用顺时针排列的(x, y)元组列表初始化边界

        Args:
            vertices: 顶点坐标列表，按顺时针顺序排列

        Raises:
            ValueError: 当顶点数量少于3个时
        """
        if len(vertices) < 3:
            raise ValueError("A boundary must have at least three vertices.")
        self._verts = np.asarray(vertices, dtype=float)  # shape (N, 2)

    # ------------------------------------------------------------
    # 只读辅助方法
    # ------------------------------------------------------------
    def get_vertices(self) -> List[Tuple[float, float]]:
        """
        返回顶点的副本作为元组列表

        Returns:
            List[Tuple[float, float]]: 顶点坐标列表的副本
        """
        return [tuple(v) for v in self._verts]

    def get_edges(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        返回边的列表，每条边由(前一个顶点, 当前顶点)元组表示

        Returns:
            List[Tuple[Tuple[float, float], Tuple[float, float]]]: 边的列表
        """
        return [(tuple(self._verts[i - 1]), tuple(self._verts[i]))
                for i in range(len(self._verts))]

    def size(self) -> int:
        """
        返回当前边界中顶点的数量

        Returns:
            int: 边界中顶点的数量
        """
        return len(self._verts)

    def get_average_edge_length(self) -> float:
        """
        计算并返回边界中所有边的平均长度

        Returns:
            float: 边界中所有边的平均长度

        Raises:
            ValueError: 当边界顶点数量少于2时
        """
        if len(self._verts) < 2:
            raise ValueError("Need at least 2 vertices to calculate edge length")

        total_length = 0.0
        num_edges = len(self._verts)

        for i in range(num_edges):
            # 计算当前顶点到下一个顶点的距离（闭合多边形）
            current_vertex = self._verts[i]
            next_vertex = self._verts[(i + 1) % num_edges]

            # 计算欧几里得距离
            edge_length = np.linalg.norm(next_vertex - current_vertex)
            total_length += edge_length

        return total_length / num_edges

    # ------------------------------------------------------------
    # 内角计算工具
    # ------------------------------------------------------------
    def _interior_angles(self) -> np.ndarray:
        """
        计算每个顶点的内角（度数）

        Returns:
            np.ndarray: 每个顶点的内角数组
        """
        v_prev = np.roll(self._verts, 1, axis=0)
        v_next = np.roll(self._verts, -1, axis=0)
        a = v_prev - self._verts  # BA
        b = v_next - self._verts  # BC
        dot = np.einsum('ij,ij->i', a, b)
        norm_prod = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        # 防止除零错误
        cos_theta = np.clip(dot / np.where(norm_prod == 0, 1, norm_prod), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def get_max_interior_angles(self) -> Tuple[Tuple[float, float], float]:
        """
        返回具有最大内角的顶点和角度

        Returns:
            Tuple[Tuple[float, float], float]: (顶点坐标, 角度值)
        """
        angles = self._interior_angles()
        idx = int(angles.argmax())
        return tuple(self._verts[idx]), float(angles[idx])

    def get_min_interior_angles(self) -> Tuple[Tuple[float, float], float]:
        """
        返回具有最小内角的顶点和角度

        Returns:
            Tuple[Tuple[float, float], float]: (顶点坐标, 角度值)
        """
        angles = self._interior_angles()
        idx = int(angles.argmin())
        return tuple(self._verts[idx]), float(angles[idx])

    # ------------------------------------------------------------
    # 修改器方法
    # ------------------------------------------------------------
    def remove_vertex(self, vertex: Tuple[float, float]):
        """
        如果存在则移除指定顶点，否则静默忽略

        Args:
            vertex: 要移除的顶点坐标(x, y)
        """
        mask = (self._verts == vertex).all(axis=1)
        if mask.any():
            self._verts = self._verts[~mask]

    def insert_vertex(self, vertex: Tuple[float, float], position: int):
        """
        在指定位置插入顶点

        Args:
            vertex: 要插入的顶点坐标(x, y)
            position: 插入位置索引(0 ≤ pos ≤ len)

        Raises:
            IndexError: 当位置超出范围时
        """
        if not (0 <= position <= len(self._verts)):
            raise IndexError("position out of range")
        self._verts = np.insert(self._verts, position, vertex, axis=0)

    def part_of_boundary(self, vertex: Tuple[float, float]) -> bool:
        """
        查找该点是否存在为当前的边界组成点

        Args:
            vertex: 要检查的点坐标(x, y)

        Returns:
            bool: 如果点是边界组成点返回True，否则返回False
        """
        vertex_array = np.array(vertex, dtype=float)
        # 检查是否与任何顶点匹配（考虑浮点数精度）
        distances = np.linalg.norm(self._verts - vertex_array, axis=1)
        return np.any(distances < 1e-10)

    def vertex_inside_boundary(self, vertex: Tuple[float, float]) -> bool:
        """
        严格判定点是否处于多边形边界的内部

        判断条件：
        - 点为边界组成点 -> False
        - 点位于边界的某个边上 -> False
        - 点位于边界内部 -> True
        - 点位于边界外部 -> False

        注意：边界self._verts为顺时针顺序，即(self._verts[n], self._verts[n+1])的右侧为内部

        Args:
            vertex: 要检查的点坐标(x, y)

        Returns:
            bool: 如果点在边界内部返回True，否则返回False
        """
        # 首先检查是否是边界组成点
        if self.part_of_boundary(vertex):
            return False

        # 检查是否在边界的某个边上
        if self._point_on_boundary_edge(vertex):
            return False

        # 使用射线投射算法判断点是否在多边形内部
        return self._point_in_polygon(vertex)

    def edge_cross(self, edge: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """
        严格判定输入的边是否与任意边界的边有相交

        判断条件：
        - 输入边的一个组成点为边界的组成点 -> False
        - 输入边与任意一个边界的边相交 -> True
        - 输入边的任意一个点位于边界的某一个边上 -> True

        Args:
            edge: 输入的边((x1, y1), (x2, y2))

        Returns:
            bool: 如果边与边界相交返回True，否则返回False
        """
        p1, p2 = edge

        # 检查输入边的组成点是否为边界组成点
        if self.part_of_boundary(p1) or self.part_of_boundary(p2):
            return False

        # 检查输入边的点是否在边界的某个边上
        if self._point_on_boundary_edge(p1) or self._point_on_boundary_edge(p2):
            return True

        # 检查输入边是否与任何边界边相交
        return self._edge_intersects_boundary(edge)

    def edge_inside_boundary(self, edge: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """
        严格判定输入的边是否完整位于边界内

        判断条件：
        - 输入边的两个点都位于边界内 -> True
        - 输入边与边界相交 -> False
        - 输入边的一个点为边界组成点，另一个点位于边界内 -> True
        - 输入边的一个点为边界组成点，另一个点位于边界外 -> False

        Args:
            edge: 输入的边((x1, y1), (x2, y2))

        Returns:
            bool: 如果边完整位于边界内返回True，否则返回False
        """
        p1, p2 = edge

        # 检查边是否与边界相交
        if self.edge_cross(edge):
            return False

        p1_on_boundary = self.part_of_boundary(p1)
        p2_on_boundary = self.part_of_boundary(p2)
        p1_inside = self.vertex_inside_boundary(p1) if not p1_on_boundary else False
        p2_inside = self.vertex_inside_boundary(p2) if not p2_on_boundary else False

        # 两个点都在边界内
        if p1_inside and p2_inside:
            return True

        # 一个点在边界上，另一个点在边界内
        if (p1_on_boundary and p2_inside) or (p2_on_boundary and p1_inside):
            return True

        # 其他情况都返回False
        return False

    # ------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------
    def _point_on_boundary_edge(self, point: Tuple[float, float]) -> bool:
        """
        检查点是否在边界的某个边上

        Args:
            point: 要检查的点坐标(x, y)

        Returns:
            bool: 如果点在边界的某个边上返回True，否则返回False
        """
        point_array = np.array(point, dtype=float)

        for i in range(len(self._verts)):
            v1 = self._verts[i]
            v2 = self._verts[(i + 1) % len(self._verts)]

            if self._point_on_line_segment(point_array, v1, v2):
                return True

        return False

    def _point_on_line_segment(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> bool:
        """
        检查点是否在线段上

        Args:
            point: 要检查的点
            line_start: 线段起点
            line_end: 线段终点

        Returns:
            bool: 如果点在线段上返回True，否则返回False
        """
        # 使用向量叉积判断点是否在线段上
        # 如果点在线段上，则叉积应该为0，且点应该在线段的范围内

        # 向量
        v1 = point - line_start
        v2 = line_end - line_start

        # 叉积（在2D中是标量）
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        # 如果叉积不为0（考虑浮点精度），点不在直线上
        if abs(cross_product) > 1e-10:
            return False

        # 检查点是否在线段范围内
        dot_product = np.dot(v1, v2)
        squared_length = np.dot(v2, v2)

        if squared_length == 0:  # 线段长度为0
            return np.allclose(point, line_start)

        param = dot_product / squared_length
        return 0 <= param <= 1

    def _point_in_polygon(self, point: Tuple[float, float]) -> bool:
        """
        使用射线投射算法判断点是否在多边形内部

        Args:
            point: 要检查的点坐标(x, y)

        Returns:
            bool: 如果点在多边形内部返回True，否则返回False
        """
        x, y = point
        n = len(self._verts)
        inside = False

        p1x, p1y = self._verts[0]
        for i in range(1, n + 1):
            p2x, p2y = self._verts[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _edge_intersects_boundary(self, edge: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """
        检查边是否与边界的任何边相交

        Args:
            edge: 要检查的边((x1, y1), (x2, y2))

        Returns:
            bool: 如果边与边界相交返回True，否则返回False
        """
        p1, p2 = edge

        for i in range(len(self._verts)):
            v1 = self._verts[i]
            v2 = self._verts[(i + 1) % len(self._verts)]

            if self._line_segments_intersect(p1, p2, tuple(v1), tuple(v2)):
                return True

        return False

    def _line_segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                                 p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """
        检查两个线段是否相交

        Args:
            p1: 第一个线段的起点
            p2: 第一个线段的终点
            p3: 第二个线段的起点
            p4: 第二个线段的终点

        Returns:
            bool: 如果两个线段相交返回True，否则返回False
        """

        def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
            """检查三个点是否按逆时针顺序排列"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # 如果两个线段相交，则它们的端点应该在对方的两侧
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))
