from math import atan2, pi, degrees
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

    def get_vertex_index(self, v: Tuple[float, float]):
        for i, vert in enumerate(self._verts):
            if abs(vert[0] - v[0]) < 1e-8 and abs(vert[1] - v[1]) < 1e-8:
                return i
        return -1

    def get_vertex_by_index(self, n: int):
        """Return the vertex at index n, supporting negative and overflow indices."""
        if not isinstance(n, int):
            raise TypeError("index must be int")
        if self.size() == 0:
            raise IndexError("no vertices in boundary")

        idx = n % self.size()  # 支持负数和越界
        return tuple(self._verts[idx])

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

    # ------------------------------------------------------------
    # 内角计算工具
    # ------------------------------------------------------------
    def get_ref_vertex(self):
        min_interior_angle = 360
        min_n = 0
        for n in range(self.size()):
            avg_interior_angle = self.get_avg_interior_angle(n)
            if avg_interior_angle < min_interior_angle:
                min_interior_angle = avg_interior_angle
                min_n = n

        return min_n

    def get_avg_interior_angle(self, n):
        """
        按照论文中的算法，ref_point 为v0，选取两个内角(v-2, v0, v+2)与(v-1, v0, v+1)的平均值
        :param n: int ref_point的index
        :return: 角度 0-360
        """
        ref_point = self.get_vertex_by_index(n)
        left_point_1 = self.get_vertex_by_index(n + 1)
        left_point_2 = self.get_vertex_by_index(n + 2)
        right_point_1 = self.get_vertex_by_index(n - 1)
        right_point_2 = self.get_vertex_by_index(n - 2)
        return (self.get_interior_angle(right_point_1, ref_point, left_point_1) +
                self.get_interior_angle(right_point_2, ref_point, left_point_2)) / 2

    @staticmethod
    def get_interior_angle(right, center, left):
        ax, ay = right[0] - center[0], right[1] - center[1]
        bx, by = left[0] - center[0], left[1] - center[1]
        theta = (atan2(ay, ax) - atan2(by, bx)) % (2 * pi)
        return degrees(theta)

    def _interior_angles(self) -> np.ndarray:
        """Return the clockwise interior angle of each vertex in degrees."""

        v_prev = np.roll(self._verts, 1, axis=0)
        v_next = np.roll(self._verts, -1, axis=0)

        # 向量指向前一个和后一个顶点
        vec_prev = v_prev - self._verts
        vec_next = v_next - self._verts

        # 计算与x轴的绝对角度然后取差值。
        ang_prev = np.arctan2(vec_prev[:, 1], vec_prev[:, 0])
        ang_next = np.arctan2(vec_next[:, 1], vec_next[:, 0])

        # 因为顶点按顺时针给出，内角 = (ang_prev - ang_next) mod 2π
        angles = (ang_prev - ang_next) % (2 * np.pi)

        return np.degrees(angles)

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
        - 输入边与任意一个边界的边相交 -> True
        - 输入边的任意一个点位于边界的某一个边上 (但不是其端点) -> True

        Args:
            edge: 输入的边((x1, y1), (x2, y2))

        Returns:
            bool: 如果边与边界相交返回True，否则返回False
        """
        p1, p2 = edge

        # [FIXED] The original check here was flawed and has been removed.
        # The intersection logic itself is now robust enough.

        # 检查输入边的点是否在边界的某个边上
        if self._point_on_boundary_edge(p1) or self._point_on_boundary_edge(p2):
            # This check is now implicitly handled by the robust _edge_intersects_boundary
            # which correctly checks for touching/collinear cases.
            pass

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

    def get_neighbor_info(self, vertex: tuple, n: int) -> dict:
        """
        本函数旨在获取指定顶点(vertex)在边界上的一个完整局部片段信息。
        一个局部片段被定义为该顶点本身，以及其沿边界前后各n个邻居点，总共2n+1个点。
        函数将返回这个片段的有序坐标列表，以及由这些点构成的局部线段的平均长度。
        """
        # 1. 定位顶点
        vertex_array = np.array(vertex, dtype=float)
        # 找到匹配的顶点索引（考虑浮点数精度）
        distances = np.linalg.norm(self._verts - vertex_array, axis=1)
        matches = np.where(distances < 1e-10)[0]

        if len(matches) == 0:
            raise ValueError(f"Vertex {vertex} not found in boundary")

        idx = matches[0]  # 取第一个匹配的索引

        # 2. 处理边界大小
        boundary_size = self.size()
        if boundary_size < 2 * n + 1:
            raise ValueError(f"Boundary has only {boundary_size} vertices, need at least {2 * n + 1} for n={n}")

        # 3. 构建局部片段列表
        local_segment_coords = []
        for i in range(-n, n + 1):
            # 使用模运算处理环形边界的索引
            neighbor_idx = (idx + i) % boundary_size
            local_segment_coords.append(tuple(self._verts[neighbor_idx]))

        # 4. 计算局部平均边长
        total_length = 0.0
        num_edges = 2 * n  # 2n条边（2n+1个点之间有2n条边）

        for i in range(num_edges):
            current_point = np.array(local_segment_coords[i])
            next_point = np.array(local_segment_coords[i + 1])
            # 计算欧几里得距离
            edge_length = np.linalg.norm(next_point - current_point)
            total_length += edge_length

        local_avg_edge_length = total_length / num_edges

        # 5. 构建并返回结果
        return {
            "local_segment_coords": local_segment_coords,
            "local_avg_edge_length": local_avg_edge_length
        }

    def get_area(self) -> float:
        """
        本函数旨在计算当前边界所围成多边形的面积。
        这个面积值是计算网格化进度指标 ρ_t (rho_t) 的基础。
        """
        if len(self._verts) < 3:
            return 0.0

        # 1. 获取顶点坐标
        x = self._verts[:, 0]
        y = self._verts[:, 1]

        # 2. 应用鞋带公式
        # 使用numpy.roll获取下一个顶点的坐标，形成闭环
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        # 计算交叉乘积：Σ(x_i * y_{i+1} - x_{i+1} * y_i)
        cross_product = x * y_next - x_next * y

        # 3. 返回结果：Area = 0.5 * |Σ(cross_product)|
        area = 0.5 * abs(np.sum(cross_product))

        return area

    def get_fan_points(self, reference_vertex_index: int, g: int, fan_radius: float) -> List[Tuple[float, float]]:
        """
        本函数旨在实现论文中定义的"前瞻性扇形区域"顶点提取功能。
        它会从给定的参考顶点向多边形内部投射一个扇形区域，将其划分为`g`个相等的角切片，
        并在每个切片中找到一个代表性的边界点，最终返回这`g`个点的列表。
        """
        # 输入验证
        boundary_size = len(self._verts)
        if not (0 <= reference_vertex_index < boundary_size):
            raise IndexError("Reference vertex index out of range")

        if g <= 0:
            raise ValueError("Number of slices g must be positive")

        # 1. 定义扇形几何边界
        v_0 = self.get_vertex_by_index(reference_vertex_index)
        v_l1 = self.get_vertex_by_index(reference_vertex_index + 1)  # 左邻居
        v_r1 = self.get_vertex_by_index(reference_vertex_index - 1)  # 右邻居

        # 计算定义扇形边界的两个向量
        vec_left = v_l1 - v_0
        vec_right = v_r1 - v_0

        # 计算角度
        angle_left = np.arctan2(vec_left[1], vec_left[0])
        angle_right = np.arctan2(vec_right[1], vec_right[0])

        # 计算总扇形角度（考虑顺时针边界）
        # 从右邻居到左邻居的内角
        total_fan_angle = angle_left - angle_right
        if total_fan_angle <= 0:
            total_fan_angle += 2 * np.pi

        # 2. 划分扇形为g个切片
        slice_angle = total_fan_angle / g
        result_points = []

        # 3. 为每个切片寻找代表点
        for i in range(g):
            # 当前切片的起始和结束角度（从右到左）
            slice_start_angle = angle_right + i * slice_angle
            slice_end_angle = angle_right + (i + 1) * slice_angle

            # a. 筛选候选点
            candidates = []
            for j, vertex in enumerate(self._verts):
                if j == reference_vertex_index:
                    continue

                # 计算到参考点的向量和距离
                vec_to_vertex = vertex - v_0
                distance = np.linalg.norm(vec_to_vertex)

                if distance > fan_radius:
                    continue

                # 计算角度
                vertex_angle = np.arctan2(vec_to_vertex[1], vec_to_vertex[0])

                # 检查是否在当前切片内
                if self._is_angle_in_slice(vertex_angle, slice_start_angle, slice_end_angle):
                    candidates.append((j, vertex, distance))

            # b. 选择最近点
            if candidates:
                # 选择距离最近的候选点
                best_candidate = min(candidates, key=lambda x: x[2])
                result_points.append(tuple(best_candidate[1]))
            else:
                # c. 处理空切片 - 使用角平分线与边界的交点
                bisector_angle = (slice_start_angle + slice_end_angle) / 2
                intersection_point = self._find_bisector_boundary_intersection(v_0, bisector_angle, fan_radius)
                result_points.append(intersection_point)

        return result_points

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

    def _edge_intersects_boundary(
            self, edge: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> bool:
        """
        Return True **only if** the interior of `edge` intersects
        any boundary edge.

        ── Allowed ──────────────────────────────────────────────
        • Sharing one or two endpoints with the boundary.
          (Typical case: make a fan from boundary vertices.)
        ── Forbidden ────────────────────────────────────────────
        • Proper cross-intersection.
        • Colinear overlap by a positive length (including
          being exactly the same as an existing boundary edge).
        """
        p1, p2 = edge

        for i in range(len(self._verts)):
            v1 = tuple(self._verts[i])
            v2 = tuple(self._verts[(i + 1) % len(self._verts)])

            # Fast-path: the two segments share at least one endpoint
            shared = {p1, p2}.intersection({v1, v2})
            if shared:
                # -- Colinear ?  If yes, still need to check real overlap
                if (
                        self._orientation(p1, p2, v1) == 0
                        and self._orientation(p1, p2, v2) == 0
                ):
                    if self._segments_overlap_interior(p1, p2, v1, v2):
                        return True  # positive-length overlap ⇒ intersection
                # Not colinear → only touch at common vertex ⇒ allowed
                continue

            # No shared endpoints – use the full intersection test
            if self._line_segments_intersect(p1, p2, v1, v2):
                return True

        return False

    def _segments_overlap_interior(
            self,
            a: Tuple[float, float],
            b: Tuple[float, float],
            c: Tuple[float, float],
            d: Tuple[float, float],
            eps: float = 1e-10,
    ) -> bool:
        """
        Segments a-b and c-d are assumed colinear.
        Return True iff they overlap by more than a single point.
        """
        # Project onto the dominant axis to measure one-dim overlap
        if abs(a[0] - b[0]) >= abs(a[1] - b[1]):
            s1 = sorted([a[0], b[0]])
            s2 = sorted([c[0], d[0]])
        else:
            s1 = sorted([a[1], b[1]])
            s2 = sorted([c[1], d[1]])

        overlap_len = min(s1[1], s2[1]) - max(s1[0], s2[0])
        return overlap_len > eps  # positive-length ⇒ real overlap

    def _is_angle_in_slice(self, angle: float, start_angle: float, end_angle: float) -> bool:
        """检查角度是否在切片内（处理角度环绕问题）"""

        # 规范化角度到[0, 2π]
        def normalize_angle(a):
            while a < 0:
                a += 2 * np.pi
            while a >= 2 * np.pi:
                a -= 2 * np.pi
            return a

        angle = normalize_angle(angle)
        start_angle = normalize_angle(start_angle)
        end_angle = normalize_angle(end_angle)

        if start_angle <= end_angle:
            # 正常情况
            return start_angle <= angle <= end_angle
        else:
            # 跨越0点的情况
            return angle >= start_angle or angle <= end_angle

    def _find_bisector_boundary_intersection(self, origin: np.ndarray, bisector_angle: float, max_distance: float) -> \
            Tuple[float, float]:
        """找到角平分线与边界的最近交点"""
        # 角平分线方向向量
        direction = np.array([np.cos(bisector_angle), np.sin(bisector_angle)])

        closest_intersection = None
        min_distance = float('inf')

        # 检查与每条边界边的交点
        for i in range(len(self._verts)):
            edge_start = self._verts[i]
            edge_end = self._verts[(i + 1) % len(self._verts)]

            # 计算射线与线段的交点
            intersection = self._ray_segment_intersection(origin, direction, edge_start, edge_end)

            if intersection is not None:
                distance = np.linalg.norm(intersection - origin)
                if distance <= max_distance and distance < min_distance and distance > 1e-10:
                    min_distance = distance
                    closest_intersection = intersection

        if closest_intersection is not None:
            return tuple(closest_intersection)
        else:
            # 如果没有找到交点，返回射线上的最远点
            farthest_point = origin + direction * max_distance
            return tuple(farthest_point)

    def _ray_segment_intersection(self, ray_origin: np.ndarray, ray_direction: np.ndarray,
                                  segment_start: np.ndarray, segment_end: np.ndarray) -> np.ndarray:
        """计算射线与线段的交点"""
        # 线段方向向量
        segment_vec = segment_end - segment_start

        # 检查平行性
        cross_product = ray_direction[0] * segment_vec[1] - ray_direction[1] * segment_vec[0]
        if abs(cross_product) < 1e-10:
            return None  # 平行或共线

        # 计算参数
        to_segment_start = segment_start - ray_origin
        t = (to_segment_start[0] * segment_vec[1] - to_segment_start[1] * segment_vec[0]) / cross_product
        s = (to_segment_start[0] * ray_direction[1] - to_segment_start[1] * ray_direction[0]) / cross_product

        # 检查交点是否在射线和线段上
        if t >= 0 and 0 <= s <= 1:
            intersection = ray_origin + t * ray_direction
            return intersection

        return None

    def _orientation(self, p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> int:
        """
        Find orientation of ordered triplet (p, q, r).
        Returns:
        0 --> p, q and r are collinear
        1 --> Clockwise
        2 --> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - \
              (q[0] - p[0]) * (r[1] - q[1])

        if abs(val) < 1e-10: return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    def _line_segments_intersect(self, p1: Tuple[float, float], q1: Tuple[float, float],
                                 p2: Tuple[float, float], q2: Tuple[float, float]) -> bool:
        """
        A robust function to check if line segment 'p1q1' and 'p2q2' intersect.
        This handles all general, collinear, and touching cases.
        """
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        # General case: segments cross each other
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases for collinear points
        # p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if o1 == 0 and self._point_on_line_segment(np.array(p2), np.array(p1), np.array(q1)):
            return True

        # p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if o2 == 0 and self._point_on_line_segment(np.array(q2), np.array(p1), np.array(q1)):
            return True

        # p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if o3 == 0 and self._point_on_line_segment(np.array(p1), np.array(p2), np.array(q2)):
            return True

        # p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if o4 == 0 and self._point_on_line_segment(np.array(q1), np.array(p2), np.array(q2)):
            return True

        return False  # Doesn't fall in any of the above cases
