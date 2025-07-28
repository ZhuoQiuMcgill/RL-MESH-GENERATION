import numpy as np
from typing import List, Tuple
from math import inf
from .fan_shape import FanShape
import math

from src.utils.angle import get_interior_angle, euclidean_distance
from src.utils.segment import (
    ray_segment_intersection,
    orientation,
    point_on_line_segment,
    line_segments_intersect,
    segments_overlap_interior,
    point_to_segment_distance
)


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
    def get_vertices(self):
        """
        返回顶点的副本作为元组列表

        Returns:
            List[Tuple[float, float]]: 顶点坐标列表的副本
        """
        return [tuple(v) for v in self._verts]

    def get_neighbors(self, ref_index, n, get_type="default"):
        vertices = []
        for i in range(-n, n + 1):
            if get_type == "exclude ref" and i == 0:
                continue
            vertices.append(self.get_vertex_by_index(ref_index + i))
        return vertices

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

    def get_edges(self):
        """
        返回边的列表，每条边由(前一个顶点, 当前顶点)元组表示

        Returns:
            List[Tuple[Tuple[float, float], Tuple[float, float]]]: 边的列表
        """
        return [(tuple(self._verts[i - 1]), tuple(self._verts[i]))
                for i in range(len(self._verts))]

    def get_closest_edge_distance(self, vertex, ignore_edges):
        """
        Return the shortest distance from *vertex* to any boundary edge, skipping
        edges listed in *ignore_edges*.  Edge comparison is direction-agnostic and
        works whether endpoints are tuples or NumPy arrays.
        """

        def to_tuple(pt):
            """Convert array-like point to plain (x, y) tuple of floats."""
            if isinstance(pt, np.ndarray):
                return tuple(float(x) for x in pt)
            return tuple(pt)

        def normalize(edge):
            """Return direction-free, hashable representation of an edge."""
            p1, p2 = edge
            return frozenset((to_tuple(p1), to_tuple(p2)))

        normalized_ignore = {normalize(edge) for edge in ignore_edges}

        min_distance = inf
        for seg_start, seg_end in self.get_edges():
            if normalize((seg_start, seg_end)) in normalized_ignore:
                continue
            distance = point_to_segment_distance(vertex, seg_start, seg_end)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def get_closest_vertex_distance(self, vertex, ignore_vertices):
        """
        Return the shortest distance from *vertex* to any boundary vertex, skipping
        vertices listed in *ignore_vertices*.
        """

        def to_tuple(pt):
            """Convert array-like point to plain (x, y) tuple of floats."""
            if isinstance(pt, np.ndarray):
                return tuple(float(x) for x in pt)
            return tuple(pt)

        # Normalize ignore_vertices to tuples for consistent comparison
        normalized_ignore = {to_tuple(v) for v in ignore_vertices}

        min_distance = inf
        vertex_array = np.asarray(vertex, dtype=float)

        for boundary_vertex in self.get_vertices():
            if to_tuple(boundary_vertex) in normalized_ignore:
                continue
            boundary_vertex_array = np.asarray(boundary_vertex, dtype=float)
            distance = float(np.linalg.norm(vertex_array - boundary_vertex_array))
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def get_avg_neighbor_length(self, v: int, n: int) -> float:
        """
        Calculate the average edge length of n neighbor edges from the center vertex v.
        
        Args:
            v: The index of the target vertex (center)
            n: The number of neighbor edges on each side
            
        Returns:
            float: The average length of 2n edges centered around vertex v
            
        Example:
            If n=2, calculates average length of edges: (v-2,v-1), (v-1,v0), (v0,v1), (v1,v2)
        """
        if not isinstance(v, int):
            raise TypeError("vertex index v must be int")
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        total_length = 0.0
        edge_count = 2 * n

        # Calculate lengths of edges from (v-n) to (v+n)
        for i in range(-n, n):
            # Get current vertex and next vertex using existing boundary methods
            current_vertex = self.get_vertex_by_index(v + i)
            next_vertex = self.get_vertex_by_index(v + i + 1)

            # Calculate edge length using existing euclidean_distance function
            edge_length = euclidean_distance(current_vertex, next_vertex)
            total_length += edge_length

        return total_length / edge_count

    def get_avg_vertex_distance(self) -> float:
        """
        Calculate the average edge length of all boundary edges.
        
        Returns:
            float: The average length of all edges in the boundary
        """
        if self.size() < 2:
            return 0.0

        total_length = 0.0
        edge_count = self.size()

        # Calculate lengths of all boundary edges
        for i in range(edge_count):
            current_vertex = self.get_vertex_by_index(i)
            next_vertex = self.get_vertex_by_index(i + 1)

            # Calculate edge length using existing euclidean_distance function
            edge_length = euclidean_distance(current_vertex, next_vertex)
            total_length += edge_length

        return total_length / edge_count

    def get_min_and_critical_area(self):
        """
        Calculate min_area and critical_area based on boundary vertex geometry.
        Returns:
            tuple: (min_area, critical_area) as float values
        """
        # Step 1 - Calculate edge lengths
        edge_lengths = []
        for i in range(self.size()):
            current_vertex = self.get_vertex_by_index(i)
            next_vertex = self.get_vertex_by_index(i + 1)  # handles wraparound via modulo
            edge_length = euclidean_distance(current_vertex, next_vertex)
            edge_lengths.append(edge_length)

        # Step 2 - Sort edge lengths
        sorted_lengths = sorted(edge_lengths)

        # Ensure we have at least 2 edges for indexing operations
        if len(sorted_lengths) < 2:
            raise ValueError("Boundary must have at least 2 edges")

        # Step 3 - Calculate base parameters
        L = sum(edge_lengths) / len(edge_lengths)  # average edge length

        min_L = min(L / math.sqrt(2), sorted_lengths[1])  # second shortest edge
        max_L = min(sorted_lengths[-2], 2 * L)  # second longest edge

        # Step 4 - Calculate area range bounds
        lower_bound = min_L
        upper_bound = (max_L + 3 * min_L) / 4

        # Step 5 - Compute final area thresholds
        min_area = lower_bound ** 2
        critical_area = upper_bound ** 2

        return min_area, critical_area

    def size(self) -> int:
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
        return (get_interior_angle(right_point_1, ref_point, left_point_1) +
                get_interior_angle(right_point_2, ref_point, left_point_2)) / 2

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

    def part_of_boundary(self, vertex: Tuple[float, float]):
        vertex_array = np.array(vertex, dtype=float)

        distances = np.linalg.norm(self._verts - vertex_array, axis=1)
        return np.any(distances < 1e-10)

    def vertex_inside_boundary(self, vertex: Tuple[float, float]) -> bool:
        if self.part_of_boundary(vertex):
            return False

        if self._point_on_boundary_edge(vertex):
            return False

        return self._point_in_polygon(vertex)

    def vertex_inside_action_space(self, vertex: Tuple[float, float], ref_index, alpha, n) -> bool:
        ref_point = self.get_vertex_by_index(ref_index)
        ref_right = self.get_vertex_by_index(ref_index - 1)
        ref_left = self.get_vertex_by_index(ref_index + 1)
        r = self.get_avg_neighbor_length(ref_index, n) * alpha

        if euclidean_distance(ref_point, vertex) > r:
            return False

        def _vec(a, b):
            return b[0] - a[0], b[1] - a[1]

        def _norm(v):
            return math.hypot(v[0], v[1])

        def _unit(v):
            l = _norm(v)
            return v[0] / l, v[1] / l

        v_r = _vec(ref_point, ref_right)
        v_l = _vec(ref_point, ref_left)
        v_c = _vec(ref_point, vertex)

        if _norm(v_c) == 0:
            return False

        u_r = _unit(v_r)
        u_l = _unit(v_l)
        u_c = _unit(v_c)

        def _clamp(x):
            return max(-1.0, min(1.0, x))

        interior = math.acos(_clamp(u_r[0] * u_l[0] + u_r[1] * u_l[1]))
        a_rc = math.acos(_clamp(u_r[0] * u_c[0] + u_r[1] * u_c[1]))
        a_lc = math.acos(_clamp(u_l[0] * u_c[0] + u_l[1] * u_c[1]))

        if abs((a_rc + a_lc) - interior) <= 1e-6 and a_rc <= interior and a_lc <= interior:
            return False

        return True

    def edge_cross(self, edge: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        return self._edge_intersects_boundary(edge)

    def get_area(self) -> float:
        if len(self._verts) < 3:
            return 0.0

        x = self._verts[:, 0]
        y = self._verts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross_product = x * y_next - x_next * y
        area = 0.5 * abs(np.sum(cross_product))

        return area

    def get_fan_points(self, reference_vertex_index: int, n, beta=6, g=3) -> List[Tuple[float, float]]:
        base_length = self.get_avg_neighbor_length(reference_vertex_index, n)
        fan_shape = FanShape(
            self.get_vertex_by_index(reference_vertex_index - 1),
            self.get_vertex_by_index(reference_vertex_index),
            self.get_vertex_by_index(reference_vertex_index + 1),
            base_length,
            beta=beta,
            g=g
        )
        return fan_shape.process(self.get_vertices())

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

            if point_on_line_segment(point_array, v1, v2):
                return True

        return False

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
        x_inters = 0

        p1x, p1y = self._verts[0]
        for i in range(1, n + 1):
            p2x, p2y = self._verts[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_inters:
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
        • Collinear overlap by a positive length (including
          being exactly the same as an existing boundary edge).
        """
        p1, p2 = edge

        for i in range(len(self._verts)):
            v1 = tuple(self._verts[i])
            v2 = tuple(self._verts[(i + 1) % len(self._verts)])

            # Fast-path: the two segments share at least one endpoint
            shared = {p1, p2}.intersection({v1, v2})
            if shared:
                # -- Collinear ?  If yes, still need to check real overlap
                if (
                        orientation(p1, p2, v1) == 0
                        and orientation(p1, p2, v2) == 0
                ):
                    if segments_overlap_interior(p1, p2, v1, v2):
                        return True  # positive-length overlap ⇒ intersection
                # Not collinear → only touch at common vertex ⇒ allowed
                continue

            # No shared endpoints – use the full intersection test
            if line_segments_intersect(p1, p2, v1, v2):
                return True

        return False
