import numpy as np
from typing import List, Tuple, Any


class Boundary:
    """Closed polygon boundary with clockwise-ordered vertices."""

    def __init__(self, vertices: List[Tuple[float, float]]):
        """Initialize with a list of (x, y) tuples in clockwise order."""
        if len(vertices) < 3:
            raise ValueError("A boundary must have at least three vertices.")
        self._verts = np.asarray(vertices, dtype=float)  # shape (N, 2)

    # ------------------------------------------------------------
    # Read‑only helpers
    # ------------------------------------------------------------
    def get_vertices(self) -> List[Tuple[float, float]]:
        """Return a *copy* of vertices as list of tuples."""
        return [tuple(v) for v in self._verts]

    def get_edges(self) -> list[tuple[tuple[Any, ...], tuple[Any, ...]]]:
        """Return edges as list of (prev, curr) vertex tuples."""
        return [(tuple(self._verts[i - 1]), tuple(self._verts[i]))
                for i in range(len(self._verts))]

    # ------------------------------------------------------------
    # Interior‑angle utilities
    # ------------------------------------------------------------
    def _interior_angles(self) -> np.ndarray:
        """Vectorised interior angles (deg) for each vertex."""
        v_prev = np.roll(self._verts, 1, axis=0)
        v_next = np.roll(self._verts, -1, axis=0)
        a = v_prev - self._verts  # BA
        b = v_next - self._verts  # BC
        dot = np.einsum('ij,ij->i', a, b)
        norm_prod = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        # guard against divide‑by‑zero
        cos_theta = np.clip(dot / np.where(norm_prod == 0, 1, norm_prod), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def get_max_interior_angles(self):
        """Return (vertex, angle) with the largest interior angle."""
        angles = self._interior_angles()
        idx = int(angles.argmax())
        return tuple(self._verts[idx]), float(angles[idx])

    def get_min_interior_angles(self):
        """Return (vertex, angle) with the smallest interior angle."""
        angles = self._interior_angles()
        idx = int(angles.argmin())
        return tuple(self._verts[idx]), float(angles[idx])

    # ------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------
    def remove_vertex(self, vertex: Tuple[float, float]):
        """Remove *vertex* if present; silently ignore otherwise."""
        mask = (self._verts == vertex).all(axis=1)
        if mask.any():
            self._verts = self._verts[~mask]

    def insert_vertex(self, vertex: Tuple[float, float], position: int):
        """Insert *vertex* at *position* (0 ≤ pos ≤ len)."""
        if not (0 <= position <= len(self._verts)):
            raise IndexError("position out of range")
        self._verts = np.insert(self._verts, position, vertex, axis=0)

    def part_of_boundary(self, vertex):
        """
        TODO: 该函数将查找该点是否存为当前的边界组成点
        :param vertex: (X, Y)
        :return: bool
        """

    def vertex_inside_boundary(self, vertex):
        """
        TODO: 该函数严格判定点是否处于多边形边界的内部。
        判断条件：
        点为边界组成点->False
        点位于边界的某个边上->False
        点位于边界内部->True
        点位于边界外部->False

        注意：边界self.vertices为顺时针顺序，即(self.vertices[n], self.vertices[n+1])的右侧为内部
        :param vertex: (X, Y)
        :return: bool
        """
        pass

    def edge_cross(self, edge):
        """
        TODO: 该函数严格判定输入的边是否与任意边界的边有相交。
        判断条件：
        输入边的一个组成点为边界的组成点->False
        输入边与任意一个边界的边相交->True
        输入边的任意一个点位于边界都某一个边上->True

        :param edge: ((X1, Y1), (X2, Y2))
        :return: bool
        """
        pass

    def edge_inside_boundary(self, edge):
        """
        TODO: 该函数严格判定输入的边是否完整位于边界内。
        判断条件：
        输入边的两个点都位于边界内->True
        输入边与边界相交->False
        输入边的一个点为边界组成点，另一个点位于边界内->True
        输入边的一个点为边界组成点，另一个点位于边界外->False

        :param edge: ((X1, Y1), (X2, Y2))
        :return: bool
        """
        pass

