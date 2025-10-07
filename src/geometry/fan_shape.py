import math
import numpy as np
from typing import List, Tuple, Optional

from src.utils.segment import ray_segment_intersection
from .sector_point import SectorPoint


class FanShape:
    def __init__(self, v_right, v_ref, v_left, base_L, beta=6, g=3):
        """
        Args:
            v_right (np.ndarray | Tuple[float, float]): right neighbor of v_ref
            v_ref   (np.ndarray | Tuple[float, float]): reference vertex v0
            v_left  (np.ndarray | Tuple[float, float]): left neighbor of v_ref
            base_L  (float):     local average edge length around v_ref
            beta    (int):       radius multiplier (default 6)
            g       (int):       number of sector slices (default 3)
        """
        self.v_ref = np.asarray(v_ref, dtype=float)
        self.v_right = np.asarray(v_right, dtype=float)
        self.v_left = np.asarray(v_left, dtype=float)

        # Parameters
        self.beta = beta
        self.g = g

        # Sector ray endpoints (g+1) and radius bound R
        self.fan_vertices: List[Tuple[float, float]] = []
        self.radius = self.beta * base_L
        self._init_fan_shapes(np.asarray(v_right, dtype=float),
                              self.v_ref,
                              np.asarray(v_left, dtype=float))

    # ------------------------------------------------------------------ #
    # internal helpers (kept local, no external symbols are exported)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _angle(vec):
        """Return polar angle in [0, 2π)."""
        return (math.atan2(vec[1], vec[0]) + 2 * math.pi) % (2 * math.pi)

    @staticmethod
    def _cw_diff(a, b):
        """Clockwise angular difference a→b in [0, 2π)."""
        return (a - b) % (2 * math.pi)

    @staticmethod
    def _ccw_diff(a, b):
        """Counter-clockwise angular difference a→b in [0, 2π)."""
        return (b - a) % (2 * math.pi)

    @staticmethod
    def _short_arc(start: float, end: float):
        """
        Return the signed shortest rotation from start→end.
        Returns (span, sign):
        - span: absolute angle in (0, π] of the short arc length.
        - sign: +1 if the short arc is CCW from start to end, -1 if CW.
        """
        two_pi = 2 * math.pi
        ccw = (end - start) % two_pi
        if ccw <= math.pi:
            return ccw, +1  # CCW short arc
        else:
            return (two_pi - ccw), -1  # CW short arc

    @staticmethod
    def _ang_center(start: float, end: float) -> float:
        """Return the clockwise mid-angle of the sector (exclusive of endpoints)."""
        cw_span = (start - end) % (2 * math.pi)
        return (start - cw_span / 2.0) % (2 * math.pi)

    @staticmethod
    def _ang_center_oriented(start: float, end: float, sign: int) -> float:
        """
        Return mid-angle along the oriented arc from start to end.
        sign = +1 means CCW; sign = -1 means CW.
        """
        two_pi = 2 * math.pi
        if sign > 0:
            span = (end - start) % two_pi
            return (start + span / 2.0) % two_pi
        else:
            span = (start - end) % two_pi
            return (start - span / 2.0) % two_pi

    @staticmethod
    def _ang_distance(a: float, b: float) -> float:
        """Smallest absolute angular distance between two angles in radians."""
        two_pi = 2 * math.pi
        diff = (a - b + math.pi) % two_pi - math.pi
        return abs(diff)

    # ------------------------------------------------------------------ #
    #               1. build g sector rays self.fan_vertices
    # ------------------------------------------------------------------ #
    def _init_fan_shapes(self, v_right, v_ref, v_left):
        """
        Build g+1 ray end points along the SHORT arc between vr1 and vl1.
        We orient the arc from angle_right (vr1 direction) towards angle_left (vl1)
        along the shortest path on the unit circle. The orientation is captured
        by self._sector_sign: +1 for CCW, -1 for CW. Sector i occupies the open
        interval between rays i and i+1 along this oriented short arc.
        """
        # Angles of the two neighbor directions relative to v_ref
        angle_right = self._angle(v_right - v_ref)
        angle_left = self._angle(v_left - v_ref)

        # Short-arc span and orientation (≤ π)
        span, sign = self._short_arc(angle_right, angle_left)
        self._sector_sign = sign
        self._sector_span = span

        slice_angle = span / self.g if self.g > 0 else 0.0

        # Generate g+1 ray endpoints on the circle of radius R following orientation
        self.fan_vertices.clear()
        for i in range(self.g + 1):
            if sign > 0:  # CCW
                ang = (angle_right + i * slice_angle) % (2 * math.pi)
            else:  # CW
                ang = (angle_right - i * slice_angle) % (2 * math.pi)
            dx, dy = math.cos(ang) * self.radius, math.sin(ang) * self.radius
            self.fan_vertices.append((v_ref[0] + dx, v_ref[1] + dy))

    # ------------------------------------------------------------------ #
    #               2. pick sector representatives from boundary
    # ------------------------------------------------------------------ #
    def process(self, boundary_vertices: List[Tuple[float, float]]):
        """
        Two-stage selection for clarity and correctness:
        1) Collect all candidates inside the SHORT arc and within radius R, and
           construct SectorPoint objects with full diagnostic attributes.
        2) Assign candidates to sectors using strict open membership; per-sector
           select the best by distance, then by closeness to sector center.
        3) Optional bisector coverage (g==3): if the coverage gate passes for S2,
           merge three-edge candidates by sector with the existing winners using
           the same per-sector tie-breaking (NO wholesale replacement).
        """
        eps = 1e-10
        two_pi = 2 * math.pi
        v_ref = self.v_ref

        # Pre-compute ray angles (g+1) for sector borders; angles follow oriented short arc
        fan_angles = [self._angle(np.asarray(v, dtype=float) - v_ref)
                      for v in self.fan_vertices]

        # Helper: oriented delta (distance along oriented arc from start→x)
        def oriented_delta(start: float, x: float) -> float:
            if self._sector_sign > 0:  # CCW
                return (x - start) % two_pi
            else:  # CW
                return (start - x) % two_pi

        # Helper: strict open-sector membership (exclude endpoints)
        def in_open_sector(angle_x: float, start: float, end: float) -> bool:
            d = oriented_delta(start, angle_x)
            span = oriented_delta(start, end)
            return (d > eps) and (d < span - eps)

        # Exclude v0, vr1, vl1 explicitly
        exclude_set = {
            tuple(v_ref),
            tuple(self.v_right),
            tuple(self.v_left)
        }

        # -------------------- Stage 1: collect all candidates --------------------
        angle_right = self._angle(self.v_right - v_ref)
        candidates: List[SectorPoint] = []
        for v in boundary_vertices:
            if v in exclude_set:
                continue
            vec = np.asarray(v, dtype=float) - v_ref
            dist = float(np.linalg.norm(vec))
            ang = self._angle(vec)
            delta = oriented_delta(angle_right, ang)
            in_r = dist <= self.radius + eps
            in_arc = (delta > eps) and (delta < self._sector_span - eps)
            if in_r and in_arc:
                candidates.append(SectorPoint(vertex=tuple(v), dist=dist, angle_abs=ang,
                                              delta_oriented=delta, in_radius=True, in_short_arc=True))

        # If no candidates, return all None
        if not candidates:
            return [None] * self.g

        # -------------------- Stage 2: sector assignment and winners -------------
        winners: List[Optional[SectorPoint]] = [None] * self.g
        for sp in candidates:
            # Determine which sector this point belongs to (strict open)
            assigned = False
            for i in range(self.g):
                start = fan_angles[i]
                end = fan_angles[i + 1]
                if in_open_sector(sp.angle_abs, start, end):
                    center = self._ang_center_oriented(start, end, self._sector_sign)
                    sp.sector_index = i
                    sp.center_delta = self._ang_distance(sp.angle_abs, center)
                    if sp.better_than(winners[i]):
                        winners[i] = sp
                    assigned = True
                    break
            # If a point is not assigned to any sector due to numeric edge cases,
            # we simply ignore it.
            if not assigned:
                continue

        # -------------------- Stage 3: optional bisector coverage (g == 3) ------
        if self.g == 3:
            # Compute mid-angle along the oriented short arc
            bisector_ang = (angle_right + self._sector_sign * (self._sector_span / 2.0)) % two_pi
            ray_origin = v_ref
            ray_dir = np.array([math.cos(bisector_ang), math.sin(bisector_ang)], dtype=float)

            # Find nearest intersection within radius (excluding endpoints)
            nearest = None  # (distance, edge_index, intersection_point)
            N = len(boundary_vertices)
            for e in range(N):
                a = np.asarray(boundary_vertices[e], dtype=float)
                b = np.asarray(boundary_vertices[(e + 1) % N], dtype=float)
                inter = ray_segment_intersection(ray_origin, ray_dir, a, b)
                if inter is None:
                    continue
                if (np.linalg.norm(inter - a) < 1e-8) or (np.linalg.norm(inter - b) < 1e-8):
                    continue
                d = float(np.linalg.norm(inter - ray_origin))
                if d <= self.radius + eps:
                    if nearest is None or d < nearest[0] - eps:
                        nearest = (d, e, inter)

            # Evaluate coverage gate for S2
            if nearest is not None:
                d_bis, edge_idx, _ = nearest
                cur_s2_dist = float("inf") if winners[1] is None else winners[1].dist
                coverage_gate = (winners[1] is None) or (d_bis + 1e-12 < cur_s2_dist)

                if coverage_gate:
                    # Build three consecutive vertices centered around the intersected edge
                    idxs = [(edge_idx - 1) % N, edge_idx % N, (edge_idx + 1) % N]
                    triple = [boundary_vertices[i] for i in idxs]

                    # Convert to SectorPoint and merge by sector
                    for v in triple:
                        if v in exclude_set:
                            continue
                        vec = np.asarray(v, dtype=float) - v_ref
                        dist = float(np.linalg.norm(vec))
                        if dist > self.radius + eps:
                            continue
                        ang = self._angle(vec)
                        delta = oriented_delta(angle_right, ang)
                        if not ((delta > eps) and (delta < self._sector_span - eps)):
                            continue
                        sp = SectorPoint(vertex=tuple(v), dist=dist, angle_abs=ang,
                                         delta_oriented=delta, in_radius=True, in_short_arc=True)
                        # Assign to exact sector and try merge
                        for i in range(self.g):
                            start = fan_angles[i]
                            end = fan_angles[i + 1]
                            if in_open_sector(sp.angle_abs, start, end):
                                center = self._ang_center_oriented(start, end, self._sector_sign)
                                sp.sector_index = i
                                sp.center_delta = self._ang_distance(sp.angle_abs, center)
                                # S2 is protected by the coverage gate; S1/S3 are merged only if better.
                                if sp.better_than(winners[i]):
                                    winners[i] = sp
                                break

        # -------------------- Finalize: ensure unique vertices across sectors ----
        seen = set()
        ordered: List[Optional[Tuple[float, float]]] = []
        for i in range(self.g):
            sp = winners[i]
            if sp is None:
                ordered.append(None)
            else:
                if sp.vertex in seen:
                    ordered.append(None)
                else:
                    ordered.append(sp.vertex)
                    seen.add(sp.vertex)

        return ordered
