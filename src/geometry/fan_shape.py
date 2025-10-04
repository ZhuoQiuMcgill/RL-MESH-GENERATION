import math
import numpy as np
from typing import List, Tuple, Optional

from src.utils.segment import ray_segment_intersection


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
    def _ang_center(start: float, end: float) -> float:
        """Return the clockwise mid-angle of the sector (exclusive of endpoints)."""
        cw_span = (start - end) % (2 * math.pi)
        return (start - cw_span / 2.0) % (2 * math.pi)

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
        Build g+1 ray end points (including the two sides) that define g CW sectors.
        fan_vertices[0] aligns with the right neighbor direction,
        fan_vertices[-1] aligns with the left neighbor direction.
        """
        # Angles of the two neighbor directions relative to v_ref
        angle_right = self._angle(v_right - v_ref)
        angle_left = self._angle(v_left - v_ref)

        # CW span and per-sector span
        total_angle = self._cw_diff(angle_right, angle_left)
        slice_angle = total_angle / self.g if self.g > 0 else 0.0

        # Generate g+1 ray endpoints on the circle of radius R
        self.fan_vertices.clear()
        for i in range(self.g + 1):
            ang = (angle_right - i * slice_angle) % (2 * math.pi)
            dx, dy = math.cos(ang) * self.radius, math.sin(ang) * self.radius
            self.fan_vertices.append((v_ref[0] + dx, v_ref[1] + dy))

    # ------------------------------------------------------------------ #
    #               2. pick sector representatives from boundary
    # ------------------------------------------------------------------ #
    def process(self, boundary_vertices: List[Tuple[float, float]]):
        """
        Select at most one representative vertex for each sector slice.
        - Open sectors: endpoints (0 and θ) are excluded; vr1 and vl1 are excluded.
        - Radius cap: candidates must satisfy ||u - v0|| <= R.
        - Tie-breaking: prefer smaller distance; if equal (within eps), pick the
          one closer to the sector center angle.
        - Optional coverage: if g==3 and the bisector ray intersects the boundary
          within R, use the intersection edge to provide a more stable triple when
          S2 is empty or the intersection is strictly closer than the chosen S2.
        """
        eps = 1e-10
        two_pi = 2 * math.pi
        v_ref = self.v_ref

        # Pre-compute ray angles (g+1) for sector borders
        fan_angles = [self._angle(np.asarray(v, dtype=float) - v_ref)
                      for v in self.fan_vertices]

        # Helper: strict open-sector membership (exclude endpoints)
        def in_open_sector(angle_x: float, start: float, end: float) -> bool:
            cw_to_x = self._cw_diff(start, angle_x)
            cw_to_end = self._cw_diff(start, end)
            # strictly inside (0, span), exclude ~0 and ~span
            return (cw_to_x > eps) and (cw_to_x < cw_to_end - eps)

        # Exclude v0, vr1, vl1 explicitly
        exclude_set = {
            tuple(v_ref),
            tuple(self.v_right),
            tuple(self.v_left)
        }

        # Per sector best candidate
        best_vertices: List[Optional[Tuple[float, float]]] = [None] * self.g
        best_dists: List[float] = [float("inf")] * self.g
        best_angles: List[Optional[float]] = [None] * self.g

        # Iterate sectors
        for i in range(self.g):
            start = fan_angles[i]
            end = fan_angles[i + 1]
            center = self._ang_center(start, end)

            for v in boundary_vertices:
                if v in exclude_set:
                    continue
                vec = np.asarray(v, dtype=float) - v_ref
                dist = float(np.linalg.norm(vec))
                if dist > self.radius + eps:
                    continue
                ang = self._angle(vec)
                if not in_open_sector(ang, start, end):
                    continue

                if best_vertices[i] is None or dist < best_dists[i] - eps:
                    best_vertices[i] = v
                    best_dists[i] = dist
                    best_angles[i] = ang
                elif abs(dist - best_dists[i]) <= eps:
                    # tie-break: closer to sector center angle
                    if best_angles[i] is None:
                        best_vertices[i] = v
                        best_dists[i] = dist
                        best_angles[i] = ang
                    else:
                        cur_delta = self._ang_distance(best_angles[i], center)
                        new_delta = self._ang_distance(ang, center)
                        if new_delta + 1e-12 < cur_delta:
                            best_vertices[i] = v
                            best_dists[i] = dist
                            best_angles[i] = ang

        # Optional bisector coverage (only meaningful when g == 3)
        if self.g == 3:
            # Compute CW mid-angle between right and left directions
            angle_right = self._angle(self.v_right - v_ref)
            angle_left = self._angle(self.v_left - v_ref)
            total_angle = self._cw_diff(angle_right, angle_left)
            bisector_ang = (angle_right - total_angle / 2.0) % two_pi

            # Ray origin and direction (unit)
            ray_origin = v_ref
            ray_dir = np.array([math.cos(bisector_ang), math.sin(bisector_ang)], dtype=float)

            # Find nearest intersection within radius, excluding endpoints
            nearest = None  # (distance, edge_index, intersection_point)
            N = len(boundary_vertices)
            for e in range(N):
                a = np.asarray(boundary_vertices[e], dtype=float)
                b = np.asarray(boundary_vertices[(e + 1) % N], dtype=float)

                inter = ray_segment_intersection(ray_origin, ray_dir, a, b)
                if inter is None:
                    continue
                # exclude touching endpoints
                if (np.linalg.norm(inter - a) < 1e-8) or (np.linalg.norm(inter - b) < 1e-8):
                    continue
                d = float(np.linalg.norm(inter - ray_origin))
                if d <= self.radius + eps:
                    if nearest is None or d < nearest[0] - eps:
                        nearest = (d, e, inter)

            # If intersection found, evaluate coverage criteria
            if nearest is not None:
                d_bis, edge_idx, _ = nearest
                s2_empty = best_vertices[1] is None
                s2_better = (not s2_empty) and (d_bis + 1e-12 < best_dists[1])
                if s2_empty or s2_better:
                    # Build three consecutive vertices centered around the intersected edge
                    idxs = [(edge_idx - 1) % N, edge_idx % N, (edge_idx + 1) % N]
                    triple = [boundary_vertices[i] for i in idxs]

                    # Filter by open angle domain and radius cap
                    angle_right = self._angle(self.v_right - v_ref)
                    angle_left = self._angle(self.v_left - v_ref)
                    # For mapping to sectors, we only need their angles
                    items = []  # (angle, vertex)
                    for v in triple:
                        if v in exclude_set:
                            continue
                        vec = np.asarray(v, dtype=float) - v_ref
                        dist = float(np.linalg.norm(vec))
                        if dist > self.radius + eps:
                            continue
                        ang = self._angle(vec)
                        # check (0, θ) open domain: CW between right and left
                        cw_to_ang = self._cw_diff(angle_right, ang)
                        cw_to_left = self._cw_diff(angle_right, angle_left)
                        if (cw_to_ang > eps) and (cw_to_ang < cw_to_left - eps):
                            items.append((ang, v))

                    # Sort by angle (strictly increasing CW from right to left)
                    items.sort(key=lambda x: x[0])

                    # Map to S1, S2, S3 by increasing angle; fill None if missing
                    rep = [None, None, None]
                    for i, (_, v) in enumerate(items[:3]):
                        rep[i] = v

                    best_vertices = rep
                    best_angles = [x[0] if x is not None else None for x in [(items[0] if len(items) > 0 else None), (items[1] if len(items) > 1 else None), (items[2] if len(items) > 2 else None)]
                    best_dists = [float(np.linalg.norm((np.asarray(v, dtype=float) - v_ref))) if v is not None else float("inf") for v in best_vertices]

        # Ensure strictly increasing angles across sectors (best-effort)
        # With open sectors and tie-breaking, duplicates should rarely happen.
        # We still enforce by dropping duplicates if any.
        seen = set()
        ordered: List[Optional[Tuple[float, float]]] = []
        for i in range(self.g):
            v = best_vertices[i]
            if v is None:
                ordered.append(None)
            else:
                if v in seen:
                    ordered.append(None)
                else:
                    ordered.append(v)
                    seen.add(v)

        return ordered
