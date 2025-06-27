import os
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from gymnasium import Env, spaces
from .utils.geometry import (
    calculate_polygon_area, generate_action_coordinates,
    calculate_reward_components, calculate_reference_vertex,
    get_state_components, calculate_element_quality, load_domain_from_file,
    transform_to_relative_coords
)


class MeshEnv(Env):
    """
    Mesh generation environment implementing Gymnasium interface.
    Modified to match original author's approach for better reproducibility.
    """

    def __init__(self, config: Dict):
        super(MeshEnv, self).__init__()
        self.config = config

        # Core parameters matching original implementation
        self.neighbor_num = config['environment'].get('neighbor_num', 6)  # Original uses 6
        self.radius_num = config['environment'].get('radius_num', 3)  # Original uses 3
        self.max_radius = config['environment'].get('max_radius', 2)  # Original uses 2
        self.radius = config['environment'].get('observation_radius', 4)  # Original uses 4

        # Original author's action space: [rule_type, x_coord, y_coord]
        self.action_space = spaces.Box(
            np.array([-1, -1.0, -1.0]),
            np.array([1, 1.0, 1.0]),
            dtype=np.float32
        )

        # Original author's observation space: flat array
        obs_dim = 2 * (self.neighbor_num + self.radius_num)
        self.observation_space = spaces.Box(
            low=-999, high=999,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Environment setup
        self.domain_file = config['domain']['training_domain']
        self.data_dir = config['paths']['data_dir']
        self.original_boundary = self._load_domain()
        self.original_area = calculate_polygon_area(self.original_boundary)
        self.original_boundary = self._ensure_clockwise(self.original_boundary)
        self.current_boundary = None
        self.generated_elements = []
        self.current_area_ratio = 1.0
        self.step_count = 0
        self.episode_reward = 0.0
        self.max_steps = int(config['environment'].get('max_steps', 1000))

        # Original author's reward method selection
        self.reward_method = config['environment'].get('reward_method', 10)

        # Additional tracking variables from original
        self.failed_num = 0
        self.not_valid_points = []
        self.TYPE_THRESHOLD = 0.3  # Original threshold for action type decision

        print(f"Environment initialized with neighbor_num={self.neighbor_num}, radius_num={self.radius_num}")

    def _ensure_clockwise(self, boundary: np.ndarray) -> np.ndarray:
        """Ensure boundary vertices are in clockwise order."""
        area = 0.0
        n = len(boundary)
        for i in range(n):
            j = (i + 1) % n
            area += (boundary[j][0] - boundary[i][0]) * (boundary[j][1] + boundary[i][1])

        if area > 0:  # Counter-clockwise, need to reverse
            return boundary[::-1]
        return boundary

    def _load_domain(self) -> np.ndarray:
        """Load domain from file."""
        domain_path = os.path.join(self.data_dir, self.domain_file)
        return load_domain_from_file(domain_path)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.current_boundary = self.original_boundary.copy()
        self.generated_elements = []
        self.current_area_ratio = 1.0
        self.step_count = 0
        self.episode_reward = 0.0
        self.failed_num = 0
        self.not_valid_points = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _find_reference_point(self) -> Optional[int]:
        """Find reference point with minimum boundary angle."""
        if len(self.current_boundary) < 3:
            return None

        n_boundary = len(self.current_boundary)
        min_angle = float('inf')
        ref_idx = 0

        for i in range(n_boundary):
            angles = []
            for j in range(1, min(3, n_boundary // 2) + 1):
                left_idx = (i - j) % n_boundary
                right_idx = (i + j) % n_boundary

                vec1 = self.current_boundary[left_idx] - self.current_boundary[i]
                vec2 = self.current_boundary[right_idx] - self.current_boundary[i]

                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)

            avg_angle = np.mean(angles)
            if avg_angle < min_angle:
                min_angle = avg_angle
                ref_idx = i

        return ref_idx

    def _get_observation(self) -> np.ndarray:
        """Get current state observation using correct dimensions."""
        if self.current_boundary is None or len(self.current_boundary) < 3:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        try:
            ref_idx = self._find_reference_point()
            if ref_idx is None:
                return np.zeros(self.observation_space.shape[0], dtype=np.float32)

            # Use the corrected state generation method
            state = self._get_state_from_boundary(ref_idx)

            # Ensure correct length
            expected_len = self.observation_space.shape[0]
            if len(state) != expected_len:
                print(f"Warning: State length mismatch. Expected {expected_len}, got {len(state)}")
                # Pad or truncate to match expected length
                if len(state) < expected_len:
                    state.extend([0.0] * (expected_len - len(state)))
                else:
                    state = state[:expected_len]

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"Error in observation: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def _get_state_from_boundary(self, ref_idx: int) -> np.ndarray:
        """Get state representation from boundary with fixed-length vector."""
        state = []
        n_boundary = len(self.current_boundary)
        ref_vertex = self.current_boundary[ref_idx]

        # Compute reference direction
        left_vertex = self.current_boundary[(ref_idx - 1) % n_boundary]
        right_vertex = self.current_boundary[(ref_idx + 1) % n_boundary]
        ref_direction = right_vertex - left_vertex
        ref_direction = ref_direction / (np.linalg.norm(ref_direction) + 1e-8)

        # Calculate base length
        distances = []
        for j in range(1, min(4, n_boundary // 2) + 1):
            left_idx = (ref_idx - j) % n_boundary
            right_idx = (ref_idx + j) % n_boundary
            left_dist = np.linalg.norm(self.current_boundary[left_idx] - ref_vertex)
            right_dist = np.linalg.norm(self.current_boundary[right_idx] - ref_vertex)
            distances.extend([left_dist, right_dist])
        base_length = np.mean(distances) if distances else 1.0

        # Get neighbor indices
        neighbor_indices = [(ref_idx - j) % n_boundary for j in range(1, self.neighbor_num // 2 + 1)]
        neighbor_indices += [(ref_idx + j) % n_boundary for j in range(1, self.neighbor_num // 2 + 1)]

        # Get radius neighbors and pad to fixed length
        radius_neighbors = []
        for i in range(n_boundary):
            if i != ref_idx and i not in neighbor_indices:
                dist = np.linalg.norm(self.current_boundary[i] - ref_vertex)
                if dist <= self.max_radius * base_length:
                    radius_neighbors.append(i)
        radius_neighbors = sorted(radius_neighbors,
                                  key=lambda x: np.linalg.norm(self.current_boundary[x] - ref_vertex))
        radius_neighbors = radius_neighbors[:self.radius_num]
        while len(radius_neighbors) < self.radius_num:
            radius_neighbors.append(ref_idx)

        # Transform neighbor vertices features
        for neighbor_idx in neighbor_indices:
            relative_pos = self.current_boundary[neighbor_idx] - ref_vertex
            distance = np.linalg.norm(relative_pos) / base_length
            if distance > 1e-8:
                cos_angle = np.dot(relative_pos, ref_direction) / (
                        np.linalg.norm(relative_pos) * np.linalg.norm(ref_direction)
                )
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                cross_product = np.cross(ref_direction, relative_pos)
                if cross_product < 0:
                    angle = -angle
            else:
                angle = 0.0
            state.extend([distance, angle])

        # Transform radius neighbors features
        for radius_idx in radius_neighbors:
            relative_pos = self.current_boundary[radius_idx] - ref_vertex
            distance = np.linalg.norm(relative_pos) / base_length
            if distance > 1e-8:
                cos_angle = np.dot(relative_pos, ref_direction) / (
                        np.linalg.norm(relative_pos) * np.linalg.norm(ref_direction)
                )
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                cross_product = np.cross(ref_direction, relative_pos)
                if cross_product < 0:
                    angle = -angle
            else:
                angle = 0.0
            state.extend([distance, angle])

        return np.array(state, dtype=np.float32)

    def step(self, action: np.ndarray, global_timestep: Optional[int] = None) -> Tuple[
        np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment following original logic with simplified intersection check."""
        self.step_count += 1

        if self.step_count >= self.max_steps:
            observation = self._get_observation()
            info = self._get_info()
            info.update({'termination_reason': 'max_steps_reached'})
            return observation, -1.0, False, True, info

        # Parse action following original approach
        rule_type = action[0]  # First component determines action type
        failed = True
        reward = 0.0

        try:
            ref_idx = self._find_reference_point()
            if ref_idx is None:
                observation = self._get_observation()
                info = self._get_info()
                return observation, -0.1, False, False, info

            n_boundary = len(self.current_boundary)

            # Check termination condition
            if n_boundary <= 5:
                reward = 10
                terminated = True
                if n_boundary == 4:
                    # Add final quadrilateral
                    final_element = self.current_boundary.copy()
                    self.generated_elements.append(final_element)
                observation = self._get_observation()
                info = self._get_info()
                info.update({'termination_reason': 'boundary_complete'})
                return observation, reward, terminated, False, info

            # Determine action type using original thresholds and generate mesh
            mesh = None
            action_type = None

            if rule_type <= -self.TYPE_THRESHOLD:
                # Create quad by connecting 4 consecutive boundary vertices
                mesh = np.array([
                    self.current_boundary[(ref_idx - 1) % n_boundary],
                    self.current_boundary[ref_idx],
                    self.current_boundary[(ref_idx + 1) % n_boundary],
                    self.current_boundary[(ref_idx + 2) % n_boundary],
                ])
                action_type = -1

            elif rule_type >= self.TYPE_THRESHOLD:
                # Create quad by connecting 4 consecutive boundary vertices
                mesh = np.array([
                    self.current_boundary[(ref_idx - 2) % n_boundary],
                    self.current_boundary[(ref_idx - 1) % n_boundary],
                    self.current_boundary[ref_idx],
                    self.current_boundary[(ref_idx + 1) % n_boundary],
                ])
                action_type = 1

            else:  # Type 0: Add new vertex
                new_point = self._action_to_point_constrained(action[1:], ref_idx)
                if new_point is not None:
                    # Create quad with new point and 3 boundary vertices
                    # Order: new_point, left_neighbor, ref_vertex, right_neighbor
                    mesh = np.array([
                        new_point,
                        self.current_boundary[(ref_idx - 1) % n_boundary],
                        self.current_boundary[ref_idx],
                        self.current_boundary[(ref_idx + 1) % n_boundary],
                    ])
                    action_type = 0
                else:
                    reward = -1.0
                    mesh = None

            # Validate mesh and check for boundary self-intersection after potential update
            if mesh is not None and self._validate_mesh_simple(mesh):
                # Additional validation for action_type 0: ensure new vertex is inside boundary
                if action_type == 0:
                    new_vertex = mesh[0]  # First vertex is the new one for type 0
                    if not self._is_point_inside_boundary_robust(new_vertex):
                        reward = -1.0 / max(len(self.generated_elements), 1)
                        terminated = False
                        failed = True
                        if self.step_count % 100 == 0:
                            print(f"Step {self.step_count}: Mesh rejected - new vertex outside boundary")
                    else:
                        # Continue with normal validation
                        reward, terminated, failed = self._validate_and_apply_mesh(ref_idx, mesh, action_type)
                else:
                    # For action_type -1 and 1, proceed with normal validation
                    reward, terminated, failed = self._validate_and_apply_mesh(ref_idx, mesh, action_type)
            else:
                reward = -1.0 / max(len(self.generated_elements), 1)
                terminated = False
                failed = True

                # Log why mesh was rejected
                if mesh is not None:
                    if not self._validate_mesh_simple(mesh):
                        if self.step_count % 100 == 0:
                            print(f"Step {self.step_count}: Mesh rejected - failed basic validation")
                else:
                    if self.step_count % 100 == 0:
                        print(f"Step {self.step_count}: No mesh generated")

        except Exception as e:
            print(f"Error in step: {e}")
            reward = -0.1
            terminated = False
            failed = True

        # Handle failed actions
        if failed:
            self.failed_num += 1
            if self.failed_num >= 100:  # Original uses 100
                terminated = True

        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'is_valid_element': not failed,
            'element_count': len(self.generated_elements),
            'boundary_vertices': len(self.current_boundary),
            'termination_reason': 'boundary_complete' if terminated and not failed else None
        })

        return observation, reward, terminated, False, info

    def _action_to_point_constrained(self, action: np.ndarray, ref_idx: int) -> Optional[np.ndarray]:
        """
        Convert action to world coordinates with strict boundary constraints.
        Uses multiple attempts with decreasing radius to ensure point stays inside.
        """
        if len(self.current_boundary) < 3:
            return None

        ref_vertex = self.current_boundary[ref_idx]
        n_boundary = len(self.current_boundary)

        # Calculate base length
        distances = []
        for j in range(1, min(4, n_boundary // 2) + 1):
            left_idx = (ref_idx - j) % n_boundary
            right_idx = (ref_idx + j) % n_boundary
            left_dist = np.linalg.norm(self.current_boundary[left_idx] - ref_vertex)
            right_dist = np.linalg.norm(self.current_boundary[right_idx] - ref_vertex)
            distances.extend([left_dist, right_dist])

        base_length = np.mean(distances) if distances else 1.0
        max_radius = self.max_radius * base_length

        # Try multiple radii with decreasing values
        for attempt in range(5):
            radius = max_radius * (1.0 - 0.2 * attempt)

            # Convert normalized action to world coordinates
            angle = action[0] * np.pi  # [-1, 1] -> [-pi, pi]
            distance = (action[1] + 1) * 0.5 * radius  # [-1, 1] -> [0, radius]

            new_point = ref_vertex + distance * np.array([np.cos(angle), np.sin(angle)])

            # Check if point is inside boundary
            if self._is_point_inside_boundary_robust(new_point):
                return new_point

        return None

    def _validate_mesh_simple(self, mesh: np.ndarray) -> bool:
        """Simple mesh validation."""
        if mesh is None or len(mesh) != 4:
            return False

        # Check for degenerate elements
        area = calculate_polygon_area(mesh)
        if area < 1e-8:
            return False

        # Check for self-intersection
        return not self._check_mesh_self_intersection(mesh)

    def _check_mesh_self_intersection(self, mesh: np.ndarray) -> bool:
        """Check if mesh has self-intersection."""
        if len(mesh) != 4:
            return True

        # Check if diagonals intersect properly for a quad
        p1, p2, p3, p4 = mesh

        # Check intersection of diagonals p1-p3 and p2-p4
        def line_intersect(p1, p2, p3, p4):
            d1 = np.cross(p3 - p1, p2 - p1)
            d2 = np.cross(p4 - p1, p2 - p1)
            d3 = np.cross(p1 - p3, p4 - p3)
            d4 = np.cross(p2 - p3, p4 - p3)

            return (d1 * d2 < 0) and (d3 * d4 < 0)

        # For a valid convex quad, diagonals should intersect
        return not line_intersect(p1, p3, p2, p4)

    def _is_point_inside_boundary_robust(self, point: np.ndarray) -> bool:
        """Check if point is inside boundary using ray casting."""
        if self.current_boundary is None or len(self.current_boundary) < 3:
            return False

        x, y = point
        n = len(self.current_boundary)
        inside = False

        p1x, p1y = self.current_boundary[0]
        for i in range(n + 1):
            p2x, p2y = self.current_boundary[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    # ---------------------------------------------------------------------
    # --------------  HELPER: geometry containment  -----------------------
    # ---------------------------------------------------------------------

    # -----------------------------------------------------------------
    #  robust point-in-polygon —— treat on-edge / on-vertex as inside
    # -----------------------------------------------------------------
    def _is_point_inside_polygon(self,
                                 point: np.ndarray,
                                 polygon: np.ndarray,
                                 eps: float = 1e-8) -> bool:
        """
        Return True if *point* is strictly inside *polygon* **or**
        lies on one of its edges / vertices.
        Ray-casting with an early on-edge test.
        """
        x, y = point
        n = len(polygon)

        # --- 1) on-vertex / on-edge quick check -----------------------
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]

            # on vertex
            if np.linalg.norm(point - a) <= eps or np.linalg.norm(point - b) <= eps:
                return True

            # on edge: colinear & projection within segment
            ab = b - a
            ap = point - a
            cross = np.abs(np.cross(ab, ap))
            if cross <= eps:
                dot = np.dot(ab, ap)
                if -eps <= dot <= np.dot(ab, ab) + eps:
                    return True

        # --- 2) standard ray-casting test ----------------------------
        inside = False
        px, py = polygon[0]
        for i in range(1, n + 1):
            qx, qy = polygon[i % n]

            if (py > y) != (qy > y):  # edge crosses horizontal line y
                x_int = (qx - px) * (y - py) / (qy - py + 1e-15) + px
                if x_int >= x - eps:  # ray to +x
                    inside = not inside
            px, py = qx, qy

        return inside

    def _is_segment_inside_polygon(
            self,
            p1: np.ndarray,
            p2: np.ndarray,
            polygon: np.ndarray,
            eps: float = 1e-8,
    ) -> bool:
        """
        Return True only if the entire segment p1–p2 lies inside (or on) *polygon*.

        Logic
        -----
        1. Both endpoints must be inside the polygon or on its boundary.
        2. The segment must not intersect any polygon edge that does not share an
           endpoint with the segment.
        3. Sample interior points at 0.25, 0.5, and 0.75 of the segment; each of
           these points must also be inside the polygon or on its boundary.
        """
        # 1) Endpoint containment check
        if (
                not self._is_point_inside_polygon(p1, polygon, eps)
                or not self._is_point_inside_polygon(p2, polygon, eps)
        ):
            return False

        # 2) Intersection check against all polygon edges
        n = len(polygon)
        for i in range(n):
            q1 = polygon[i]
            q2 = polygon[(i + 1) % n]

            # Skip edges that share any endpoint with the segment
            if (
                    np.allclose(p1, q1, atol=eps)
                    or np.allclose(p1, q2, atol=eps)
                    or np.allclose(p2, q1, atol=eps)
                    or np.allclose(p2, q2, atol=eps)
            ):
                continue

            if self._segments_intersect(p1, p2, q1, q2):
                return False

        # 3) Interior sampling check (handles concave-polygon chord cases)
        direction = p2 - p1
        for t in (0.25, 0.5, 0.75):
            sample_pt = p1 + t * direction
            if not self._is_point_inside_polygon(sample_pt, polygon, eps):
                return False

        return True

    def _is_polygon_inside_original(self, poly: np.ndarray) -> bool:
        """Return True if poly (convex quad or boundary) is fully inside original domain."""
        # 1) vertex check
        for pt in poly:
            if not self._is_point_inside_polygon(pt, self.original_boundary):
                return False

        # 2) edge check
        m = len(poly)
        for i in range(m):
            if not self._is_segment_inside_polygon(
                    poly[i], poly[(i + 1) % m], self.original_boundary):
                return False
        return True

    # ---------------------------------------------------------------------
    # --------------  MAIN: validate & apply mesh  ------------------------
    # ---------------------------------------------------------------------

    def _validate_and_apply_mesh(self,
                                 ref_idx: int,
                                 mesh: np.ndarray,
                                 action_type: int) -> Tuple[float, bool, bool]:
        """Validate mesh, ensure containment, then apply if valid."""
        # Step-1: local boundary update
        temp_boundary = self._update_boundary(ref_idx, mesh, action_type)

        # Step-2: geometric validations
        valid = (
                temp_boundary is not None and
                not self._check_boundary_self_intersection(temp_boundary) and
                self._is_polygon_inside_original(mesh) and
                self._is_polygon_inside_original(temp_boundary)
        )

        if valid:
            # commit
            self.current_boundary = temp_boundary
            self.generated_elements.append(mesh)

            reward = self._calculate_reward_original(mesh, action_type)
            self.episode_reward += reward
            self.failed_num = 0
            terminated = len(self.current_boundary) <= 5
            if terminated and len(self.current_boundary) == 4:
                self.generated_elements.append(self.current_boundary.copy())
            if terminated:
                reward += 10
            return reward, terminated, False  # failed=False
        else:
            reward = -1.0 / max(len(self.generated_elements), 1)
            return reward, False, True  # failed=True

    # ------------------------------------------------------------
    #  Update boundary after cutting off a quadrilateral element
    # ------------------------------------------------------------
    def _update_boundary(
            self,
            ref_idx: int,
            mesh: np.ndarray,
            action_type: int
    ) -> Optional[np.ndarray]:
        """
        Update self.current_boundary according to the primitive rule
        just applied at reference index `ref_idx`.

        Parameters
        ----------
        ref_idx : int
            Index of the reference vertex Vi (before cutting).
        mesh : np.ndarray
            4×2 array of vertices of the new quadrilateral (clock-wise).
            For action_type == 0 the first vertex is the newly inserted one.
        action_type : int
            -1 : forward cut  - remove Vi and Vi+1
             1 : backward cut - remove Vi-1 and Vi
             0 : insert one new vertex at Vi (replace Vi)

        Returns
        -------
        Optional[np.ndarray]
            New boundary (clock-wise) or None if the update is degenerate.
        """
        n = len(self.current_boundary)

        # ---------- (A) rule type 0 : replace the reference vertex ----------
        if action_type == 0:
            new_boundary = self.current_boundary.copy()
            new_boundary[ref_idx] = mesh[0]  # mesh[0] is the new point
            return np.array(new_boundary)

        # ---------- (B) rule type –1 / 1 : delete two vertices ----------
        if action_type == -1:
            # forward cut → remove Vi , Vi+1
            remove_set = {ref_idx % n, (ref_idx + 1) % n}
        elif action_type == 1:
            # backward cut → remove Vi-1 , Vi
            remove_set = {(ref_idx - 1) % n, ref_idx % n}
        else:  # unknown action
            return None

        new_boundary = [
            self.current_boundary[i]
            for i in range(n)
            if i not in remove_set
        ]

        # a legal boundary needs at least 3 vertices
        if len(new_boundary) < 3:
            return None

        return np.array(new_boundary)

    def _check_boundary_self_intersection(self, boundary: np.ndarray) -> bool:
        """Check if boundary has self-intersection."""
        if len(boundary) < 4:
            return False

        n = len(boundary)
        for i in range(n):
            for j in range(i + 2, n):
                if j == (i - 1) % n:  # Adjacent segments
                    continue

                # Check intersection between segments i and j
                p1, p2 = boundary[i], boundary[(i + 1) % n]
                p3, p4 = boundary[j], boundary[(j + 1) % n]

                if self._segments_intersect(p1, p2, p3, p4):
                    return True

        return False

    def _segments_intersect(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
        """Check if two line segments intersect."""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _calculate_reward_original(self, mesh: np.ndarray, action_type: int) -> float:
        """
        Paper-consistent reward:
            m_t = η_e + η_b + μ_t
        Invalid element → -0.1   (paper Fig. 5)
        """
        # ---------- invalid element ----------
        if mesh is None:
            return -0.1

        # ---------- η_e : element quality ----------
        eta_e = calculate_element_quality(mesh)  # ∈ [0, 1]

        # ---------- η_b : remaining boundary quality ----------
        n = len(self.current_boundary)
        if n <= 4:
            eta_b = 0.0
        else:
            angles = []
            for i in range(n):
                p_prev = self.current_boundary[(i - 1) % n]
                p_curr = self.current_boundary[i]
                p_next = self.current_boundary[(i + 1) % n]
                v1 = p_prev - p_curr
                v2 = p_next - p_curr
                cos_a = np.dot(v1, v2) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                )
                angle = np.arccos(np.clip(cos_a, -1.0, 1.0))
                angles.append(angle)
            mean_dev = np.mean(np.abs(np.array(angles) - np.pi / 2) / (np.pi / 2))
            eta_b = -float(np.clip(mean_dev, 0.0, 1.0))  # ∈ [-1, 0]

        # ---------- μ_t : density term ----------
        v_density = float(self.config['environment'].get('v_density', 1.0))

        edges = np.linalg.norm(
            self.current_boundary - np.roll(self.current_boundary, -1, axis=0),
            axis=1,
        )
        e_min, e_max = edges.min(), edges.max()
        kappa = 4.0
        A_min = v_density * (e_min ** 2)
        A_max = v_density * (((e_max - e_min) / kappa) + e_min) ** 2
        A_t = calculate_polygon_area(mesh)

        if A_t < A_min:
            mu_t = -1.0
        elif A_t < A_max:
            mu_t = (A_t - A_min) / (A_max - A_min)
        else:
            mu_t = 0.0
        mu_t = float(np.clip(mu_t, -1.0, 1.0))

        # ---------- total reward ----------
        return eta_e + eta_b + mu_t

    def _get_info(self) -> Dict:
        """Get environment info."""
        return {
            'boundary_area': calculate_polygon_area(self.current_boundary) if self.current_boundary is not None and len(
                self.current_boundary) >= 3 else 0,
            'boundary_vertices': len(self.current_boundary) if self.current_boundary is not None else 0,
            'generated_elements': len(self.generated_elements),
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'area_ratio': self.current_area_ratio
        }

    def get_current_mesh(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Get current mesh state."""
        return self.current_boundary.copy(), self.generated_elements.copy()

    def set_domain(self, domain_file: str):
        """Set domain for evaluation."""
        self.domain_file = domain_file
        self.original_boundary = self._load_domain()
        self.original_boundary = self._ensure_clockwise(self.original_boundary)
        self.original_area = calculate_polygon_area(self.original_boundary)

    def get_mesh_quality_metrics(self) -> Dict:
        """Get mesh quality metrics."""
        if not self.generated_elements:
            return {}

        element_qualities = []
        for elem in self.generated_elements:
            if elem is not None and len(elem) > 0:
                quality = calculate_element_quality(elem)
                element_qualities.append(quality)

        if not element_qualities:
            return {}

        return {
            'mean_element_quality': np.mean(element_qualities),
            'min_element_quality': np.min(element_qualities),
            'max_element_quality': np.max(element_qualities),
            'std_element_quality': np.std(element_qualities),
            'num_elements': len(self.generated_elements)
        }


class MultiDomainMeshEnv(MeshEnv):
    """Multi-domain environment for curriculum learning."""

    def __init__(self, config: Dict, domain_files: List[str]):
        self.domain_files = domain_files
        self.current_domain_idx = 0
        super().__init__(config)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset with random domain selection."""
        if len(self.domain_files) > 1:
            self.current_domain_idx = np.random.randint(0, len(self.domain_files))
            self.set_domain(self.domain_files[self.current_domain_idx])
        return super().reset(seed, options)

    def get_current_domain_info(self) -> Dict:
        """Get current domain information."""
        return {
            'domain_file': self.domain_files[self.current_domain_idx],
            'domain_index': self.current_domain_idx,
            'total_domains': len(self.domain_files)
        }
