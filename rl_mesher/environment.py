import os
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from gymnasium import Env, spaces
from .utils.geometry import (
    calculate_polygon_area, generate_action_coordinates,
    calculate_reward_components, calculate_reference_vertex,
    get_state_components, calculate_element_quality, load_domain_from_file,
    transform_to_relative_coords, calculate_boundary_quality
)


class MeshEnv(Env):
    """
    Mesh generation environment implementing Gymnasium interface.
    Modified to match original author's approach for better reproducibility.
    """

    def __init__(self, config: Dict):
        super(MeshEnv, self).__init__()
        self.config = config

        self.n_neighbors = self.config['environment'].get('n_neighbors', 2)
        self.n_fan_points = self.config['environment'].get('n_fan_points', 3)
        self.beta_obs = self.config['environment'].get('beta_obs', 6.0)
        self.alpha_action = self.config['environment'].get('alpha_action', 2.0)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        obs_dim = (self.n_neighbors * 2 + self.n_fan_points) * 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # --- MODIFICATION START ---
        # Ensure domain_file is always a string for initialization
        domain_setting = config['domain']['training_domain']
        if isinstance(domain_setting, list):
            # If it's a list, just pick the first one for the initial setup.
            # The MultiDomainEnv will handle switching later.
            self.domain_file = domain_setting[0]
        else:
            self.domain_file = domain_setting
        # --- MODIFICATION END ---

        self.data_dir = config['paths']['data_dir']
        self.original_boundary = self._load_domain()
        if self.original_boundary is not None:
            self.original_area = calculate_polygon_area(self.original_boundary)
            self.original_boundary = self._ensure_clockwise(self.original_boundary)
        else:
            self.original_area = 0.0

        self.current_boundary = None
        self.generated_elements = []
        self.current_area_ratio = 1.0
        self.step_count = 0
        self.episode_reward = 0.0
        self.max_steps = int(config['environment'].get('max_steps', 1000))

        self.reward_method = config['environment'].get('reward_method', 10)
        self.failed_num = 0
        self.not_valid_points = []

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
        if self.current_boundary is None or len(self.current_boundary) < 3:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        try:
            ref_idx = self._find_reference_point()
            if ref_idx is None:
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            n_boundary = len(self.current_boundary)
            V0 = self.current_boundary[ref_idx]

            # 1. Get Neighboring Vertices
            left_neighbors = np.array(
                [self.current_boundary[(ref_idx - i) % n_boundary] for i in range(1, self.n_neighbors + 1)])
            right_neighbors = np.array(
                [self.current_boundary[(ref_idx + i) % n_boundary] for i in range(1, self.n_neighbors + 1)])

            # 2. Get Fan Points (as described in Paper Figure 6)
            Vl1 = left_neighbors[0]
            Vr1 = right_neighbors[0]

            # Base length L for radius calculation (Eq. 2)
            L = 0.0
            n_L = min(self.n_neighbors, n_boundary // 2)
            if n_L > 0:
                len_sum = 0
                for i in range(n_L):
                    len_sum += np.linalg.norm(self.current_boundary[(ref_idx - i) % n_boundary] - self.current_boundary[
                        (ref_idx - i - 1) % n_boundary])
                    len_sum += np.linalg.norm(self.current_boundary[(ref_idx + i) % n_boundary] - self.current_boundary[
                        (ref_idx + i + 1) % n_boundary])
                L = len_sum / (2 * n_L)
            if L < 1e-6: L = 1.0

            Lr = self.beta_obs * L  # Radius Lr for fan area (Eq. 3)

            # Define fan area
            vec_l = Vl1 - V0
            vec_r = Vr1 - V0
            angle_l = np.arctan2(vec_l[1], vec_l[0])
            angle_r = np.arctan2(vec_r[1], vec_r[0])

            # Handle angle wrapping
            if angle_l < angle_r:
                angle_l += 2 * np.pi

            slice_angle = (angle_l - angle_r) / self.n_fan_points
            fan_points = []

            for i in range(self.n_fan_points):
                start_angle = angle_r + i * slice_angle
                end_angle = start_angle + slice_angle

                closest_point = None
                min_dist = float('inf')

                # Find closest vertex within the slice
                for j in range(n_boundary):
                    if j == ref_idx: continue
                    point = self.current_boundary[j]
                    dist = np.linalg.norm(point - V0)
                    if dist <= Lr:
                        vec_p = point - V0
                        p_angle = np.arctan2(vec_p[1], vec_p[0])
                        if p_angle < angle_r: p_angle += 2 * np.pi

                        if start_angle <= p_angle <= end_angle:
                            if dist < min_dist:
                                min_dist = dist
                                closest_point = point

                # If no point in slice, find intersection with boundary (as per paper)
                if closest_point is None:
                    bisector_angle = start_angle + slice_angle / 2
                    ray_dir = np.array([np.cos(bisector_angle), np.sin(bisector_angle)])

                    intersect_dist = float('inf')
                    intersect_point = None

                    for j in range(n_boundary):
                        p1 = self.current_boundary[j]
                        p2 = self.current_boundary[(j + 1) % n_boundary]
                        edge_dir = p2 - p1

                        if np.cross(ray_dir, edge_dir) != 0:
                            t = np.cross(p1 - V0, edge_dir) / np.cross(ray_dir, edge_dir)
                            u = np.cross(p1 - V0, ray_dir) / np.cross(ray_dir, edge_dir)
                            if t > 0 and 0 <= u <= 1:
                                if t < intersect_dist:
                                    intersect_dist = t
                                    intersect_point = V0 + t * ray_dir

                    if intersect_point is not None and intersect_dist <= Lr:
                        closest_point = intersect_point
                    else:
                        # Fallback: furthest point along bisector
                        closest_point = V0 + Lr * ray_dir

                fan_points.append(closest_point)

            fan_points = np.array(fan_points)

            # 3. Calculate Area Ratio
            current_area = calculate_polygon_area(self.current_boundary)
            self.current_area_ratio = current_area / self.original_area if self.original_area > 0 else 0.0

            # 4. Convert to relative polar coordinates and create state vector
            # Reference direction is V0 -> Vr1 (as per paper page 9)
            ref_direction = Vr1 - V0
            ref_angle = np.arctan2(ref_direction[1], ref_direction[0])

            def to_polar_relative(p):
                vec = p - V0
                dist = np.linalg.norm(vec)
                angle = np.arctan2(vec[1], vec[0])
                relative_angle = angle - ref_angle
                # Normalize angle to [-pi, pi]
                relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
                return [dist, relative_angle]

            state_components = []
            all_points = np.concatenate([left_neighbors, right_neighbors, fan_points])
            for p in all_points:
                state_components.extend(to_polar_relative(p))

            state_components.append(self.current_area_ratio)

            observation = np.array(state_components, dtype=np.float32)

            # Final check for dimension consistency
            if observation.shape[0] != self.observation_space.shape[0]:
                # Fallback to zero vector if something went wrong
                print(
                    f"Error: Observation dimension mismatch. Expected {self.observation_space.shape[0]}, got {observation.shape[0]}.")
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            return observation

        except Exception as e:
            print(f"Error during observation generation: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(self.observation_space.shape, dtype=np.float32)

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

    def step(self, action: np.ndarray, global_timestep: Optional[int] = None, block_validation: bool = False) -> Tuple[
        np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        failed = True
        successful_termination = False

        if self.step_count >= self.max_steps:
            truncated = True
            reward = -1.0
            failed = True

        elif self.current_boundary is not None and len(self.current_boundary) <= 5:
            reward = 10.0
            terminated = True
            successful_termination = True
            if len(self.current_boundary) == 4:
                self.generated_elements.append(self.current_boundary.copy())
            failed = False

        else:
            ref_idx = self._find_reference_point()
            mesh = None
            action_type_code = -99  # Default to an invalid code

            if ref_idx is not None:
                n_boundary = len(self.current_boundary)
                rule_type = action[0]
                TYPE2_THRESHOLD, TYPE0_THRESHOLD, TYPEM1_THRESHOLD = 0.5, 0.0, -0.5

                if rule_type >= TYPE2_THRESHOLD:
                    action_type_code = 2
                    V0, V1 = self.current_boundary[ref_idx], self.current_boundary[(ref_idx + 1) % n_boundary]
                    V2, V3 = self._action_to_points_type2(action[1:], V0, V1)
                    mesh = np.array([V0, V1, V2, V3])
                elif rule_type >= TYPE0_THRESHOLD:
                    action_type_code = 1
                    mesh = np.array([
                        self.current_boundary[(ref_idx - 2) % n_boundary],
                        self.current_boundary[(ref_idx - 1) % n_boundary],
                        self.current_boundary[ref_idx], self.current_boundary[(ref_idx + 1) % n_boundary]
                    ])
                elif rule_type >= TYPEM1_THRESHOLD:
                    action_type_code = 0
                    new_point = self._action_to_point_type0(action[1:3], ref_idx)
                    if new_point is not None:
                        mesh = np.array([
                            new_point, self.current_boundary[(ref_idx - 1) % n_boundary],
                            self.current_boundary[ref_idx], self.current_boundary[(ref_idx + 1) % n_boundary]
                        ])
                else:
                    action_type_code = -1
                    mesh = np.array([
                        self.current_boundary[(ref_idx - 1) % n_boundary], self.current_boundary[ref_idx],
                        self.current_boundary[(ref_idx + 1) % n_boundary],
                        self.current_boundary[(ref_idx + 2) % n_boundary]
                    ])

            # --- MODIFICATION START ---
            # This is the new logic branch that handles validation blocking
            if block_validation:
                failed = False  # Force failure to false to prevent failed_num from incrementing
                try:
                    if mesh is not None and ref_idx is not None:
                        # Attempt to update the boundary directly, without validation
                        temp_boundary = None
                        if action_type_code == 2:
                            V2, V3 = mesh[2], mesh[3]
                            temp_boundary = self._update_boundary_type2(ref_idx, V2, V3)
                        else:
                            temp_boundary = self._update_boundary(ref_idx, mesh, action_type_code)

                        # Even if the update returns None (degenerate), we proceed
                        if temp_boundary is not None:
                            self.current_boundary = temp_boundary

                        # We always add the element to visualize the agent's raw action
                        self.generated_elements.append(mesh)
                        reward = self._calculate_reward_original(ref_idx, mesh, action_type_code)

                        # Check for successful termination only
                        if self.current_boundary is not None and len(self.current_boundary) <= 5:
                            terminated = True
                            successful_termination = True
                            reward += 10.0
                    else:
                        # Action was invalid from the start (e.g., could not find ref_idx)
                        reward = -1.0  # Assign penalty but continue

                except Exception as e:
                    # A crash happened during the unvalidated update.
                    # Log it, assign a penalty, but DO NOT terminate or mark as failed.
                    print(f"  [Eval-NoBlock Mode] Caught exception during unvalidated step: {e}. Continuing.")
                    reward = -1.0  # Give a penalty for the crash
                    # Keep the old boundary and continue the episode

            # This is the original logic for normal training/evaluation
            else:
                try:
                    if mesh is not None and ref_idx is not None:
                        if action_type_code == 2:
                            reward, terminated, failed = self._validate_and_apply_type2(ref_idx, mesh)
                        else:
                            reward, terminated, failed = self._validate_and_apply_mesh(ref_idx, mesh, action_type_code)
                        if terminated and not failed:
                            successful_termination = True
                    else:
                        failed = True
                except Exception as e:
                    print(f"Error in step validation: {e}")
                    failed = True
            # --- MODIFICATION END ---

        # This block now only runs if validation is ON
        if failed and not block_validation and not terminated and not truncated:
            self.failed_num += 1
            reward = -0.1
            if self.failed_num >= 100:
                terminated = True
                successful_termination = False

        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'is_valid_element': not failed,
            'element_count': len(self.generated_elements),
            'boundary_vertices': len(self.current_boundary) if self.current_boundary is not None else 0,
            'successful_termination': successful_termination
        })

        return observation, reward, terminated, truncated, info

    def _action_to_point_type0(self, action_vec: np.ndarray, ref_idx: int) -> Optional[np.ndarray]:
        if len(self.current_boundary) < 3: return None
        ref_vertex = self.current_boundary[ref_idx]

        # Base length L (Eq. 2)
        L = np.linalg.norm(self.current_boundary[(ref_idx + 1) % len(self.current_boundary)] - self.current_boundary[
            (ref_idx - 1) % len(self.current_boundary)]) / 2.0
        if L < 1e-6: L = 1.0
        radius = self.alpha_action * L

        relative_coords = action_vec * radius
        new_point = ref_vertex + relative_coords

        if self._is_point_inside_polygon(new_point, self.current_boundary):
            return new_point
        return None

    def _action_to_points_type2(self, action_vec: np.ndarray, V0: np.ndarray, V1: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        L = np.linalg.norm(V1 - V0)
        radius = self.alpha_action * L

        # Action vector gives coords relative to the midpoint of V0-V1 edge
        midpoint = (V0 + V1) / 2.0

        # First point from action[0:2]
        x1_rel, y1_rel = action_vec[0] * radius, action_vec[1] * radius
        V2 = midpoint + np.array([x1_rel, y1_rel])

        # Second point from action[2:4]
        x2_rel, y2_rel = action_vec[2] * radius, action_vec[3] * radius
        V3 = midpoint + np.array([x2_rel, y2_rel])

        return V2, V3

    def _validate_and_apply_type2(self, ref_idx: int, mesh: np.ndarray) -> Tuple[float, bool, bool]:
        V0, V1, V2, V3 = mesh[0], mesh[1], mesh[2], mesh[3]

        # Validation 1: New points must be inside the current boundary
        if not self._is_point_inside_polygon(V2, self.current_boundary) or \
                not self._is_point_inside_polygon(V3, self.current_boundary):
            return -0.1, False, True  # Failed

        # Validation 2: New internal edge V3-V2 must be inside and not intersect boundary
        new_internal_edges = [np.array([V0, V3]), np.array([V3, V2]), np.array([V2, V1])]
        for edge in new_internal_edges:
            if not self._is_segment_inside_polygon(edge[0], edge[1], self.current_boundary, ref_idx):
                return -0.1, False, True  # Failed

        # Validation 3: Update boundary and check for self-intersection
        new_boundary = self._update_boundary_type2(ref_idx, V2, V3)
        if new_boundary is None or self._check_boundary_self_intersection(new_boundary):
            return -0.1, False, True  # Failed

        # All checks passed, commit the changes
        self.current_boundary = new_boundary
        self.generated_elements.append(mesh)

        # --- THIS IS THE MODIFIED LINE ---
        reward = self._calculate_reward_original(ref_idx, mesh, 2)
        # ---------------------------------

        self.episode_reward += reward
        self.failed_num = 0

        terminated = len(self.current_boundary) <= 5
        if terminated:
            reward += 10.0

        return reward, terminated, False  # Success

    def _update_boundary_type2(self, ref_idx: int, V2: np.ndarray, V3: np.ndarray) -> Optional[np.ndarray]:
        """
        Update the boundary for a type 2 action.
        This corrected version replaces the edge (V0, V1) on the boundary
        with the new path (V0, V3, V2, V1).

        Args:
            ref_idx: The index of the reference vertex (V0).
            V2: The second new vertex.
            V3: The first new vertex.

        Returns:
            The new boundary as a numpy array, or None if the update is degenerate.
        """
        n = len(self.current_boundary)
        if n < 3:
            return None

        v0_idx = ref_idx
        v1_idx = (ref_idx + 1) % n

        # Convert the numpy array to a list for easier manipulation (insertion)
        boundary_list = list(self.current_boundary)

        # The new path from V0 to V1 is via V3 and V2. The element created is (V0, V1, V2, V3).
        # We need to insert V3 and V2 between V0 and V1 in the boundary list.
        # The correct insertion point is at the index of V1.

        # Create a new list to build the updated boundary
        new_boundary_list = []

        # Iterate through the original boundary list to construct the new one
        for i in range(n):
            # Add the current vertex
            new_boundary_list.append(boundary_list[i])

            # When we are at V0 (ref_idx), insert the new path points (V3, V2)
            # before moving on. V1 will be added later in its original turn.
            if i == v0_idx:
                new_boundary_list.append(V3)
                new_boundary_list.append(V2)

        # A quick validation to ensure the length is correct.
        # The new boundary should have n+2 vertices.
        if len(new_boundary_list) != n + 2:
            # This case might happen with complex indexing, fallback to a more robust method
            # This robust method rebuilds the list from scratch.
            new_boundary_list = []
            # 1. Add all points up to and including V0
            for i in range(v0_idx + 1):
                new_boundary_list.append(self.current_boundary[i])

            # 2. Add the new points that form the new part of the boundary
            new_boundary_list.append(V3)
            new_boundary_list.append(V2)

            # 3. Add all points from V1 to the end of the original boundary
            for i in range(v1_idx, n):
                new_boundary_list.append(self.current_boundary[i])

        if len(new_boundary_list) < 3:
            return None

        return np.array(new_boundary_list)

    def _segments_intersect(self, p1, p2, p3, p4, tol=1e-9) -> bool:
        def cross_product(a, b):
            return a[0] * b[1] - a[1] * b[0]

        r = p2 - p1
        s = p4 - p3
        q_minus_p = p3 - p1

        r_cross_s = cross_product(r, s)
        q_minus_p_cross_r = cross_product(q_minus_p, r)

        if abs(r_cross_s) < tol:  # Collinear or parallel
            return abs(q_minus_p_cross_r) < tol  # Check if they are on the same line

        t = cross_product(q_minus_p, s) / r_cross_s
        u = q_minus_p_cross_r / r_cross_s

        return (0 <= t <= 1) and (0 <= u <= 1)

    def _is_segment_inside_polygon(self, p1: np.ndarray, p2: np.ndarray, polygon: np.ndarray, ref_idx: int) -> bool:
        # 1. Check if endpoints are inside
        if not self._is_point_inside_polygon(p1, polygon) or \
                not self._is_point_inside_polygon(p2, polygon):
            return False

        # 2. Check if midpoint is inside
        if not self._is_point_inside_polygon((p1 + p2) / 2.0, polygon):
            return False

        # 3. Check for intersection with boundary edges
        n = len(polygon)
        v1_idx = (ref_idx + 1) % n
        for i in range(n):
            q1 = polygon[i]
            q2 = polygon[(i + 1) % n]

            # Skip the edge this action is based on
            if (i == ref_idx and (i + 1) % n == v1_idx) or (i == v1_idx and (i + 1) % n == ref_idx):
                continue

            # If the new segment intersects a boundary edge, it's invalid
            if self._segments_intersect(p1, p2, q1, q2):
                # Allow intersection only at shared endpoints
                if not (np.allclose(p1, q1) or np.allclose(p1, q2) or np.allclose(p2, q1) or np.allclose(p2, q2)):
                    return False

        return True

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
            boundary_before_update = self.current_boundary
            self.current_boundary = temp_boundary
            self.generated_elements.append(mesh)

            # --- THIS IS THE MODIFIED LINE ---
            reward = self._calculate_reward_original(ref_idx, mesh, action_type)
            # ---------------------------------

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

    def _calculate_reward_original(self, ref_idx: int, mesh: np.ndarray, action_type: int) -> float:
        """
        Paper-consistent reward calculation.
        This version is modified to call the new geometry.calculate_boundary_quality function.
        m_t = eta_e + eta_b + mu_t
        Invalid element -> -0.1
        """
        if mesh is None:
            return -0.1

        # eta_e: element quality (Equation 7)
        eta_e = calculate_element_quality(mesh)

        # eta_b: remaining boundary quality (Equation 8)
        # This now calls the detailed implementation from geometry.py
        boundary_before = self.current_boundary
        M_angle = float(self.config['environment'].get('M_angle', 60.0))
        eta_b = calculate_boundary_quality(
            element_vertices=mesh,
            boundary_before=boundary_before,
            action_type=action_type,
            ref_idx=ref_idx,
            M_angle=M_angle
        )

        # mu_t: density term (Equation 9)
        v_density = float(self.config['environment'].get('v_density', 1.0))

        edges = np.linalg.norm(
            self.current_boundary - np.roll(self.current_boundary, -1, axis=0),
            axis=1,
        )
        if len(edges) == 0:
            return eta_e + eta_b  # Cannot calculate density term

        e_min, e_max = edges.min(), edges.max()
        kappa = 4.0
        A_min = v_density * (e_min ** 2)
        A_max = v_density * (((e_max - e_min) / kappa) + e_min) ** 2
        A_t = calculate_polygon_area(mesh)

        mu_t = 0.0
        if A_max > A_min:
            if A_t < A_min:
                mu_t = -1.0
            elif A_t < A_max:
                mu_t = (A_t - A_min) / (A_max - A_min)

        mu_t = float(np.clip(mu_t, -1.0, 1.0))

        # total reward
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
        """
        Initializes the multi-domain environment.

        Args:
            config: The main configuration dictionary.
            domain_files: A list of domain file names to be used for training.
        """
        self.domain_files = domain_files
        self.current_domain_idx = 0

        # --- MODIFICATION START ---
        # Crucially, we must pass the modified config with a SINGLE domain file
        # to the parent constructor to ensure it initializes correctly.
        # We can store a copy of the original config if needed, but for now,
        # modifying it before the super() call is sufficient.

        # We set 'training_domain' to the first file in the list for the initial
        # super() call. The `reset` method will then handle randomizing the domain.
        config['domain']['training_domain'] = domain_files[0]

        super().__init__(config)

        # After the parent is initialized, we restore the list of domains
        # for this instance's use in the `reset` method.
        self.config['domain']['training_domain'] = domain_files
        # --- MODIFICATION END ---
