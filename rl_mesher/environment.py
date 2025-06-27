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
        return boundary[::-1] if area > 0 else boundary

    def _load_domain(self) -> np.ndarray:
        """Load domain from file."""
        domain_path = os.path.join(self.data_dir, self.domain_file)
        return load_domain_from_file(domain_path)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
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

    def _get_observation(self) -> np.ndarray:
        """Get observation as flat array following original author's approach."""
        if self.current_boundary is None or len(self.current_boundary) < 3:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        try:
            # Find reference point using original logic
            ref_idx = self._find_reference_point()
            if ref_idx is None:
                return np.zeros(self.observation_space.shape[0], dtype=np.float32)

            # Get state following original PointEnvironment logic
            state = self._get_point_environment_state(ref_idx)
            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"Error in observation: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def _find_reference_point(self) -> Optional[int]:
        """Find reference point using original author's method."""
        n_vertices = len(self.current_boundary)
        if n_vertices < 3:
            return None

        # Original method: find vertex with minimum angle or use other criteria
        min_angle = float('inf')
        ref_idx = 0

        for i in range(n_vertices):
            # Calculate angle at vertex i
            p1 = self.current_boundary[(i - 1) % n_vertices]
            p2 = self.current_boundary[i]
            p3 = self.current_boundary[(i + 1) % n_vertices]

            # Calculate angle
            v1 = p1 - p2
            v2 = p3 - p2

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 1e-8 and v2_norm > 1e-8:
                cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))

                if angle < min_angle:
                    min_angle = angle
                    ref_idx = i

        # Avoid recently failed points
        if self.not_valid_points and ref_idx in [self.current_boundary.tolist().index(p.tolist()) for p in
                                                 self.not_valid_points if p.tolist() in self.current_boundary.tolist()]:
            # Find alternative
            for i in range(n_vertices):
                if i not in [self.current_boundary.tolist().index(p.tolist()) for p in self.not_valid_points if
                             p.tolist() in self.current_boundary.tolist()]:
                    ref_idx = i
                    break

        return ref_idx

    def _get_point_environment_state(self, ref_idx: int) -> List[float]:
        """Generate state following original PointEnvironment logic."""
        n_vertices = len(self.current_boundary)
        ref_vertex = self.current_boundary[ref_idx]

        # Get neighbors (original approach)
        neighbors = []

        # Get left and right neighbors
        for i in range(1, min(self.neighbor_num // 2 + 1, n_vertices // 2)):
            left_idx = (ref_idx - i) % n_vertices
            right_idx = (ref_idx + i) % n_vertices
            neighbors.extend([left_idx, right_idx])

        # Pad if not enough neighbors
        while len(neighbors) < self.neighbor_num:
            neighbors.append(ref_idx)  # Use reference as fallback

        neighbors = neighbors[:self.neighbor_num]

        # Get radius neighbors (points within observation radius)
        radius_neighbors = []
        for i, vertex in enumerate(self.current_boundary):
            if i != ref_idx:
                distance = np.linalg.norm(vertex - ref_vertex)
                if distance <= self.radius and len(radius_neighbors) < self.radius_num:
                    radius_neighbors.append(i)

        # Pad radius neighbors
        while len(radius_neighbors) < self.radius_num:
            radius_neighbors.append(ref_idx)

        radius_neighbors = radius_neighbors[:self.radius_num]

        # Transform to local coordinate system (original approach)
        state = []

        # Calculate base length and reference direction
        if len(neighbors) >= 2:
            v1 = self.current_boundary[neighbors[0]]
            v2 = self.current_boundary[neighbors[1]]
            base_length = np.linalg.norm(v2 - v1)
            if base_length < 1e-8:
                base_length = 1.0

            # Reference direction
            ref_direction = v2 - v1
            ref_direction = ref_direction / (np.linalg.norm(ref_direction) + 1e-8)
        else:
            base_length = 1.0
            ref_direction = np.array([1.0, 0.0])

        # Store base_length for action transformation
        self.current_base_length = base_length
        self.current_ref_direction = ref_direction
        self.current_ref_vertex = ref_vertex
        self.current_ref_idx = ref_idx

        # Transform neighbor points to relative coordinates
        for neighbor_idx in neighbors:
            neighbor_vertex = self.current_boundary[neighbor_idx]
            relative_pos = neighbor_vertex - ref_vertex

            # Convert to polar-like coordinates (original approach)
            distance = np.linalg.norm(relative_pos) / base_length
            if distance > 1e-8:
                # Angle relative to reference direction
                cos_angle = np.dot(relative_pos, ref_direction) / (
                        np.linalg.norm(relative_pos) * np.linalg.norm(ref_direction))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                # Check if point is on left or right side
                cross_product = np.cross(ref_direction, relative_pos)
                if cross_product < 0:
                    angle = -angle
            else:
                angle = 0.0

            state.extend([distance, angle])

        # Transform radius neighbors
        for radius_idx in radius_neighbors:
            radius_vertex = self.current_boundary[radius_idx]
            relative_pos = radius_vertex - ref_vertex

            distance = np.linalg.norm(relative_pos) / base_length
            if distance > 1e-8:
                cos_angle = np.dot(relative_pos, ref_direction) / (
                        np.linalg.norm(relative_pos) * np.linalg.norm(ref_direction))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                cross_product = np.cross(ref_direction, relative_pos)
                if cross_product < 0:
                    angle = -angle
            else:
                angle = 0.0

            state.extend([distance, angle])

        return state

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

            # Determine action type using original thresholds
            mesh = None
            action_type = None

            if rule_type <= -0.5:  # Type -1: Remove 2 vertices
                # Create quad by connecting 4 consecutive boundary vertices
                mesh = np.array([
                    self.current_boundary[(ref_idx - 1) % n_boundary],
                    self.current_boundary[ref_idx],
                    self.current_boundary[(ref_idx + 1) % n_boundary],
                    self.current_boundary[(ref_idx + 2) % n_boundary],
                ])
                action_type = -1

            elif rule_type >= 0.5:  # Type 1: Remove 2 vertices (different direction)
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
                        reward = -1.0 / len(self.generated_elements) if len(self.generated_elements) else -1.0
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
                reward = -1.0 / len(self.generated_elements) if len(self.generated_elements) else -1.0
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
        x, y = action[0], action[1]

        if not hasattr(self, 'current_base_length'):
            return None

        # Calculate reference direction rotation matrix
        cos_theta = self.current_ref_direction[0]
        sin_theta = self.current_ref_direction[1]
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # Try multiple scales to find a valid point inside boundary
        max_attempts = 5
        scale_factors = [0.15, 0.25, 0.4, 0.6, 0.8]  # Even more conservative scaling

        for attempt in range(max_attempts):
            scale = scale_factors[min(attempt, len(scale_factors) - 1)]

            # Scale by base length with conservative scaling
            x_world = x * self.current_base_length * self.max_radius * scale
            y_world = y * self.current_base_length * self.max_radius * scale

            local_coords = np.array([x_world, y_world])
            world_coords = self.current_ref_vertex + rotation_matrix @ local_coords

            # Check if point is inside boundary
            if self._is_point_inside_boundary_robust(world_coords):
                # Additional check: ensure point is not too close to boundary edges
                if self._is_point_safe_distance_from_boundary(world_coords):
                    return world_coords

        # If all attempts fail, try constrained projection
        # Use very small scale as fallback
        x_world = x * self.current_base_length * self.max_radius * 0.1
        y_world = y * self.current_base_length * self.max_radius * 0.1
        local_coords = np.array([x_world, y_world])
        fallback_coords = self.current_ref_vertex + rotation_matrix @ local_coords

        if self._is_point_inside_boundary_robust(fallback_coords):
            return fallback_coords

        # Final fallback: project to safe interior point
        safe_point = self._find_safe_interior_point(ref_idx)
        return safe_point

    def _is_point_inside_boundary_robust(self, point: np.ndarray) -> bool:
        """
        Robust point-in-polygon test using winding number algorithm.
        More reliable than simple ray casting for edge cases.
        """
        if len(self.current_boundary) < 3:
            return False

        x, y = point
        vertices = self.current_boundary
        n = len(vertices)

        # Winding number algorithm
        winding_number = 0

        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]

            if y1 <= y:
                if y2 > y:  # Upward crossing
                    if self._is_left(x1, y1, x2, y2, x, y) > 0:
                        winding_number += 1
            else:
                if y2 <= y:  # Downward crossing
                    if self._is_left(x1, y1, x2, y2, x, y) < 0:
                        winding_number -= 1

        return winding_number != 0

    def _is_left(self, x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
        """
        Test if point P is left/on/right of line P1P2.
        Returns: >0 for P left of the line, =0 for P on the line, <0 for P right of the line
        """
        return (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)

    def _is_point_safe_distance_from_boundary(self, point: np.ndarray, min_distance_factor: float = 0.02) -> bool:
        """
        Check if point is at safe distance from boundary edges.

        Args:
            point: Point to check
            min_distance_factor: Minimum distance as factor of base_length

        Returns:
            True if point is at safe distance from all boundary edges
        """
        min_distance = self.current_base_length * min_distance_factor

        n = len(self.current_boundary)
        for i in range(n):
            edge_start = self.current_boundary[i]
            edge_end = self.current_boundary[(i + 1) % n]

            # Calculate distance from point to edge
            dist = self._point_to_line_distance(point, edge_start, edge_end)
            if dist < min_distance:
                return False

        return True

    def _find_safe_interior_point(self, ref_idx: int) -> np.ndarray:
        """
        Find a safe interior point near the reference vertex.
        Uses centroid of local triangle as fallback.
        """
        n = len(self.current_boundary)

        # Create a small triangle around reference vertex
        ref_vertex = self.current_boundary[ref_idx]
        left_vertex = self.current_boundary[(ref_idx - 1) % n]
        right_vertex = self.current_boundary[(ref_idx + 1) % n]

        # Calculate centroid of triangle formed by ref and neighbors
        # Move it slightly towards interior
        centroid = (ref_vertex + left_vertex + right_vertex) / 3.0

        # Move centroid towards ref_vertex to ensure it's safe
        direction = ref_vertex - centroid
        if np.linalg.norm(direction) > 1e-8:
            direction = direction / np.linalg.norm(direction)
            # Create point that's 10% of base_length towards ref_vertex (more conservative)
            safe_point = centroid + direction * (self.current_base_length * 0.1)
        else:
            safe_point = ref_vertex

        # Final verification - if still outside, use reference vertex
        if not self._is_point_inside_boundary_robust(safe_point):
            # Create a very conservative point very close to reference vertex
            offset = np.array([self.current_base_length * 0.005, 0])  # Even smaller offset
            safe_point = ref_vertex + offset

            # If still outside, just use reference vertex (this should never happen)
            if not self._is_point_inside_boundary_robust(safe_point):
                safe_point = ref_vertex

        return safe_point

    def _is_point_inside_boundary(self, point: np.ndarray) -> bool:
        """Wrapper for backward compatibility - use robust version."""
        return self._is_point_inside_boundary_robust(point)

    def _check_intersection_with_boundary(self, mesh: np.ndarray, ref_idx: int, action_type: int) -> bool:
        """
        Check if mesh intersects with current boundary.
        Simplified but comprehensive approach for all action types.

        Args:
            mesh: Generated mesh vertices
            ref_idx: Reference vertex index
            action_type: Type of action (-1, 0, 1)

        Returns:
            True if intersection detected, False otherwise
        """
        if len(mesh) < 3:
            return False

        # Get mesh edges
        mesh_edges = []
        for i in range(len(mesh)):
            mesh_edges.append((mesh[i], mesh[(i + 1) % len(mesh)]))

        # Get current boundary edges
        n_boundary = len(self.current_boundary)
        boundary_edges = []
        for i in range(n_boundary):
            boundary_edges.append((self.current_boundary[i], self.current_boundary[(i + 1) % n_boundary]))

        # Check each mesh edge against each boundary edge
        for mesh_edge in mesh_edges:
            mesh_start, mesh_end = mesh_edge

            for boundary_edge in boundary_edges:
                boundary_start, boundary_end = boundary_edge

                # Skip if the boundary edge shares vertices with the mesh
                # This is important for avoiding false positives
                shares_vertex = False
                for mesh_vertex in mesh:
                    if (np.allclose(boundary_start, mesh_vertex, atol=1e-6) or
                            np.allclose(boundary_end, mesh_vertex, atol=1e-6)):
                        shares_vertex = True
                        break

                if shares_vertex:
                    continue

                # Check for proper intersection (not just touching at endpoints)
                if self._line_segments_intersect_proper(mesh_start, mesh_end,
                                                        boundary_start, boundary_end):
                    return True

        # Additional check for action_type 0: ensure new vertex doesn't create invalid geometry
        if action_type == 0:
            new_vertex = mesh[0]  # First vertex is the new one for type 0

            # Check if new vertex is too close to existing boundary (might cause numerical issues)
            for i in range(n_boundary):
                boundary_start = self.current_boundary[i]
                boundary_end = self.current_boundary[(i + 1) % n_boundary]

                # Skip edges that are part of the mesh
                skip_edge = False
                for mesh_vertex in mesh[1:]:  # Skip the new vertex itself
                    if (np.allclose(boundary_start, mesh_vertex, atol=1e-6) or
                            np.allclose(boundary_end, mesh_vertex, atol=1e-6)):
                        skip_edge = True
                        break

                if skip_edge:
                    continue

                # Check distance from new vertex to boundary edge
                dist = self._point_to_line_distance(new_vertex, boundary_start, boundary_end)
                if dist < 1e-3:  # Too close to boundary edge
                    return True

        return False

    def _line_segments_intersect_proper(self, p1: np.ndarray, p2: np.ndarray,
                                        p3: np.ndarray, p4: np.ndarray) -> bool:
        """
        Check if two line segments intersect properly (not just touching at endpoints).

        Args:
            p1, p2: First line segment endpoints
            p3, p4: Second line segment endpoints

        Returns:
            True if segments intersect properly, False otherwise
        """

        def orientation(p, q, r):
            """Find orientation of ordered triplet (p, q, r).
            Returns:
            0 --> p, q and r are colinear
            1 --> Clockwise
            2 --> Counterclockwise
            """
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-10:
                return 0
            return 1 if val > 0 else 2

        def on_segment(p, q, r):
            """Check if point q lies on segment pr"""
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        # General case - proper intersection
        if o1 != o2 and o3 != o4:
            return True

        # Special cases - we want to avoid these as they might be valid touching
        # Check if endpoints are the same (valid connection)
        if (np.allclose(p1, p3, atol=1e-6) or np.allclose(p1, p4, atol=1e-6) or
                np.allclose(p2, p3, atol=1e-6) or np.allclose(p2, p4, atol=1e-6)):
            return False

        # Colinear cases
        if (o1 == 0 and on_segment(p1, p3, p2)) or \
                (o2 == 0 and on_segment(p1, p4, p2)) or \
                (o3 == 0 and on_segment(p3, p1, p4)) or \
                (o4 == 0 and on_segment(p3, p2, p4)):
            return True

        return False

    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        Calculate the distance from a point to a line segment.

        Args:
            point: The point
            line_start, line_end: Line segment endpoints

        Returns:
            Distance from point to line segment
        """
        line_vec = line_end - line_start
        point_vec = point - line_start

        line_length_sq = np.dot(line_vec, line_vec)

        if line_length_sq < 1e-10:
            # Line segment is actually a point
            return np.linalg.norm(point_vec)

        # Project point onto line
        t = np.dot(point_vec, line_vec) / line_length_sq

        if t < 0:
            # Closest point is line_start
            return np.linalg.norm(point_vec)
        elif t > 1:
            # Closest point is line_end
            return np.linalg.norm(point - line_end)
        else:
            # Closest point is on the line segment
            projection = line_start + t * line_vec
            return np.linalg.norm(point - projection)

    def _validate_mesh_simple(self, mesh: np.ndarray) -> bool:
        """Simple mesh validation following original criteria."""
        if mesh is None or len(mesh) < 3:
            return False

        # Check area
        area = calculate_polygon_area(mesh)
        if area < 1e-6:
            return False

        # Check for duplicate vertices
        for i in range(len(mesh)):
            for j in range(i + 1, len(mesh)):
                if np.linalg.norm(mesh[i] - mesh[j]) < 1e-6:
                    return False

        return True

    def _check_mesh_intersection_with_existing_meshes(self, new_mesh: np.ndarray) -> bool:
        """
        Check if new mesh intersects with any existing generated meshes.
        This is crucial to prevent the mesh crossing issues shown in the training images.

        Args:
            new_mesh: New mesh vertices to check

        Returns:
            True if intersection detected, False otherwise
        """
        if len(new_mesh) < 3 or not self.generated_elements:
            return False

        # Get edges of new mesh
        new_mesh_edges = []
        for i in range(len(new_mesh)):
            new_mesh_edges.append((new_mesh[i], new_mesh[(i + 1) % len(new_mesh)]))

        # Check against all existing meshes
        for existing_mesh in self.generated_elements:
            if len(existing_mesh) < 3:
                continue

            # Get edges of existing mesh
            existing_edges = []
            for i in range(len(existing_mesh)):
                existing_edges.append((existing_mesh[i], existing_mesh[(i + 1) % len(existing_mesh)]))

            # Check each new mesh edge against each existing mesh edge
            for new_edge in new_mesh_edges:
                new_start, new_end = new_edge

                for existing_edge in existing_edges:
                    existing_start, existing_end = existing_edge

                    # Skip if edges share vertices (valid adjacency)
                    shares_vertex = False
                    for new_vertex in [new_start, new_end]:
                        for existing_vertex in [existing_start, existing_end]:
                            if np.allclose(new_vertex, existing_vertex, atol=1e-6):
                                shares_vertex = True
                                break
                        if shares_vertex:
                            break

                    if shares_vertex:
                        continue

                    # Check for proper intersection
                    if self._line_segments_intersect_proper(new_start, new_end,
                                                            existing_start, existing_end):
                        return True

        return False

    def _check_mesh_self_intersection(self, mesh: np.ndarray) -> bool:
        """
        Check if a mesh has self-intersecting edges.

        Args:
            mesh: Mesh vertices to check

        Returns:
            True if mesh self-intersects, False otherwise
        """
        if len(mesh) < 4:  # Triangles cannot self-intersect
            return False

        n = len(mesh)

        # Check each edge against all non-adjacent edges
        for i in range(n):
            edge1_start = mesh[i]
            edge1_end = mesh[(i + 1) % n]

            # Check against non-adjacent edges
            for j in range(i + 2, n):
                # Skip the closing edge (last edge with first edge)
                if j == n - 1 and i == 0:
                    continue

                edge2_start = mesh[j]
                edge2_end = mesh[(j + 1) % n]

                if self._line_segments_intersect_proper(edge1_start, edge1_end, edge2_start, edge2_end):
                    return True

        return False

    def _get_updated_boundary(self, ref_idx: int, mesh: np.ndarray, action_type: int) -> Optional[np.ndarray]:
        """
        Get the updated boundary without modifying the current state.

        Args:
            ref_idx: Reference vertex index
            mesh: Generated mesh vertices
            action_type: Type of action (-1, 0, 1)

        Returns:
            Updated boundary or None if invalid
        """
        n_boundary = len(self.current_boundary)

        if action_type == -1:  # Remove current and next vertex
            indices_to_remove = {ref_idx, (ref_idx + 1) % n_boundary}
            new_boundary = []
            for i in range(n_boundary):
                if i not in indices_to_remove:
                    new_boundary.append(self.current_boundary[i])
            return np.array(new_boundary) if len(new_boundary) >= 3 else None

        elif action_type == 1:  # Remove previous and current vertex
            indices_to_remove = {(ref_idx - 1) % n_boundary, ref_idx}
            new_boundary = []
            for i in range(n_boundary):
                if i not in indices_to_remove:
                    new_boundary.append(self.current_boundary[i])
            return np.array(new_boundary) if len(new_boundary) >= 3 else None

        else:  # action_type == 0: Add new vertex
            new_point = mesh[0]
            new_boundary = []
            for i in range(n_boundary):
                if i == ref_idx:
                    new_boundary.append(new_point)
                else:
                    new_boundary.append(self.current_boundary[i])
            return np.array(new_boundary)

    def _check_boundary_self_intersection(self, boundary: np.ndarray) -> bool:
        """
        Check if a boundary has self-intersections.

        Args:
            boundary: Boundary vertices to check

        Returns:
            True if boundary self-intersects, False otherwise
        """
        if len(boundary) < 4:  # Need at least 4 vertices to have self-intersection
            return False

        n = len(boundary)

        # Check each edge against all non-adjacent edges
        for i in range(n):
            edge1_start = boundary[i]
            edge1_end = boundary[(i + 1) % n]

            # Check against non-adjacent edges
            for j in range(i + 2, n):
                # Skip adjacent edges and the edge that closes the boundary
                if j == (i + n - 1) % n:
                    continue

                edge2_start = boundary[j]
                edge2_end = boundary[(j + 1) % n]

                if self._line_segments_intersect_proper(edge1_start, edge1_end, edge2_start, edge2_end):
                    return True

        return False

    def _update_boundary(self, ref_idx: int, mesh: np.ndarray, action_type: int):
        """Update boundary following original logic with proper element integration."""
        # This method is now simplified since boundary is pre-validated in step()
        # Update area ratio
        current_area = calculate_polygon_area(self.current_boundary)
        self.current_area_ratio = current_area / self.original_area if self.original_area > 0 else 0.0

    def _validate_and_apply_mesh(self, ref_idx: int, mesh: np.ndarray, action_type: int) -> Tuple[float, bool, bool]:
        """
        Validate mesh and apply it if valid.

        Args:
            ref_idx: Reference vertex index
            mesh: Generated mesh vertices
            action_type: Type of action (-1, 0, 1)

        Returns:
            Tuple of (reward, terminated, failed)
        """
        temp_boundary = self._get_updated_boundary(ref_idx, mesh, action_type)

        if (temp_boundary is not None and
                not self._check_intersection_with_boundary(mesh, ref_idx, action_type) and
                not self._check_mesh_intersection_with_existing_meshes(mesh) and
                not self._check_mesh_self_intersection(mesh) and
                not self._check_boundary_self_intersection(temp_boundary)):

            # Store old boundary for validation
            old_boundary = self.current_boundary.copy()
            old_boundary_size = len(old_boundary)

            self.generated_elements.append(mesh)
            self.current_boundary = temp_boundary  # Use the pre-validated boundary
            self._update_boundary(ref_idx, mesh, action_type)  # Update area ratio

            new_boundary_size = len(self.current_boundary)

            # Debug output for boundary updates
            if self.step_count % 100 == 0:  # Print every 100 steps
                print(f"Step {self.step_count}: Action type {action_type}, "
                      f"Boundary: {old_boundary_size} -> {new_boundary_size} vertices, "
                      f"Elements: {len(self.generated_elements)}, Valid mesh generated")

            # Calculate reward using original methods
            reward = self._calculate_reward_original(mesh, action_type)
            failed = False

            self.episode_reward += reward
            self.failed_num = 0

            # Check if completed
            if len(self.current_boundary) <= 5:
                reward += 10
                terminated = True
                if len(self.current_boundary) == 4:
                    final_element = self.current_boundary.copy()
                    self.generated_elements.append(final_element)
            else:
                terminated = False

            return reward, terminated, failed
        else:
            reward = -1.0 / len(self.generated_elements) if len(self.generated_elements) else -1.0
            terminated = False
            failed = True

            # Log why mesh was rejected
            if temp_boundary is None:
                if self.step_count % 100 == 0:
                    print(f"Step {self.step_count}: Mesh rejected - invalid boundary update")
            elif self._check_intersection_with_boundary(mesh, ref_idx, action_type):
                if self.step_count % 100 == 0:
                    print(
                        f"Step {self.step_count}: Mesh rejected - mesh-boundary intersection (action_type: {action_type})")
            elif self._check_mesh_intersection_with_existing_meshes(mesh):
                if self.step_count % 100 == 0:
                    print(
                        f"Step {self.step_count}: Mesh rejected - mesh-mesh intersection (action_type: {action_type})")
            elif self._check_mesh_self_intersection(mesh):
                if self.step_count % 100 == 0:
                    print(
                        f"Step {self.step_count}: Mesh rejected - mesh self-intersection (action_type: {action_type})")
            elif self._check_boundary_self_intersection(temp_boundary):
                if self.step_count % 100 == 0:
                    print(
                        f"Step {self.step_count}: Mesh rejected - boundary self-intersection after update (action_type: {action_type})")

            return reward, terminated, failed

    def _calculate_reward_original(self, mesh: np.ndarray, action_type: int) -> float:
        """Calculate reward using original author's methods."""

        if self.reward_method == 10:  # Original method 10
            # Calculate mesh quality
            quality = calculate_element_quality(mesh)

            # Add speed penalty based on area
            mesh_area = calculate_polygon_area(mesh)
            speed_penalty = self._get_speed_penalty(mesh_area)

            reward = quality + speed_penalty

        elif self.reward_method == 2:  # Alternative method
            quality = calculate_element_quality(mesh)
            transition_quality = self._get_transition_quality(mesh)
            forward_quality = 5 / len(self.current_boundary)
            reward = quality * transition_quality + forward_quality

        else:  # Default method
            reward = calculate_element_quality(mesh)

        return reward

    def _get_speed_penalty(self, mesh_area: float) -> float:
        """Calculate speed penalty following original approach."""
        # Estimate area range based on current boundary
        edge_lengths = []
        n_vertices = len(self.current_boundary)

        for i in range(n_vertices):
            edge_length = np.linalg.norm(
                self.current_boundary[(i + 1) % n_vertices] - self.current_boundary[i]
            )
            edge_lengths.append(edge_length)

        if edge_lengths:
            min_edge = min(edge_lengths)
            max_edge = max(edge_lengths)

            min_area = (min_edge ** 2) * 0.5
            critical_area = (max_edge ** 2) * 0.5

            if min_area <= mesh_area < critical_area:
                speed_penalty = (mesh_area - critical_area) / (critical_area - min_area)
            elif mesh_area < min_area:
                speed_penalty = -1
            else:
                speed_penalty = 0
        else:
            speed_penalty = 0

        return speed_penalty

    def _get_transition_quality(self, mesh: np.ndarray) -> float:
        """Calculate transition quality following original approach."""
        # Simplified transition quality based on mesh angles
        angles = []
        n_vertices = len(mesh)

        for i in range(n_vertices):
            p1 = mesh[(i - 1) % n_vertices]
            p2 = mesh[i]
            p3 = mesh[(i + 1) % n_vertices]

            v1 = p1 - p2
            v2 = p3 - p2

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 1e-8 and v2_norm > 1e-8:
                cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)

        if angles:
            # Quality based on how close angles are to 90 degrees
            angle_quality = np.mean([1.0 - abs(angle - 90.0) / 90.0 for angle in angles])
            return max(0.2, angle_quality)  # Minimum quality threshold
        else:
            return 0.2

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
