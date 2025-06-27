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
        """Take a step in the environment following original logic."""
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

            # Validate and add mesh - CRITICAL FIX: Only use current boundary vertices
            if mesh is not None and self._validate_mesh(mesh) and self._validate_mesh_uses_current_boundary(mesh):
                # Store old boundary for validation
                old_boundary = self.current_boundary.copy()
                old_boundary_size = len(old_boundary)

                self.generated_elements.append(mesh)
                self._update_boundary(ref_idx, mesh, action_type)

                new_boundary_size = len(self.current_boundary)

                # Debug output for boundary updates
                if self.step_count % 100 == 0:  # Print every 100 steps
                    print(f"Step {self.step_count}: Action type {action_type}, "
                          f"Boundary: {old_boundary_size} -> {new_boundary_size} vertices, "
                          f"Elements: {len(self.generated_elements)}")

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
            else:
                reward = -1.0 / len(self.generated_elements) if len(self.generated_elements) else -1.0
                terminated = False
                failed = True

                # Log why mesh was rejected
                if mesh is not None:
                    if not self._validate_mesh(mesh):
                        if self.step_count % 100 == 0:
                            print(f"Step {self.step_count}: Mesh rejected - failed basic validation")
                    elif not self._validate_mesh_uses_current_boundary(mesh):
                        print(f"Step {self.step_count}: Mesh rejected - uses vertices not in current boundary")
                        print(f"  Action type: {action_type}, Current boundary size: {len(self.current_boundary)}")
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

    def _validate_mesh_uses_current_boundary(self, mesh: np.ndarray) -> bool:
        """
        Critical validation: ensure all mesh vertices come from current boundary.
        This prevents elements from using vertices from previous boundary states.
        """
        tolerance = 1e-6

        # Check each mesh vertex
        for mesh_vertex in mesh:
            # Check if this vertex is in current boundary
            found_in_boundary = False
            for boundary_vertex in self.current_boundary:
                if np.linalg.norm(mesh_vertex - boundary_vertex) < tolerance:
                    found_in_boundary = True
                    break

            # If it's not in current boundary, it might be a newly generated point
            # For action_type 0, the first vertex is the new point
            if not found_in_boundary:
                # Check if it's inside the current boundary (valid new point)
                if not self._is_point_inside_boundary(mesh_vertex):
                    return False

        return True

    def _action_to_point_constrained(self, action: np.ndarray, ref_idx: int) -> Optional[np.ndarray]:
        """
        Convert action to world coordinates with boundary constraints.

        This is the key fix - ensure generated points are always inside the boundary.
        """
        x, y = action[0], action[1]

        if not hasattr(self, 'current_base_length'):
            return None

        # Get local boundary constraints
        max_distance = self._get_max_valid_distance(ref_idx, x, y)

        # Constrain the action to valid range
        action_distance = np.sqrt(x * x + y * y)
        if action_distance > 1e-8:
            # Scale down if needed to stay within boundary
            scale_factor = min(1.0, max_distance / (action_distance * self.current_base_length * self.max_radius))
            x *= scale_factor
            y *= scale_factor

        # Scale by base length and max radius
        x_world = x * self.current_base_length * self.max_radius
        y_world = y * self.current_base_length * self.max_radius

        # Rotate by reference direction
        cos_theta = self.current_ref_direction[0]
        sin_theta = self.current_ref_direction[1]

        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        local_coords = np.array([x_world, y_world])
        world_coords = self.current_ref_vertex + rotation_matrix @ local_coords

        # Final safety check - if still outside, project to nearest valid point
        if not self._is_point_inside_boundary(world_coords):
            world_coords = self._project_point_inside(world_coords, ref_idx)

        return world_coords

    def _get_max_valid_distance(self, ref_idx: int, x: float, y: float) -> float:
        """
        Calculate maximum valid distance in the direction of (x, y) from reference vertex.
        """
        if abs(x) < 1e-8 and abs(y) < 1e-8:
            return self.current_base_length * self.max_radius

        # Normalize direction
        direction = np.array([x, y])
        direction = direction / np.linalg.norm(direction)

        # Rotate direction to world coordinates
        cos_theta = self.current_ref_direction[0]
        sin_theta = self.current_ref_direction[1]
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        world_direction = rotation_matrix @ direction

        # Find intersection with boundary edges
        min_distance = self.current_base_length * self.max_radius
        ref_vertex = self.current_ref_vertex

        n_vertices = len(self.current_boundary)
        for i in range(n_vertices):
            p1 = self.current_boundary[i]
            p2 = self.current_boundary[(i + 1) % n_vertices]

            # Check intersection with edge p1-p2
            intersection_distance = self._ray_edge_intersection(ref_vertex, world_direction, p1, p2)
            if intersection_distance is not None and intersection_distance > 1e-8:
                min_distance = min(min_distance, intersection_distance * 0.9)  # Safety margin

        return max(min_distance, self.current_base_length * 0.1)  # Minimum distance

    def _ray_edge_intersection(self, ray_origin: np.ndarray, ray_direction: np.ndarray,
                               edge_p1: np.ndarray, edge_p2: np.ndarray) -> Optional[float]:
        """
        Calculate intersection distance between a ray and an edge.
        Returns None if no intersection or intersection is behind ray origin.
        """
        edge_vector = edge_p2 - edge_p1
        edge_length = np.linalg.norm(edge_vector)
        if edge_length < 1e-8:
            return None

        # Solve ray-line intersection
        # ray: ray_origin + t * ray_direction
        # edge: edge_p1 + s * edge_vector
        # Solve: ray_origin + t * ray_direction = edge_p1 + s * edge_vector

        A = np.column_stack([ray_direction, -edge_vector])
        b = edge_p1 - ray_origin

        try:
            params = np.linalg.solve(A, b)
            t, s = params[0], params[1]

            # Check if intersection is valid
            if t > 1e-8 and 0 <= s <= 1:  # t > 0 (forward ray), 0 <= s <= 1 (on edge)
                return t
        except np.linalg.LinAlgError:
            pass

        return None

    def _project_point_inside(self, point: np.ndarray, ref_idx: int) -> np.ndarray:
        """
        Project a point to the nearest valid location inside the boundary.
        """
        if self._is_point_inside_boundary(point):
            return point

        # Find closest point on boundary
        min_distance = float('inf')
        closest_point = self.current_ref_vertex

        n_vertices = len(self.current_boundary)
        for i in range(n_vertices):
            p1 = self.current_boundary[i]
            p2 = self.current_boundary[(i + 1) % n_vertices]

            # Project point onto edge
            projected = self._project_point_on_edge(point, p1, p2)
            distance = np.linalg.norm(projected - point)

            if distance < min_distance:
                min_distance = distance
                closest_point = projected

        # Move slightly inside from boundary
        direction = closest_point - point
        if np.linalg.norm(direction) > 1e-8:
            direction = direction / np.linalg.norm(direction)
            closest_point = closest_point - direction * (self.current_base_length * 0.01)  # Small offset

        return closest_point

    def _project_point_on_edge(self, point: np.ndarray, edge_p1: np.ndarray, edge_p2: np.ndarray) -> np.ndarray:
        """Project a point onto an edge (line segment)."""
        edge_vector = edge_p2 - edge_p1
        edge_length_sq = np.dot(edge_vector, edge_vector)

        if edge_length_sq < 1e-8:
            return edge_p1

        # Calculate projection parameter
        t = np.dot(point - edge_p1, edge_vector) / edge_length_sq
        t = np.clip(t, 0.0, 1.0)  # Clamp to edge

        return edge_p1 + t * edge_vector

    def _is_point_inside_boundary(self, point: np.ndarray) -> bool:
        """Check if point is inside current boundary using ray casting."""
        x, y = point
        n = len(self.current_boundary)
        inside = False

        p1x, p1y = self.current_boundary[0]
        for i in range(1, n + 1):
            p2x, p2y = self.current_boundary[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _validate_mesh(self, mesh: np.ndarray) -> bool:
        """Validate mesh following original criteria."""
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

    def _update_boundary(self, ref_idx: int, mesh: np.ndarray, action_type: int):
        """Update boundary following original logic with proper element integration."""
        n_boundary = len(self.current_boundary)

        if action_type == -1:  # Remove current and next vertex
            # Remove vertices that are now interior to the generated quad
            indices_to_remove = {ref_idx, (ref_idx + 1) % n_boundary}
            new_boundary = []
            for i in range(n_boundary):
                if i not in indices_to_remove:
                    new_boundary.append(self.current_boundary[i])
            self.current_boundary = np.array(new_boundary)

        elif action_type == 1:  # Remove previous and current vertex
            # Remove vertices that are now interior to the generated quad
            indices_to_remove = {(ref_idx - 1) % n_boundary, ref_idx}
            new_boundary = []
            for i in range(n_boundary):
                if i not in indices_to_remove:
                    new_boundary.append(self.current_boundary[i])
            self.current_boundary = np.array(new_boundary)

        else:  # action_type == 0: Add new vertex and properly update boundary
            # For type 0, we created a quad: [new_point, left_neighbor, ref_vertex, right_neighbor]
            # The ref_vertex is now interior to the quad and should be removed from boundary
            # The new_point becomes part of the boundary

            new_point = mesh[0]

            # Create new boundary by replacing ref_vertex with new_point
            new_boundary = []
            for i in range(n_boundary):
                if i == ref_idx:
                    # Replace reference vertex with new point
                    new_boundary.append(new_point)
                else:
                    new_boundary.append(self.current_boundary[i])

            self.current_boundary = np.array(new_boundary)

        # Update area ratio
        current_area = calculate_polygon_area(self.current_boundary)
        self.current_area_ratio = current_area / self.original_area if self.original_area > 0 else 0.0

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
