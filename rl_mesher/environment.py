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

        # Reward function parameters from paper
        self.v_density = config['environment'].get('v_density', 1.0)
        self.M_angle = config['environment'].get('M_angle', 60.0)

        # Reward function parameters from paper
        self.v_density = config['environment'].get('v_density', 1.0)
        self.M_angle = config['environment'].get('M_angle', 60.0)

        # Additional tracking variables from original
        self.failed_num = 0
        self.not_valid_points = []
        self.TYPE_THRESHOLD = 0.3  # Original threshold for action type decision

        print(f"Environment initialized with neighbor_num={self.neighbor_num}, radius_num={self.radius_num}")

    def _ensure_clockwise(self, boundary: np.ndarray) -> np.ndarray:
        """Ensure boundary vertices are in clockwise order."""
        if len(boundary) < 3:
            return boundary

        # Calculate area using shoelace formula
        area = 0.0
        for i in range(len(boundary)):
            j = (i + 1) % len(boundary)
            area += boundary[i, 0] * boundary[j, 1]
            area -= boundary[j, 0] * boundary[i, 1]

        area = area / 2.0

        # If area is positive, boundary is counter-clockwise, so reverse it
        if area > 0:
            return boundary[::-1].copy()
        else:
            return boundary

    def _load_domain(self) -> np.ndarray:
        """Load domain from file."""
        domain_path = os.path.join(self.data_dir, self.domain_file)
        return load_domain_from_file(domain_path)

    def _get_observation(self, ref_idx: int) -> np.ndarray:
        """Get observation for current state following original author's method."""
        boundary = self.current_boundary

        if boundary is None or len(boundary) < 3:
            # Return zero observation if no valid boundary
            obs_dim = 2 * (self.neighbor_num + self.radius_num)
            return np.zeros(obs_dim, dtype=np.float32)

        # Ensure ref_idx is valid for current boundary
        if ref_idx >= len(boundary):
            ref_idx = len(boundary) - 1
        if ref_idx < 0:
            ref_idx = 0

        # Get state components following original approach
        # neighbor_num is split between left and right neighbors
        n_neighbors_per_side = self.neighbor_num // 2
        state_components = get_state_components(
            boundary=boundary,
            ref_idx=ref_idx,
            n_neighbors=n_neighbors_per_side,
            n_fan_points=self.radius_num,
            beta_obs=self.radius
        )

        # Extract coordinates in the correct order
        observation_coords = []

        # Add left neighbors first
        left_neighbors = state_components.get('left_neighbors', [])
        for i in range(n_neighbors_per_side):
            if i < len(left_neighbors):
                coord = left_neighbors[i]
                observation_coords.extend([coord[0], coord[1]])
            else:
                observation_coords.extend([0.0, 0.0])

        # Add right neighbors
        right_neighbors = state_components.get('right_neighbors', [])
        for i in range(n_neighbors_per_side):
            if i < len(right_neighbors):
                coord = right_neighbors[i]
                observation_coords.extend([coord[0], coord[1]])
            else:
                observation_coords.extend([0.0, 0.0])

        # Add fan points
        fan_points = state_components.get('fan_points', [])
        for i in range(self.radius_num):
            if i < len(fan_points):
                coord = fan_points[i]
                observation_coords.extend([coord[0], coord[1]])
            else:
                observation_coords.extend([0.0, 0.0])

        # Ensure observation has correct size
        expected_size = 2 * (self.neighbor_num + self.radius_num)
        while len(observation_coords) < expected_size:
            observation_coords.append(0.0)

        observation_coords = observation_coords[:expected_size]

        return np.array(observation_coords, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset to original boundary
        self.current_boundary = self.original_boundary.copy()
        self.generated_elements = []
        self.current_area_ratio = 1.0
        self.step_count = 0
        self.episode_reward = 0.0
        self.failed_num = 0

        # Calculate initial reference vertex with safety checks
        try:
            ref_idx = calculate_reference_vertex(self.current_boundary, nrv=2)
            # Ensure ref_idx is valid
            if ref_idx >= len(self.current_boundary):
                ref_idx = len(self.current_boundary) - 1
            if ref_idx < 0:
                ref_idx = 0
        except Exception as e:
            print(f"Error calculating reference vertex in reset: {e}")
            ref_idx = 0  # Fallback to first vertex

        observation = self._get_observation(ref_idx)
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray, timestep: Optional[int] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.step_count += 1

        # Prevent infinite episodes
        if self.step_count >= self.max_steps:
            observation = self._get_observation(0)
            return observation, -10.0, True, False, self._get_info()

        if self.current_boundary is None or len(self.current_boundary) < 3:
            observation = self._get_observation(0)
            return observation, -1.0, True, False, self._get_info()

        # Calculate reference vertex with safety checks
        try:
            ref_idx = calculate_reference_vertex(self.current_boundary, nrv=2)
            # Ensure ref_idx is valid for current boundary
            if ref_idx >= len(self.current_boundary):
                ref_idx = len(self.current_boundary) - 1
            if ref_idx < 0:
                ref_idx = 0
        except Exception as e:
            print(f"Error calculating reference vertex: {e}")
            ref_idx = 0  # Fallback to first vertex

        # Generate action coordinates with proper constraints
        ref_vertex = self.current_boundary[ref_idx]

        # Generate action coordinates with proper constraints for quadrilateral generation
        ref_vertex = self.current_boundary[ref_idx]

        # Get neighboring vertices for local coordinate system
        left_idx = (ref_idx - 1) % len(self.current_boundary)
        right_idx = (ref_idx + 1) % len(self.current_boundary)
        left_vertex = self.current_boundary[left_idx]
        right_vertex = self.current_boundary[right_idx]

        # Calculate local coordinate system based on boundary edges
        edge_to_left = left_vertex - ref_vertex
        edge_to_right = right_vertex - ref_vertex

        # Use average edge length as base scale, but be more conservative
        left_length = np.linalg.norm(edge_to_left)
        right_length = np.linalg.norm(edge_to_right)
        base_scale = min(left_length, right_length) / 6.0  # Much more conservative

        # Create local coordinate system
        # Primary direction: towards interior (perpendicular to boundary)
        edge_direction = (edge_to_right - edge_to_left)
        if np.linalg.norm(edge_direction) > 1e-8:
            edge_direction = edge_direction / np.linalg.norm(edge_direction)
        else:
            edge_direction = np.array([1.0, 0.0])

        # Perpendicular direction (towards interior)
        perp_direction = np.array([-edge_direction[1], edge_direction[0]])

        # Convert action to local coordinates
        local_x = action[1] * base_scale  # Along edge direction
        local_y = action[2] * base_scale  # Perpendicular to edge (towards interior)

        # Ensure y is always towards interior (negative y for clockwise boundary)
        if local_y > 0:
            local_y = -abs(local_y)  # Force towards interior

        action_offset = local_x * edge_direction + local_y * perp_direction

        if mesh is None:
            observation = self._get_observation(ref_idx)
            reward = -1.0 / max(len(self.generated_elements), 1)
            return observation, reward, False, True, self._get_info()

        # Determine action type and create mesh following paper's Fig. 5
        rule_type = action[0]  # First component determines action type
        mesh = None
        action_type = None

        # Determine action type and create QUADRILATERAL following paper's Fig. 5
        rule_type = action[0]  # First component determines action type
        mesh = None
        action_type = None
        n_boundary = len(self.current_boundary)

        if rule_type <= -self.TYPE_THRESHOLD and n_boundary >= 5:  # Type 0: 添加0个新顶点 (约20%)
            # Use 4 consecutive boundary vertices to form quadrilateral
            # Need at least 5 boundary vertices to safely remove 2
            mesh = np.array([
                self.current_boundary[ref_idx],  # V0 (reference)
                self.current_boundary[(ref_idx + 1) % n_boundary],  # V1 (right neighbor)
                self.current_boundary[(ref_idx + 2) % n_boundary],  # V2 (right+1)
                self.current_boundary[(ref_idx - 1) % n_boundary]  # V3 (left neighbor)
            ])
            action_type = 0

        elif rule_type >= self.TYPE_THRESHOLD:  # Type 1: 添加1个新顶点 (约80%)
            # Add 1 new vertex + 3 boundary vertices to form quadrilateral
            # Generate new vertex in fan-shaped coordinate space
            max_attempts = 3
            new_vertex = None

            for attempt in range(max_attempts):
                scale_factor = 1.0 / (attempt + 1)  # Scale down if attempts fail
                current_offset = action_offset * scale_factor
                candidate_vertex = ref_vertex + current_offset

                # Check if new vertex is inside boundary
                if self._is_point_inside_boundary_robust(candidate_vertex):
                    new_vertex = candidate_vertex
                    break

            # If all attempts failed, create very conservative vertex
            if new_vertex is None:
                tiny_offset = action_offset * 0.05  # Very small offset
                new_vertex = ref_vertex + tiny_offset

            # Create quadrilateral: new_vertex + 3 boundary vertices
            mesh = np.array([
                new_vertex,  # New vertex
                self.current_boundary[(ref_idx - 1) % n_boundary],  # Left neighbor
                self.current_boundary[ref_idx],  # Reference vertex
                self.current_boundary[(ref_idx + 1) % n_boundary]  # Right neighbor
            ])
            action_type = 1

        else:  # Fallback to Type 1 for edge cases or insufficient boundary vertices
            # Always try Type 1 as fallback since it only needs 3 boundary vertices
            max_attempts = 3
            new_vertex = None

            for attempt in range(max_attempts):
                scale_factor = 0.1 / (attempt + 1)  # Very conservative scaling for fallback
                current_offset = action_offset * scale_factor
                candidate_vertex = ref_vertex + current_offset

                if self._is_point_inside_boundary_robust(candidate_vertex):
                    new_vertex = candidate_vertex
                    break

            # If still failed, use very small offset from reference vertex
            if new_vertex is None:
                new_vertex = ref_vertex + np.array([base_scale * 0.01, base_scale * 0.01])

            mesh = np.array([
                new_vertex,  # New vertex
                self.current_boundary[(ref_idx - 1) % n_boundary],  # Left neighbor
                self.current_boundary[ref_idx],  # Reference vertex
                self.current_boundary[(ref_idx + 1) % n_boundary]  # Right neighbor
            ])
            action_type = 1

        # Validate mesh and apply if valid
        if mesh is not None and self._validate_mesh_simple(mesh):
            reward, terminated, failed = self._validate_and_apply_mesh(ref_idx, mesh, action_type)
        else:
            reward = -1.0 / max(len(self.generated_elements), 1)
            terminated = False
            failed = True

            if self.step_count % 100 == 0:
                if mesh is None:
                    print(f"Step {self.step_count}: Failed to generate valid quadrilateral - mesh is None")
                else:
                    mesh_area = calculate_polygon_area(mesh)
                    print(
                        f"Step {self.step_count}: Invalid quadrilateral - area: {mesh_area:.6f}, vertices: {len(mesh)}")

        # Get next observation
        if not terminated and not failed:
            # Recalculate reference vertex for updated boundary
            if len(self.current_boundary) > 0:
                try:
                    next_ref_idx = calculate_reference_vertex(self.current_boundary, nrv=2)
                    # Ensure the index is valid
                    if next_ref_idx >= len(self.current_boundary):
                        next_ref_idx = len(self.current_boundary) - 1
                    if next_ref_idx < 0:
                        next_ref_idx = 0
                except Exception as e:
                    print(f"Error calculating next reference vertex: {e}")
                    next_ref_idx = 0
                observation = self._get_observation(next_ref_idx)
            else:
                observation = self._get_observation(0)
        else:
            # For terminated/failed cases, use safe index
            safe_ref_idx = min(ref_idx, len(self.current_boundary) - 1) if len(self.current_boundary) > 0 else 0
            observation = self._get_observation(safe_ref_idx)

        info = self._get_info()
        return observation, reward, terminated, failed, info

    def _validate_mesh_simple(self, mesh: np.ndarray) -> bool:
        """Validate quadrilateral mesh."""
        if len(mesh) != 4:  # Must be quadrilateral
            return False

        # Check if mesh area is positive and reasonable
        area = calculate_polygon_area(mesh)
        if area <= 1e-8:  # Too small area
            return False

        # Check if all vertices are different
        for i in range(4):
            for j in range(i + 1, 4):
                if np.allclose(mesh[i], mesh[j], atol=1e-6):
                    return False

        # Check diagonal lengths - they shouldn't be too different (avoid degenerate quads)
        diag1 = np.linalg.norm(mesh[2] - mesh[0])
        diag2 = np.linalg.norm(mesh[3] - mesh[1])
        if max(diag1, diag2) > 20 * min(diag1, diag2):  # Too distorted
            return False

        # Check that it's a valid convex quadrilateral (no self-intersection)
        return not self._check_mesh_self_intersection(mesh)

    def _is_point_inside_boundary_robust(self, point: np.ndarray) -> bool:
        """Check if point is inside boundary using ray casting with robustness."""
        if self.current_boundary is None or len(self.current_boundary) < 3:
            return False

        boundary = self.current_boundary
        x, y = point[0], point[1]
        n = len(boundary)
        inside = False

        p1x, p1y = boundary[0]
        for i in range(1, n + 1):
            p2x, p2y = boundary[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _point_inside_boundary(self, point: np.ndarray) -> bool:
        """Check if point is inside current boundary - wrapper for compatibility."""
        return self._is_point_inside_boundary_robust(point)

    def _check_intersection_with_boundary(self, mesh: np.ndarray, ref_idx: int, action_type: int) -> bool:
        """
        Check if quadrilateral mesh intersects with current boundary.

        Args:
            mesh: Generated quadrilateral mesh vertices
            ref_idx: Reference vertex index
            action_type: Type of action (0=no new vertex, 1=one new vertex)

        Returns:
            True if intersection detected, False otherwise
        """
        if len(mesh) != 4:  # Must be quadrilateral
            return True  # Invalid mesh

        # Get mesh edges
        mesh_edges = []
        for i in range(4):
            mesh_edges.append((mesh[i], mesh[(i + 1) % 4]))

        # Get current boundary edges
        n_boundary = len(self.current_boundary)
        boundary_edges = []
        for i in range(n_boundary):
            boundary_edges.append((self.current_boundary[i], self.current_boundary[(i + 1) % n_boundary]))

        # Define which boundary vertices/edges are used by the mesh
        if action_type == 0:  # Type 0: uses 4 consecutive boundary vertices
            used_vertices = {
                ref_idx,
                (ref_idx + 1) % n_boundary,
                (ref_idx + 2) % n_boundary,
                (ref_idx - 1) % n_boundary
            }
        elif action_type == 1:  # Type 1: uses 3 boundary vertices + 1 new vertex
            used_vertices = {
                (ref_idx - 1) % n_boundary,
                ref_idx,
                (ref_idx + 1) % n_boundary
            }
        else:
            used_vertices = set()

        # Check each mesh edge against boundary edges
        for mesh_edge in mesh_edges:
            mesh_start, mesh_end = mesh_edge

            for i, boundary_edge in enumerate(boundary_edges):
                boundary_start, boundary_end = boundary_edge

                # Skip boundary edges that involve vertices used in the mesh
                if i in used_vertices or (i + 1) % n_boundary in used_vertices:
                    continue

                # Check for proper intersection
                if self._line_segments_intersect_proper(mesh_start, mesh_end,
                                                        boundary_start, boundary_end):
                    return True

        return False

    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate distance from point to line segment."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-8:
            return np.linalg.norm(point_vec)

        line_unit = line_vec / line_len
        proj_length = np.dot(point_vec, line_unit)

        if proj_length < 0:
            return np.linalg.norm(point_vec)
        elif proj_length > line_len:
            return np.linalg.norm(point - line_end)
        else:
            proj_point = line_start + proj_length * line_unit
            return np.linalg.norm(point - proj_point)

    def _check_mesh_intersection_with_existing_meshes(self, mesh: np.ndarray) -> bool:
        """Check if new mesh intersects with existing generated meshes."""
        for existing_mesh in self.generated_elements:
            if self._meshes_intersect(mesh, existing_mesh):
                return True
        return False

    def _meshes_intersect(self, mesh1: np.ndarray, mesh2: np.ndarray) -> bool:
        """Check if two meshes intersect."""
        # Get edges for both meshes
        edges1 = []
        for i in range(len(mesh1)):
            edges1.append((mesh1[i], mesh1[(i + 1) % len(mesh1)]))

        edges2 = []
        for i in range(len(mesh2)):
            edges2.append((mesh2[i], mesh2[(i + 1) % len(mesh2)]))

        # Check all edge pairs
        for edge1 in edges1:
            for edge2 in edges2:
                if self._line_segments_intersect_proper(edge1[0], edge1[1], edge2[0], edge2[1]):
                    return True
        return False

    def _line_segments_intersect_proper(self, p1: np.ndarray, q1: np.ndarray,
                                        p2: np.ndarray, q2: np.ndarray) -> bool:
        """Check if two line segments intersect properly (not just touching at endpoints)."""

        def orientation(p, q, r):
            """Find orientation of ordered triplet (p, q, r)."""
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-8:
                return 0  # collinear
            return 1 if val > 0 else 2  # clockwise or counter-clockwise

        def on_segment(p, q, r):
            """Check if point q lies on line segment pr."""
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case - proper intersection
        if o1 != o2 and o3 != o4:
            return True

        # Special cases - we consider these as non-intersecting for our purposes
        # to avoid false positives when segments share endpoints
        return False

    def _check_mesh_self_intersection(self, mesh: np.ndarray) -> bool:
        """
        Check if mesh has self-intersections.

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
        Get updated boundary for quadrilateral mesh generation.

        Args:
            ref_idx: Reference vertex index
            mesh: Generated quadrilateral mesh vertices
            action_type: Type of action (0=no new vertex, 1=one new vertex)

        Returns:
            Updated boundary or None if invalid
        """
        n_boundary = len(self.current_boundary)

        if action_type == 0:  # Type 0: Used 4 consecutive boundary vertices
            # Remove the 2 middle vertices (ref+1, ref+2) from the 4-vertex sequence
            # Keep ref_idx and (ref_idx-1), remove (ref_idx+1) and (ref_idx+2)
            indices_to_remove = {
                (ref_idx + 1) % n_boundary,
                (ref_idx + 2) % n_boundary
            }

            new_boundary = []
            for i in range(n_boundary):
                if i not in indices_to_remove:
                    new_boundary.append(self.current_boundary[i])

            return np.array(new_boundary) if len(new_boundary) >= 3 else None

        elif action_type == 1:  # Type 1: Added 1 new vertex + 3 boundary vertices
            # Replace ref_idx with the new vertex (first vertex in mesh)
            new_vertex = mesh[0]  # First vertex is the new vertex

            new_boundary = []
            for i in range(n_boundary):
                if i == ref_idx:
                    new_boundary.append(new_vertex)  # Replace ref with new vertex
                else:
                    new_boundary.append(self.current_boundary[i])

            return np.array(new_boundary)

        else:
            return None

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

            # Store old boundary for reward calculation
            old_boundary = self.current_boundary.copy()
            old_boundary_size = len(old_boundary)

            self.generated_elements.append(mesh)
            self.current_boundary = temp_boundary  # Use the pre-validated boundary
            self._update_boundary(ref_idx, mesh, action_type)  # Update area ratio

            new_boundary_size = len(self.current_boundary)

            # Debug output for successful quadrilateral generation
            if self.step_count % 100 == 0:
                mesh_area = calculate_polygon_area(mesh)
                action_desc = "Type 0 (4 boundary vertices)" if action_type == 0 else "Type 1 (1 new + 3 boundary)"
                print(f"Step {self.step_count}: Generated quadrilateral - {action_desc}, "
                      f"area: {mesh_area:.4f}, boundary: {old_boundary_size} -> {new_boundary_size}")

            # Calculate reward using corrected paper's methods
            reward = self._calculate_reward_original(mesh, action_type, old_boundary, self.current_boundary)
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

    def _calculate_reward_original(self, mesh: np.ndarray, action_type: int,
                                   boundary_before: np.ndarray, boundary_after: np.ndarray) -> float:
        """
        Calculate reward using paper's formula (6): m_t = η_t^e + η_t^b + μ_t

        Args:
            mesh: Generated element vertices
            action_type: Type of action (-1, 0, 1)
            boundary_before: Boundary before element generation
            boundary_after: Boundary after element generation

        Returns:
            Calculated reward following paper's equation (6)
        """
        # Use paper's reward components function
        eta_e, eta_b, mu_t = calculate_reward_components(
            element_vertices=mesh,
            boundary_before=boundary_before,
            boundary_after=boundary_after,
            area_ratio=self.current_area_ratio,
            v_density=self.v_density,
            M_angle=self.M_angle
        )

        # Apply paper's formula (6): m_t = η_t^e + η_t^b + μ_t
        reward = eta_e + eta_b + mu_t

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
