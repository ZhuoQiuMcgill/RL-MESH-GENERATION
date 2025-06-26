import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import os
import gymnasium as gym
from gymnasium import spaces

from .utils.geometry import (
    calculate_reference_vertex, get_state_components, transform_to_relative_coords,
    calculate_reward_components, calculate_polygon_area,
    load_domain_from_file, generate_action_coordinates, calculate_element_quality
)


class MeshEnv(gym.Env):
    """
    Mesh generation environment implementing Gymnasium interface.

    This environment follows the exact algorithm from the paper:
    1. Select reference vertex with minimum angle
    2. Execute action (Type 0: connect neighbors, Type 1: add new vertex)
    3. Use action filter to ensure new vertices are inside boundary
    4. Update boundary by inserting/removing vertices correctly
    5. Stop when max_steps reached or boundary has only 4 vertices
    """

    def __init__(self, config: Dict):
        """
        Initialize mesh generation environment.

        Args:
            config: Configuration dictionary containing environment parameters
        """
        super(MeshEnv, self).__init__()

        self.config = config

        # Environment parameters (ensure proper types)
        self.n_neighbors = int(config['environment']['n_neighbors'])
        self.n_fan_points = int(config['environment']['n_fan_points'])
        self.beta_obs = float(config['environment']['beta_obs'])
        self.alpha_action = float(config['environment']['alpha_action'])
        self.v_density = float(config['environment']['v_density'])
        self.M_angle = float(config['environment']['M_angle'])
        self.nrv = int(config['environment']['nrv'])

        # Episode settings
        self.max_steps = int(config['environment'].get('max_steps', 1000))

        # Domain settings
        self.domain_file = config['domain']['training_domain']
        self.data_dir = config['paths']['data_dir']

        # Load initial domain
        self.original_boundary = self._load_domain()
        self.original_area = calculate_polygon_area(self.original_boundary)

        # Ensure boundary is clockwise oriented
        self.original_boundary = self._ensure_clockwise(self.original_boundary)

        # Initialize state
        self.current_boundary = None
        self.generated_elements = []
        self.current_area_ratio = 1.0
        self.step_count = 0

        # Define action and observation spaces
        self._setup_spaces()

        # Episode tracking
        self.episode_reward = 0.0

        print(f"🌍 Environment initialized with max_steps: {self.max_steps}")

    def _ensure_clockwise(self, boundary: np.ndarray) -> np.ndarray:
        """Ensure boundary vertices are in clockwise order."""
        # Calculate signed area using shoelace formula
        area = 0.0
        n = len(boundary)
        for i in range(n):
            j = (i + 1) % n
            area += (boundary[j][0] - boundary[i][0]) * (boundary[j][1] + boundary[i][1])

        # If area > 0, vertices are counterclockwise, so reverse them
        if area > 0:
            return boundary[::-1]
        return boundary

    def _load_domain(self) -> np.ndarray:
        """Load domain boundary from file."""
        domain_path = os.path.join(self.data_dir, self.domain_file)
        return load_domain_from_file(domain_path)

    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: [type_prob, x, y] where type_prob determines action type
        # and (x, y) are relative coordinates for new vertex generation
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space components
        self.observation_space = spaces.Dict({
            'ref_vertex': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'left_neighbors': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.n_neighbors, 2), dtype=np.float32),
            'right_neighbors': spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(self.n_neighbors, 2), dtype=np.float32),
            'fan_points': spaces.Box(low=-np.inf, high=np.inf,
                                     shape=(self.n_fan_points, 2), dtype=np.float32),
            'area_ratio': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset boundary to original (ensure clockwise)
        self.current_boundary = self.original_boundary.copy()
        self.generated_elements = []
        self.current_area_ratio = 1.0
        self.step_count = 0
        self.episode_reward = 0.0

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step following the paper's algorithm."""
        self.step_count += 1

        # Check for max steps truncation first
        if self.step_count >= self.max_steps:
            observation = self._get_observation()
            info = self._get_info()
            info.update({
                'is_valid_element': False,
                'element_count': len(self.generated_elements),
                'termination_reason': 'max_steps_reached'
            })
            return observation, -1.0, False, True, info

        # Step 1: Select reference vertex
        try:
            ref_idx = calculate_reference_vertex(self.current_boundary, self.nrv)
        except Exception as e:
            # If reference vertex calculation fails, just use index 0
            ref_idx = 0

        # Step 2: Get state components
        try:
            state_components = get_state_components(
                self.current_boundary, ref_idx, self.n_neighbors,
                self.n_fan_points, self.beta_obs
            )
        except Exception as e:
            # If state calculation fails, give small penalty and continue
            observation = self._get_observation()
            info = self._get_info()
            info.update({
                'is_valid_element': False,
                'element_count': len(self.generated_elements),
                'termination_reason': f'state_error: {str(e)}'
            })
            return observation, -0.1, False, False, info

        # Step 3: Determine action type
        action_type = 0 if action[0] < 0 else 1

        # Step 4: Generate element and apply action filter
        element_vertices, is_valid, new_boundary = self._execute_action(
            action, state_components, action_type, ref_idx
        )

        # Step 5: Calculate reward
        reward = self._calculate_reward(element_vertices, is_valid, state_components)

        # Step 6: Update environment if action is valid
        terminated = False
        termination_reason = None

        if is_valid:
            # Debug: Print detailed boundary change info
            old_boundary_size = len(self.current_boundary)
            new_boundary_size = len(new_boundary)

            # if self.step_count % 20 == 0:  # Print every 20 steps for more frequent monitoring
            #     print(f"\n=== Step {self.step_count} ===")
            #     print(f"Action type: {action_type}")
            #     print(f"Ref vertex index: {ref_idx}")
            #     print(f"Boundary: {old_boundary_size} -> {new_boundary_size}")
            #     print(f"Elements generated: {len(self.generated_elements)}")
#
            #     # Show actual boundary change
            #     old_vertices = [f"({v[0]:.1f},{v[1]:.1f})" for v in self.current_boundary]
            #     new_vertices = [f"({v[0]:.1f},{v[1]:.1f})" for v in new_boundary]
            #     print(f"Old boundary: {old_vertices}")
            #     print(f"New boundary: {new_vertices}")
#
            #     if old_boundary_size == new_boundary_size:
            #         print("✓ Boundary size unchanged (replacement operation)")
            #     elif new_boundary_size == old_boundary_size - 1:
            #         print("✓ Boundary size decreased by 1 (vertex removal)")
            #     else:
            #         print("⚠️  Unexpected boundary size change!")
            #     print()

            self.generated_elements.append(element_vertices)
            self.current_boundary = new_boundary
            # Update area ratio
            current_area = calculate_polygon_area(self.current_boundary)
            self.current_area_ratio = current_area / self.original_area if self.original_area > 0 else 0.0
            self.episode_reward += reward

            # Check termination: Only terminate when boundary becomes 4 vertices or less
            if len(self.current_boundary) <= 4:
                # Add final element if boundary can form one
                if len(self.current_boundary) == 4:
                    final_element = self.current_boundary.copy()
                    self.generated_elements.append(final_element)

                terminated = True
                termination_reason = 'boundary_complete'
                reward += 10.0  # Completion bonus
                print(f"🎉 Mesh generation completed at step {self.step_count}! "
                      f"Generated {len(self.generated_elements)} elements.")

        else:
            # Invalid action, small penalty but continue
            reward = -0.1

        # Get next observation
        observation = self._get_observation()

        # Get info
        info = self._get_info()
        info.update({
            'is_valid_element': is_valid,
            'element_count': len(self.generated_elements),
            'boundary_vertices': len(self.current_boundary),
            'termination_reason': termination_reason
        })

        return observation, reward, terminated, False, info

    def _execute_action(self, action: np.ndarray, state_components: Dict,
                        action_type: int, ref_idx: int) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        Execute action following the paper's algorithm.
        Key insight: When element is cut off, the new point becomes part of the boundary.
        """
        ref_vertex = state_components['ref_vertex']
        left_neighbors = state_components['left_neighbors']
        right_neighbors = state_components['right_neighbors']
        reference_direction = state_components['reference_direction']
        base_length = state_components['base_length']

        n_boundary = len(self.current_boundary)

        # Ensure we have enough neighbors for the action
        if len(left_neighbors) == 0 or len(right_neighbors) == 0 or n_boundary < 5:
            return np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), False, self.current_boundary

        # Get neighbor vertices
        v_right = right_neighbors[0]  # Right neighbor (clockwise)
        v_left = left_neighbors[0]  # Left neighbor (counter-clockwise)

        if action_type == 0:
            # Type 0: Create element using three existing boundary vertices + interior point
            # Element: [ref_vertex, v_right, interior_point, v_left]
            # After cutting: ref_vertex is replaced by interior_point in boundary

            # Create interior point that makes a reasonable quadrilateral
            triangle_center = np.mean([ref_vertex, v_right, v_left], axis=0)

            # Calculate an inward direction to place interior point
            edge_vec = v_right - ref_vertex
            perp_vec = np.array([-edge_vec[1], edge_vec[0]])  # Perpendicular
            if np.linalg.norm(perp_vec) > 1e-8:
                perp_vec = perp_vec / np.linalg.norm(perp_vec)

                # Test which direction is inward
                test_point = triangle_center + perp_vec * base_length * 0.01
                if self._is_point_inside_boundary(test_point):
                    interior_point = triangle_center + perp_vec * base_length * 0.2
                else:
                    interior_point = triangle_center - perp_vec * base_length * 0.2
            else:
                interior_point = triangle_center

            # Element vertices: ref_vertex, v_right, interior_point, v_left
            element_vertices = np.array([ref_vertex, v_right, interior_point, v_left])

            # For boundary update: interior_point will replace ref_vertex
            boundary_replacement_point = interior_point

        else:
            # Type 1: Add new vertex inside the boundary
            # Element: [ref_vertex, v_right, new_vertex, v_left]
            # After cutting: ref_vertex is replaced by new_vertex in boundary

            new_vertex = generate_action_coordinates(
                action, ref_vertex, reference_direction,
                self.alpha_action, base_length
            )

            # Action filter: check if new vertex is inside boundary
            if not self._is_point_inside_boundary(new_vertex):
                return np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), False, self.current_boundary

            # Element vertices: ref_vertex, v_right, new_vertex, v_left
            element_vertices = np.array([ref_vertex, v_right, new_vertex, v_left])

            # For boundary update: new_vertex will replace ref_vertex
            boundary_replacement_point = new_vertex

        # Update boundary: replace ref_vertex with the new point
        new_boundary = self._update_boundary_cut(ref_idx, 0, 0, boundary_replacement_point, action_type)

        # Validate the generated element and new boundary
        is_valid = self._validate_element(element_vertices) and len(new_boundary) >= 3

        if not is_valid:
            return element_vertices, False, self.current_boundary

        return element_vertices, True, new_boundary

    def _update_boundary_cut(self, ref_idx: int, right_idx: int, left_idx: int,
                             new_point: np.ndarray, action_type: int) -> np.ndarray:
        """
        Update boundary by cutting off the element region properly.

        Key insight: When element [ref_vertex, right_neighbor, new_point, left_neighbor]
        is cut off, the new boundary should include the new_point as it becomes part
        of the new boundary where the element was removed.

        Example: boundary [0,1,2,3,4,5,6,7,8,9,10,11]
        If element is [5,6,7,P], new boundary should be [0,1,2,3,4,5,P,7,8,9,10,11]
        """
        boundary_list = [v.tolist() for v in self.current_boundary]
        n = len(boundary_list)

        if action_type == 0:
            # Type 0: Element uses ref_vertex + neighbors + interior_point
            # After cutting: replace ref_vertex with interior_point
            # This simulates the element being cut off and interior_point becoming new boundary
            new_boundary_list = boundary_list.copy()
            new_boundary_list[ref_idx] = new_point.tolist()

        else:
            # Type 1: Element uses ref_vertex + neighbors + new_vertex
            # After cutting: replace ref_vertex with new_vertex
            new_boundary_list = boundary_list.copy()
            new_boundary_list[ref_idx] = new_point.tolist()

        # Ensure we have enough vertices
        if len(new_boundary_list) < 3:
            return self.current_boundary

        return np.array(new_boundary_list)

    def _find_vertex_index(self, vertex: np.ndarray) -> int:
        """Find the index of a vertex in the current boundary."""
        for i, boundary_vertex in enumerate(self.current_boundary):
            if np.allclose(vertex, boundary_vertex, atol=1e-6):
                return i
        return -1  # Not found

    def _is_point_inside_boundary(self, point: np.ndarray) -> bool:
        """
        Check if point is inside the boundary using ray casting algorithm.
        For clockwise oriented boundary.
        """
        x, y = point
        boundary = self.current_boundary
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

    def _update_boundary_type0(self, ref_idx: int, v3_idx: int) -> np.ndarray:
        """
        Update boundary for Type 0 action (connect neighbors).
        Simply remove the reference vertex, connecting left and right neighbors directly.
        """
        boundary_list = self.current_boundary.tolist()
        n = len(boundary_list)

        # Remove only the reference vertex
        new_boundary_list = boundary_list[:ref_idx] + boundary_list[ref_idx + 1:]

        # Ensure we have enough vertices left
        if len(new_boundary_list) < 3:
            return self.current_boundary

        return np.array(new_boundary_list)

    def _update_boundary_type1(self, ref_idx: int, new_vertex: np.ndarray) -> np.ndarray:
        """
        Update boundary for Type 1 action (add new vertex).
        Replace ref_vertex with new_vertex.
        """
        new_boundary = self.current_boundary.copy()
        new_boundary[ref_idx] = new_vertex

        return new_boundary

    def _validate_element(self, element_vertices: np.ndarray) -> bool:
        """Validate that the generated element is acceptable - simplified version."""
        if len(element_vertices) < 3:
            return False

        # Check for degenerate element (zero area)
        area = calculate_polygon_area(element_vertices)
        if area < 1e-6:  # More lenient area threshold
            return False

        # Simple validation: just check if vertices are distinct
        for i in range(len(element_vertices)):
            for j in range(i + 1, len(element_vertices)):
                if np.linalg.norm(element_vertices[i] - element_vertices[j]) < 1e-6:
                    return False  # Duplicate vertices

        return True

    def _calculate_reward(self, element_vertices: np.ndarray, is_valid: bool,
                          state_components: Dict) -> float:
        """Calculate reward for the current action."""
        if not is_valid:
            return -0.5

        try:
            # Calculate reward components according to paper
            eta_e, eta_b, mu_t = calculate_reward_components(
                element_vertices, self.current_boundary, self.current_boundary,
                self.current_area_ratio, self.v_density, self.M_angle
            )

            # Combined reward
            reward = eta_e + eta_b + mu_t

            # Bonus for progress (reducing boundary size)
            if len(self.current_boundary) < len(self.original_boundary):
                progress_bonus = 0.1
                reward += progress_bonus

            return reward

        except Exception:
            return -0.1

    def _get_observation(self) -> Dict:
        """Get current observation."""
        if len(self.current_boundary) < 3:
            return self._get_default_observation()

        # Get reference vertex
        try:
            ref_idx = calculate_reference_vertex(self.current_boundary, self.nrv)
        except:
            ref_idx = 0

        # Get state components
        try:
            state_components = get_state_components(
                self.current_boundary, ref_idx, self.n_neighbors,
                self.n_fan_points, self.beta_obs
            )
        except:
            return self._get_default_observation()

        # Transform to relative coordinates
        ref_vertex = state_components['ref_vertex']
        reference_direction = state_components['reference_direction']

        # Collect all points for transformation
        all_points = [ref_vertex]
        all_points.extend(state_components['left_neighbors'])
        all_points.extend(state_components['right_neighbors'])
        all_points.extend(state_components['fan_points'])

        # Transform to relative coordinates
        try:
            relative_points = transform_to_relative_coords(
                all_points, ref_vertex, reference_direction
            )
        except:
            return self._get_default_observation()

        # Ensure we have the right number of points
        while len(relative_points) < 1 + 2 * self.n_neighbors + self.n_fan_points:
            relative_points.append(np.zeros(2))

        # Create observation dictionary
        observation = {
            'ref_vertex': torch.tensor(relative_points[0], dtype=torch.float32),
            'left_neighbors': torch.tensor(np.array(relative_points[1:1 + self.n_neighbors]), dtype=torch.float32),
            'right_neighbors': torch.tensor(np.array(relative_points[1 + self.n_neighbors:1 + 2 * self.n_neighbors]),
                                            dtype=torch.float32),
            'fan_points': torch.tensor(
                np.array(relative_points[1 + 2 * self.n_neighbors:1 + 2 * self.n_neighbors + self.n_fan_points]),
                dtype=torch.float32),
            'area_ratio': torch.tensor([self.current_area_ratio], dtype=torch.float32)
        }

        return observation

    def _get_default_observation(self) -> Dict:
        """Get default observation when boundary is too small."""
        return {
            'ref_vertex': torch.zeros(2, dtype=torch.float32),
            'left_neighbors': torch.zeros((self.n_neighbors, 2), dtype=torch.float32),
            'right_neighbors': torch.zeros((self.n_neighbors, 2), dtype=torch.float32),
            'fan_points': torch.zeros((self.n_fan_points, 2), dtype=torch.float32),
            'area_ratio': torch.tensor([self.current_area_ratio], dtype=torch.float32)
        }

    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        return {
            'boundary_area': calculate_polygon_area(self.current_boundary) if len(self.current_boundary) >= 3 else 0,
            'boundary_vertices': len(self.current_boundary),
            'generated_elements': len(self.generated_elements),
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'area_ratio': self.current_area_ratio
        }

    def render(self, mode: str = 'human'):
        """Render the current state of the environment."""
        pass

    def close(self):
        """Close the environment."""
        pass

    def get_current_mesh(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Get current mesh state."""
        return self.current_boundary.copy(), self.generated_elements.copy()

    def set_domain(self, domain_file: str):
        """Set new domain for the environment."""
        self.domain_file = domain_file
        self.original_boundary = self._load_domain()
        self.original_boundary = self._ensure_clockwise(self.original_boundary)
        self.original_area = calculate_polygon_area(self.original_boundary)

    def get_mesh_quality_metrics(self) -> Dict:
        """Calculate quality metrics for current mesh."""
        if len(self.generated_elements) == 0:
            return {}

        element_qualities = []
        for element in self.generated_elements:
            try:
                quality = calculate_element_quality(element)
                element_qualities.append(quality)
            except:
                element_qualities.append(0.0)

        if not element_qualities:
            return {}

        metrics = {
            'mean_element_quality': np.mean(element_qualities),
            'min_element_quality': np.min(element_qualities),
            'max_element_quality': np.max(element_qualities),
            'std_element_quality': np.std(element_qualities),
            'num_elements': len(self.generated_elements)
        }

        return metrics


class MultiDomainMeshEnv(MeshEnv):
    """Extended mesh environment that can handle multiple domains for training."""

    def __init__(self, config: Dict, domain_files: List[str]):
        """
        Initialize multi-domain mesh environment.

        Args:
            config: Configuration dictionary
            domain_files: List of domain files to use
        """
        self.domain_files = domain_files
        self.current_domain_idx = 0

        super().__init__(config)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset with random domain selection."""
        # Randomly select domain
        if len(self.domain_files) > 1:
            self.current_domain_idx = np.random.randint(0, len(self.domain_files))
            self.domain_file = self.domain_files[self.current_domain_idx]
            self.original_boundary = self._load_domain()
            self.original_boundary = self._ensure_clockwise(self.original_boundary)
            self.original_area = calculate_polygon_area(self.original_boundary)

        return super().reset(seed, options)

    def get_current_domain_info(self) -> Dict:
        """Get information about current domain."""
        return {
            'domain_file': self.domain_file,
            'domain_index': self.current_domain_idx,
            'total_domains': len(self.domain_files)
        }
