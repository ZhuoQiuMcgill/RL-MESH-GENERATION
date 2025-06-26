import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Polygon, Point
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

from rl_mesher.utils.geometry import (
    calculate_reference_vertex, get_state_components, transform_to_relative_coords,
    calculate_reward_components, is_element_valid, update_boundary,
    calculate_polygon_area, load_domain_from_file, generate_action_coordinates,
    calculate_element_quality, calculate_boundary_quality, calculate_density_reward,
    find_boundary_neighbors, calculate_fan_points, is_element_valid_enhanced,
    update_boundary_with_polygon_ops
)


class MeshEnv(gym.Env):
    """
    Mesh generation environment implementing Gymnasium interface.

    This environment formulates quadrilateral mesh generation as a
    Markov Decision Process where the agent sequentially generates
    mesh elements by making local decisions.
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

        # Episode settings - read from config instead of hardcoding
        self.max_steps = int(config['environment'].get('max_steps', 1000))

        # Domain settings
        self.domain_file = config['domain']['training_domain']
        self.data_dir = config['paths']['data_dir']

        # Load initial domain
        self.original_boundary = self._load_domain()
        self.original_area = calculate_polygon_area(self.original_boundary)

        # Initialize state
        self.current_boundary = None
        self.generated_elements = []
        self.current_area_ratio = 1.0
        self.step_count = 0

        # Define action and observation spaces
        self._setup_spaces()

        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0

        print(f"🌍 Environment initialized with max_steps: {self.max_steps}")

    def _load_domain(self) -> np.ndarray:
        """Load domain boundary from file."""
        domain_path = os.path.join(self.data_dir, self.domain_file)
        return load_domain_from_file(domain_path)

    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: [type_prob, x_coord, y_coord]
        # type_prob in [0, 1], coordinates in [-1, 1] (normalized)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space: flattened state vector
        # ref_vertex(2) + left_neighbors(n*2) + right_neighbors(n*2) +
        # fan_points(g*2) + area_ratio(1)
        state_dim = 2 + (self.n_neighbors * 2) + (self.n_neighbors * 2) + \
                    (self.n_fan_points * 2) + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset environment state
        self.current_boundary = self.original_boundary.copy()
        self.generated_elements = []
        self.current_area_ratio = 1.0
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action vector [type_prob, x_coord, y_coord]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        self.episode_length += 1

        # Determine action type based on action[0]
        action_type = 0 if action[0] < 0.5 else 1

        # Get current state components
        try:
            state_components = get_state_components(
                self.current_boundary, self.n_neighbors,
                self.n_fan_points, self.beta_obs
            )
        except Exception as e:
            # If state computation fails, terminate with penalty
            observation = self._get_observation()
            info = self._get_info()
            info.update({
                'is_valid_element': False,
                'element_count': len(self.generated_elements),
                'area_ratio': self.current_area_ratio,
                'termination_reason': f'state_error: {str(e)}'
            })
            return observation, -1.0, True, False, info

        # Generate element based on action
        element_vertices, is_valid = self._generate_element_enhanced(action, state_components, action_type)

        # Calculate reward using enhanced reward function
        reward = self._calculate_enhanced_reward(element_vertices, is_valid, state_components)

        # Update environment if element is valid
        terminated = False
        termination_reason = None

        if is_valid:
            self.generated_elements.append(element_vertices)

            # Update boundary using improved polygon operations
            try:
                new_boundary = update_boundary_with_polygon_ops(self.current_boundary, element_vertices)

                # Validate the new boundary
                if len(new_boundary) >= 3 and calculate_polygon_area(new_boundary) > 0:
                    self.current_boundary = new_boundary

                    # Update area ratio
                    current_area = calculate_polygon_area(self.current_boundary)
                    self.current_area_ratio = current_area / self.original_area

                    # Check if mesh generation is complete
                    if len(self.current_boundary) <= 4 or self.current_area_ratio < 0.05:
                        reward += 10.0  # Completion bonus
                        terminated = True
                        termination_reason = 'mesh_complete'

                else:
                    # Invalid boundary update, penalize and terminate
                    reward -= 5.0
                    terminated = True
                    termination_reason = 'invalid_boundary_update'

            except Exception as e:
                # Boundary update failed, penalize and terminate
                reward -= 5.0
                terminated = True
                termination_reason = f'boundary_update_error: {str(e)}'
        else:
            # Invalid element generated, small penalty but continue
            reward -= 0.5

        # Check for truncation (max steps reached)
        truncated = self.step_count >= self.max_steps

        if truncated and not terminated:
            termination_reason = 'max_steps_reached'

        # Get next observation
        observation = self._get_observation()

        # Update episode tracking
        self.episode_reward += reward

        # Get info
        info = self._get_info()
        info.update({
            'is_valid_element': is_valid,
            'element_count': len(self.generated_elements),
            'area_ratio': self.current_area_ratio,
            'boundary_vertices': len(self.current_boundary),
            'termination_reason': termination_reason
        })

        return observation, reward, terminated, truncated, info

    def _generate_element_enhanced(self, action: np.ndarray, state_components: Dict,
                                   action_type: int) -> Tuple[np.ndarray, bool]:
        """
        Generate high-quality quadrilateral element with enhanced validation.
        """
        ref_vertex = state_components['ref_vertex']
        left_neighbors = state_components['left_neighbors']
        right_neighbors = state_components['right_neighbors']
        reference_direction = state_components.get('reference_direction', np.array([1.0, 0.0]))
        base_length = state_components.get('base_length', 1.0)

        # Ensure we have neighbors
        if len(left_neighbors) == 0 or len(right_neighbors) == 0:
            # Generate a small, well-formed quadrilateral
            offset = base_length * 0.2
            element_vertices = np.array([
                ref_vertex,
                ref_vertex + np.array([offset, 0]),
                ref_vertex + np.array([offset, offset]),
                ref_vertex + np.array([0, offset])
            ])
        else:
            if action_type == 0:
                # Type 0: Use existing vertices to form quadrilateral
                try:
                    v_left = left_neighbors[0]
                    v_right = right_neighbors[0]

                    # Calculate interior point to create proper quadrilateral
                    center = (ref_vertex + v_left + v_right) / 3.0

                    # Create inward offset for interior point
                    normal_offset = base_length * 0.3
                    boundary_normal = self._calculate_inward_normal(ref_vertex, v_left, v_right)
                    interior_point = center + boundary_normal * normal_offset

                    # Create quadrilateral with proper vertex ordering
                    element_vertices = np.array([
                        ref_vertex,
                        v_right,
                        interior_point,
                        v_left
                    ])

                except Exception:
                    # Fallback to simple quadrilateral
                    offset = base_length * 0.2
                    element_vertices = np.array([
                        ref_vertex,
                        ref_vertex + np.array([offset, 0]),
                        ref_vertex + np.array([offset, offset]),
                        ref_vertex + np.array([0, offset])
                    ])
            else:
                # Type 1: Generate new vertex based on action
                new_vertex = generate_action_coordinates(
                    action, ref_vertex, reference_direction,
                    self.alpha_action, base_length
                )

                v_left = left_neighbors[0] if len(left_neighbors) > 0 else ref_vertex + np.array([0, base_length * 0.2])
                v_right = right_neighbors[0] if len(right_neighbors) > 0 else ref_vertex + np.array(
                    [base_length * 0.2, 0])

                # Create quadrilateral with proper vertex ordering
                element_vertices = np.array([
                    ref_vertex,
                    v_right,
                    new_vertex,
                    v_left
                ])

        # Enhanced validation
        is_valid = is_element_valid_enhanced(element_vertices, self.current_boundary)

        return element_vertices, is_valid

    def _calculate_inward_normal(self, ref_vertex: np.ndarray, v_left: np.ndarray, v_right: np.ndarray) -> np.ndarray:
        """Calculate inward normal direction for boundary vertex."""
        # Calculate edge vectors
        edge_left = v_left - ref_vertex
        edge_right = v_right - ref_vertex

        # Calculate bisector direction
        edge_left_norm = edge_left / (np.linalg.norm(edge_left) + 1e-8)
        edge_right_norm = edge_right / (np.linalg.norm(edge_right) + 1e-8)

        bisector = edge_left_norm + edge_right_norm
        if np.linalg.norm(bisector) < 1e-8:
            # Edges are opposite, use perpendicular
            bisector = np.array([-edge_left_norm[1], edge_left_norm[0]])
        else:
            bisector = bisector / np.linalg.norm(bisector)

        return bisector

    def _calculate_enhanced_reward(self, element_vertices: np.ndarray, is_valid: bool,
                                   state_components: Dict) -> float:
        """Enhanced reward calculation with better quality metrics."""
        if not is_valid:
            return -1.0  # Strong penalty for invalid elements

        # Element quality reward
        try:
            element_quality = calculate_element_quality(element_vertices)
            quality_reward = element_quality * 2.0  # Scale up quality reward
        except:
            quality_reward = -0.5

        # Boundary quality reward
        try:
            boundary_quality = calculate_boundary_quality(
                element_vertices, self.current_boundary, self.M_angle
            )
            boundary_reward = boundary_quality * 1.0
        except:
            boundary_reward = -0.5

        # Density reward
        try:
            element_area = calculate_polygon_area(element_vertices)
            density_reward = calculate_density_reward(
                element_area, self.current_boundary, self.v_density
            )
        except:
            density_reward = 0.0

        # Progress reward
        progress_reward = 0.1  # Small reward for valid progress

        total_reward = quality_reward + boundary_reward + density_reward + progress_reward
        return total_reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        try:
            state_components = get_state_components(
                self.current_boundary, self.n_neighbors,
                self.n_fan_points, self.beta_obs
            )
        except:
            # Return zero observation if state computation fails
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Build observation vector
        obs_parts = []

        # Reference vertex
        obs_parts.extend(state_components['ref_vertex'])

        # Left neighbors (pad if needed)
        left_neighbors = state_components['left_neighbors']
        for i in range(self.n_neighbors):
            if i < len(left_neighbors):
                obs_parts.extend(left_neighbors[i])
            else:
                obs_parts.extend([0.0, 0.0])

        # Right neighbors (pad if needed)
        right_neighbors = state_components['right_neighbors']
        for i in range(self.n_neighbors):
            if i < len(right_neighbors):
                obs_parts.extend(right_neighbors[i])
            else:
                obs_parts.extend([0.0, 0.0])

        # Fan points
        fan_points = state_components.get('fan_points', [])
        for i in range(self.n_fan_points):
            if i < len(fan_points):
                obs_parts.extend(fan_points[i])
            else:
                obs_parts.extend([0.0, 0.0])

        # Area ratio
        obs_parts.append(self.current_area_ratio)

        return np.array(obs_parts, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        return {
            'boundary_vertices': len(self.current_boundary),
            'elements_generated': len(self.generated_elements),
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'area_ratio': self.current_area_ratio,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length
        }

    def render(self, mode: str = 'human'):
        """
        Render the current state of the environment.

        Args:
            mode: Rendering mode
        """
        # This would integrate with visualization.py for rendering
        pass

    def close(self):
        """Close the environment."""
        pass

    def get_current_mesh(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Get current mesh state.

        Returns:
            Tuple of (current_boundary, generated_elements)
        """
        return self.current_boundary.copy(), self.generated_elements.copy()

    def set_domain(self, domain_file: str):
        """
        Set new domain for the environment.

        Args:
            domain_file: Path to domain file
        """
        self.domain_file = domain_file
        self.original_boundary = self._load_domain()
        self.original_area = calculate_polygon_area(self.original_boundary)

    def get_mesh_quality_metrics(self) -> Dict:
        """
        Calculate quality metrics for current mesh.

        Returns:
            Dictionary of quality metrics
        """
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
    """
    Extended mesh environment that can handle multiple domains for training.
    """

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
            self.original_area = calculate_polygon_area(self.original_boundary)

        return super().reset(seed, options)

    def get_current_domain_info(self) -> Dict:
        """Get information about current domain."""
        return {
            'domain_file': self.domain_file,
            'domain_index': self.current_domain_idx,
            'total_domains': len(self.domain_files)
        }
