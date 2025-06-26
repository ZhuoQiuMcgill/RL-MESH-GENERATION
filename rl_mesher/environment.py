import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import os
import gymnasium as gym
from gymnasium import spaces

from .utils.geometry import (
    calculate_reference_vertex, get_state_components, transform_to_relative_coords,
    calculate_reward_components, is_element_valid, update_boundary,
    calculate_polygon_area, load_domain_from_file, generate_action_coordinates
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
            action: Action to take [type_prob, x, y]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        self.episode_length += 1

        # Extract action components
        type_prob = action[0]
        action_type = 0 if type_prob < 0.5 else 1  # Binary decision for now

        # Get current state components
        ref_idx = calculate_reference_vertex(self.current_boundary, self.nrv)
        state_components = get_state_components(
            self.current_boundary, ref_idx, self.n_neighbors,
            self.n_fan_points, self.beta_obs
        )

        # Generate element based on action
        element_vertices, is_valid = self._generate_element(action, state_components, action_type)

        # Calculate reward
        reward = self._calculate_reward(element_vertices, is_valid, state_components)

        # Update environment if element is valid
        terminated = False
        if is_valid:
            self.generated_elements.append(element_vertices)

            # Update boundary
            self.current_boundary = update_boundary(self.current_boundary, element_vertices)

            # Update area ratio
            current_area = calculate_polygon_area(self.current_boundary)
            self.current_area_ratio = current_area / self.original_area

            # Check if meshing is complete
            if len(self.current_boundary) <= 4:  # Final element
                reward += 10.0  # Completion bonus
                terminated = True

        # Check for truncation (max steps reached)
        truncated = self.step_count >= self.max_steps

        # Get next observation
        observation = self._get_observation()

        # Update episode tracking
        self.episode_reward += reward

        # Get info
        info = self._get_info()
        info.update({
            'is_valid_element': is_valid,
            'element_count': len(self.generated_elements),
            'area_ratio': self.current_area_ratio
        })

        return observation, reward, terminated, truncated, info

    def _generate_element(self, action: np.ndarray, state_components: Dict,
                          action_type: int) -> Tuple[np.ndarray, bool]:
        """
        Generate element based on action and current state.

        Args:
            action: Action vector
            state_components: Current state components
            action_type: Type of action (0 or 1)

        Returns:
            Tuple of (element_vertices, is_valid)
        """
        ref_vertex = state_components['ref_vertex']
        left_neighbors = state_components['left_neighbors']
        right_neighbors = state_components['right_neighbors']
        reference_direction = state_components['reference_direction']
        base_length = state_components['base_length']

        if action_type == 0:
            # Type 0: Use existing vertices to form quadrilateral
            # Use ref_vertex, left_neighbor[0], right_neighbor[0], and their common neighbor
            if len(left_neighbors) > 0 and len(right_neighbors) > 0:
                # Find common neighbor (simplified)
                # In practice, this requires more complex topology analysis
                element_vertices = np.array([
                    ref_vertex,
                    right_neighbors[0],
                    (left_neighbors[0] + right_neighbors[0]) / 2,  # Simplified
                    left_neighbors[0]
                ])
            else:
                # Fallback to type 1
                action_type = 1

        if action_type == 1:
            # Type 1: Generate new vertex based on action
            new_vertex = generate_action_coordinates(
                action, ref_vertex, reference_direction,
                self.alpha_action, base_length
            )

            element_vertices = np.array([
                ref_vertex,
                right_neighbors[0] if len(right_neighbors) > 0 else ref_vertex + [1, 0],
                new_vertex,
                left_neighbors[0] if len(left_neighbors) > 0 else ref_vertex + [0, 1]
            ])

        # Check if element is valid
        is_valid = is_element_valid(element_vertices, self.current_boundary)

        return element_vertices, is_valid

    def _calculate_reward(self, element_vertices: np.ndarray, is_valid: bool,
                          state_components: Dict) -> float:
        """
        Calculate reward for the current action.

        Args:
            element_vertices: Generated element vertices
            is_valid: Whether element is valid
            state_components: Current state components

        Returns:
            Reward value
        """
        if not is_valid:
            return -0.1  # Invalid element penalty

        # Calculate reward components
        eta_e, eta_b, mu_t = calculate_reward_components(
            element_vertices, self.current_boundary, self.current_boundary,
            self.current_area_ratio, self.v_density, self.M_angle
        )

        # Combined reward
        reward = eta_e + eta_b + mu_t

        return reward

    def _get_observation(self) -> Dict:
        """
        Get current observation from environment state.

        Returns:
            Observation dictionary
        """
        if len(self.current_boundary) < 3:
            # Handle edge case of very small boundary
            return self._get_default_observation()

        # Get reference vertex
        try:
            ref_idx = calculate_reference_vertex(self.current_boundary, self.nrv)
        except:
            ref_idx = 0  # Fallback

        # Get state components
        state_components = get_state_components(
            self.current_boundary, ref_idx, self.n_neighbors,
            self.n_fan_points, self.beta_obs
        )

        # Transform to relative coordinates
        ref_vertex = state_components['ref_vertex']
        reference_direction = state_components['reference_direction']

        # Collect all points for transformation
        all_points = [ref_vertex]
        all_points.extend(state_components['left_neighbors'])
        all_points.extend(state_components['right_neighbors'])
        all_points.extend(state_components['fan_points'])

        # Transform to relative coordinates
        relative_points = transform_to_relative_coords(
            all_points, ref_vertex, reference_direction
        )

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

        from .utils.geometry import calculate_element_quality

        element_qualities = []
        for element in self.generated_elements:
            quality = calculate_element_quality(element)
            element_qualities.append(quality)

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