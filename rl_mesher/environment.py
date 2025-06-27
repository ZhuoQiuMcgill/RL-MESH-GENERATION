import os
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from gymnasium import Env, spaces
from rl_mesher.utils.geometry import (
    calculate_polygon_area, generate_action_coordinates,
    calculate_reward_components, calculate_reference_vertex,
    get_state_components, calculate_element_quality, load_domain_from_file
)


class MeshEnv(Env):
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

        # Observation space components - Dict type for compatibility
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
        observation = self._get_observation_dict()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray, global_timestep: Optional[int] = None) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step following the paper's algorithm."""
        self.step_count += 1

        # Truncate if max_steps is reached
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
            ref_idx = 0

        # Step 2: Get state components
        try:
            state_components = get_state_components(
                self.current_boundary, ref_idx, self.n_neighbors,
                self.n_fan_points, self.beta_obs
            )
        except Exception as e:
            observation = self._get_observation()
            info = self._get_info()
            info.update({
                'is_valid_element': False,
                'element_count': len(self.generated_elements),
                'termination_reason': f'state_error: {str(e)}'
            })
            return observation, -0.1, False, False, info

        # Step 3: Determine action type with forced exploration during early training
        exploration_steps = self.config['training'].get('forced_exploration_steps', 0)

        if global_timestep is not None and global_timestep < exploration_steps:
            action_type = np.random.choice([0, 1])
        else:
            action_type = 0 if action[0] < 0 else 1

        # Step 4: Execute the action to get element and new boundary
        element_vertices, is_valid, new_boundary = self._execute_action(
            action, state_components, action_type, ref_idx
        )

        # Step 5: Calculate reward (strictly based on the paper)
        reward = self._calculate_reward(element_vertices, is_valid, state_components)

        # Step 6: Update environment state
        terminated = False
        termination_reason = None

        if is_valid:
            self.generated_elements.append(element_vertices)
            self.current_boundary = new_boundary
            current_area = calculate_polygon_area(self.current_boundary)
            self.current_area_ratio = current_area / self.original_area if self.original_area > 0 else 0.0
            self.episode_reward += reward

            if len(self.current_boundary) <= 4:
                if len(self.current_boundary) == 4:
                    self.generated_elements.append(self.current_boundary.copy())
                terminated = True
                termination_reason = 'boundary_complete'
                reward += 10.0
        else:
            reward = -0.1

        # Get next state and info
        observation = self._get_observation()
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
        Execute action with geometrically correct boundary updates for both types,
        based on the paper's diagrams (Fig. 8a and 8b).
        """
        n_boundary = len(self.current_boundary)
        new_boundary = None
        element_vertices = None

        if action_type == 0:
            # TYPE 0: CONNECT action (Shrink-by-2). Based on Fig. 8a of the paper.
            # Forms an element from 4 consecutive vertices and removes 2 of them.
            if n_boundary < 5:  # This action is only possible if the boundary is large enough
                return np.array([]), False, self.current_boundary

            # Get indices of the 4 consecutive vertices, handling wrap-around.
            # V3, V0, V1, V4 in the paper correspond to left, ref, right, right_right.
            left_idx = (ref_idx - 1 + n_boundary) % n_boundary
            right_idx = (ref_idx + 1) % n_boundary
            right_right_idx = (ref_idx + 2) % n_boundary

            # Ensure we have 4 unique vertices before proceeding
            if len(set([left_idx, ref_idx, right_idx, right_right_idx])) < 4:
                return np.array([]), False, self.current_boundary

            v_left = self.current_boundary[left_idx]
            ref_vertex = self.current_boundary[ref_idx]
            v_right = self.current_boundary[right_idx]
            v_right_right = self.current_boundary[right_right_idx]

            element_vertices = np.array([ref_vertex, v_right, v_right_right, v_left])

            # --- Boundary Update for Type 0 ---
            # The new boundary connects v_left and v_right_right.
            # We must remove ref_vertex and v_right from the boundary.
            # A boolean mask is the most robust way to handle index removal.
            keep_mask = np.ones(n_boundary, dtype=bool)
            keep_mask[ref_idx] = False
            keep_mask[right_idx] = False

            new_boundary = self.current_boundary[keep_mask]

        else:  # action_type == 1
            # TYPE 1: INSERT action (Replace). Based on Fig. 8b of the paper.
            # Forms an element with a new vertex, replacing the reference vertex.
            # The boundary size remains unchanged.
            if n_boundary < 3:
                return np.array([]), False, self.current_boundary

            ref_vertex = state_components['ref_vertex']
            v_left = state_components['left_neighbors'][0]
            v_right = state_components['right_neighbors'][0]
            base_length = state_components['base_length']
            reference_direction = state_components['reference_direction']

            new_vertex = generate_action_coordinates(
                action, ref_vertex, reference_direction,
                self.alpha_action, base_length
            )

            if not self._is_point_inside_boundary(new_vertex):
                return np.array([]), False, self.current_boundary

            element_vertices = np.array([ref_vertex, v_right, new_vertex, v_left])

            # --- Boundary Update for Type 1 ---
            new_boundary = self.current_boundary.copy()
            new_boundary[ref_idx] = new_vertex

        # --- Final Validation ---
        if new_boundary is None or element_vertices is None or len(new_boundary) < 3:
            return np.array([]), False, self.current_boundary

        is_valid = self._validate_element(element_vertices)

        if not is_valid:
            return element_vertices, False, self.current_boundary

        return element_vertices, True, new_boundary

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

    def _validate_element(self, element_vertices: np.ndarray) -> bool:
        """Validate that the generated element is acceptable."""
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
        """
        Calculate reward for the current action, strictly following the paper's
        three-component reward function (Eq. 6).
        """
        if not is_valid:
            # A penalty for creating a geometrically invalid element.
            return -0.5

        try:
            # Calculate the three reward components as defined in the paper:
            # element quality (eta_e), boundary quality (eta_b), and density (mu_t).
            eta_e, eta_b, mu_t = calculate_reward_components(
                element_vertices, self.current_boundary, self.current_boundary,
                self.current_area_ratio, self.v_density, self.M_angle
            )

            # The total step reward is the sum of the three quality metrics.
            # This implementation now perfectly matches the paper's Eq. 6.
            # No additional bonuses are applied, allowing the agent to learn the
            # intrinsic trade-offs between element quality, boundary quality, and density.
            reward = eta_e + eta_b + mu_t
            return reward

        except Exception:
            # A fallback penalty if reward calculation fails for any reason.
            return -0.1

    def _get_observation_dict(self) -> Dict:
        """Get current observation as dictionary with torch tensors."""
        if len(self.current_boundary) < 5:
            # Minimal boundary case - return zeros
            return {
                'ref_vertex': torch.zeros(2, dtype=torch.float32),
                'left_neighbors': torch.zeros((self.n_neighbors, 2), dtype=torch.float32),
                'right_neighbors': torch.zeros((self.n_neighbors, 2), dtype=torch.float32),
                'fan_points': torch.zeros((self.n_fan_points, 2), dtype=torch.float32),
                'area_ratio': torch.tensor([self.current_area_ratio], dtype=torch.float32)
            }

        try:
            # Calculate reference vertex
            ref_idx = calculate_reference_vertex(self.current_boundary, self.nrv)

            # Get state components
            state_components = get_state_components(
                self.current_boundary, ref_idx, self.n_neighbors,
                self.n_fan_points, self.beta_obs
            )

            # Convert to torch tensors
            ref_vertex = torch.zeros(2, dtype=torch.float32)

            # Pad left neighbors
            left_neighbors = torch.zeros((self.n_neighbors, 2), dtype=torch.float32)
            if len(state_components['left_neighbors']) > 0:
                n_left = min(len(state_components['left_neighbors']), self.n_neighbors)
                left_neighbors[:n_left] = torch.tensor(state_components['left_neighbors'][:n_left], dtype=torch.float32)

            # Pad right neighbors
            right_neighbors = torch.zeros((self.n_neighbors, 2), dtype=torch.float32)
            if len(state_components['right_neighbors']) > 0:
                n_right = min(len(state_components['right_neighbors']), self.n_neighbors)
                right_neighbors[:n_right] = torch.tensor(state_components['right_neighbors'][:n_right],
                                                         dtype=torch.float32)

            # Pad fan points
            fan_points = torch.zeros((self.n_fan_points, 2), dtype=torch.float32)
            if len(state_components['fan_points']) > 0:
                n_fan = min(len(state_components['fan_points']), self.n_fan_points)
                fan_points[:n_fan] = torch.tensor(state_components['fan_points'][:n_fan], dtype=torch.float32)

            return {
                'ref_vertex': ref_vertex,
                'left_neighbors': left_neighbors,
                'right_neighbors': right_neighbors,
                'fan_points': fan_points,
                'area_ratio': torch.tensor([self.current_area_ratio], dtype=torch.float32)
            }

        except Exception as e:
            # Fallback observation
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
            domain_files: List of domain file names
        """
        self.domain_files = domain_files
        self.current_domain_idx = 0

        # Initialize with first domain
        config = config.copy()
        config['domain']['training_domain'] = domain_files[0]

        super().__init__(config)

        # Load all domains
        self.all_domains = []
        for domain_file in domain_files:
            domain_path = os.path.join(self.data_dir, domain_file)
            boundary = load_domain_from_file(domain_path)
            boundary = self._ensure_clockwise(boundary)
            self.all_domains.append(boundary)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment, optionally switching to next domain."""
        # Switch domain randomly or in sequence
        if options and 'domain_idx' in options:
            self.current_domain_idx = options['domain_idx']
        else:
            # Random domain selection
            self.current_domain_idx = np.random.choice(len(self.domain_files))

        # Set current domain
        self.original_boundary = self.all_domains[self.current_domain_idx].copy()
        self.original_area = calculate_polygon_area(self.original_boundary)

        return super().reset(seed=seed, options=options)

    def get_current_domain_info(self) -> Dict:
        """Get information about current domain."""
        return {
            'domain_idx': self.current_domain_idx,
            'domain_file': self.domain_files[self.current_domain_idx],
            'num_domains': len(self.domain_files),
            'original_vertices': len(self.original_boundary),
            'original_area': self.original_area
        }


# Compatibility alias
MeshEnvironment = MeshEnv
