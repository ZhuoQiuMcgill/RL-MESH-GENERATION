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
    This version includes the definitive fix for the out-of-boundary edge generation issue.
    """

    def __init__(self, config: Dict):
        super(MeshEnv, self).__init__()
        self.config = config
        self.n_neighbors = int(config['environment']['n_neighbors'])
        self.n_fan_points = int(config['environment']['n_fan_points'])
        self.beta_obs = float(config['environment']['beta_obs'])
        self.alpha_action = float(config['environment']['alpha_action'])
        self.v_density = float(config['environment']['v_density'])
        self.M_angle = float(config['environment']['M_angle'])
        self.nrv = int(config['environment']['nrv'])
        self.max_steps = int(config['environment'].get('max_steps', 1000))
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
        self._setup_spaces()
        print(f"🌍 Environment initialized with max_steps: {self.max_steps}")

    def _ensure_clockwise(self, boundary: np.ndarray) -> np.ndarray:
        area = 0.0
        n = len(boundary)
        for i in range(n):
            j = (i + 1) % n
            area += (boundary[j][0] - boundary[i][0]) * (boundary[j][1] + boundary[i][1])
        return boundary[::-1] if area > 0 else boundary

    def _load_domain(self) -> np.ndarray:
        domain_path = os.path.join(self.data_dir, self.domain_file)
        return load_domain_from_file(domain_path)

    def _setup_spaces(self):
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]),
                                       dtype=np.float32)
        self.observation_space = spaces.Dict({
            'ref_vertex': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'left_neighbors': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_neighbors, 2), dtype=np.float32),
            'right_neighbors': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_neighbors, 2), dtype=np.float32),
            'fan_points': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_fan_points, 2), dtype=np.float32),
            'area_ratio': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.current_boundary = self.original_boundary.copy()
        self.generated_elements = []
        self.current_area_ratio = 1.0
        self.step_count = 0
        self.episode_reward = 0.0
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def _get_observation(self) -> Dict:
        if self.current_boundary is None or len(self.current_boundary) < 3:
            return self._get_default_observation()
        try:
            ref_idx = calculate_reference_vertex(self.current_boundary, self.nrv)
            state_components = get_state_components(
                self.current_boundary, ref_idx, self.n_neighbors,
                self.n_fan_points, self.beta_obs
            )
            ref_vertex = torch.tensor(state_components['ref_vertex'], dtype=torch.float32)
            left_neighbors = torch.zeros((self.n_neighbors, 2), dtype=torch.float32)
            if 'left_neighbors' in state_components and len(state_components['left_neighbors']) > 0:
                n_left = min(len(state_components['left_neighbors']), self.n_neighbors)
                left_neighbors_data = np.array(state_components['left_neighbors'][:n_left])
                left_neighbors[:n_left] = torch.from_numpy(left_neighbors_data)
            right_neighbors = torch.zeros((self.n_neighbors, 2), dtype=torch.float32)
            if 'right_neighbors' in state_components and len(state_components['right_neighbors']) > 0:
                n_right = min(len(state_components['right_neighbors']), self.n_neighbors)
                right_neighbors_data = np.array(state_components['right_neighbors'][:n_right])
                right_neighbors[:n_right] = torch.from_numpy(right_neighbors_data)
            fan_points = torch.zeros((self.n_fan_points, 2), dtype=torch.float32)
            if 'fan_points' in state_components and len(state_components['fan_points']) > 0:
                n_fan = min(len(state_components['fan_points']), self.n_fan_points)
                fan_points_data = np.array(state_components['fan_points'][:n_fan])
                fan_points[:n_fan] = torch.from_numpy(fan_points_data)
            return {
                'ref_vertex': ref_vertex, 'left_neighbors': left_neighbors,
                'right_neighbors': right_neighbors, 'fan_points': fan_points,
                'area_ratio': torch.tensor([self.current_area_ratio], dtype=torch.float32)
            }
        except Exception as e:
            print(f"Error during get_observation: {e}")
            return self._get_default_observation()

    def _get_default_observation(self) -> Dict:
        return {
            'ref_vertex': torch.zeros(2, dtype=torch.float32),
            'left_neighbors': torch.zeros((self.n_neighbors, 2), dtype=torch.float32),
            'right_neighbors': torch.zeros((self.n_neighbors, 2), dtype=torch.float32),
            'fan_points': torch.zeros((self.n_fan_points, 2), dtype=torch.float32),
            'area_ratio': torch.tensor([1.0], dtype=torch.float32)
        }

    # ==============================================================================
    #  CRITICAL FIX: The _execute_action function is replaced with a version that
    #  includes robust checks for out-of-boundary edges for BOTH action types.
    # ==============================================================================
    def _execute_action(self, action: np.ndarray, state_components: Dict, action_type: int, ref_idx: int) -> Tuple[
        np.ndarray, bool, np.ndarray]:
        n_boundary = len(self.current_boundary)
        new_boundary = None
        element_vertices = None

        if action_type == 0:
            # TYPE 0: CONNECT action (Shrink-by-2).
            if n_boundary < 5: return np.array([]), False, self.current_boundary

            left_idx = (ref_idx - 1 + n_boundary) % n_boundary
            right_idx = (ref_idx + 1) % n_boundary
            right_right_idx = (ref_idx + 2) % n_boundary

            if len(set([left_idx, ref_idx, right_idx, right_right_idx])) < 4:
                return np.array([]), False, self.current_boundary

            v_left = self.current_boundary[left_idx]
            v_right_right = self.current_boundary[right_right_idx]

            # Check if the new connecting edge is outside the boundary
            midpoint_of_new_edge = (v_left + v_right_right) / 2.0
            if not self._is_point_inside_boundary(midpoint_of_new_edge):
                return np.array([[-2, -2]]), False, self.current_boundary  # Invalid topological change marker

            element_vertices = np.array([self.current_boundary[ref_idx], self.current_boundary[right_idx],
                                         v_right_right, v_left])
            keep_mask = np.ones(n_boundary, dtype=bool)
            keep_mask[ref_idx], keep_mask[right_idx] = False, False
            new_boundary = self.current_boundary[keep_mask]
        else:
            # TYPE 1: INSERT action.
            if n_boundary < 3: return np.array([]), False, self.current_boundary

            ref_vertex = state_components['ref_vertex']
            v_left = state_components['left_neighbors'][0]
            v_right = state_components['right_neighbors'][0]

            new_vertex = generate_action_coordinates(action, state_components['ref_vertex'],
                                                     state_components['reference_direction'], self.alpha_action,
                                                     state_components['base_length'])

            # Check 1: Point must be inside
            if not self._is_point_inside_boundary(new_vertex):
                return np.array([[-1, -1]]), False, self.current_boundary  # Out-of-boundary point marker

            # Check 2: New edges formed by the point must also be inside
            midpoint_edge1 = (v_left + new_vertex) / 2.0
            midpoint_edge2 = (v_right + new_vertex) / 2.0
            if not self._is_point_inside_boundary(midpoint_edge1) or not self._is_point_inside_boundary(midpoint_edge2):
                return np.array([[-2, -2]]), False, self.current_boundary  # Invalid topological change marker

            element_vertices = np.array([ref_vertex, v_right, new_vertex, v_left])
            new_boundary = self.current_boundary.copy()
            new_boundary[ref_idx] = new_vertex

        is_valid = self._validate_element(element_vertices)
        if not is_valid or new_boundary is None or len(new_boundary) < 3:
            return element_vertices, False, self.current_boundary
        return element_vertices, True, new_boundary

    def step(self, action: np.ndarray, global_timestep: Optional[int] = None) -> Tuple[Dict, float, bool, bool, Dict]:
        self.step_count += 1
        if self.step_count >= self.max_steps:
            observation = self._get_observation()
            info = self._get_info()
            info.update({'termination_reason': 'max_steps_reached'})
            return observation, -1.0, False, True, info

        try:
            ref_idx = calculate_reference_vertex(self.current_boundary, self.nrv)
            state_components = get_state_components(self.current_boundary, ref_idx, self.n_neighbors, self.n_fan_points,
                                                    self.beta_obs)
        except Exception as e:
            observation = self._get_observation()
            info = self._get_info()
            info.update({'termination_reason': f'state_error: {e}'})
            return observation, -0.1, False, False, info

        exploration_steps = self.config['training'].get('forced_exploration_steps', 0)
        action_type = np.random.choice(
            [0, 1]) if global_timestep is not None and global_timestep < exploration_steps else (
            0 if action[0] < 0 else 1)

        element_vertices, is_valid, new_boundary = self._execute_action(action, state_components, action_type, ref_idx)
        reward = self._calculate_reward(element_vertices, is_valid, state_components)

        terminated = False
        if is_valid:
            if action_type == 1: reward += 0.2
            self.generated_elements.append(element_vertices)
            self.current_boundary = new_boundary
            self.current_area_ratio = calculate_polygon_area(
                self.current_boundary) / self.original_area if self.original_area > 0 else 0.0
            self.episode_reward += reward
            if len(self.current_boundary) <= 4:
                terminated = True
                if len(self.current_boundary) == 4: self.generated_elements.append(self.current_boundary.copy())
                total_meshed_area = sum([calculate_polygon_area(elem) for elem in self.generated_elements])
                completion_ratio = total_meshed_area / self.original_area if self.original_area > 0 else 0
                if completion_ratio >= 0.80:
                    reward += 2.0
                else:
                    reward -= 10.0
        else:
            if element_vertices is not None:
                if np.array_equal(element_vertices, np.array([[-1, -1]])):
                    reward = -1.0
                elif np.array_equal(element_vertices, np.array([[-2, -2]])):
                    reward = -1.0
                else:
                    reward = -0.1
            else:
                reward = -0.1

        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'is_valid_element': is_valid, 'element_count': len(self.generated_elements),
            'boundary_vertices': len(self.current_boundary),
            'termination_reason': 'boundary_complete' if terminated else None
        })
        return observation, reward, terminated, False, info

    def _is_point_inside_boundary(self, point: np.ndarray) -> bool:
        x, y = point;
        n = len(self.current_boundary);
        inside = False
        p1x, p1y = self.current_boundary[0]
        for i in range(1, n + 1):
            p2x, p2y = self.current_boundary[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = x
                if p1x == p2x or x <= xinters: inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _validate_element(self, element_vertices: np.ndarray) -> bool:
        if element_vertices is None or len(element_vertices) < 3 or calculate_polygon_area(
                element_vertices) < 1e-6: return False
        for i in range(len(element_vertices)):
            for j in range(i + 1, len(element_vertices)):
                if np.linalg.norm(element_vertices[i] - element_vertices[j]) < 1e-6: return False
        return True

    def _calculate_reward(self, element_vertices: np.ndarray, is_valid: bool, state_components: Dict) -> float:
        if not is_valid: return -0.5
        try:
            eta_e, eta_b, mu_t = calculate_reward_components(element_vertices, self.current_boundary,
                                                             self.current_boundary, self.current_area_ratio,
                                                             self.v_density, self.M_angle)
            return eta_e + eta_b + mu_t
        except Exception:
            return -0.1

    def _get_info(self) -> Dict:
        return {
            'boundary_area': calculate_polygon_area(self.current_boundary) if self.current_boundary is not None and len(
                self.current_boundary) >= 3 else 0,
            'boundary_vertices': len(self.current_boundary) if self.current_boundary is not None else 0,
            'generated_elements': len(self.generated_elements), 'episode_reward': self.episode_reward,
            'step_count': self.step_count, 'area_ratio': self.current_area_ratio}

    def get_current_mesh(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        return self.current_boundary.copy(), self.generated_elements.copy()

    def set_domain(self, domain_file: str):
        self.domain_file = domain_file;
        self.original_boundary = self._load_domain()
        self.original_boundary = self._ensure_clockwise(self.original_boundary)
        self.original_area = calculate_polygon_area(self.original_boundary)

    def get_mesh_quality_metrics(self) -> Dict:
        if not self.generated_elements: return {}
        element_qualities = [calculate_element_quality(elem) for elem in self.generated_elements if
                             elem is not None and len(elem) > 0]
        if not element_qualities: return {}
        return {'mean_element_quality': np.mean(element_qualities), 'min_element_quality': np.min(element_qualities),
                'max_element_quality': np.max(element_qualities), 'std_element_quality': np.std(element_qualities),
                'num_elements': len(self.generated_elements)}


class MultiDomainMeshEnv(MeshEnv):
    def __init__(self, config: Dict, domain_files: List[str]):
        self.domain_files = domain_files
        self.current_domain_idx = 0
        super().__init__(config)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        if len(self.domain_files) > 1:
            self.current_domain_idx = np.random.randint(0, len(self.domain_files))
            self.set_domain(self.domain_files[self.current_domain_idx])
        return super().reset(seed, options)

    def get_current_domain_info(self) -> Dict:
        return {'domain_file': self.domain_files[self.current_domain_idx],
                'domain_index': self.current_domain_idx, 'total_domains': len(self.domain_files)}
