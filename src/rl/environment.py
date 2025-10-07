import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from typing import Any

from src.geometry import Mesh, Boundary
from src.rl.action.action_manager import ActionManager
from .config import load_config
from src.utils import euclidean_distance, normalize_coordinates_cartesian, calculate_polygon_area


class MeshEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, initial_boundary: Boundary, n=None, g=None, alpha=None, beta=None, max_steps=None, config=None, eval_mode=False):
        super(MeshEnv, self).__init__()
        cfg = load_config() if config is None else config
        env_cfg = cfg.get("environment", {})
        
        # Evaluation mode flag
        self.eval_mode = eval_mode

        self.initial_boundary = initial_boundary
        self.n = n if n is not None else env_cfg.get("n", 2)
        self.g = g if g is not None else env_cfg.get("g", 3)
        self.alpha = alpha if alpha is not None else env_cfg.get("alpha", 2)
        self.beta = beta if beta is not None else env_cfg.get("beta", 6)
        self.max_steps = max_steps if max_steps is not None else env_cfg.get("max_steps", 1000)
        self.upsilon = env_cfg.get("upsilon", 1.0)
        self.kappa = env_cfg.get("kappa", 4.0)
        self.M_angle = env_cfg.get("M_angle", 60.0)

        action_config = env_cfg.get("actions", {
            "enabled": ["type0_left", "type0_right", "type1"],
            "auto_remap": False
        })
        self.action_manager = ActionManager(
            alpha=self.alpha,
            n=self.n,
            max_steps=self.max_steps,
            action_config=action_config
        )

        # Calculate dynamic bounds from boundary points
        boundary_vertices = initial_boundary.get_vertices()
        if boundary_vertices:
            all_coords = np.array(boundary_vertices)
            min_coord = np.min(all_coords)
            max_coord = np.max(all_coords)
        else:
            min_coord, max_coord = -1.0, 1.0
        print(f"min_coord: {min_coord}, max_coord: {max_coord}")

        # State: (n_left + n_right + g_points) * 2 (coords) + qt
        state_dim = (self.n * 2 + self.g) * 2
        self.observation_space = spaces.Box(low=min_coord, high=max_coord, shape=(state_dim,), dtype=np.float32)

        # Action space from ActionManager
        self.action_space = self.action_manager.get_action_space()

        self.boundary = None
        self.mesh = None
        self.last_reference_info = None
        self.total_initial_area = initial_boundary.get_area()
        self.current_step = 0
        self.generated_elements = 0
        self.first_invalid_action = True

        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0

        self.invalid_action_count = 0
        self.total_element_quality = 0.0
        self.min_area, self.critical_area = self.initial_boundary.get_min_and_critical_area()
        self.action_count = {}
        self.invalid_points_index = set()
        self.stoped = False
        
    def set_eval_mode(self, eval_mode: bool):
        """Set evaluation mode flag"""
        self.eval_mode = eval_mode
        
    def is_eval_mode(self) -> bool:
        """Check if environment is in evaluation mode"""
        return self.eval_mode

    def _reset_episode_stats(self) -> None:
        self.episode_reward = 0.0
        self.episode_length = 0

    def _update_episode_stats(self, reward: float) -> None:
        self.episode_reward += float(reward)
        self.episode_length += 1

    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.boundary = copy.deepcopy(self.initial_boundary)
        self.mesh = Mesh(self.boundary)
        self.total_initial_area = self.boundary.get_area()
        self.current_step = 0
        self.generated_elements = 0
        self.total_element_quality = 0.0
        self.invalid_action_count = 0
        self.action_count = {}
        self.invalid_points_index = set()
        self.stoped = False
        self.min_area, self.critical_area = self.initial_boundary.get_min_and_critical_area()

        self._reset_episode_stats()

        if hasattr(self, '_initialized'):
            self.episode_count += 1
        else:
            self._initialized = True

        observation = self._get_obs()
        info = {"step": self.current_step, "boundary_vertices": len(self.boundary.get_vertices())}

        return observation, info

    def step(self, action):
        """
        Execute one environment step using ActionManager.
        """
        # Evaluation mode specific logic can be added here
        if self.eval_mode:
            # Add any evaluation-specific preprocessing
            pass
        # Get reference vertex
        reference_vertex_idx = self._get_reference_vertex()

        # Process action using ActionManager
        action_result = self.action_manager.process_action(
            action, self.mesh, self.boundary, reference_vertex_idx, self.M_angle
        )

        action_valid = action_result['action_valid']
        action_name = action_result['action_name']
        action_attempted = action_result['action_attempted']
        element_quality_reward = action_result['element_quality_reward']
        boundary_quality_reward = action_result['boundary_quality_reward']
        generated_element = action_result['generated_element']

        self.current_step += 1

        # Local flag: only when exceeding invalid threshold we will request bootstrap
        bootstrap_needed = False

        # Calculate reward and termination
        if action_valid:
            reward = (
                    element_quality_reward
                    + 1 * (boundary_quality_reward - 1)
                    + self.speed_penalty(generated_element)
            )
            self.invalid_points_index = set()
            self.generated_elements += 1
            self.total_element_quality += element_quality_reward
            terminated = self._is_terminated()
            complete = terminated
            truncated = self.current_step >= self.max_steps
            term_reason = "task_complete" if terminated else None
            trunc_reason = "time_limit" if truncated else None
        else:
            reward = self.invalid_penalty()
            self.invalid_action_count += 1

            complete = False
            truncated = False
            terminated = False
            trunc_reason = None
            term_reason = None


            # MARK: I DON'T KNOW WHY BUT IT WORKS IN THE ORIGINAL PROJECT
            # =============================================================================
            if self.eval_mode:
                self.invalid_points_index.add(reference_vertex_idx)
            if self.invalid_action_count >= 100:
                truncated = True
                self.stoped = True
                trunc_reason = "invalid_action"
                # Request bootstrap only in this specific case (not time-limit based)
                bootstrap_needed = True
            # =============================================================================



        # Update episode statistics
        self._update_episode_stats(reward)
        self._collect_action_info(action_name, action_valid, reward)
        observation = self._get_obs()

        info = {
            "action_valid": action_valid,
            "action_name": action_name,
            "boundary_vertices": len(self.boundary.get_vertices()),
            "element_generated": generated_element is not None,
            "term_reason": term_reason,
            "trunc_reason": trunc_reason,
            "eval_mode": self.eval_mode
        }

        # Only when exceeding 100 invalid actions, signal bootstrap to the trainer/algorithm
        # Note: We deliberately use SB3's timeout-compatible keys for bootstrapping semantics
        if bootstrap_needed:
            info["TimeLimit.truncated"] = True  # used by SB3 to treat this as bootstrap-able truncation
            info["terminal_observation"] = observation  # provide terminal observation for bootstrap

        if terminated or truncated:
            avg_element_quality = 0
            if self.generated_elements > 0:
                avg_element_quality = self.total_element_quality / self.generated_elements

            info["episode"] = {"r": float(self.episode_reward),
                               "l": int(self.current_step)}
            info["detail"] = {"r": float(self.episode_reward),
                              "l": int(self.current_step),
                              "mesh_data": self.get_mesh_data(),
                              "boundary_vertices_data": self.boundary.get_vertices(),
                              "last_ref_point": self.get_last_reference_info(),
                              "is_completed": complete,
                              "generated_elements": self.generated_elements,
                              "action_count": self.action_count,
                              "avg_element_quality": avg_element_quality,
                              "action_attempted": action_attempted}

        return observation, reward, terminated, truncated, info

    def _collect_action_info(self, action_name, action_valid, reward):
        if action_name in self.action_count:
            if action_valid:
                self.action_count[action_name]["valid"] += 1
            else:
                self.action_count[action_name]["invalid"] += 1
            self.action_count[action_name]["rewards"].append(reward)
        else:
            if action_valid:
                self.action_count[action_name] = {"valid": 1, "invalid": 0}
            else:
                self.action_count[action_name] = {"valid": 0, "invalid": 1}
            self.action_count[action_name]["rewards"] = [reward]

    def _get_reference_vertex(self):
        return self.boundary.get_ref_vertex(self.n, self.invalid_points_index)

    def _get_obs(self):
        """
        Build the state vector following Eq.(4) in the paper.
        Order: neighbors first, then fan points, finally area ratio ρ_t.
        """
        if self.boundary.size() < 3:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # 1. reference vertex and geometric info
        get_type = "exclude ref"
        reference_idx = self._get_reference_vertex()
        ref_v = self.boundary.get_vertex_by_index(reference_idx)
        right_neighbor_v = self.boundary.get_vertex_by_index(reference_idx - 1)
        neighbor_coords = self.boundary.get_neighbors(reference_idx, self.n, get_type=get_type)

        # 2. collect fan-sector vertices (global coords)
        try:
            fan_coords = list(self.boundary.get_fan_points(reference_idx, self.n, self.beta, self.g))
        except Exception:
            fan_coords = [None] * self.g

        # 3. scale factor (based on average neighbor length)
        base_len = self.boundary.get_avg_neighbor_length(reference_idx, self.n)
        scale_factor = 1.0 / base_len if base_len > 0 else 1.0

        # 4. normalize coordinates in one shot
        normalized_vertex = normalize_coordinates_cartesian(
            neighbor_coords + fan_coords, ref_v, right_neighbor_v, scale_factor
        )

        # 5. construct state vector
        state_components = []
        for r, theta in normalized_vertex:
            state_components.extend([r, theta])

        # 6. area ratio ρ_t
        # area_ratio = (
        #     self.boundary.get_area() / self.total_initial_area
        #     if self.total_initial_area > 0 else 1.0
        # )
        # state_components.append(area_ratio)

        # 7. cache info for debugging/visualization
        if get_type == "exclude ref":
            neighbor_coords.insert(self.n, ref_v)

        if not self.stoped:
            self.last_reference_info = {
                "ref_vertex": tuple(ref_v),
                "local_env_vertices": [tuple(v) for v in neighbor_coords]
            }

        return np.array(state_components, dtype=np.float32)

    def speed_penalty(self, element):
        element_area = calculate_polygon_area(element)
        if element_area < self.min_area:
            return -1
        if self.min_area <= element_area < self.critical_area:
            return (element_area - self.critical_area) / (self.critical_area - self.min_area)
        return 0

    def invalid_penalty(self):
        # Negative reward for invalid action
        punish = -1
        if self.generated_elements == 0:
            return punish
        return punish / self.generated_elements

    def _is_terminated(self):
        return self.boundary.size() <= 5

    def render(self):
        pass

    def close(self):
        pass

    def get_last_reference_info(self):
        return self.last_reference_info

    def get_mesh_data(self):
        if self.mesh is None:
            return {}

        try:
            return self.mesh.get_adjacency_dict()
        except Exception as e:
            print(f"Fail to get mesh data: {e}")
            return {}
