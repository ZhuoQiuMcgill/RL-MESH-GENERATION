import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from typing import Any

# 导入几何模块和动作模块
from src.geometry import Mesh, Boundary
from src.rl.action.action_manager import ActionManager
from .config import load_config
from src.utils import euclidean_distance, normalize_coordinates, calculate_polygon_area


class MeshEnv(gym.Env):
    """
    网格生成的强化学习环境
    实现了基于论文的MDP formulation
    现在包含SB3兼容的episode统计功能
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, initial_boundary: Boundary, n=None, g=None, alpha=None, beta=None, max_steps=None, config=None):
        """
        初始化网格生成环境

        Args:
            initial_boundary: 初始边界对象
            n: 参考顶点左右邻居数量
            g: 扇形区域内观察点数量
            alpha: 动作空间半径因子
            beta: 状态观察半径因子
        """
        super(MeshEnv, self).__init__()
        cfg = load_config() if config is None else config
        env_cfg = cfg.get("environment", {})

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
            "enabled": ["type0_left", "type0_right", "type1", "type2"],
            "auto_remap": True
        })
        self.action_manager = ActionManager(
            alpha=self.alpha,
            n=self.n,
            max_steps=self.max_steps,
            action_config=action_config
        )

        # State: (n_left + n_right + g_points) * 2 (coords) + qt
        state_dim = (self.n * 2 + self.g) * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # Action space from ActionManager
        self.action_space = self.action_manager.get_action_space()

        self.boundary = None
        self.mesh = None
        self.last_reference_info = None
        self.total_initial_area = initial_boundary.get_area()
        self.current_step = 0
        self.generated_elements = 0
        self.first_invalid_action = True

        # SB3兼容的episode统计属性
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0

        self.invalid_action_count = 0
        self.min_area, self.critical_area = self.initial_boundary.get_min_and_critical_area()

    def _reset_episode_stats(self) -> None:
        """重置episode级别的统计信息"""
        self.episode_reward = 0.0
        self.episode_length = 0

    def _update_episode_stats(self, reward: float) -> None:
        """更新episode级别的统计信息"""
        self.episode_reward += float(reward)
        self.episode_length += 1

    def get_wrapper_attr(self, name: str) -> Any:
        """获取环境属性，兼容SB3/gymnasium wrapper标准"""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)

        # 创建边界和网格的深拷贝
        self.boundary = copy.deepcopy(self.initial_boundary)
        self.mesh = Mesh(self.boundary)
        self.total_initial_area = self.boundary.get_area()
        self.current_step = 0
        self.generated_elements = 0
        self.invalid_action_count = 0

        self.min_area, self.critical_area = self.initial_boundary.get_min_and_critical_area()

        # 重置episode统计
        self._reset_episode_stats()

        # 增加episode计数（仅在非首次reset时）
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
        # Get reference vertex
        reference_vertex_idx = self._get_reference_vertex()

        # Process action using ActionManager
        action_result = self.action_manager.process_action(
            action, self.mesh, self.boundary, reference_vertex_idx, self.M_angle
        )

        action_valid = action_result['action_valid']
        action_name = action_result['action_name']
        element_quality_reward = action_result['element_quality_reward']
        boundary_quality_reward = action_result['boundary_quality_reward']
        generated_element = action_result['generated_element']

        self.current_step += 1
        # Calculate reward and termination
        if action_valid:
            reward = (
                    element_quality_reward
                    + 1 * (boundary_quality_reward - 1)
                    + self.speed_penalty(generated_element)
            )
            self.generated_elements += 1
            terminated = self._is_terminated()
            complete = terminated
            truncated = self.current_step >= self.max_steps
            term_reason = "task_complete" if terminated else None
            trunc_reason = "time_limit" if truncated else None
        else:
            reward = self.invalid_penalty()
            self.invalid_action_count += 1
            if self.invalid_action_count >= 100:
                terminated = True
            else:
                terminated = False
            complete = False
            truncated = False
            term_reason = "invalid_action"
            trunc_reason = None

        # Update episode statistics
        self._update_episode_stats(reward)
        observation = self._get_obs()

        info = {
            "action_valid": action_valid,
            "action_name": action_name,
            "boundary_vertices": len(self.boundary.get_vertices()),
            "element_generated": generated_element is not None,
            "term_reason": term_reason,
            "trunc_reason": trunc_reason
        }

        if terminated or truncated:
            info["episode"] = {"r": float(self.episode_reward),
                               "l": int(self.current_step)}
            info["detail"] = {"r": float(self.episode_reward),
                              "l": int(self.current_step),
                              "mesh_data": self.get_mesh_data(),
                              "boundary_vertices_data": self.boundary.get_vertices(),
                              "last_ref_point": self.get_last_reference_info(),
                              "is_completed": complete,
                              "generated_elements": self.generated_elements}

        return observation, reward, terminated, truncated, info

    def _get_reference_vertex(self):
        """
        根据公式(1)选择具有最小平均内角的参考顶点

        Returns:
            int: 参考顶点在边界中的索引
        """
        return self.boundary.get_ref_vertex()

    def _get_obs(self):
        """
        Build the state vector following Eq.(4) in the paper.
        Order: neighbors first, then fan points, finally area ratio ρ_t.
        """
        if self.boundary.size() < 3:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # 1. reference vertex and geometric info
        get_type = "exclude ref"
        reference_idx = self.boundary.get_ref_vertex()
        ref_v = self.boundary.get_vertex_by_index(reference_idx)
        right_neighbor_v = self.boundary.get_vertex_by_index(reference_idx - 1)
        neighbor_coords = self.boundary.get_neighbors(reference_idx, self.n, get_type=get_type)

        # 2. collect fan-sector vertices (global coords)
        try:
            fan_coords = list(self.boundary.get_fan_points(reference_idx))
        except Exception:
            fan_coords = [None] * self.g

        # 3. scale factor (based on average neighbor length)
        base_len = self.boundary.get_avg_neighbor_length(reference_idx, self.n)
        scale_factor = 1.0 / base_len if base_len > 0 else 1.0

        # 4. normalize coordinates in one shot
        normalized_vertex = normalize_coordinates(
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
        return self.boundary.size() <= 4

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
