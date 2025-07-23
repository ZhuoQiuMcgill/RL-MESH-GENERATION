import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
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

        # 初始化动作管理器
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

        # 定义状态空间和动作空间
        # State: (n_left + n_right + 1 + g_points) * 2 (coords) + 1 (area_ratio)
        state_dim = (self.n + self.n + 1 + self.g) * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # Action space from ActionManager
        self.action_space = self.action_manager.get_action_space()

        # 环境状态变量
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

        def invalid_penalty() -> float:
            # Negative reward for invalid action
            punish = ((self.generated_elements - self.max_steps) * 0.1)
            if self.generated_elements == 0:
                return punish
            return punish / self.generated_elements

        # Process action using ActionManager
        action_result = self.action_manager.process_action(
            action, self.mesh, self.boundary, reference_vertex_idx, self.M_angle
        )

        action_valid = action_result['action_valid']
        action_name = action_result['action_name']
        element_quality_reward = action_result['element_quality_reward']
        boundary_quality_reward = action_result['boundary_quality_reward']
        generated_element = action_result['generated_element']

        # Calculate reward and termination
        if action_valid:
            reward = (
                    element_quality_reward
                    + boundary_quality_reward
                    + self._calculate_density_reward(generated_element)
            )
            self.generated_elements += 1
            self.current_step += 1
            terminated = self._is_terminated()
            complete = terminated
            truncated = self.current_step >= self.max_steps
            term_reason = "task_complete" if terminated else None
            trunc_reason = "time_limit" if truncated else None
        else:
            reward = invalid_penalty()
            terminated = True  # illegal action → failure
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
                              "is_completed": complete}

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
        获取当前状态观察，实现公式(4)的状态表示
        严格按照论文方法进行坐标标准化

        Returns:
            np.ndarray: 状态向量
        """

        if self.boundary.size() < 3:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # 获取参考顶点
        reference_idx = self.boundary.get_ref_vertex()

        ref_vertex_coords = self.boundary.get_vertex_by_index(reference_idx)
        local_env_coords = [self.boundary.get_vertex_by_index(reference_idx + i) for i in
                            range(-self.n, self.n + 1)]

        self.last_reference_info = {
            "ref_vertex": tuple(ref_vertex_coords),
            "local_env_vertices": [tuple(v) for v in local_env_coords]
        }

        state_components = []

        # 按论文方法标准化邻居顶点坐标
        normalized_neighbors = normalize_coordinates(local_env_coords, self.n, self.boundary, self.n)  # 参考点在中间位置

        # 添加邻居顶点的标准化坐标到状态
        for r, theta in normalized_neighbors:
            state_components.extend([r, theta])

        # 获取扇形区域内的观察点并标准化
        try:
            fan_points = self.boundary.get_fan_points(
                reference_idx, self.g,
                self.beta * self.boundary.get_avg_neighbor_length(reference_idx, self.n)
            )

            # 将参考顶点加入fan_points列表的开头以便标准化
            fan_vertices_with_ref = [ref_vertex_coords] + list(fan_points)
            normalized_fan = normalize_coordinates(fan_vertices_with_ref, 0, self.boundary, self.n)  # 参考点在位置0

            # 跳过参考点本身，只添加扇形点的坐标
            for r, theta in normalized_fan[1:]:
                state_components.extend([r, theta])

            # 如果扇形点不足，用零填充
            while len(normalized_fan) - 1 < self.g:
                state_components.extend([0.0, 0.0])
                if len(normalized_fan) - 1 >= self.g:
                    break

        except Exception:
            # 如果获取扇形点失败，用零填充
            for _ in range(self.g):
                state_components.extend([0.0, 0.0])

        # 添加面积比 ρt
        current_area = self.boundary.get_area()
        area_ratio = current_area / self.total_initial_area if self.total_initial_area > 0 else 1.0
        state_components.append(area_ratio)

        return np.array(state_components, dtype=np.float32)

    def _calculate_density_reward(self, element):
        """
        计算密度奖励 μ_t，实现公式(9)

        Args:
            element: 生成的元素

        Returns:
            float: 密度奖励值
        """
        if element is None:
            return 0.0

        # 计算元素面积
        element_area = calculate_polygon_area(element)

        # 获取边界信息计算最小和最大面积
        vertices = self.boundary.get_vertices()
        if len(vertices) < 2:
            return 0.0

        # 计算边长范围
        edge_lengths = []
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            edge_lengths.append(euclidean_distance(v1, v2))

        e_min = min(edge_lengths)
        e_max = max(edge_lengths)

        # 计算最小和最大允许面积
        A_min = self.upsilon * e_min ** 2
        A_max = self.upsilon * ((e_max - e_min) / self.kappa + e_min) ** 2

        # 计算密度奖励
        if element_area < A_min:
            return -1.0
        elif A_min <= element_area < A_max:
            return (element_area - A_min) / (A_max - A_min)
        else:
            return 0.0

    def _is_terminated(self):
        """
        判断是否完成网格生成（边界变成四边形或更少顶点）

        Returns:
            bool: 是否终止
        """
        return self.boundary.size() <= 5

    def render(self):
        """可视化当前状态（可选实现）"""
        pass

    def close(self):
        """清理资源"""
        pass

    def get_last_reference_info(self):
        """返回上一步的参考点及其局部环境信息"""
        return self.last_reference_info

    def get_mesh_data(self):
        """
        获取当前网格的邻接关系数据，用于前端可视化

        Returns:
            dict: 网格邻接关系字典，格式为 {vertex_str: [adjacent_vertices]}
        """
        if self.mesh is None:
            return {}

        try:
            return self.mesh.get_adjacency_dict()
        except Exception as e:
            print(f"获取mesh数据失败: {e}")
            return {}
