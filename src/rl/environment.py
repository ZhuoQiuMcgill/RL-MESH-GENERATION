import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import copy

# 导入几何模块和动作模块
from src.geometry import Mesh
from src.rl.action.type0 import ActionType0
from src.rl.action.type1 import ActionType1
from src.rl.action import ActionType2
from .config import load_config


class MeshEnv(gym.Env):
    """
    网格生成的强化学习环境
    实现了基于论文的MDP formulation
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, initial_boundary, n=None, g=None, alpha=None, beta=None, config=None):
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
        self.max_steps = env_cfg.get("max_steps", 1000)
        self.upsilon = env_cfg.get("upsilon", 1.0)
        self.kappa = env_cfg.get("kappa", 4.0)
        self.M_angle = env_cfg.get("M_angle", 60.0)

        # 初始化动作类型
        self.action_type_0 = ActionType0()
        self.action_type_1 = ActionType1()
        self.action_type_2 = ActionType2()

        # 定义状态空间和动作空间
        # State: (n_left + n_right + 1 + g_points) * 2 (coords) + 1 (area_ratio)
        state_dim = (self.n + self.n + 1 + self.g) * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # Action: [type_logit, x_coord, y_coord]
        action_dim = 3
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

        # 环境状态变量
        self.boundary = None
        self.mesh = None
        self.total_initial_area = 0.0
        self.current_step = 0

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

        observation = self._get_obs()
        info = {"step": self.current_step, "boundary_vertices": len(self.boundary.get_vertices())}

        return observation, info

    def step(self, action):
        """
        执行一步动作

        Args:
            action: 连续动作向量 [type_logit, x_coord, y_coord]

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # 解码动作
        action_type, new_coords, reference_vertex_idx = self._decode_action(action)

        # 检查动作有效性并执行
        action_valid = False
        generated_element = None
        old_boundary = copy.deepcopy(self.boundary)

        try:
            if action_type == 0:
                if self.action_type_0.is_valid(self.boundary, reference_vertex_idx):
                    generated_element = self.action_type_0.execute(
                        self.mesh, self.boundary, reference_vertex_idx)
                    action_valid = True
            elif action_type == 1:
                if self.action_type_1.is_valid(self.boundary, reference_vertex_idx, new_coords[0]):
                    generated_element = self.action_type_1.execute(
                        self.mesh, self.boundary, reference_vertex_idx, new_coords[0])
                    action_valid = True
            elif action_type == 2:
                if self.action_type_2.is_valid(self.boundary, reference_vertex_idx, new_coords[0], new_coords[1]):
                    generated_element = self.action_type_2.execute(
                        self.mesh, self.boundary, reference_vertex_idx, new_coords[0], new_coords[1])
                    action_valid = True
        except Exception as e:
            # 动作执行失败
            action_valid = False

        # 计算奖励
        reward = self._calculate_reward(action_valid, generated_element, old_boundary)

        # 判断结束条件
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        # 获取新状态
        observation = self._get_obs()

        info = {
            "step": self.current_step,
            "action_valid": action_valid,
            "action_type": action_type,
            "boundary_vertices": len(self.boundary.get_vertices()),
            "element_generated": generated_element is not None
        }

        return observation, reward, terminated, truncated, info

    def _decode_action(self, action):
        """
        解码连续动作向量

        Args:
            action: 连续动作向量 [type_logit, x_coord, y_coord]

        Returns:
            tuple: (action_type, new_coords, reference_vertex_idx)
        """
        # 解码动作类型（使用tanh输出范围映射到{0,1,2}）
        type_logit = action[0]
        if type_logit < -0.33:
            action_type = 0
        elif type_logit < 0.33:
            action_type = 1
        else:
            action_type = 2

        # 获取参考顶点（选择最小平均内角的顶点）
        reference_vertex_idx = self._get_reference_vertex()

        # 解码新顶点坐标（如果需要）
        new_coords = []
        if action_type > 0:
            # 计算动作空间半径
            base_length = self._calculate_base_length(reference_vertex_idx)
            radius = self.alpha * base_length

            # 将[-1,1]范围的动作映射到扇形区域内的坐标
            vertices = self.boundary.get_vertices()
            reference_vertex = vertices[reference_vertex_idx]

            # 第一个新顶点坐标
            angle = action[1] * math.pi  # 将[-1,1]映射到[-π,π]
            distance = (action[2] + 1) / 2 * radius  # 将[-1,1]映射到[0,radius]

            x = reference_vertex[0] + distance * math.cos(angle)
            y = reference_vertex[1] + distance * math.sin(angle)
            new_coords.append((x, y))

            # 第二个新顶点坐标（仅Type 2需要）
            if action_type == 2:
                # 为简化，第二个顶点使用固定偏移
                angle2 = angle + math.pi / 6  # 偏移30度
                distance2 = distance * 0.8
                x2 = reference_vertex[0] + distance2 * math.cos(angle2)
                y2 = reference_vertex[1] + distance2 * math.sin(angle2)
                new_coords.append((x2, y2))

        return action_type, new_coords, reference_vertex_idx

    def _get_reference_vertex(self):
        """
        根据公式(1)选择具有最小平均内角的参考顶点

        Returns:
            int: 参考顶点在边界中的索引
        """
        vertices = self.boundary.get_vertices()
        boundary_size = len(vertices)

        if boundary_size < 3:
            return 0

        min_avg_angle = float('inf')
        reference_idx = 0

        for i in range(boundary_size):
            total_angle = 0.0
            count = 0

            # 计算该顶点周围的平均角度
            for j in range(1, min(self.n + 1, boundary_size // 2)):
                left_idx = (i - j) % boundary_size
                right_idx = (i + j) % boundary_size

                # 计算角度 ∠Vl,j Vi Vr,j
                left_vertex = vertices[left_idx]
                center_vertex = vertices[i]
                right_vertex = vertices[right_idx]

                angle = self._calculate_angle(left_vertex, center_vertex, right_vertex)
                total_angle += angle
                count += 1

            if count > 0:
                avg_angle = total_angle / count
                if avg_angle < min_avg_angle:
                    min_avg_angle = avg_angle
                    reference_idx = i

        return reference_idx

    def _calculate_angle(self, p1, center, p2):
        """
        计算三点构成的角度

        Args:
            p1, center, p2: 三个点的坐标

        Returns:
            float: 角度值（度数）
        """
        v1 = (p1[0] - center[0], p1[1] - center[1])
        v2 = (p2[0] - center[0], p2[1] - center[1])

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # 防止数值误差

        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    def _calculate_base_length(self, reference_vertex_idx):
        """
        根据公式(2)计算基础长度L

        Args:
            reference_vertex_idx: 参考顶点索引

        Returns:
            float: 基础长度
        """
        vertices = self.boundary.get_vertices()
        boundary_size = len(vertices)

        total_length = 0.0
        count = 0

        n = min(self.n, boundary_size // 2)

        for j in range(n):
            # 左侧边长度
            left_idx1 = (reference_vertex_idx - j) % boundary_size
            left_idx2 = (reference_vertex_idx - j - 1) % boundary_size
            left_length = self._euclidean_distance(vertices[left_idx1], vertices[left_idx2])

            # 右侧边长度
            right_idx1 = (reference_vertex_idx + j) % boundary_size
            right_idx2 = (reference_vertex_idx + j + 1) % boundary_size
            right_length = self._euclidean_distance(vertices[right_idx1], vertices[right_idx2])

            total_length += left_length + right_length
            count += 2

        return total_length / count if count > 0 else 1.0

    def _euclidean_distance(self, p1, p2):
        """计算两点间欧几里得距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _get_obs(self):
        """
        获取当前状态观察，实现公式(4)的状态表示

        Returns:
            np.ndarray: 状态向量
        """
        vertices = self.boundary.get_vertices()
        boundary_size = len(vertices)

        if boundary_size < 3:
            # 边界太小，返回零状态
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # 获取参考顶点
        reference_idx = self._get_reference_vertex()
        reference_vertex = vertices[reference_idx]

        state_components = []

        # 获取左右邻居顶点（相对坐标）
        for i in range(-self.n, self.n + 1):
            if i == 0:
                # 参考顶点，相对坐标为(0, 0)
                state_components.extend([0.0, 0.0])
            else:
                neighbor_idx = (reference_idx + i) % boundary_size
                neighbor_vertex = vertices[neighbor_idx]

                # 转换为相对坐标
                rel_x = neighbor_vertex[0] - reference_vertex[0]
                rel_y = neighbor_vertex[1] - reference_vertex[1]
                state_components.extend([rel_x, rel_y])

        # 获取扇形区域内的观察点
        try:
            fan_points = self.boundary.get_fan_points(
                reference_idx, self.g,
                self.beta * self._calculate_base_length(reference_idx)
            )

            for point in fan_points:
                rel_x = point[0] - reference_vertex[0]
                rel_y = point[1] - reference_vertex[1]
                state_components.extend([rel_x, rel_y])

        except Exception:
            # 如果获取扇形点失败，用零填充
            for _ in range(self.g):
                state_components.extend([0.0, 0.0])

        # 添加面积比 ρt
        current_area = self.boundary.get_area()
        area_ratio = current_area / self.total_initial_area if self.total_initial_area > 0 else 1.0
        state_components.append(area_ratio)

        return np.array(state_components, dtype=np.float32)

    def _calculate_reward(self, action_was_valid, generated_element, old_boundary):
        """
        计算奖励函数，实现公式(5)-(9)

        Args:
            action_was_valid: 动作是否有效
            generated_element: 生成的元素
            old_boundary: 执行动作前的边界

        Returns:
            float: 奖励值
        """
        # 无效动作的惩罚
        if not action_was_valid:
            return -0.1

        # 检查是否生成了最后一个元素（完成网格）
        if self._is_terminated():
            return 10.0

        # 计算三个奖励组成部分 mt = η_e + η_b + μ_t
        eta_e = self._calculate_element_quality(generated_element)
        eta_b = self._calculate_boundary_quality(generated_element, old_boundary)
        mu_t = self._calculate_density_reward(generated_element)

        return eta_e + eta_b + mu_t

    def _calculate_element_quality(self, element):
        """
        计算元素质量 η_e，实现公式(7)

        Args:
            element: 四边形元素的四个顶点

        Returns:
            float: 元素质量值
        """
        if element is None or len(element) != 4:
            return 0.0

        # 计算边长
        edges = []
        for i in range(4):
            v1 = element[i]
            v2 = element[(i + 1) % 4]
            edge_length = self._euclidean_distance(v1, v2)
            edges.append(edge_length)

        # 计算对角线长度
        diag1 = self._euclidean_distance(element[0], element[2])
        diag2 = self._euclidean_distance(element[1], element[3])
        max_diagonal = max(diag1, diag2)

        # 计算边质量 q_edge
        min_edge = min(edges)
        q_edge = (math.sqrt(2) * min_edge) / max_diagonal if max_diagonal > 0 else 0

        # 计算内角
        angles = []
        for i in range(4):
            v_prev = element[(i - 1) % 4]
            v_curr = element[i]
            v_next = element[(i + 1) % 4]
            angle = self._calculate_angle(v_prev, v_curr, v_next)
            angles.append(angle)

        # 计算角度质量 q_angle
        min_angle = min(angles)
        max_angle = max(angles)
        q_angle = min_angle / max_angle if max_angle > 0 else 0

        # 元素质量
        eta_e = math.sqrt(q_edge * q_angle)
        return min(1.0, max(0.0, eta_e))

    def _calculate_boundary_quality(self, element, old_boundary):
        """
        计算剩余边界质量 η_b，实现公式(8)

        Args:
            element: 生成的元素
            old_boundary: 旧边界

        Returns:
            float: 边界质量值（-1到0之间）
        """
        # 简化实现：基于新形成的角度
        M_angle = self.M_angle  # 最小角度阈值

        # 计算新形成的角度（这里简化处理）
        # 在实际实现中，需要分析新边界的角度变化
        min_angle = 90.0  # 默认值，实际需要从边界几何计算

        # 角度质量部分
        angle_quality = min(min_angle, M_angle) / M_angle

        # 距离质量部分（如果有新顶点添加）
        q_dist = 1.0  # 简化假设

        eta_b = math.sqrt(angle_quality) * q_dist - 1
        return max(-1.0, min(0.0, eta_b))

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
        element_area = self._calculate_polygon_area(element)

        # 获取边界信息计算最小和最大面积
        vertices = self.boundary.get_vertices()
        if len(vertices) < 2:
            return 0.0

        # 计算边长范围
        edge_lengths = []
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            edge_lengths.append(self._euclidean_distance(v1, v2))

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

    def _calculate_polygon_area(self, vertices):
        """
        使用鞋带公式计算多边形面积

        Args:
            vertices: 顶点列表

        Returns:
            float: 多边形面积
        """
        if len(vertices) < 3:
            return 0.0

        area = 0.0
        n = len(vertices)

        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        return abs(area) / 2.0

    def _is_terminated(self):
        """
        判断是否完成网格生成（边界变成四边形或更少顶点）

        Returns:
            bool: 是否终止
        """
        vertices = self.boundary.get_vertices()
        return len(vertices) <= 4

    def render(self):
        """可视化当前状态（可选实现）"""
        pass

    def close(self):
        """清理资源"""
        pass
