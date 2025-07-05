import numpy as np
import random
from collections import deque


class SumTree:
    """
    高效的Sum Tree数据结构，用于优先级经验回放的快速采样

    这个数据结构允许O(log n)的采样和更新操作
    """

    def __init__(self, capacity):
        """
        初始化Sum Tree

        Args:
            capacity (int): 缓冲区容量
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = 0

    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """检索具有给定累积优先级的叶节点"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """返回所有优先级的总和"""
        return self.tree[0]

    def add(self, p, data):
        """
        添加新的经验数据

        Args:
            p (float): 优先级
            data: 经验数据
        """
        idx = self.pending_idx + self.capacity - 1

        self.data[self.pending_idx] = data
        self.update(idx, p)

        self.pending_idx += 1
        if self.pending_idx >= self.capacity:
            self.pending_idx = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """
        更新指定索引的优先级

        Args:
            idx (int): tree中的索引
            p (float): 新的优先级
        """
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """
        根据累积优先级获取数据

        Args:
            s (float): 累积优先级值

        Returns:
            tuple: (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class ReplayBuffer:
    """
    经验回放缓冲区

    用于存储和采样强化学习过程中的经验数据，支持SAC算法的训练过程。
    """

    def __init__(self, capacity):
        """
        初始化经验回放缓冲区

        Args:
            capacity (int): 缓冲区的最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """
        向缓冲区添加一条经验

        Args:
            state (np.ndarray): 当前状态
            action (np.ndarray): 执行的动作
            reward (float): 获得的奖励
            next_state (np.ndarray): 下一个状态
            done (bool): 是否结束episode
        """
        # 将经验存储为元组
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        从缓冲区中随机采样一个批次的经验

        Args:
            batch_size (int): 批次大小

        Returns:
            tuple: 包含(states, actions, rewards, next_states, dones)的元组，
                   每个元素都是numpy数组
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区大小({len(self.buffer)})小于批次大小({batch_size})")

        # 随机采样
        batch = random.sample(self.buffer, batch_size)

        # 将批次数据转换为numpy数组
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        返回缓冲区当前存储的经验数量

        Returns:
            int: 当前缓冲区大小
        """
        return len(self.buffer)

    def clear(self):
        """
        清空缓冲区
        """
        self.buffer.clear()
        self.position = 0

    def is_full(self):
        """
        检查缓冲区是否已满

        Returns:
            bool: 如果缓冲区已满返回True，否则返回False
        """
        return len(self.buffer) == self.capacity

    def get_capacity(self):
        """
        获取缓冲区的最大容量

        Returns:
            int: 缓冲区最大容量
        """
        return self.capacity

    def get_statistics(self):
        """
        获取缓冲区的统计信息

        Returns:
            dict: 包含缓冲区统计信息的字典
        """
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "avg_reward": 0.0,
                "avg_episode_length": 0.0
            }

        rewards = [experience[2] for experience in self.buffer]
        dones = [experience[4] for experience in self.buffer]

        # 计算平均奖励
        avg_reward = np.mean(rewards)

        # 计算平均episode长度（近似）
        done_count = sum(dones)
        avg_episode_length = len(self.buffer) / max(done_count, 1)

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "avg_reward": avg_reward,
            "avg_episode_length": avg_episode_length
        }


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区

    使用Sum Tree实现高效的优先级采样，支持基于TD误差的重要性采样。
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, epsilon=1e-6):
        """
        初始化优先级经验回放缓冲区

        Args:
            capacity (int): 缓冲区最大容量
            alpha (float): 优先级指数，控制优先级的强度 (0=uniform, 1=full prioritization)
            beta_start (float): 重要性采样权重的初始值
            beta_frames (int): beta从beta_start增长到1.0所需的帧数
            epsilon (float): 小常数，防止优先级为0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1

        # 使用Sum Tree实现高效的优先级采样
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """
        向缓冲区添加经验，新经验使用最大优先级

        Args:
            state (np.ndarray): 当前状态
            action (np.ndarray): 执行的动作
            reward (float): 获得的奖励
            next_state (np.ndarray): 下一个状态
            done (bool): 是否结束episode
        """
        experience = (state, action, reward, next_state, done)

        # 新经验使用最大优先级，确保至少被采样一次
        priority = (self.max_priority + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """
        基于优先级采样经验

        Args:
            batch_size (int): 批次大小

        Returns:
            tuple: (states, actions, rewards, next_states, dones, weights, indices)
                  其中weights是重要性采样权重，indices是在tree中的索引
        """
        if self.tree.n_entries < batch_size:
            raise ValueError(f"缓冲区大小({self.tree.n_entries})小于批次大小({batch_size})")

        batch = []
        indices = []
        priorities = []

        # 计算优先级采样的区间
        priority_segment = self.tree.total() / batch_size

        # 采样
        for i in range(batch_size):
            # 在每个区间内随机采样
            a = priority_segment * i
            b = priority_segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # 计算重要性采样权重
        beta = self._get_beta()
        priorities = np.array(priorities)

        # 防止除零错误
        sampling_probabilities = priorities / (self.tree.total() + 1e-7)
        weights = (self.tree.n_entries * sampling_probabilities) ** (-beta)

        # 归一化权重，使最大权重为1
        weights = weights / (weights.max() + 1e-7)

        # 转换为numpy数组
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        """
        更新指定经验的优先级

        Args:
            indices (list): tree中的索引列表
            priorities (list): 新的优先级值列表（通常是TD误差的绝对值）
        """
        for idx, priority in zip(indices, priorities):
            # 确保优先级为正数并应用alpha
            priority = abs(priority) + self.epsilon
            priority = priority ** self.alpha

            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def _get_beta(self):
        """
        计算当前的beta值，随训练进程线性增长到1.0

        Returns:
            float: 当前的beta值
        """
        progress = min(1.0, self.frame / self.beta_frames)
        beta = self.beta_start + (1.0 - self.beta_start) * progress
        self.frame += 1
        return beta

    def __len__(self):
        """
        返回缓冲区当前存储的经验数量

        Returns:
            int: 当前缓冲区大小
        """
        return self.tree.n_entries

    def clear(self):
        """
        清空缓冲区和优先级
        """
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.frame = 1

    def get_capacity(self):
        """
        获取缓冲区的最大容量

        Returns:
            int: 缓冲区最大容量
        """
        return self.capacity

    def get_statistics(self):
        """
        获取缓冲区的统计信息

        Returns:
            dict: 包含缓冲区统计信息的字典
        """
        if self.tree.n_entries == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "avg_priority": 0.0,
                "max_priority": self.max_priority,
                "beta": self._get_beta()
            }

        # 计算平均优先级
        avg_priority = self.tree.total() / self.tree.n_entries if self.tree.n_entries > 0 else 0

        return {
            "size": self.tree.n_entries,
            "capacity": self.capacity,
            "utilization": self.tree.n_entries / self.capacity,
            "avg_priority": avg_priority,
            "max_priority": self.max_priority,
            "beta": self._get_beta()
        }
