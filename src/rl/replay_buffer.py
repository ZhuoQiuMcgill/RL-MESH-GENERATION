import numpy as np
import random
from collections import deque


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


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    优先级经验回放缓冲区

    继承自ReplayBuffer，增加了基于TD误差的优先级采样功能。
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        初始化优先级经验回放缓冲区

        Args:
            capacity (int): 缓冲区最大容量
            alpha (float): 优先级指数，控制优先级的强度
            beta_start (float): 重要性采样权重的初始值
            beta_frames (int): beta从beta_start增长到1.0所需的帧数
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        # 存储优先级，使用简单列表实现
        self.priorities = deque(maxlen=capacity)
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
        super().add(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        """
        基于优先级采样经验

        Args:
            batch_size (int): 批次大小

        Returns:
            tuple: (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区大小({len(self.buffer)})小于批次大小({batch_size})")

        # 计算采样概率
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()

        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # 获取经验
        batch = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        beta = self._get_beta()
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()  # 归一化权重

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
            indices (list): 要更新的经验索引
            priorities (list): 新的优先级值
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def _get_beta(self):
        """
        计算当前的beta值

        Returns:
            float: 当前的beta值
        """
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def clear(self):
        """
        清空缓冲区和优先级
        """
        super().clear()
        self.priorities.clear()
        self.max_priority = 1.0
        self.frame = 1