"""
经验回放缓冲区工厂函数

提供简单的接口来根据配置创建不同类型的缓冲区
"""

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .config import load_config


class NoReplayBuffer:
    """
    空缓冲区类，用于禁用经验回放的在线学习模式

    当replay buffer类型设置为"off"时使用，提供与正常缓冲区
    相同的接口但不实际存储任何数据
    """

    def __init__(self):
        """初始化空缓冲区"""
        self.capacity = 0
        self._size = 0

    def add(self, state, action, reward, next_state, done):
        """
        添加经验（空操作）

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束episode
        """
        # 在线学习模式下不存储经验
        pass

    def sample(self, batch_size):
        """
        采样经验（抛出异常）

        Args:
            batch_size: 批次大小

        Raises:
            RuntimeError: 在线学习模式下不支持采样操作
        """
        raise RuntimeError("在线学习模式(replay_buffer.type='off')下不支持从缓冲区采样")

    def __len__(self):
        """
        返回缓冲区大小（始终为0）

        Returns:
            int: 缓冲区大小（0）
        """
        return 0

    def clear(self):
        """清空缓冲区（空操作）"""
        pass

    def is_full(self):
        """
        检查缓冲区是否已满（始终返回False）

        Returns:
            bool: 始终返回False
        """
        return False

    def get_capacity(self):
        """
        获取缓冲区最大容量

        Returns:
            int: 缓冲区最大容量（0）
        """
        return self.capacity

    def get_statistics(self):
        """
        获取缓冲区统计信息

        Returns:
            dict: 包含缓冲区统计信息的字典
        """
        return {
            "size": 0,
            "capacity": 0,
            "utilization": 0.0,
            "mode": "online_learning"
        }


def create_replay_buffer(config=None, capacity=None, buffer_type=None):
    """
    根据配置创建经验回放缓冲区

    Args:
        config (dict, optional): 配置字典，如果为None则从config.yaml加载
        capacity (int, optional): 缓冲区容量，如果为None则从配置中读取
        buffer_type (str, optional): 缓冲区类型，如果为None则从配置中读取

    Returns:
        ReplayBuffer, PrioritizedReplayBuffer, 或 NoReplayBuffer: 创建的缓冲区实例

    Raises:
        ValueError: 当buffer_type不是"normal", "prioritized", 或 "off"时
    """
    # 加载配置
    if config is None:
        config = load_config()

    buffer_cfg = config.get("replay_buffer", {})
    sac_cfg = config.get("sac_agent", {})

    # 确定缓冲区参数
    if capacity is None:
        capacity = buffer_cfg.get("capacity", sac_cfg.get("buffer_size", 1000000))

    if buffer_type is None:
        buffer_type = buffer_cfg.get("type", "normal")

    # 创建缓冲区
    if buffer_type == "normal":
        return ReplayBuffer(capacity)

    elif buffer_type == "prioritized":
        per_cfg = buffer_cfg.get("prioritized", {})

        alpha = per_cfg.get("alpha", 0.6)
        beta_start = per_cfg.get("beta_start", 0.4)
        beta_frames = per_cfg.get("beta_frames", 100000)
        epsilon = per_cfg.get("epsilon", 1e-6)

        return PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
            epsilon=epsilon
        )

    elif buffer_type == "off":
        return NoReplayBuffer()

    else:
        raise ValueError(f"不支持的缓冲区类型: {buffer_type}. 支持的类型: 'normal', 'prioritized', 'off'")


def get_buffer_info(replay_buffer):
    """
    获取缓冲区的详细信息

    Args:
        replay_buffer: 缓冲区实例

    Returns:
        dict: 包含缓冲区类型和统计信息的字典
    """
    if isinstance(replay_buffer, NoReplayBuffer):
        buffer_type = "off"
    elif hasattr(replay_buffer, 'update_priorities'):
        buffer_type = "prioritized"
    else:
        buffer_type = "normal"

    stats = replay_buffer.get_statistics()

    return {
        "type": buffer_type,
        "statistics": stats
    }
