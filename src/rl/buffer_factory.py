"""
经验回放缓冲区工厂函数

提供简单的接口来根据配置创建不同类型的缓冲区
"""

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .config import load_config


def create_replay_buffer(config=None, capacity=None, buffer_type=None):
    """
    根据配置创建经验回放缓冲区

    Args:
        config (dict, optional): 配置字典，如果为None则从config.yaml加载
        capacity (int, optional): 缓冲区容量，如果为None则从配置中读取
        buffer_type (str, optional): 缓冲区类型，如果为None则从配置中读取

    Returns:
        ReplayBuffer or PrioritizedReplayBuffer: 创建的缓冲区实例

    Raises:
        ValueError: 当buffer_type不是"normal"或"prioritized"时
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

    else:
        raise ValueError(f"不支持的缓冲区类型: {buffer_type}. 支持的类型: 'normal', 'prioritized'")


def get_buffer_info(replay_buffer):
    """
    获取缓冲区的详细信息

    Args:
        replay_buffer: 缓冲区实例

    Returns:
        dict: 包含缓冲区类型和统计信息的字典
    """
    buffer_type = "prioritized" if hasattr(replay_buffer, 'update_priorities') else "normal"
    stats = replay_buffer.get_statistics()

    return {
        "type": buffer_type,
        "statistics": stats
    }
