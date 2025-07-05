"""
强化学习模块

本模块提供网格生成的强化学习相关功能。
主要包含SAC算法实现、神经网络结构、经验回放缓冲区和训练环境。

主要类:
    SACAgent: SAC(Soft Actor-Critic)强化学习算法的实现
    Actor: 策略网络
    Critic: Q值网络
    ReplayBuffer: 经验回放缓冲区
    PrioritizedReplayBuffer: 优先级经验回放缓冲区
    MeshEnv: 网格生成环境

主要函数:
    create_replay_buffer: 根据配置创建经验回放缓冲区的工厂函数
    get_buffer_info: 获取缓冲区信息
    load_config: 加载配置文件

用法示例:
    from src.rl import SACAgent, MeshEnv, create_replay_buffer
    from src.rl.config import load_config

    # 创建环境和智能体
    env = MeshEnv(boundary)
    agent = SACAgent(state_dim, action_dim, max_action, device)

    # 根据配置创建缓冲区（自动选择普通或优先级回放）
    replay_buffer = create_replay_buffer()

    # 或手动指定类型
    per_buffer = create_replay_buffer(buffer_type="prioritized", capacity=100000)

    # 训练
    state, _ = env.reset()
    for step in range(total_steps):
        action = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)

        if len(replay_buffer) > batch_size:
            # SAC agent会自动检测缓冲区类型并处理PER的权重和优先级更新
            agent.train(replay_buffer, batch_size)

        state = next_state if not done else env.reset()[0]
"""

from .agent.sac_agent import SACAgent
from .agent.network import Actor, Critic
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .buffer_factory import create_replay_buffer, get_buffer_info
from .environment import MeshEnv
from .config import load_config

# 定义模块的公共API
__all__ = [
    'SACAgent',
    'Actor',
    'Critic',
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'create_replay_buffer',
    'get_buffer_info',
    'MeshEnv',
    'load_config'
]

# 版本信息
__version__ = '1.0.0'

# 模块作者信息
__author__ = 'ZhuoQiuMcgill'