# 导入智能体相关类
try:
    from .agent.sac_agent import SACAgent
except ImportError:
    # 如果自制SAC未实现，则跳过
    SACAgent = None

from .agent.sb3_sac_agent import SB3SACAgent

# 导入网络结构
try:
    from .agent.network import Actor, Critic
except ImportError:
    # 如果网络结构未实现，则跳过
    Actor = None
    Critic = None

# 导入缓冲区相关类
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .buffer_factory import create_replay_buffer, get_buffer_info, NoReplayBuffer

# 导入训练器
from .training.sb3_sac_trainer import SB3SACTrainer

# 导入环境
from .environment import MeshEnv

# 导入配置加载函数
from .config import load_config

# 定义模块的公共API
__all__ = [
    # 智能体
    'SB3SACAgent',
    'SB3SACTrainer',

    # 经验回放缓冲区
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'NoReplayBuffer',
    'create_replay_buffer',
    'get_buffer_info',

    # 环境
    'MeshEnv',

    # 配置
    'load_config'
]

# 条件导入的类（如果实现了则添加到__all__中）
if SACAgent is not None:
    __all__.append('SACAgent')

if Actor is not None and Critic is not None:
    __all__.extend(['Actor', 'Critic'])

# 版本信息
__version__ = '1.1.0'

# 模块作者信息
__author__ = 'ZhuoQiuMcgill'

# 模块说明
__description__ = '网格生成强化学习模块，提供SAC算法和训练环境'
