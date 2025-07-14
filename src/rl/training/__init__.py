"""
训练模块

提供统一的SAC训练接口，支持自制SAC和SB3 SAC的无缝切换
"""

from .base_trainer import BaseTrainer
from .custom_sac_trainer import CustomSACTrainer
from .unified_trainer import UnifiedTrainer, create_trainer, get_available_backends

# 尝试导入SB3训练器
try:
    from .sb3_sac_trainer import SB3SACTrainer, SB3_AVAILABLE

    __all__ = [
        'BaseTrainer',
        'CustomSACTrainer',
        'SB3SACTrainer',
        'UnifiedTrainer',
        'create_trainer',
        'get_available_backends',
        'SB3_AVAILABLE'
    ]
except ImportError:
    # SB3不可用时的fallback
    SB3SACTrainer = None
    SB3_AVAILABLE = False
    __all__ = [
        'BaseTrainer',
        'CustomSACTrainer',
        'UnifiedTrainer',
        'create_trainer',
        'get_available_backends',
        'SB3_AVAILABLE'
    ]

# 版本信息
__version__ = '2.0.0'

# 模块作者信息
__author__ = 'ZhuoQiuMcgill'

# 模块描述
__doc__ = """
统一训练模块 v2.0

主要特性:
- 统一的训练器接口，支持多种SAC实现
- 自动后端选择和切换
- 模块化设计，易于扩展
- 完整的回调系统和状态管理
- 前端友好的API设计

基本用法:
    # 创建训练器（自动选择后端）
    trainer = create_trainer()

    # 手动指定后端
    trainer = create_trainer(backend="sb3")  # 或 "custom"

    # 异步训练
    training_id = trainer.start_training_async(max_timesteps=100000)

    # 获取状态
    status = trainer.get_training_status()

    # 停止训练
    trainer.stop_training()

高级用法:
    # 检查可用后端
    backends = get_available_backends()

    # 切换后端
    if backends["sb3"]:
        trainer = create_trainer(backend="sb3")
    else:
        trainer = create_trainer(backend="custom")

    # 添加回调
    def on_episode_complete(episode_data):
        print(f"Episode {episode_data['episode']} 完成")

    trainer.add_episode_callback(on_episode_complete)
"""