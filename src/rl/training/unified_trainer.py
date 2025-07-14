"""
统一训练器接口

提供统一的训练器接口，根据配置自动选择合适的SAC实现
"""
import os
from typing import Dict, Any, Optional, Union

from .base_trainer import BaseTrainer
from .custom_sac_trainer import CustomSACTrainer
from .sb3_sac_trainer import SB3SACTrainer, SB3_AVAILABLE
from ..config import load_config
from src.geometry import Boundary


class UnifiedTrainer:
    """
    统一训练器

    根据配置自动选择合适的SAC实现，提供统一的接口
    """

    def __init__(self, boundary_source: Union[Boundary, str, Dict[str, str]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        初始化统一训练器

        Args:
            boundary_source: 边界数据源
            config: 配置字典
            device: 训练设备
        """
        # 加载配置
        self.config = config if config is not None else load_config()

        # 确定SAC后端类型
        sac_backend_config = self.config.get("sac_backend", {})
        self.sac_backend = sac_backend_config.get("type", "custom")

        # 验证后端类型
        if self.sac_backend not in ["custom", "sb3"]:
            raise ValueError(f"不支持的SAC后端类型: {self.sac_backend}. "
                             f"支持的类型: 'custom', 'sb3'")

        # 检查SB3可用性
        if self.sac_backend == "sb3" and not SB3_AVAILABLE:
            print("警告: stable_baselines3未安装，将回退到自制SAC实现")
            self.sac_backend = "custom"

        print(f"SAC后端: {self.sac_backend}")

        # 创建具体的训练器实例
        if self.sac_backend == "sb3":
            self.trainer = SB3SACTrainer(
                boundary_source=boundary_source,
                config=self.config,
                device=device
            )
        else:
            self.trainer = CustomSACTrainer(
                boundary_source=boundary_source,
                config=self.config,
                device=device
            )

    def __getattr__(self, name):
        """
        代理方法调用到具体的训练器实例

        Args:
            name: 方法或属性名

        Returns:
            具体训练器的方法或属性
        """
        return getattr(self.trainer, name)

    def get_backend_type(self) -> str:
        """
        获取当前使用的SAC后端类型

        Returns:
            str: 后端类型 ("custom" 或 "sb3")
        """
        return self.sac_backend

    def get_trainer_info(self) -> Dict[str, Any]:
        """
        获取训练器信息

        Returns:
            Dict[str, Any]: 训练器信息
        """
        return {
            "backend_type": self.sac_backend,
            "trainer_class": self.trainer.__class__.__name__,
            "device": str(self.trainer.device),
            "state_dim": getattr(self.trainer, 'state_dim', None),
            "action_dim": getattr(self.trainer, 'action_dim', None),
            "boundary_vertices": len(self.trainer.initial_boundary.get_vertices()) if hasattr(self.trainer,
                                                                                              'initial_boundary') else None
        }

    def train(self, **kwargs) -> Dict[str, Any]:
        """
        执行训练

        Args:
            **kwargs: 训练参数

        Returns:
            Dict[str, Any]: 训练统计信息
        """
        return self.trainer.train(**kwargs)

    def start_training_async(self, **kwargs) -> str:
        """
        异步启动训练

        Args:
            **kwargs: 训练参数

        Returns:
            str: 训练ID
        """
        return self.trainer.start_training_async(**kwargs)

    def stop_training(self):
        """停止训练"""
        return self.trainer.stop_training()

    def get_training_status(self) -> Dict[str, Any]:
        """
        获取训练状态

        Returns:
            Dict[str, Any]: 训练状态信息
        """
        status = self.trainer.get_training_status()
        # 添加后端信息
        status['backend_type'] = self.sac_backend
        return status

    def load_boundary(self, boundary_source: Union[Boundary, str, Dict[str, str]]):
        """
        加载新边界

        Args:
            boundary_source: 边界数据源
        """
        return self.trainer.load_boundary(boundary_source)

    def add_episode_callback(self, callback):
        """
        添加episode完成回调

        Args:
            callback: 回调函数
        """
        return self.trainer.add_episode_callback(callback)

    def add_step_callback(self, callback):
        """
        添加训练步骤回调

        Args:
            callback: 回调函数
        """
        return self.trainer.add_step_callback(callback)

    def save_model(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        return self.trainer._save_model(path)

    def load_model(self, path: str):
        """
        加载模型

        Args:
            path: 模型路径
        """
        return self.trainer._load_model(path)


# 便捷函数
def create_trainer(boundary_source: Union[Boundary, str, Dict[str, str]] = None,
                   config: Optional[Dict[str, Any]] = None,
                   device: Optional[str] = None,
                   backend: Optional[str] = None) -> UnifiedTrainer:
    """
    创建训练器的便捷函数

    Args:
        boundary_source: 边界数据源
        config: 配置字典
        device: 训练设备
        backend: 强制指定后端类型 ("custom" 或 "sb3")

    Returns:
        UnifiedTrainer: 统一训练器实例
    """
    # 如果指定了后端，临时修改配置
    if backend is not None:
        if config is None:
            config = load_config()
        else:
            config = config.copy()  # 避免修改原配置

        if "sac_backend" not in config:
            config["sac_backend"] = {}
        config["sac_backend"]["type"] = backend

    return UnifiedTrainer(
        boundary_source=boundary_source,
        config=config,
        device=device
    )


def get_available_backends() -> Dict[str, bool]:
    """
    获取可用的SAC后端

    Returns:
        Dict[str, bool]: 后端可用性
    """
    return {
        "custom": True,  # 自制SAC总是可用
        "sb3": SB3_AVAILABLE  # SB3依赖于是否安装了stable_baselines3
    }
